"""
TrackNetV4 — ball tracking model for squash.

Architecture: TrackNetV2/V3 U-Net backbone (multi-scale CBAM attention) extended
with the V4 motion prompt layer (Raj et al., arXiv:2409.14543).

Input tensor:  (B, 11, 288, 512)
  channels 0-8  : 3 consecutive RGB frames stacked (frame t-1, t, t+1), /255
  channels 9-10 : 2 grayscale difference maps, /255
                  ch9  = |gray(frame_t)   - gray(frame_{t-1})|
                  ch10 = |gray(frame_{t+1}) - gray(frame_t)  |

Output tensor: (B, 3, 288, 512)
  Per-frame Gaussian heatmap in [0, 1].
  Channel 1 (centre frame) is used for single-frame ball position prediction
  during inference.

Model resolution: HEIGHT=288, WIDTH=512 (matches original TrackNet standard).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# ── Constants ──────────────────────────────────────────────────────────────────
HEIGHT = 288
WIDTH  = 512
N_FRAMES     = 3
IN_CHANNELS  = N_FRAMES * 3 + 2   # 9 RGB + 2 diff = 11
OUT_CHANNELS = N_FRAMES            # one heatmap per input frame


# ── Building blocks ────────────────────────────────────────────────────────────

class Conv2DBlock(nn.Module):
    def __init__(self, in_c, out_c, kernel_size, padding="same"):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size, padding=padding, bias=True),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class Double2DConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            Conv2DBlock(in_c, out_c, 3),
            Conv2DBlock(out_c, out_c, 3),
        )

    def forward(self, x):
        return self.seq(x)


class Double2DConv2(nn.Module):
    """Multi-scale inception-style encoder block."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.path1 = nn.Sequential(Conv2DBlock(in_c, out_c, 1), Conv2DBlock(out_c, out_c, 3))
        self.path2 = nn.Sequential(Conv2DBlock(in_c, out_c, 3), Conv2DBlock(out_c, out_c, 3))
        self.path3 = nn.Sequential(Conv2DBlock(in_c, out_c, 5), Conv2DBlock(out_c, out_c, 3))
        self.fuse  = Conv2DBlock(out_c * 3, out_c, 3)

    def forward(self, x):
        p1 = self.path1(x)
        p2 = self.path2(x)
        p3 = self.path3(x)
        return self.fuse(torch.cat([p1, p2, p3], dim=1)) + p2


class Triple2DConv(nn.Module):
    """Three Conv2DBlocks — used as bottleneck."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            Conv2DBlock(in_c, out_c, 3),
            Conv2DBlock(out_c, out_c, 3),
            Conv2DBlock(out_c, out_c, 3),
        )

    def forward(self, x):
        return self.seq(x)


class ChannelAttention(nn.Module):
    def __init__(self, channels, ratio=16):
        super().__init__()
        mid = max(1, channels // ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, mid, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid, channels, 1, bias=False),
        )
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.max = nn.AdaptiveMaxPool2d(1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        return self.sig(self.mlp(self.avg(x)) + self.mlp(self.max(x)))


class SpatialAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.sig  = nn.Sigmoid()

    def forward(self, x):
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        return self.sig(self.conv(torch.cat([avg, mx], dim=1)))


class CBAM(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.ca = ChannelAttention(channels)
        self.sa = SpatialAttention()

    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x


# ── V4 Motion Prompt Layer ─────────────────────────────────────────────────────

class MotionPromptLayer(nn.Module):
    """Converts 2-channel frame-difference maps into spatial attention weights
    at three encoder scales (64, 128, 256 channels)."""
    def __init__(self, n_diff=2):
        super().__init__()
        self.scale1 = nn.Sequential(
            Conv2DBlock(n_diff, 16, 3),
            Conv2DBlock(16, 64, 3),
            nn.Sigmoid(),
        )
        self.scale2 = nn.Sequential(
            Conv2DBlock(n_diff, 32, 3),
            Conv2DBlock(32, 128, 3),
            nn.Sigmoid(),
        )
        self.scale3 = nn.Sequential(
            Conv2DBlock(n_diff, 64, 3),
            Conv2DBlock(64, 256, 3),
            nn.Sigmoid(),
        )
        self.pool = nn.MaxPool2d(2, stride=2)

    def forward(self, diff):
        d2 = self.pool(diff)
        d4 = self.pool(d2)
        return self.scale1(diff), self.scale2(d2), self.scale3(d4)


# ── TrackNetV4 ─────────────────────────────────────────────────────────────────

class TrackNetV4(nn.Module):
    def __init__(self, in_dim=9, out_dim=3, n_diff=2):
        super().__init__()

        self.down1 = Double2DConv2(in_dim, 64)
        self.down2 = Double2DConv2(64, 128)
        self.down3 = Double2DConv2(128, 256)
        self.pool  = nn.MaxPool2d(2, stride=2)

        self.bottleneck = Triple2DConv(256, 512)

        self.up1 = Double2DConv(512 + 256, 256)
        self.up2 = Double2DConv(256 + 128, 128)
        self.up3 = Double2DConv(128 + 64,  64)
        self.head = nn.Conv2d(64, out_dim, kernel_size=1)
        self.sig  = nn.Sigmoid()

        self.cbam_skip1 = CBAM(256)
        self.cbam_skip2 = CBAM(128)
        self.cbam_skip3 = CBAM(64)
        self.cbam_dec1  = CBAM(256)
        self.cbam_dec2  = CBAM(128)
        self.cbam_dec3  = CBAM(64)

        self.motion_prompt = MotionPromptLayer(n_diff)

    def forward(self, x):
        visual = x[:, :9]
        diff   = x[:, 9:]

        m1, m2, m3 = self.motion_prompt(diff)

        e1 = self.down1(visual) * (1.0 + m1)
        e2 = self.down2(self.pool(e1)) * (1.0 + m2)
        e3 = self.down3(self.pool(e2)) * (1.0 + m3)
        bn = self.bottleneck(self.pool(e3))

        s3 = self.cbam_skip1(e3)
        d = self.up1(torch.cat([F.interpolate(bn, scale_factor=2, mode="bilinear", align_corners=False), s3], dim=1))
        d = self.cbam_dec1(d)

        s2 = self.cbam_skip2(e2)
        d = self.up2(torch.cat([F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False), s2], dim=1))
        d = self.cbam_dec2(d)

        s1 = self.cbam_skip3(e1)
        d = self.up3(torch.cat([F.interpolate(d, scale_factor=2, mode="bilinear", align_corners=False), s1], dim=1))
        d = self.cbam_dec3(d)

        return self.sig(self.head(d))


# ── Utilities ──────────────────────────────────────────────────────────────────

def build_input_tensor(frames_bgr: list, device="cpu") -> torch.Tensor:
    """Convert a list of 3 BGR frames into a (1, 11, 288, 512) model input tensor."""
    import cv2
    import numpy as np

    resized = [cv2.resize(f, (WIDTH, HEIGHT)) for f in frames_bgr]
    visual = np.concatenate(
        [np.moveaxis(f, -1, 0) for f in resized], axis=0
    ).astype(np.float32) / 255.0
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0 for f in resized]
    diff = np.stack([np.abs(g[1] - g[0]), np.abs(g[2] - g[1])], axis=0)
    inp = np.concatenate([visual, diff], axis=0)
    return torch.from_numpy(inp).unsqueeze(0).to(device)


def heatmap_to_pixel(heatmap: torch.Tensor, orig_h: int, orig_w: int,
                     threshold: float = 0.5):
    """Extract (x, y, confidence) from a (H, W) heatmap in original video coordinates."""
    import cv2
    import numpy as np

    hm   = heatmap.cpu().numpy()
    conf = float(hm.max())
    if conf < threshold:
        return None

    binary = (hm >= threshold).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cx_orig = (x + w / 2.0) * orig_w / WIDTH
    cy_orig = (y + h / 2.0) * orig_h / HEIGHT
    return cx_orig, cy_orig, conf


if __name__ == "__main__":
    model = TrackNetV4()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TrackNetV4 params: {n_params:,}")
    dummy = torch.zeros(1, 11, HEIGHT, WIDTH)
    out = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
