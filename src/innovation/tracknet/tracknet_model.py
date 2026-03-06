"""
TrackNetV4 — ball tracking model for squash.

Architecture: TrackNetV2/V3 U-Net backbone (multi-scale CBAM attention) extended
with the V4 motion prompt layer (Raj et al., arXiv:2409.14543).

The key V4 innovation is a lightweight side branch that converts frame-differencing
maps into spatial attention weights, which are fused with encoder features at every
scale.  This lets the network explicitly "see" where things are moving, dramatically
improving recall when the ball is small or dark.

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
N_FRAMES = 3          # number of frames in one input triplet
IN_CHANNELS  = N_FRAMES * 3 + 2   # 9 RGB + 2 diff = 11
OUT_CHANNELS = N_FRAMES            # one heatmap per input frame


# ── Building blocks (from TrackNetV2/V3) ───────────────────────────────────────

class Conv2DBlock(nn.Module):
    """Conv2d → BatchNorm → ReLU."""
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
    """Two successive Conv2DBlocks with 3×3 kernels."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.seq = nn.Sequential(
            Conv2DBlock(in_c, out_c, 3),
            Conv2DBlock(out_c, out_c, 3),
        )

    def forward(self, x):
        return self.seq(x)


class Double2DConv2(nn.Module):
    """Multi-scale inception-style block used in encoder down-path.

    Runs three parallel conv paths (1×1→3×3, 3×3→3×3, 5×5→3×3), concatenates
    them, fuses, then adds a residual from the 3×3 path.
    """
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
        out = self.fuse(torch.cat([p1, p2, p3], dim=1))
        return out + p2   # residual from the 3×3 path


class Triple2DConv(nn.Module):
    """Three successive Conv2DBlocks — used as bottleneck."""
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
    """Convolutional Block Attention Module."""
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
    at three encoder scales (64, 128, 256 channels).

    At each scale the diff map is downsampled to match the encoder feature map
    resolution, then a small Conv stack produces per-channel attention weights
    in [0, 1].  The fusion rule  feat * (1 + attn)  adds a positive bias to
    regions where motion was detected while never suppressing static regions
    (residual-style, gentler than pure multiplicative attention).
    """
    def __init__(self, n_diff=2):
        super().__init__()
        self.scale1 = nn.Sequential(       # full resolution  → 64 ch
            Conv2DBlock(n_diff, 16, 3),
            Conv2DBlock(16, 64, 3),
            nn.Sigmoid(),
        )
        self.scale2 = nn.Sequential(       # ×0.5 resolution  → 128 ch
            Conv2DBlock(n_diff, 32, 3),
            Conv2DBlock(32, 128, 3),
            nn.Sigmoid(),
        )
        self.scale3 = nn.Sequential(       # ×0.25 resolution → 256 ch
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
    """Full TrackNetV4 model.

    Parameters
    ----------
    in_dim  : int  — visual channels (default 9 = 3 frames × RGB)
    out_dim : int  — output heatmap channels (default 3 = one per input frame)
    n_diff  : int  — number of difference map channels (default 2)
    """
    def __init__(self, in_dim=9, out_dim=3, n_diff=2):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────────────
        self.down1 = Double2DConv2(in_dim, 64)
        self.down2 = Double2DConv2(64, 128)
        self.down3 = Double2DConv2(128, 256)
        self.pool  = nn.MaxPool2d(2, stride=2)

        # ── Bottleneck ────────────────────────────────────────────────────────
        self.bottleneck = Triple2DConv(256, 512)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.up1 = Double2DConv(512 + 256, 256)
        self.up2 = Double2DConv(256 + 128, 128)
        self.up3 = Double2DConv(128 + 64,  64)
        self.head = nn.Conv2d(64, out_dim, kernel_size=1)
        self.sig  = nn.Sigmoid()

        # ── CBAM (skip connections + decoder outputs) ─────────────────────────
        self.cbam_skip1 = CBAM(256)
        self.cbam_skip2 = CBAM(128)
        self.cbam_skip3 = CBAM(64)
        self.cbam_dec1  = CBAM(256)
        self.cbam_dec2  = CBAM(128)
        self.cbam_dec3  = CBAM(64)

        # ── V4: Motion Prompt Layer ───────────────────────────────────────────
        self.motion_prompt = MotionPromptLayer(n_diff)

    def forward(self, x):
        """
        x : (B, 11, H, W)  channels 0-8 = visual, 9-10 = diff maps
        returns : (B, 3, H, W) heatmaps in [0, 1]
        """
        visual = x[:, :9]    # (B, 9, H, W)
        diff   = x[:, 9:]    # (B, 2, H, W)

        # Motion attention maps at 3 scales
        m1, m2, m3 = self.motion_prompt(diff)

        # Encoder — fuse motion attention after each block
        e1 = self.down1(visual)          # (B, 64,  H,    W)
        e1 = e1 * (1.0 + m1)            # motion-aware fusion

        e2 = self.down2(self.pool(e1))   # (B, 128, H/2,  W/2)
        e2 = e2 * (1.0 + m2)

        e3 = self.down3(self.pool(e2))   # (B, 256, H/4,  W/4)
        e3 = e3 * (1.0 + m3)

        bn = self.bottleneck(self.pool(e3))   # (B, 512, H/8,  W/8)

        # Decoder — CBAM on both skip and decoder feature maps before concat
        s3 = self.cbam_skip1(e3)
        d = self.up1(torch.cat([F.interpolate(bn, scale_factor=2, mode="bilinear",
                                               align_corners=False), s3], dim=1))
        d = self.cbam_dec1(d)

        s2 = self.cbam_skip2(e2)
        d = self.up2(torch.cat([F.interpolate(d, scale_factor=2, mode="bilinear",
                                               align_corners=False), s2], dim=1))
        d = self.cbam_dec2(d)

        s1 = self.cbam_skip3(e1)
        d = self.up3(torch.cat([F.interpolate(d, scale_factor=2, mode="bilinear",
                                               align_corners=False), s1], dim=1))
        d = self.cbam_dec3(d)

        return self.sig(self.head(d))   # (B, 3, H, W)


# ── Utilities ──────────────────────────────────────────────────────────────────

def build_input_tensor(frames_bgr: list, device="cpu") -> torch.Tensor:
    """Convert a list of 3 BGR frames (numpy H×W×3, uint8) into a (1, 11, 288, 512)
    model input tensor.

    Parameters
    ----------
    frames_bgr : list of 3 numpy arrays — [frame_{t-1}, frame_t, frame_{t+1}]
    device     : torch device string

    Returns
    -------
    tensor : (1, 11, 288, 512) float32 on `device`
    """
    import cv2
    import numpy as np

    resized = [cv2.resize(f, (WIDTH, HEIGHT)) for f in frames_bgr]

    # Stack RGB channels: shape (9, H, W)
    visual = np.concatenate(
        [np.moveaxis(f, -1, 0) for f in resized], axis=0
    ).astype(np.float32) / 255.0

    # Grayscale difference maps: shape (2, H, W)
    g = [cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
         for f in resized]
    diff = np.stack([
        np.abs(g[1] - g[0]),   # |t - t-1|
        np.abs(g[2] - g[1]),   # |t+1 - t|
    ], axis=0)

    inp = np.concatenate([visual, diff], axis=0)   # (11, H, W)
    return torch.from_numpy(inp).unsqueeze(0).to(device)


def heatmap_to_pixel(heatmap: torch.Tensor, orig_h: int, orig_w: int,
                     threshold: float = 0.5):
    """Extract (x, y, confidence) from a single (H, W) heatmap.

    Returns (cx_orig, cy_orig, conf) in original video pixel coordinates,
    or None if no blob is found above `threshold`.
    """
    import cv2
    import numpy as np

    hm = heatmap.cpu().numpy()
    conf = float(hm.max())
    if conf < threshold:
        return None

    binary = (hm >= threshold).astype(np.uint8) * 255
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    # Largest blob
    cnt = max(cnts, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cx_model = x + w / 2.0
    cy_model = y + h / 2.0

    # Scale back to original resolution
    cx_orig = cx_model * orig_w / WIDTH
    cy_orig = cy_model * orig_h / HEIGHT

    return cx_orig, cy_orig, conf


if __name__ == "__main__":
    model = TrackNetV4()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"TrackNetV4  params: {n_params:,}")
    dummy = torch.zeros(1, 11, HEIGHT, WIDTH)
    out = model(dummy)
    print(f"Input:  {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
