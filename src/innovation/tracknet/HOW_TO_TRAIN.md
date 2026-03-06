# Training TrackNetV4 on Google Colab

Train on a free T4 GPU in ~30 minutes without uploading your whole repo.

---

## Files to upload (4 total)

| File | Where it lives locally |
|------|------------------------|
| `tracknet_model.py` | `src/innovation/tracknet/tracknet_model.py` |
| `tracknet_dataset.py` | `src/innovation/tracknet/tracknet_dataset.py` |
| `tracknet_train.py` | `src/innovation/tracknet/tracknet_train.py` |
| `ball_labels.csv` | `assets/ball_labels.csv` |

Your video file(s) (e.g. `men360.mp4`) also need to be reachable in Colab — see Step 3.

---

## Step-by-step

### Step 1 — Open a new Colab notebook

Go to [colab.research.google.com](https://colab.research.google.com) and create a new notebook.

Change the runtime to GPU:
**Runtime > Change runtime type > T4 GPU > Save**

---

### Step 2 — Upload the 4 Python/CSV files

In the left sidebar click the **Files** icon (folder), then the **Upload** button.
Upload all four files listed above. They will land in `/content/`.

---

### Step 3 — Get the video file into Colab

**Option A — Upload directly** (simplest, works for files under ~1 GB):
Upload the video the same way as Step 2. It will be at `/content/men360.mp4`.

**Option B — Mount Google Drive** (better for large files or multiple videos):
Put the video in your Drive, then run this in a Colab cell:

```python
from google.colab import drive
drive.mount('/content/drive')
# Video will be at something like:
# /content/drive/MyDrive/squash/men360.mp4
```

Set `video_dir` accordingly in Step 5.

---

### Step 4 — Install dependencies

Run this in a Colab cell:

```python
!pip install torch torchvision opencv-python-headless pandas numpy --quiet
```

---

### Step 5 — Train

Paste this into a code cell and run it:

```python
import sys
sys.path.insert(0, '/content')

from tracknet_train import train

train(
    label_path = '/content/ball_labels.csv',
    video_dir  = '/content',          # folder containing your video file(s)
    epochs     = 50,
    batch_size = 8,
    lr         = 1e-3,
    out_dir    = '/content/weights',
)
```

If your video is in Drive, change `video_dir` to the Drive path, e.g.:
```python
video_dir = '/content/drive/MyDrive/squash'
```

Training prints loss and val-loss each epoch. `best.pt` is saved whenever val-loss improves.

---

### Step 6 — Download the trained weights

After training finishes, run:

```python
from google.colab import files
files.download('/content/weights/best.pt')
```

Save `best.pt` to `assets/weights/tracknet_best.pt` in your local repo.

---

### Step 7 — Run inference locally

```bash
python src/innovation/tracknet/tracknet_infer.py
```

The script reads `assets/weights/tracknet_best.pt` by default and writes ball positions to `output/`.

---

## Tips

- **Free T4 GPU** is enough. You don't need Colab Pro for this model size.
- **Session timeout**: Colab disconnects after ~90 min of inactivity. If training takes longer, add a keep-alive snippet in your browser console:
  ```javascript
  setInterval(() => console.log('ping'), 60000);
  ```
- **More labels = better accuracy.** Aim for 300+ visible-ball labels spread across different rallies and shot types before training seriously.
- **Resuming**: pass `checkpoint='/content/weights/last.pt'` to `train()` to continue from a previous run instead of starting over.
- **Multiple videos**: add labels for more videos to `ball_labels.csv` (via `tracknet_label.py`), upload all the video files to `/content/`, and re-run Step 5. The dataset automatically picks up every video listed in the CSV.

---

## Quick reference — local labeling

Label frames before going to Colab:

```bash
python src/innovation/tracknet/tracknet_label.py --start 0 --end 3000 --step 5
```

Controls: **left-click** = label ball and advance, **A/D** = prev/next frame, **U** = not visible, **Del** = clear, **S** = save, **Q** = quit.
