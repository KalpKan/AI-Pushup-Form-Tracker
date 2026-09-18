# AI Pushup Form Tracker (Python original)

Counts pushups from a video or a webcam and grades each frame's form as **good** or **bad**.
MediaPipe Pose finds 12 body landmarks (wrists, elbows, shoulders, hips, knees, ankles), a small
Keras neural network (`model/pushup_model_augmented.h5`, 36 inputs → 128 → 64 → 32 → 1) classifies
the form, and a shoulder-height state machine counts a rep only when the form was good at the top
**and** the bottom.

## Try it in your browser instead

The tracker now runs entirely in the browser (no install, no upload; works on a phone's front camera):

**https://pushups.kalpkan.com** — source: https://github.com/KalpKan/pushup-tracker-web

The web version uses the same model (converted to TensorFlow.js), the same 12 landmarks and a
line-for-line port of the rep counter, verified against this repo's Python pipeline with recorded
landmark fixtures.

## What is in this repo

| Path | What it is |
|---|---|
| `scripts/extract_keypoints.py` | runs MediaPipe Pose over `data/good_form/*.mp4` and `data/bad_form/*.mp4`, writes one CSV row per frame (36 floats + label) |
| `scripts/augment_data.py` | jitters every row by ±0.05 to make the augmented CSV |
| `scripts/train_model.py` | fits a `StandardScaler` on the training split and trains the MLP (the scaler is not saved; the web port recomputes it from the same split, `random_state=42`) |
| `scripts/real_time_pushup_form.py` | webcam tracker |
| `test_pushup_form.py` | the same tracker on a test video |
| `model/pushup_model_augmented.h5` | the trained model (220 KB) |

The training videos, the extracted CSVs (156 MB), the test videos (150 MB) and the unused
RandomForest pickle (64 MB) were removed from the tree in the "purge data files" PR to keep the
repo cloneable; they are still in the git history of `main` before that merge if ever needed.

## Run the Python version

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/real_time_pushup_form.py     # webcam
```

Note: the inference scripts feed raw landmark coordinates to the model, but the model was trained
on standardised inputs. On the held-out split the model scores 63 % with raw inputs and 95 % with
the scaler applied; the web version applies the scaler.
