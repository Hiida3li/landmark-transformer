# landmark-transformer

**Treating hand landmarks as tokens: gesture recognition with a small attention-based model.**

*Do landmarks beat pixels?* This repository benchmarks a from-scratch Transformer encoder
operating on MediaPipe hand landmarks against simpler baselines, on a dataset collected
from scratch. Everything trains on a CPU/GPU.

---

## 1. Motivation

Vision Transformers treat an image as a sequence of patch tokens. If the useful signal in a
gesture is the *configuration of the hand* rather than the pixels around it, then a much
cheaper tokenization is available: run a pretrained keypoint detector, and treat each frame's
21 hand landmarks as a single token.

This trades a 921,600-dimensional input (640x480x3) for a 63-dimensional one, a ~14,000x
reduction, at the cost of discarding everything the keypoint model does not capture. The
question this repo investigates is what that trade buys and what it costs.

Concretely:

1. Does a Transformer over landmark sequences classify gestures reliably?
2. Does attention earn its parameters, compared to a baseline with no temporal modeling?
3. What breaks when the model leaves the dataset it was trained on?

## 2. Method

### 2.1 Pipeline

```
webcam frame (480, 640, 3)
    -> MediaPipe HandLandmarker      -> (21, 3) landmarks, or None
    -> normalisation                 -> (21, 3) wrist-centred, scale-normalised
    -> flatten                       -> (63,) one token
    -> stack over T = 30 frames      -> (30, 63) one sample
    -> Transformer encoder           -> (5,) class logits
```

### 2.2 Normalisation

Raw MediaPipe landmarks are expressed in normalised *image* coordinates, so the same gesture
produces different numbers depending on where the hand sits in frame and how far it is from
the camera. Both nuisance factors are removed with two operations on the landmark vectors
`p_i` in R^3:

```
p~_i = (p_i - p_0) / || p_9 - p_0 ||
```

- **Translation.** Subtracting the wrist `p_0` moves the origin to the hand, removing
  absolute position.
- **Scaling.** Dividing by the wrist-to-middle-knuckle distance `|| p_9 - p_0 ||` removes
  apparent size, and therefore distance from the camera.

By construction, the wrist maps to the origin and the reference length maps to 1. Rotation is
deliberately *not* removed; see [Limitations](#6-limitations).

### 2.3 Model

A standard pre-norm Transformer encoder, implemented from scratch rather than via
`nn.TransformerEncoderLayer`, so that every matrix operation is visible:

| Component | Choice |
|---|---|
| Token embedding | `Linear(63 -> 64)` |
| Positional encoding | learned, `(1, 30, 64)` |
| Blocks | 2 |
| Attention heads | 4 (d_k = 16) |
| Feed-forward width | 128, GELU |
| Normalisation | pre-norm LayerNorm, residual connections |
| Pooling | mean over the 30 tokens |
| Head | `Linear(64 -> 5)` |
| Dropout | 0.1 |

Scaled dot-product attention is computed explicitly as
`softmax(QK^T / sqrt(d_k)) V`, with head splitting done by reshape and transpose.

### 2.4 Baseline

`MLPBaseline` mean-pools the 30 frames into a single 63-dimensional vector and applies a
two-layer MLP. It has **no** attention, no positional information, and no ability to model
temporal order — averaging destroys it. This makes it the right control for the question
"does attention help here?"

## 3. Dataset

Collected with `src/collect.py`; not distributed with the repository.

| Property | Value |
|---|---|
| Classes | `open_palm`, `fist`, `peace`, `thumbs_up`, `point` |
| Samples | 300 (60 per class) |
| Sample shape | `(30, 21, 3)` — 30 frames at ~30 fps, i.e. ~1 s |
| Recording sessions | 15 (3 rounds x 5 gestures) |
| Subject | 1 |

Each recording run is tagged with a timestamp, and the three rounds deliberately vary
position, lighting, distance, and hand (left/right) so that no single nuisance variable
predicts the label.

**Missing detections are discarded, not imputed.** If MediaPipe loses the hand partway
through a sample, the whole sample is dropped and re-recorded. Zero-padding would insert a
physically impossible hand pose into the training signal.

### 3.1 Evaluation protocol

The train/validation split is **stratified by recording session**: for each class, one entire
session is held out. This is the central methodological choice in the repo.

Frames within a single recording run are near-duplicates of each other — same lighting, same
position, same pose, seconds apart. A random per-sample split would place near-identical
samples on both sides of the boundary, and the resulting accuracy would measure memorization
rather than generalization. Splitting by session makes the validation set a genuinely unseen
recording condition.

Resulting split: 200 train / 100 validation, with all five classes present in validation.

## 4. Results

Identical data, seed, optimizer (AdamW, lr 1e-3, weight decay 1e-2), batch size 32, and 40
epochs for both models.

| Model | Parameters | Best val. accuracy | Epochs to reach 1.000 |
|---|---|---|---|
| Transformer | 73,413 | **1.000** | 2 |
| MLP (mean-pooled) | 25,349 | **1.000** | 27 |

**Both models saturate the task.** Attention provides no accuracy advantage at 2.9x the
parameter count. Its only measurable benefit is convergence speed: the Transformer reaches
100% by epoch 2 and stays there, while the MLP climbs gradually and still oscillates between
0.95 and 1.00 late in training.

### 4.1 Interpretation

This is a negative result for the Transformer, and it is a property of the *task*, not a bug.
The five gestures are **static poses** — the hand barely changes across the 30 frames — so
mean-pooling discards almost no information. Attention models relationships between positions
in a sequence; when every position carries the same content, there is nothing to model.

The correct reading is not "attention does not work" but "this task does not require temporal
modelling, and therefore cannot distinguish the two architectures." A benchmark that cannot
separate the methods under test is a benchmark problem.

### 4.2 Live behaviour

`src/live_demo.py` runs the trained model over a rolling 30-frame buffer. All five trained
gestures are recognised in real time. Presented with an *untrained* gesture, the model still
assigns it to one of the five classes, often with high confidence — the expected consequence
of a softmax over a closed label set. Confidence is not a measure of correctness.

## 5. Reproducing

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
curl -L -o models/hand_landmarker.task \
  https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task

# collect data (repeat, varying position/lighting/hand between rounds)
for g in open_palm fist peace thumbs_up point; do
  python -m src.collect --gesture $g --samples 20
done

python -m src.train --model transformer
python -m src.train --model mlp
python -m src.live_demo
```

**Environment note.** MediaPipe 1.0.x crashes on macOS with a Metal initialisation failure
(`DrishtiMetalHelper` / `Check failed: service_ Service is unavailable`) regardless of the
requested delegate. This repository pins `mediapipe==0.10.21`, which works with the CPU
delegate. The Tasks API is identical across both versions, so no code changes are required.
Note that 0.10.21 requires `numpy<2`, which conflicts with `opencv-python>=5`; use the
`opencv-contrib-python` build that MediaPipe pulls in.

## 6. Limitations

- **Single subject, single camera, single day.** The session-stratified split measures
  generalisation to a new *recording session*, not to a new person, camera, or environment.
  The 100% figure should be read with that scope in mind.
- **The task is too easy to be discriminative.** Five gestures with distinct extended-finger
  counts are close to linearly separable after normalisation.
- **Closed label set.** The model cannot express "unknown"; every input is forced into one of
  five classes.
- **Rotation is not normalised.** A tilted gesture yields different features from an upright
  one. Whether removing rotation helps or hurts is an open question, it would add invariance,
  but orientation is genuinely informative for some gestures (`thumbs_up` versus
  `thumbs_down`).
- **No image-based baseline yet.** The "landmarks versus pixels" comparison in the title is
  not yet answered; only landmark-based models have been run.

## 7. Next steps

1. **Dynamic gestures.** Swipe left/right, wave, circle, beckon. The mean of a left-swipe and
   a right-swipe is the same vector, so a mean-pooled baseline cannot in principle separate
   them. This is the experiment that can actually test whether attention earns its parameters.
2. **A `none` class** of arbitrary hand poses, to give the closed-set model somewhere to put
   unfamiliar input.
3. **Sequence-length ablation.** How few of the 30 frames are needed?
4. **Coordinate ablation.** Does dropping `z` hurt, given that MediaPipe's depth estimate is
   monocular and relative?
5. **A CNN baseline over raw frames**, to answer the landmarks-versus-pixels question.
6. **Attention map inspection.** With static poses the maps are expected to be near-uniform;
   under dynamic gestures they become interpretable and worth visualising.

## 8. Repository layout

```
src/
  capture.py      webcam loop, mirroring, FPS counter
  landmarks.py    MediaPipe HandLandmarker wrapper -> (21, 3) per frame
  collect.py      labelled data collection CLI, session-tagged
  normalize.py    wrist-centring and scale normalisation
  dataset.py      .npz loading, session-stratified split, DataLoaders
  model.py        Transformer encoder, attention implemented from scratch
  baseline.py     mean-pooled MLP control
  train.py        training loop for either model
  live_demo.py    real-time inference over a rolling buffer
configs/config.yaml   sequence length, class list, paths
```