# Fire-and-Smoke-Detection-Using-YOLOv8-Optical-Flow-and-LSTM-Classification

A three-stage computer-vision pipeline that detects **fire** and **smoke** in camera frames and outputs a single `cctv_confidence` score. The score is designed to feed a larger multi-sensor **fusion engine** for early fire warning.

Everything is built and run as Kaggle notebooks (GPU: Tesla T4).

---

## 📌 Overview

| Stage | Notebook | What it does |
|------|----------|--------------|
| 1 | `notebook1.ipynb` | Dataset exploration, label statistics, image-size analysis, augmentation preview, config export |
| 2 | `notebook2.ipynb` | Trains and evaluates a **YOLOv8n** detector for `smoke` / `fire` |
| 3 | `notebook3.ipynb` | Extracts per-frame features (YOLO + **Farneback optical flow**), trains an **LSTM** over 10-frame sequences, and exposes `get_cctv_confidence()` |

```
 Frames ──► YOLOv8n ──► detections (count, confidence)  ─┐
   │                                                     ├─► 8 features/frame ─► LSTM (10 frames) ─┐
   └────► Optical flow ──► upward-motion score ──────────┘                                          │
                                                                                                    ▼
                                         cctv_confidence = 0.6 · LSTM  +  0.4 · max YOLO conf  ──► Fusion Engine
```

---

## 📂 Dataset

**D-Fire** (YOLO format), from Kaggle: [`sayedgamal99/smoke-fire-detection-yolo`](https://www.kaggle.com/datasets/sayedgamal99/smoke-fire-detection-yolo)

| Split | Images | Smoke boxes | Fire boxes | Empty (no-object) images |
|-------|-------:|------------:|-----------:|-------------------------:|
| Train | 14,122 | 7,794 | 9,638 | 6,458 |
| Val   | 3,099  | 1,756 | 2,176 | 1,375 |
| Test  | 4,306  | 2,315 | 2,878 | 2,005 |

Classes: `0 = smoke`, `1 = fire`.

---

## 🧠 Method

### 1. Data exploration & augmentation (Notebook 1)
- Verifies dataset paths and writes the Ultralytics `dfire_dataset.yaml`.
- Visualises sample images with ground-truth boxes and plots class distribution per split.
- Previews an Albumentations pipeline (horizontal flip, brightness/contrast, noise, fog, hue/saturation).
- Saves `config.json` for the next notebooks.

### 2. YOLOv8 detector (Notebook 2)
- Base model: `yolov8n.pt` (pretrained), fine-tuned for 30 epochs.
- Settings: `imgsz=640`, `batch=16`, Adam, `lr0=0.001`, early-stopping patience 10.
- Exports `best.pt`, `nb2_config.json`, and a reusable `yolo_inference.py` (`run_yolo_on_frame()`).

### 3. Temporal model: optical flow + LSTM (Notebook 3)
- **Per-frame feature vector (8 values):** `n_fire`, `n_smoke`, `n_detect`, `max_conf`, `avg_conf`, `upward_motion_score`, `flow_magnitude`, `detected`.
- **Optical flow:** Farneback dense flow between consecutive frames. The "upward motion" score is the share of motion magnitude pointing upward (45°–135°), since smoke rises.
- **LSTM:** `LSTM(64) → Dropout → LSTM(32) → Dropout → Dense(16) → Dropout → Dense(1, sigmoid)` on sequences of 10 frames × 8 features, MinMax-scaled.
- **Final score:** `cctv_confidence = 0.6 × LSTM + 0.4 × max YOLO confidence (last 5 frames)`. A score above `0.5` is treated as danger.

---

## 📊 Results

### YOLOv8n on the validation set

| Class | Precision | Recall | mAP@50 | mAP@50-95 |
|-------|----------:|-------:|-------:|----------:|
| **All** | 0.739 | 0.693 | **0.755** | 0.436 |
| Smoke | 0.784 | 0.752 | 0.811 | 0.502 |
| Fire  | 0.693 | 0.634 | 0.699 | 0.371 |

- Model size: 3.0M parameters, 8.1 GFLOPs.
- Speed on a T4: ~0.5 ms preprocess, **~2.3 ms inference**, 0.7 ms postprocess per image, so it's fast enough for real-time use.

### LSTM (validation, 40 sequences)
Accuracy was 0.97, but the validation split contained only **1 positive sequence**, and the model did not detect it. See the limitations below. Treat this result as a working proof of concept, not a validated classifier.

---

## ⚠️ Known Limitations

Being upfront about these so results are interpreted correctly:

- **YOLO mAP@50 (0.755) is below the 0.80 target** set in the notebook. Fire is the weaker class (0.699). Likely next steps: a larger model (YOLOv8s/m), more epochs, tuned augmentation.
- **D-Fire consists of independent still images, not video.** The LSTM sequences are built from *consecutive sorted filenames*, so they are not true temporal sequences, and optical flow between unrelated frames is not physically meaningful. Real video clips are needed for the temporal stage to work as intended.
- **LSTM training set is small and imbalanced:** 199 sequences from the first 2,000 sorted training images, with very few positives. The validation AUC is unreliable, and early stopping restored epoch 1 weights.
- Validation logs show some "corrupt image" warnings caused by Ultralytics trying to write a cache to Kaggle's read-only input folder. Evaluation completed (3,094 of 3,099 images).
- Notebook 1 uses `GaussNoise(var_limit=...)`, which newer Albumentations versions no longer accept (a warning is shown, and the argument is ignored).

---

## 🚀 Getting Started

### Requirements
```bash
pip install ultralytics albumentations tensorflow scikit-learn opencv-python seaborn pandas matplotlib tqdm
```
Python 3.10+ and a CUDA GPU are recommended.

### Run order
1. **`notebook1.ipynb`** → produces `config.json` and `dfire_dataset.yaml`
2. **`notebook2.ipynb`** → produces `best.pt` and `nb2_config.json` (~75 min for 30 epochs on a T4)
3. **`notebook3.ipynb`** → produces `lstm_model.h5`, `lstm_scaler.pkl`, `nb3_config.json` (~6 min feature extraction + short LSTM training)

> **Paths:** the notebooks use Kaggle paths such as `/kaggle/input/...` and `/kaggle/working`. Running locally means updating `DATASET_ROOT`, the config paths, and `BEST_MODEL_PATH` to match your machine.

### Inference example
```python
output = get_cctv_confidence(list_of_10_image_paths)
# {
#   'cctv_confidence': 0.39,
#   'lstm_confidence': 0.46,
#   'yolo_confidence': 0.28,
#   'fire_detected': False,
#   'smoke_detected': False,
#   'n_fire': 0, 'n_smoke': 0,
#   'bbox': []
# }
```
Needs at least 10 frames. `cctv_confidence` is the value passed on to the fusion engine.

---

## 📁 Repository Structure

```
├── notebook1.ipynb      # EDA, augmentation, config
├── notebook2.ipynb      # YOLOv8 training & evaluation
├── notebook3.ipynb      # Optical flow + LSTM + CCTV confidence
└── README.md
```

Generated artifacts (not committed): `best.pt`, `lstm_model.h5`, `lstm_scaler.pkl`, `*_config.json`, `dfire_dataset.yaml`, result plots.

---

## 🔮 Future Work
- Train and evaluate on real video datasets for a genuine temporal model
- Larger YOLO variants and longer training to improve fire recall
- Balanced sequence sampling and a larger LSTM training set
- Replace the fixed 0.6/0.4 weighting with a learned fusion
- Export to ONNX/TensorRT for edge deployment

---

## 🙏 Acknowledgements
- [D-Fire dataset](https://github.com/gaiasd/DFireDataset) and its Kaggle mirror by `sayedgamal99`
- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- OpenCV, TensorFlow/Keras, Albumentations

## 📄 License
Add a license of your choice (e.g. MIT) before publishing. Note that the dataset and Ultralytics have their own licenses (Ultralytics is AGPL-3.0).

## 👤 Author
**Asmi Dagar**
