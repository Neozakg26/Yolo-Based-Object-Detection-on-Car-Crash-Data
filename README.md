# YOLO-Based Causal Accident Anticipation Framework

## Overview

This repository implements a **probabilistic accident anticipation system** that integrates **deep learning-based perception** with **causal structure learning and Dynamic Bayesian Networks (DBNs)**.

The pipeline processes egocentric dash-cam video to:

1. Detect and track objects in real-time
2. Extract spatio-temporal traffic features
3. Learn temporal dependencies using causal discovery (PCMCI via Tigramite)
4. Perform probabilistic inference over a hierarchical DBN
5. Predict accident risk trajectories and scene-level accident outcomes

---

## Key Contributions

* Integration of **YOLO-based detection + DeepSORT tracking** with probabilistic modelling
* Use of **PCMCI (Tigramite)** for temporal causal structure learning
* Design of a **hierarchical DBN** with interpretable latent variables:

  * Collision Imminence
  * Behavioural Risk
  * Environmental Hazard
* Real-time **risk trajectory prediction (Safe / Elevated / Critical)**
* Scene-level accident prediction using **probabilistic thresholds**
* Visual interpretability via **risk trajectories + frame alignment**

---

## System Pipeline

```
Dashcam Video
      ↓
YOLO Object Detection
      ↓
DeepSORT Tracking
      ↓
Feature Extraction (TTC, proximity, velocity, etc.)
      ↓
PCMCI (Causal Structure Learning)
      ↓
Hierarchical DBN
      ↓
Probabilistic Inference
      ↓
Risk Trajectory + Accident Prediction
```


---

##  Datasets

### 1. BDD100K

* Used for **training detection and tracking**
* Diverse conditions: day/night, weather, occlusion
* Classes:

  * car (0), bus (1), truck (2), person (3), bike (4), traffic light (5)

### 2. Car Crash Dataset (CCD)

* Used for **accident anticipation**
* 4,500 clips (1,500 accident, 3,000 normal)
* 50 frames per clip at 10 FPS

---

##  Installation

### Requirements

* Python 3.10+
* PyTorch
* OpenCV
* NumPy / Pandas
* NetworkX
* Tigramite
* Matplotlib

### Install dependencies

```bash
pip install -r requirements.txt
```

---

##  Usage

### 1. Run Object Detection + Tracking

```bash
python -m execute.execute_tracker --path C_000001_
```

---

### 2. Train Global DBN Model

```bash
python -m execute.train_global_dbn \
    --results_dir path/to/results \
    --meta_csv path/to/meta.csv \
    --features_path path/to/pcmci_graph.pkl \
    --out_model global_model.pkl
```

Supports:

* 5-fold cross-validation
* Risk + ego-involvement classifiers
* Evaluation metrics (F1, accuracy)

---

### 3. Predict Risk Trajectories

```bash
python -m execute.predict_accident \
    --model_path global_model.pkl \
    --input_dir path/to/scenes
```

Outputs:

* Risk trajectory plots
* Frame-level probabilities
* Scene-level predictions

---

##  Evaluation Strategy

### Frame-Level Risk Classification

* Classes: Safe / Elevated / Critical
* Metrics:

  * Accuracy
  * Macro F1-score
  * Confusion Matrix

---

### Scene-Level Accident Prediction

#### Rule-Based Evaluation:

* **Rule 1:** Any frame with risk_score ≥ threshold
* **Rule 2:** ≥2 consecutive frames above threshold

Metrics:

* Accuracy
* Precision
* Recall
* F1-score

---

### Cross Validation

* 5-fold scene-level split
* Report:

  * Mean ± standard deviation
  * Robustness across folds

---

##  Feature Definitions

Key features extracted per tracked object:

* `proximity` – object depth proxy
* `closing_rate` – relative approach speed
* `risk_speed` – speed × proximity
* `ttc_proxy` – estimated time-to-collision
* `ttc_relative` – TTC using relative velocity
* `ttc_rate` – change in TTC over time
* `ego_speed`, `ego_accel` – ego motion

---

##  Causal Structure Learning

* Algorithm: **PCMCI**
* Library: **Tigramite**
* Learns:

  * Temporal dependencies (t-1, t-2)
  * Cross-variable relationships
* Output:

  * Directed temporal graph used in DBN

---

##  Inference

* Methods:

  * Supervised classification
  * Belief Propagation
  * Variable Elimination

Outputs:

* Risk probabilities
* MAP risk state
* Risk trajectories over time

---

## Limitations

* No ground-truth tracking annotations (BDD100K limitation)
* Detection errors propagate into feature extraction
* Causal structure reflects statistical dependencies, not true causality
* Class imbalance in accident vs non-accident scenes

---

##  Citation

If you use this work, please cite accordingly:


##  Reproducibility

* All experiments use fixed random seeds
* Models and evaluation scripts are included
* Results can be reproduced using provided commands (run_batch_files)
