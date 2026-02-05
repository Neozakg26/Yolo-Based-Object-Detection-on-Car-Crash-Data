# Experiment Progress Report: Explainable Accident Anticipation Framework

## Overview

This report documents the implementation progress of the explainable accident anticipation framework as outlined in the research proposal. The framework aims to predict traffic accidents from egocentric dash-cam video using a hierarchical probabilistic model that provides structural explainability.

---

## Research Question 1 (RQ1): Object Detection & Tracking

> **RQ1**: How accurately and efficiently can objects be detected and tracked in real time using egocentric dash-cam video under diverse real-world driving conditions to support probabilistic accident anticipation?

**A1:** : Detecting and Tracking objects under various egocecntric driving conditions to support probabilistic accident anticipation

### Implementation Status: ✅ COMPLETE

### Components Implemented

#### 1. Object Detection Pipeline
- **Model**: YOLOv11 (via Ultralytics)
- **Location**: `downloaded_model/weights/best.pt`
- **Integration**: `execute_tracker.py:34-35`

```python
detector = YOLO("downloaded_model/weights/best.pt")
```

#### 2. Object Tracking (DeepSORT)
- **Location**: `tracking/deepsort_tracker.py`
- **Features Implemented**:
  - Multi-object tracking with persistent track IDs
  - Kalman filter-based motion prediction
  - Track lifecycle management (tentative → confirmed → deleted)

#### 3. Feature Extraction from Tracks
- **Location**: `tracking/deepsort_tracker.py`
- **Extracted Features**:

| Feature | Description | Purpose |
|---------|-------------|---------|
| `x`, `y` | Bounding box center position | Spatial location |
| `w`, `h` | Bounding box dimensions | Object size/depth proxy |
| `vx`, `vy` | Velocity components | Motion dynamics |
| `ax`, `ay` | Acceleration components | Behavioral intent |
| `speed` | Velocity magnitude | Risk indicator |
| `proximity` | Scale-based depth proxy (`h/frame_height`) | Distance estimation |
| `ttc_proxy` | Time-to-collision estimate | Primary risk metric |
| `closing_rate` | Rate of proximity change | Approach dynamics |
| `risk_speed` | Combined speed-proximity risk | Composite risk |

#### 4. Environment Feature Builder
- **Location**: `explainability/environment_builder.py`
- **Frame-level Aggregations**:
  - `min_distance_t`: Minimum distance across all objects
  - `mean_rel_speed_t`: Average relative speed in scene
  - `min_ttc_t`: Minimum TTC (most dangerous object)
  - `num_objects_close_t`: Count of nearby objects

### Output Files
- `{scene_id}_tracks.parquet`: Per-object, per-frame tracking data
- `{scene_id}_env.parquet`: Per-frame environment features

### Evaluation Metrics (Proposed)
- Mean Average Precision (mAP)
- Intersection over Union (IoU)
- Multiple Object Tracking Accuracy (MOTA)
- ID Switch Count
- Feature temporal consistency

### ⚠️ Evaluation Limitation

**Ground Truth Unavailable**: The Car Crash Dataset (CCD) does not provide ground truth bounding box annotations or tracking labels. As a result, formal quantitative evaluation of the object detection and tracking pipeline (mAP, IoU, MOTA, ID switches) cannot be performed within this study.

**Implication**: RQ1 evaluation is limited to:
1. Qualitative assessment of tracking consistency
2. Downstream task performance (i.e., how well extracted features support RQ2/RQ3)
3. Feature plausibility checks (e.g., TTC decreasing as objects approach)

This is a known limitation of using the CCD dataset for detection/tracking research, as it was designed primarily for accident anticipation benchmarking with accident frame labels rather than object-level annotations.

---

## Research Question 2 (RQ2): Structure Learning

> **RQ2**: To what extent can structure learning methods recover the spatio-temporal dependencies among observed, latent risk states and other latent factors from ego-centric visual data?

### Implementation Status: ⚠️ PARTIAL (PCMCI instead of DYNOTEARS)

### Components Implemented

#### 1. Causal Discovery with PCMCI
- **Location**: `explainability/feature_extractor.py`
- **Algorithm**: PCMCI (Peter-Clark Momentary Conditional Independence)
- **Library**: Tigramite

**Note**: The proposal specified DYNOTEARS, but PCMCI was implemented instead. Both are valid causal discovery methods for time-series data.

```python
# feature_extractor.py:206-212
pcmci = PCMCI(
    dataframe=tig_df,
    cond_ind_test=ParCorr(significance='analytic')
)
results = pcmci.run_pcmci(tau_max=self.tau_max, pc_alpha=0.05)
```

#### 2. Causal Edge Extraction
- **Location**: `explainability/feature_extractor.py:extract_edges()`
- **Features**:
  - Configurable time lag (`tau_max`, default=2)
  - Domain constraint filtering (forbidden edges)
  - Bootstrap stability analysis (optional)
  - Direction consistency checking

**Variables Used for Causal Discovery**:
```python
var_names = [
    "vx", "vy", "speed",           # Motion state
    "ax", "ay",                     # Acceleration
    "proximity", "risk_speed",      # Risk indicators
    "ttc_proxy", "ttc_rate",        # Time-to-collision
    "closing_rate",                 # Approach dynamics
    "min_distance_t", "mean_rel_speed_t", "min_ttc_t"  # Environment
]
```

#### 3. Domain Knowledge Integration
- **Location**: `explainability/feature_extractor.py:17-34`
- **Forbidden Edges** (physically implausible):
  ```python
  FORBIDDEN_EDGES = {
      ("x", "ttc_proxy"),
      ("y", "ttc_proxy"),
      ("min_distance_t", "proximity"),
      ("mean_rel_speed_t", "risk_speed"),
      ("min_ttc_t", "ttc_proxy"),
  }
  ```
- **Expected Directions** (for validation):
  ```python
  EXPECTED_DIRECTIONS = {
      ("vx", "x"): "positive",
      ("vy", "y"): "positive",
      ("proximity", "ttc_proxy"): "negative",
      ("risk_speed", "ttc_proxy"): "negative",
  }
  ```

#### 4. Hierarchical DBN Structure
- **Location**: `explainability/hierarchical_dbn/dbn_structure.py`
- **Three-Level Hierarchy**:

```
Level 1: Observable Features (from tracking)
    ↓
Level 2: Intermediate Latent Factors
    - Collision Imminence
    - Behavioural Risk
    - Environmental Hazard
    ↓
Level 3: Accident Risk (Safe, Elevated, Critical)
```

#### 5. Observable-to-Latent Mappings
- **Location**: `explainability/hierarchical_dbn/latent_model.py:57-96`

| Latent Factor | Parent Observables |
|---------------|-------------------|
| **Collision Imminence** | ttc_proxy, ttc_relative, ttc_smoothed, ttc_rate, closing_rate, proximity, min_distance_t |
| **Behavioural Risk** | vx, vy, speed, rel_vx, rel_vy, ax, ay, ego_speed, ego_accel, risk_speed |
| **Environmental Hazard** | num_objects_close_t, mean_rel_speed_t, min_ttc_t |

#### 6. Feature Discretization
- **Location**: `explainability/hierarchical_dbn/discretizer.py`
- **Method**: Domain-informed thresholds + percentile-based binning

Example discretization:
```python
"ttc_proxy": DiscretizationConfig(
    bins=[1.5, 3.0, 5.0],
    labels=["Critical", "Warning", "Caution", "Safe"],
)
```

### Output Files
- `{scene_id}_edge_stats.parquet`: PCMCI-discovered causal edges with statistics
- `{scene_id}_causal_graph.parquet`: Consolidated causal graph
- `{scene_id}_causal_graph.png`: Visualization (optional)

### Gap vs. Proposal
| Proposed | Implemented |
|----------|-------------|
| DYNOTEARS | PCMCI |
| Continuous optimization | Conditional independence testing |
| Expectation-Maximization for latents | Domain-informed priors + Bayesian estimation |

---

## Research Question 3 (RQ3): Inference & Explainability

> **RQ3**: Given a learned dynamic probabilistic graph structure, how effective are inference methods in real-time accident anticipation, and how do the hierarchical explanations provide interpretable reasoning?

### Implementation Status: ✅ COMPLETE (Multiple Methods)

### Components Implemented

#### 1. Supervised Classifier (Primary Method)
- **Location**: `explainability/hierarchical_dbn/risk_assessor.py`
- **Algorithm**: Gradient Boosting Classifier
- **Training Target**: Time-To-Accident (TTA) based labels

**Label Assignment**:
```python
def get_risk_label(tta_seconds):
    if tta_seconds >= 2.5:
        return 0  # SAFE
    elif tta_seconds >= 1.5:
        return 1  # ELEVATED
    else:
        return 2  # CRITICAL
```

**Training Process**:
1. Merge tracking + environment features
2. Discretize continuous features
3. Create TTA-based labels from `accident_start_frame`
4. Train GradientBoostingClassifier on discretized features
5. Output: P(Safe), P(Elevated), P(Critical) per frame

#### 2. Belief Propagation Inference
- **Location**: `explainability/hierarchical_dbn/dbn_inference.py:115-239`
- **Method**: Loopy Belief Propagation on single time-slice BN
- **Status**: Implemented but may have convergence issues

#### 3. Variable Elimination Inference
- **Location**: `explainability/hierarchical_dbn/dbn_inference.py:242-334`
- **Method**: Exact inference via variable elimination
- **Status**: Implemented (slower but exact)

#### 4. Direct Feature Scoring (Fallback)
- **Location**: `explainability/hierarchical_dbn/risk_assessor.py:_compute_direct_risk_score()`
- **Method**: Weighted combination of discretized features
- **Feature Weights**:
  ```python
  risk_weights = {
      "ttc_proxy_d": (2.0, True),      # High weight, invert (low TTC = high risk)
      "proximity_d": (1.5, False),     # Higher state = closer = more risk
      "closing_rate_d": (1.3, False),  # Higher = approaching faster
      ...
  }
  ```

#### 5. CPT Estimation
- **Location**: `explainability/hierarchical_dbn/cpt_estimator.py`
- **Method**: Semi-supervised with domain-informed priors
- **Features**:
  - Prior initialization based on domain knowledge
  - Bayesian estimation with equivalent sample size
  - TTA-based pseudo-labels for supervision

#### 6. Risk Assessment Pipeline
- **Location**: `explainability/hierarchical_dbn/risk_assessor.py:AccidentRiskAssessor`
- **Main Entry Point**: `execute_tracker.py`

**Pipeline Flow**:
```
Video → YOLO Detection → DeepSORT Tracking → Feature Extraction
    → Discretization → Classifier/DBN Inference → Risk Trajectory
```

### Output Files
- `{scene_id}_risk_trajectory.parquet`: Frame-by-frame risk predictions
  - Columns: `frame`, `P_Safe`, `P_Elevated`, `P_Critical`, `risk_score`, `MAP_Risk`
- `{scene_id}_risk_model.parquet`: Model configuration
- `{scene_id}_risk_model.classifier.pkl`: Trained classifier

### Explainability Features

#### 1. Risk Propagation Pathways
The hierarchical structure allows tracing:
```
Observable (e.g., ttc_proxy=Critical)
  → Latent Factor (Collision Imminence=High)
    → Accident Risk (Critical)
```

#### 2. Contributing Factor Explanations
- **Location**: `risk_assessor.py:_generate_explanations()`
- Generates human-readable explanations:
  ```
  Collision Imminence: High (87%) - TTC: Critical, Proximity: Close
  Behavioural Risk: Moderate (62%) - Ego speed: Fast, Ego accel: Braking
  Environmental Hazard: Low (45%) - Objects nearby: Sparse
  ```

#### 3. Risk Score Interpretation
- Score range: 0.0 (Safe) → 1.0 (Critical)
- Formula: `0×P_Safe + 0.5×P_Elevated + 1.0×P_Critical`

---

## Execution Pipeline

### Running the Full Pipeline

```bash
python execute_tracker.py --path "video_folder" [OPTIONS]

Options:
  --skip_viz          Skip causal graph visualization
  --skip_dbn          Skip hierarchical DBN risk assessment
  --tau_max N         Max time lag for PCMCI (default: 2)
  --inference_method  supervised|belief_propagation|variable_elimination
```

### Pipeline Output Summary

```
Pipeline outputs per video:
├── {scene_id}_tracks.parquet         # Object tracks
├── {scene_id}_env.parquet            # Environment features
├── {scene_id}_edge_stats.parquet     # Causal edges
├── {scene_id}_causal_graph.parquet   # Graph structure
├── {scene_id}_causal_graph.png       # Visualization
├── {scene_id}_risk_trajectory.parquet # Risk predictions
├── {scene_id}_risk_model.parquet     # Model config
└── {scene_id}_risk_model.classifier.pkl # Trained classifier
```

---

## Summary Table

| Component | Proposed | Implemented | Status |
|-----------|----------|-------------|--------|
| Object Detection | YOLOv11 | YOLOv11 (Ultralytics) | ✅ (unevaluated*) |
| Object Tracking | DeepSORT | DeepSORT | ✅ (unevaluated*) |
| Ego-motion Compensation | Optical flow | Not implemented | ❌ |
| Feature Extraction | Motion, proximity, TTC | Full set | ✅ |
| Structure Learning | DYNOTEARS | PCMCI | ⚠️ Different method |
| Hierarchical Latents | 3-level DBN | 3-level DBN | ✅ |
| Inference (BP) | Loopy BP | pgmpy BP | ✅ |
| Inference (VE) | Not specified | pgmpy VE | ✅ |
| Supervised Classifier | Not specified | GradientBoosting | ✅ Added |
| Semi-supervised Labels | TTA-based | TTA-based | ✅ |
| Discretization | Domain-informed | Domain-informed | ✅ |
| Explainability | Hierarchical pathways | Implemented | ✅ |

*\*RQ1 components are implemented but cannot be formally evaluated due to lack of ground truth annotations in CCD dataset.*

---

## Study Limitations

### 1. No Ground Truth for RQ1 Evaluation
- **Limitation**: The CCD dataset lacks bounding box and tracking ground truth annotations
- **Impact**: Cannot compute standard detection/tracking metrics (mAP, IoU, MOTA)
- **Mitigation**: RQ1 is evaluated indirectly through downstream task performance (RQ2/RQ3)
- **Note**: This is an inherent limitation of the dataset, not an implementation gap

---

## Gaps and Future Work

### 1. Ego-Motion Compensation
- **Gap**: Proposal mentions optical flow-based visual odometry
- **Impact**: Current features may include camera-induced motion artifacts
- **Recommendation**: Implement ego-motion estimation to extract true object motion

### 2. DYNOTEARS vs PCMCI
- **Gap**: Different structure learning algorithm used
- **Impact**: Both discover temporal dependencies, but DYNOTEARS is continuous optimization-based while PCMCI is conditional independence-based
- **Recommendation**: Implement DYNOTEARS for comparison

### 3. Evaluation Metrics
- **Gap**: No formal evaluation against baseline methods yet
- **Required Metrics**:
  - AUC, F1-score for risk state prediction
  - Time-to-Accident (TTA) at various horizons
  - Comparison with Yi et al., Bao et al., Karim et al.

### 4. Cross-Video Structure Learning
- **Gap**: Currently learns structure per-video
- **Recommendation**: Aggregate causal graphs across videos (see `aggregate_causal_graphs.py`)

### 5. Real-time Performance
- **Gap**: Not benchmarked for real-time inference
- **Required**: Measure inference latency per frame

---

## File Structure

```
yolo-detector/
├── execute_tracker.py              # Main pipeline entry point
├── tracking/
│   ├── deepsort_tracker.py         # DeepSORT implementation
│   ├── track_runner.py             # Tracking orchestration
│   └── test_feature_extractor.py   # Feature extraction tests
├── explainability/
│   ├── feature_extractor.py        # PCMCI causal discovery
│   ├── environment_builder.py      # Frame-level feature aggregation
│   ├── metadata.py                 # Crash table parsing
│   └── hierarchical_dbn/
│       ├── __init__.py
│       ├── latent_model.py         # Latent factor definitions
│       ├── discretizer.py          # Feature discretization
│       ├── dbn_structure.py        # DBN structure builder
│       ├── cpt_estimator.py        # CPT learning
│       ├── dbn_inference.py        # BP and VE inference
│       └── risk_assessor.py        # End-to-end risk assessment
└── car_crash_dataset/
    └── CCD_images/
        ├── Crash_Table.csv         # Metadata with accident frames
        └── results/                # Output directory
```

---

## Conclusion

The core framework for explainable accident anticipation has been implemented with:
- Complete object detection and tracking pipeline (RQ1) — *note: formal evaluation limited by lack of ground truth in CCD dataset*
- Causal structure learning via PCMCI with hierarchical DBN (RQ2)
- Multiple inference methods including supervised classification (RQ3)
- Semi-supervised labeling using Time-To-Accident

**Study Limitation**: RQ1 evaluation cannot be performed quantitatively because the CCD dataset does not include ground truth bounding box or tracking annotations. Detection and tracking performance is instead validated indirectly through the quality of downstream causal discovery (RQ2) and risk prediction (RQ3).

Key remaining work:
1. Formal evaluation of RQ2/RQ3 against benchmark datasets and baseline methods
2. Ego-motion compensation for improved feature accuracy
3. Cross-video causal graph aggregation
4. Real-time performance optimization
