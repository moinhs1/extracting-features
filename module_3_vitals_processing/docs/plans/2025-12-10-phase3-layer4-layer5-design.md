# Phase 3 Design: Layers 4-5 (Embeddings & World States)

**Date:** 2025-12-10
**Status:** Approved
**Author:** Claude + Q

---

## Overview

Build Layers 4-5 to create dense patient representations for trajectory phenotyping, outcome prediction, and future treatment optimization.

### Components (Implementation Order)

| # | Component | Output | Dependencies |
|---|-----------|--------|--------------|
| 1 | **FPCA** | 10 scores × 7 vitals = 70 dims | scikit-fda |
| 2 | **LSTM-VAE** | 32-dim latent per timestep | PyTorch |
| 3 | **DTW Clustering** | Cluster IDs + distances | tslearn |
| 4 | **Embedding Clustering** | HDBSCAN on FPCA+VAE | hdbscan |
| 5 | **World States** | ~100 dim state vectors | Layers 1-4 |

### Output Files

```
outputs/layer4/
├── fpca_scores.parquet        # (7,689 patients × 70 features)
├── fpca_components.h5         # Fitted FPCA model artifacts
├── vae_latents.h5             # (7,689 × 745 × 32) latent sequences
├── vae_model.pt               # Trained VAE weights
├── clusters_dtw.parquet       # DTW cluster assignments
├── clusters_embedding.parquet # HDBSCAN cluster assignments

outputs/layer5/
├── world_states.h5            # (7,689 × 745 × ~100) state vectors
├── state_schema.json          # Documents each dimension
```

### Key Design Principle

**Extensibility:** World states reserve dimensions for treatment context (labs, meds, imaging, procedures) - filled with zeros/NaN now, populated when those modules are ready.

---

## Component 1: FPCA

**Purpose:** Extract interpretable trajectory shape features - FPC1 = overall level, FPC2 = slope/trend, FPC3+ = oscillation patterns.

### Approach

```python
from skfda import FDataGrid
from skfda.preprocessing.dim_reduction import FPCA

for vital in ['HR', 'SBP', 'DBP', 'MAP', 'RR', 'SPO2', 'TEMP']:
    # Create functional data from hourly grid (745 timepoints)
    fd = FDataGrid(data_matrix=patient_trajectories[vital])

    # Fit FPCA, extract 10 components
    fpca = FPCA(n_components=10)
    scores = fpca.fit_transform(fd)  # (7689, 10)
```

### Handling Missing Data

Use imputed values directly (Layer 2 already handled imputation). FPCA on complete trajectories is more stable. Track quality via separate observation percentage columns.

### Output Schema (`fpca_scores.parquet`)

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient ID |
| HR_fpc1...HR_fpc10 | float | HR component scores |
| SBP_fpc1...SBP_fpc10 | float | SBP component scores |
| ... | ... | (7 vitals × 10 = 70 columns) |
| HR_obs_pct | float | % hours with Tier 1-2 data |

---

## Component 2: LSTM-VAE

**Purpose:** Learn nonlinear temporal embeddings that capture cross-vital dependencies and complex patterns FPCA misses.

### Architecture

```
Input: (batch, 745, 7) - hourly vitals + (batch, 745, 7) masks

Encoder:
├── Input projection: Linear(7 → 64)
├── LSTM: (64 → 128, bidirectional, 2 layers)
├── Temporal attention pooling → (batch, 256)
├── μ head: Linear(256 → 32)
└── logσ² head: Linear(256 → 32)

Latent: z ~ N(μ, σ²) - 32 dimensions

Decoder:
├── Latent projection: Linear(32 → 256)
├── LSTM: (256 → 128, 2 layers)
└── Output: Linear(128 → 7) per timestep

Loss: Reconstruction (MSE) + β * KL divergence
```

### Key Design Choices

1. **Bidirectional encoder** - Sees full trajectory context
2. **Attention pooling** - Learns which timepoints matter most
3. **Mask integration** - Reconstruction loss weighted by observation mask
4. **β-VAE** - Tunable β controls latent disentanglement

### Training Strategy

- **Batch size:** 64 patients
- **Epochs:** 100-200 with early stopping
- **Learning rate:** 1e-3 with cosine annealing
- **Validation:** 80/20 split, monitor reconstruction + KL

### Output (`vae_latents.h5`)

```
/latents      (7689, 745, 32)  # Per-timestep latent
/mu           (7689, 32)       # Patient-level mean embedding
/logvar       (7689, 32)       # Patient-level variance
/recon_error  (7689,)          # Per-patient reconstruction quality
```

---

## Component 3: DTW Clustering (Validation)

**Purpose:** Validate that learned embeddings preserve trajectory similarity.

```python
from tslearn.clustering import TimeSeriesKMeans

model = TimeSeriesKMeans(
    n_clusters=k,
    metric="dtw",
    max_iter=50,
    n_jobs=-1
)
```

**Computational strategy:**
- Downsample to key windows (acute 0-24h, early 24-72h)
- Run on 2-3 vitals (HR, MAP)
- Use as validation baseline, not primary clustering

---

## Component 4: HDBSCAN on Embeddings (Primary)

**Purpose:** Identify trajectory phenotypes with automatic cluster detection.

```python
import hdbscan

# Combine FPCA scores (70) + VAE mu (32) = 102 features
combined = np.hstack([fpca_scores, vae_mu])

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=50,
    min_samples=10,
    metric='euclidean',
    cluster_selection_method='eom'
)
labels = clusterer.fit_predict(combined)
```

**Benefits:** Automatic k, identifies outliers, soft probabilities.

### Output Schema

| File | Columns |
|------|---------|
| `clusters_dtw.parquet` | EMPI, cluster_dtw_hr, cluster_dtw_map, silhouette_score |
| `clusters_embedding.parquet` | EMPI, cluster_id, cluster_prob, is_outlier, nearest_cluster |

---

## Component 5: World Model States

**Purpose:** Create unified state vectors for downstream tasks.

### State Vector Composition (~100 dimensions)

| Block | Dims | Source | Description |
|-------|------|--------|-------------|
| **Raw vitals** | 7 | Layer 2 | Current HR, SBP, DBP, MAP, RR, SPO2, TEMP |
| **Vital masks** | 7 | Layer 2 | Observation indicators |
| **Time deltas** | 7 | Layer 2 | Hours since last observation |
| **Temporal position** | 2 | - | hours_from_pe, normalized_time |
| **FPCA scores** | 21 | Layer 4 | Top 3 FPCs per vital |
| **VAE latent** | 32 | Layer 4 | Current timestep latent |
| **Trend features** | 7 | Layer 3 | 6h slopes per vital |
| **Variability** | 7 | Layer 3 | RMSSD per vital |
| **Cluster context** | 5 | Layer 4 | One-hot cluster + outlier flag |
| **Treatment (reserved)** | ~15 | Future | For meds/labs/imaging |
| **Total** | ~100 | | |

### Schema Documentation (`state_schema.json`)

```json
{
  "version": "1.0",
  "dimensions": 100,
  "blocks": [
    {"name": "raw_vitals", "start": 0, "end": 7, "source": "layer2"},
    {"name": "vital_masks", "start": 7, "end": 14, "source": "layer2"},
    {"name": "time_deltas", "start": 14, "end": 21, "source": "layer2"},
    {"name": "temporal_position", "start": 21, "end": 23, "source": "computed"},
    {"name": "fpca_top3", "start": 23, "end": 44, "source": "layer4_fpca"},
    {"name": "vae_latent", "start": 44, "end": 76, "source": "layer4_vae"},
    {"name": "trend_slopes", "start": 76, "end": 83, "source": "layer3"},
    {"name": "variability", "start": 83, "end": 90, "source": "layer3"},
    {"name": "cluster_context", "start": 90, "end": 95, "source": "layer4_cluster"},
    {"name": "treatment_reserved", "start": 95, "end": 100, "source": "future"}
  ],
  "treatment_slots": {
    "medications": [95, 97],
    "labs": [97, 99],
    "imaging_procedures": [99, 100]
  }
}
```

---

## Implementation Plan

### File Structure

```
processing/
├── layer4/
│   ├── __init__.py
│   ├── fpca_builder.py
│   ├── vae_model.py
│   ├── vae_trainer.py
│   ├── clustering_dtw.py
│   └── clustering_embedding.py
├── layer4_builder.py
├── layer5_builder.py
tests/
├── test_layer4/
│   ├── test_fpca_builder.py
│   ├── test_vae_model.py
│   ├── test_clustering.py
└── test_layer5_builder.py
```

### Implementation Order

| Step | Component | Est. Time | Validates |
|------|-----------|-----------|-----------|
| 1 | FPCA builder | 1 day | Explained variance >80% |
| 2 | VAE model + trainer | 2-3 days | Reconstruction loss converges |
| 3 | DTW clustering | 1 day | Silhouette >0.3 |
| 4 | HDBSCAN clustering | 0.5 day | Clusters clinically distinct |
| 5 | Layer 4 builder | 0.5 day | All outputs generated |
| 6 | Layer 5 builder | 1 day | State vectors complete |
| 7 | Integration tests | 1 day | End-to-end pipeline |

### Dependencies

```bash
pip install scikit-fda torch tslearn hdbscan
```

### Success Criteria

- **FPCA:** First 10 components explain >90% variance per vital
- **VAE:** Reconstruction MSE < 0.1 (normalized), KL < 10
- **Clustering:** 5-15 distinct phenotypes, <5% outliers
- **States:** No NaN in non-reserved dimensions

---

## Future Extensions

When treatment package modules are ready:

1. **Medications (Module 4):** Encode active drugs, doses, timing into reserved slots
2. **Labs:** Key values (troponin, BNP, D-dimer, creatinine)
3. **Imaging:** PE severity scores, RV strain indicators
4. **Procedures:** IVC filter, catheter-directed therapy flags

State schema version will increment when treatment slots are populated.

---

**Document Version:** 1.0
**Approved:** 2025-12-10
