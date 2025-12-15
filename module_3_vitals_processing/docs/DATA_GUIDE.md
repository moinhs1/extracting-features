# Module 3: Vitals Data Guide

**Purpose:** Quick reference for finding the right vitals data for your analysis task.

---

## Quick Decision Tree

```
What analysis are you doing?
│
├─► Traditional ML (Logistic Regression, XGBoost, Random Forest)
│   └─► Use: summary_features.parquet (7,696 patients × 4,426 features)
│
├─► Time Series / Sequential Models (LSTM, Transformer, RNN)
│   └─► Use: hourly_tensors.h5 (7,696 × 745 hours × 7 vitals)
│
├─► Topological Data Analysis (TDA)
│   └─► Use: hourly_tensors.h5 or timeseries_features.parquet
│
├─► World Model / Reinforcement Learning
│   └─► Use: world_states.h5 (7,696 × 745 × 100 state dims)
│
├─► Trajectory Clustering / Phenotyping
│   └─► Use: fpca_scores.parquet + vae_latents.h5 + clusters_*.parquet
│
├─► Custom Feature Engineering
│   └─► Use: hourly_grid.parquet or canonical_vitals.parquet
│
└─► Raw Text Extraction Analysis
    └─► Use: outputs/discovery/*_raw.parquet
```

---

## Data Files Reference

### Layer 1: Canonical Records (Raw Merged)

**File:** `outputs/layer1/canonical_vitals.parquet`

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient ID |
| timestamp | datetime | Measurement time |
| hours_from_pe | float | Hours relative to PE diagnosis |
| vital_type | str | HR, SBP, DBP, MAP, RR, SPO2, TEMP |
| value | float | Measurement value |
| units | str | bpm, mmHg, %, °C |
| source | str | 'phy', 'hnp', 'prg' |
| confidence | float | Extraction confidence (0.65-1.0) |
| is_flagged_abnormal | bool | Outside normal range |

**Use when:** You need raw individual measurements with timestamps, want to do custom aggregation, or need to trace data provenance.

```python
import pandas as pd
df = pd.read_parquet('outputs/layer1/canonical_vitals.parquet')

# Filter to specific patient and vital
patient_hr = df[(df['EMPI'] == '12345') & (df['vital_type'] == 'HR')]
```

---

### Layer 2: Hourly Grid (Time-Aligned)

**File:** `outputs/layer2/hourly_grid.parquet`

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient ID |
| hour_from_pe | int | Hour relative to PE (-24 to +720) |
| vital_type | str | Vital type |
| mean, median, std, min, max | float | Aggregated values |
| count | int | Observations in hour |
| mask | int | 1=observed, 0=imputed |

**File:** `outputs/layer2/hourly_tensors.h5`

```
/values           (7696, 745, 7)   float32  # Vital values
/masks            (7696, 745, 7)   int8     # 1=observed, 0=imputed
/time_deltas      (7696, 745, 7)   float32  # Hours since last observation
/imputation_tier  (7696, 745, 7)   int8     # 1-4 tier indicators
/patient_index    (7696,)          str      # EMPI mapping
/vital_index      (7,)             str      # ['HR','SBP','DBP','MAP','RR','SPO2','TEMP']
/hour_index       (745,)           int      # [-24 to 720]
```

**Use when:** LSTM/RNN models, TDA on time series, any analysis needing regular time intervals.

```python
import h5py
import numpy as np

with h5py.File('outputs/layer2/hourly_tensors.h5', 'r') as f:
    values = f['values'][:]        # (7696, 745, 7)
    masks = f['masks'][:]          # Which values are real vs imputed
    patients = f['patient_index'][:].astype(str)
    vitals = f['vital_index'][:].astype(str)

# Get HR for all patients (index 0)
hr_idx = list(vitals).index('HR')
all_hr = values[:, :, hr_idx]  # (7696, 745)

# Get acute phase (hours 0-24) for patient 0
acute_hr = values[0, 24:48, hr_idx]  # 24 hours (hour 0 is at index 24)
```

---

### Layer 3: Engineered Features

**File:** `outputs/layer3/timeseries_features.parquet` (5.7M rows × 315 cols)

Per patient-hour features including:
- Rolling statistics (6h/12h/24h windows)
- Trend features (slope, R², direction)
- Variability (RMSSD, successive variance)
- Composite vitals (shock_index, pulse_pressure)
- Threshold hours (cumulative abnormal time)

**File:** `outputs/layer3/summary_features.parquet` (7,696 rows × 4,426 cols)

Per-patient summary across 5 clinical windows:
- `pre` (-24 to 0h): Pre-PE baseline
- `acute` (0 to 24h): Acute phase
- `early` (24 to 72h): Early treatment
- `stab` (72 to 168h): Stabilization
- `recov` (168 to 720h): Recovery

**Use when:** Traditional ML (logistic regression, XGBoost), feature importance analysis, interpretable models.

```python
import pandas as pd

# For traditional ML - one row per patient
summary = pd.read_parquet('outputs/layer3/summary_features.parquet')

# Example features
X = summary[[
    'HR_acute_mean', 'HR_acute_max',
    'SBP_acute_min', 'shock_index_acute_max',
    'SPO2_acute_min', 'hours_hypoxemia_acute_mean',
    'RR_slope6h_acute_mean'
]]

# For time-series analysis
ts_features = pd.read_parquet('outputs/layer3/timeseries_features.parquet')
```

---

### Layer 4: Embeddings & Clusters

**File:** `outputs/layer4/fpca_scores.parquet` (7,696 × 77 features)

Functional PCA scores capturing trajectory shapes:
- `{vital}_fpc1` to `{vital}_fpc10` for each vital
- `{vital}_obs_pct` observation percentage

**File:** `outputs/layer4/vae_latents.h5`

```
/mu           (7696, 32)   # Mean latent embedding
/logvar       (7696, 32)   # Log variance
/recon_error  (7696,)      # Reconstruction error
/patient_index (7696,)     # EMPI mapping
```

**File:** `outputs/layer4/clusters_embedding.parquet`

| Column | Description |
|--------|-------------|
| EMPI | Patient ID |
| cluster_id | Cluster assignment (-1=outlier) |
| cluster_prob | Soft probability |
| is_outlier | Outlier flag |

**Use when:** Phenotype discovery, trajectory clustering, dimensionality reduction, unsupervised learning.

```python
import pandas as pd
import h5py

# FPCA features
fpca = pd.read_parquet('outputs/layer4/fpca_scores.parquet')
hr_shape = fpca[['HR_fpc1', 'HR_fpc2', 'HR_fpc3']]  # Trajectory shape

# VAE latents
with h5py.File('outputs/layer4/vae_latents.h5', 'r') as f:
    latent = f['mu'][:]  # (7696, 32) learned representation

# Clusters
clusters = pd.read_parquet('outputs/layer4/clusters_embedding.parquet')
```

---

### Layer 5: World Model States

**File:** `outputs/layer5/world_states.h5`

```
/states         (7696, 745, 100)  float32  # State vectors
/patient_index  (7696,)           str      # EMPI mapping
```

**State vector structure (100 dimensions):**

| Block | Dims | Range | Content |
|-------|------|-------|---------|
| raw_vitals | 7 | 0-6 | HR, SBP, DBP, MAP, RR, SPO2, TEMP |
| vital_masks | 7 | 7-13 | Observation indicators |
| time_deltas | 7 | 14-20 | Hours since last observation |
| temporal_position | 2 | 21-22 | hours_from_pe, normalized_time |
| fpca_top3 | 21 | 23-43 | Top 3 FPCs per vital |
| vae_latent | 32 | 44-75 | VAE embedding |
| trend_slopes | 7 | 76-82 | 6h slopes |
| variability | 7 | 83-89 | RMSSD |
| cluster_context | 5 | 90-94 | Cluster one-hot |
| treatment_reserved | 5 | 95-99 | Future: meds/labs |

**Use when:** World models, RL state representation, transformer models needing rich state.

```python
import h5py
import json

with h5py.File('outputs/layer5/world_states.h5', 'r') as f:
    states = f['states'][:]  # (7696, 745, 100)
    patients = f['patient_index'][:].astype(str)

# Load schema for dimension mapping
with open('outputs/layer5/state_schema.json') as f:
    schema = json.load(f)

# Extract specific blocks
raw_vitals = states[:, :, 0:7]       # Current vital values
vae_latent = states[:, :, 44:76]     # Learned embeddings
trend_slopes = states[:, :, 76:83]   # Recent trends
```

---

## Use Case Examples

### 1. Logistic Regression for PE Mortality

```python
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load summary features (one row per patient)
features = pd.read_parquet('outputs/layer3/summary_features.parquet')

# Select clinically relevant features
X = features[[
    'HR_acute_mean', 'HR_acute_max',
    'SBP_acute_min', 'SBP_acute_mean',
    'shock_index_acute_max',
    'SPO2_acute_min', 'SPO2_acute_mean',
    'RR_acute_max',
    'hours_hypoxemia_acute_mean',
    'hours_tachycardia_acute_mean'
]].fillna(0)

# y = your outcome labels
# model = LogisticRegression().fit(X_train, y_train)
```

### 2. LSTM for Vital Trajectory Prediction

```python
import h5py
import torch
from torch.utils.data import Dataset, DataLoader

class VitalsDataset(Dataset):
    def __init__(self, h5_path):
        with h5py.File(h5_path, 'r') as f:
            self.values = f['values'][:]      # (N, 745, 7)
            self.masks = f['masks'][:]

    def __len__(self):
        return len(self.values)

    def __getitem__(self, idx):
        # Return sequence and mask
        return torch.tensor(self.values[idx]), torch.tensor(self.masks[idx])

dataset = VitalsDataset('outputs/layer2/hourly_tensors.h5')
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Your LSTM model here
# for batch_values, batch_masks in loader:
#     output = model(batch_values)
```

### 3. TDA (Topological Data Analysis)

```python
import h5py
import numpy as np
from ripser import ripser
from persim import plot_diagrams

with h5py.File('outputs/layer2/hourly_tensors.h5', 'r') as f:
    values = f['values'][:]

# Extract HR trajectories for acute phase (hours 0-24)
hr_idx = 0  # HR is first vital
acute_start = 24  # hour 0 is at index 24
acute_end = 48    # hour 24

hr_acute = values[:, acute_start:acute_end, hr_idx]  # (7696, 24)

# Compute persistence diagrams
# result = ripser(hr_acute, maxdim=1)
# plot_diagrams(result['dgms'])
```

### 4. World Model / Decision Transformer

```python
import h5py
import torch

with h5py.File('outputs/layer5/world_states.h5', 'r') as f:
    states = f['states'][:]  # (7696, 745, 100)

# States are pre-computed feature vectors per timestep
# Perfect for sequence models expecting fixed-dim input

class WorldModelDataset(torch.utils.data.Dataset):
    def __init__(self, states, seq_len=24):
        self.states = states
        self.seq_len = seq_len

    def __getitem__(self, idx):
        patient_idx = idx // (745 - self.seq_len)
        start_hour = idx % (745 - self.seq_len)

        sequence = self.states[patient_idx, start_hour:start_hour+self.seq_len]
        next_state = self.states[patient_idx, start_hour+self.seq_len]

        return torch.tensor(sequence), torch.tensor(next_state)
```

### 5. Phenotype Discovery with Clustering

```python
import pandas as pd
import h5py
from sklearn.cluster import KMeans

# Combine FPCA and VAE features
fpca = pd.read_parquet('outputs/layer4/fpca_scores.parquet')

with h5py.File('outputs/layer4/vae_latents.h5', 'r') as f:
    vae_mu = f['mu'][:]
    patient_ids = f['patient_index'][:].astype(str)

# Combine embeddings
combined = np.hstack([
    fpca.drop('EMPI', axis=1).values,  # FPCA features
    vae_mu                              # VAE latents
])

# Cluster
kmeans = KMeans(n_clusters=5)
clusters = kmeans.fit_predict(combined)

# Analyze clusters by vital patterns
```

---

## File Sizes Reference

| File | Size | Shape | Best For |
|------|------|-------|----------|
| `canonical_vitals.parquet` | 30 MB | 3.5M rows | Custom aggregation |
| `hourly_grid.parquet` | 35 MB | 5.7M rows | Pandas time series |
| `hourly_tensors.h5` | 24 MB | 7696×745×7 | LSTM, TDA |
| `timeseries_features.parquet` | 988 MB | 5.7M×315 | Time-varying features |
| `summary_features.parquet` | 96 MB | 7696×4426 | Traditional ML |
| `fpca_scores.parquet` | 4.7 MB | 7696×77 | Trajectory shape |
| `vae_latents.h5` | 2.0 MB | 7696×32 | Learned embedding |
| `world_states.h5` | 109 MB | 7696×745×100 | World models, RL |

---

## Temporal Reference

All timestamps are relative to PE diagnosis (time zero):

| Hour | Meaning | Index in tensors |
|------|---------|------------------|
| -24 | 24h before PE | 0 |
| 0 | PE diagnosis time | 24 |
| +24 | 24h after PE | 48 |
| +720 | 30 days after PE | 744 |

**Clinical windows:**
- Pre-PE: hours -24 to 0 (baseline)
- Acute: hours 0 to 24 (first day)
- Early: hours 24 to 72 (days 1-3)
- Stabilization: hours 72 to 168 (days 3-7)
- Recovery: hours 168 to 720 (days 7-30)

---

## Vital Index Reference

For tensor indexing:

| Index | Vital | Units | Normal Range |
|-------|-------|-------|--------------|
| 0 | HR | bpm | 60-100 |
| 1 | SBP | mmHg | 90-140 |
| 2 | DBP | mmHg | 60-90 |
| 3 | MAP | mmHg | 70-100 |
| 4 | RR | /min | 12-20 |
| 5 | SPO2 | % | 95-100 |
| 6 | TEMP | °C | 36.5-37.5 |

---

## Questions?

- **Need more features?** Check `timeseries_features.parquet` columns
- **Need raw measurements?** Use `canonical_vitals.parquet`
- **Need to trace extraction?** Check `outputs/discovery/*_raw.parquet`
- **Need patient IDs?** All files have EMPI column or patient_index dataset
