# Multi-Scale Conv1D VAE Design

**Date:** 2025-12-17
**Status:** Approved
**Problem:** LSTM-VAE suffers from posterior collapse (KL → 0), producing useless latent representations

## Context

- Current LSTM-VAE collapses despite: β-annealing, free bits, weaker decoder
- Need fixed-size patient embeddings for risk prediction and clinical assist tools
- Must capture both short-term fluctuations AND long-term trends
- FPCA works well (85% variance, 19 clusters) but misses cross-vital patterns

## Solution: Multi-Scale Conv1D VAE

Parallel convolutional branches operating at different temporal scales, merged before latent bottleneck.

### Architecture Overview

```
Input: (batch, 745, 7)  [time, vitals]
         │
    ┌────┴────┬────────┬────────┐
    ▼         ▼        ▼        ▼
 Branch 1  Branch 2  Branch 3  Branch 4
 k=3,5     k=15,31   k=63,127  k=255
 (local)   (hourly)  (daily)   (multi-day)
    │         │        │        │
    └────┬────┴────────┴────────┘
         ▼
   Concatenate + Pool
         │
         ▼
    FC → μ, logσ² (latent_dim=32)
         │
         ▼
   Decoder (symmetric multi-branch)
         │
         ▼
Output: (batch, 745, 7)
```

### Encoder Design

Each branch progressively downsamples while extracting scale-specific features:

**Branch 1 (Local - kernels 3,5):**
```
Conv1d(7→32, k=3, stride=2)  → 373 timesteps
Conv1d(32→64, k=5, stride=2) → 187 timesteps
Conv1d(64→64, k=5, stride=2) → 94 timesteps
→ Output: (batch, 64, 94)
```

**Branch 2 (Hourly - kernels 15,31):**
```
Conv1d(7→32, k=15, stride=4)  → 186 timesteps
Conv1d(32→64, k=31, stride=4) → 47 timesteps
→ Output: (batch, 64, 47)
```

**Branch 3 (Daily - kernels 63,127):**
```
Conv1d(7→32, k=63, stride=8)   → 93 timesteps
Conv1d(32→64, k=127, stride=8) → 12 timesteps
→ Output: (batch, 64, 12)
```

**Branch 4 (Multi-day - kernel 255):**
```
Conv1d(7→64, k=255, stride=16) → 47 timesteps
→ Output: (batch, 64, 47)
```

**Merging:**
```
Each branch → AdaptiveAvgPool1d(1) → (batch, 64)
Concatenate all branches → (batch, 256)
FC(256 → 64) → LeakyReLU
FC(64 → 32) for μ
FC(64 → 32) for logσ²
```

### Decoder Design

Mirrors encoder with transposed convolutions. Key insight: multi-branch decoding prevents collapse by requiring multi-scale information.

```
Latent z (batch, 32)
    │
    ▼
FC(32 → 256) → Reshape to (batch, 64, 4) × 4 branches
    │
    ┌───┴────┬────────┬────────┐
    ▼        ▼        ▼        ▼
 Branch 1  Branch 2  Branch 3  Branch 4
 (local)   (hourly)  (daily)   (multi-day)
    │        │        │        │
    └───┬────┴────────┴────────┘
        ▼
  Concat along channel dim → (batch, 256, 745)
        │
        ▼
  Conv1d(256 → 64, k=3) → Conv1d(64 → 7, k=3)
        │
        ▼
  Output: (batch, 745, 7)
```

### Training Strategy

**Loss Function:**
```
L = L_recon + β * L_kl

Where:
- L_recon = (L_local + L_hourly + L_daily + L_multiday) / 4
- L_kl = KL divergence with free bits (λ=2.0)
```

**Anti-collapse measures:**

| Technique | Setting | Purpose |
|-----------|---------|---------|
| β-annealing | 0→0.5 over 30 epochs | Let encoder learn before KL kicks in |
| Free bits | λ=2.0 per dimension | Guarantee minimum information in latent |
| Cyclical annealing | Reset β every 40 epochs | Escape local minima |
| Per-branch recon loss | Equal weights | Force multi-scale encoding |

**Training config:**
- Epochs: 150
- Batch size: 64
- Learning rate: 1e-3 with cosine decay
- Early stopping: patience=30, monitor val_loss

### Integration

**Config in layer4_builder.py:**
```python
'vae': {
    'model_type': 'multiscale_conv1d',
    'latent_dim': 32,
    'base_channels': 64,
    'scales': [3, 15, 63, 255],
    'beta': 0.5,
    'beta_warmup_epochs': 30,
    'cyclical_period': 40,
    'free_bits': 2.0,
    'epochs': 150,
    'patience': 30,
}
```

**Output artifacts:**
- `vae_latents.h5` - mu, logvar, patient_ids (same format)
- `vae_model.pt` - Model checkpoint
- `vae_multiscale_components.h5` - Per-branch reconstructions (debug)

### Success Criteria

**Primary (must pass):**

| Metric | Failure | Success | Target |
|--------|---------|---------|--------|
| KL divergence | 0.000 | >0.01 | 0.1-1.0 |
| Latent std | <0.01 | >0.1 | 0.5-1.0 |
| Recon loss | N/A | Converges | <0.02 |

**Validation steps:**
1. Check KL didn't collapse (>0.01 at convergence)
2. Check latent statistics: `mu.std() > 0.1`, `logvar.mean() > -5`
3. Run HDBSCAN on FPCA+VAE, compare to FPCA-only
4. Spot-check reconstructions for a few patients

### Fallback Plan

If Multi-Scale Conv1D still collapses → implement VQ-VAE (discrete latent space), which eliminates collapse by design.
