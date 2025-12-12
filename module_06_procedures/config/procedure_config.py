"""
Module 6: Procedure Processing Configuration
============================================

Central configuration for the procedure encoding pipeline.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import yaml


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

PROJECT_ROOT = Path("/home/moin/TDA_11_25")
MODULE_ROOT = PROJECT_ROOT / "module_06_procedures"
DATA_DIR = PROJECT_ROOT / "Data"

# Module 1 outputs (for Time Zero reference)
MODULE_1_OUTPUT = PROJECT_ROOT / "module_1_core_infrastructure" / "outputs"
PATIENT_TIMELINES_PKL = MODULE_1_OUTPUT / "patient_timelines.pkl"

# Input data
PRC_FILE = DATA_DIR / "Prc.txt"

# Vocabulary data
VOCAB_DIR = MODULE_ROOT / "data" / "vocabularies"
CCS_CROSSWALK = VOCAB_DIR / "ccs_crosswalk.csv"
SNOMED_DB = VOCAB_DIR / "snomed_procedures.db"

# Output directories
BRONZE_DIR = MODULE_ROOT / "data" / "bronze"
SILVER_DIR = MODULE_ROOT / "data" / "silver"
GOLD_DIR = MODULE_ROOT / "data" / "gold"
EMBEDDINGS_DIR = MODULE_ROOT / "data" / "embeddings"
EXPORTS_DIR = MODULE_ROOT / "exports"

# Config files
CONFIG_DIR = MODULE_ROOT / "config"
CCS_MAPPINGS_YAML = CONFIG_DIR / "ccs_mappings.yaml"
PE_PROCEDURE_CODES_YAML = CONFIG_DIR / "pe_procedure_codes.yaml"
SURGICAL_RISK_YAML = CONFIG_DIR / "surgical_risk.yaml"
DISCRETION_WEIGHTS_YAML = CONFIG_DIR / "discretion_weights.yaml"


# =============================================================================
# TEMPORAL CONFIGURATION
# =============================================================================

@dataclass
class TemporalConfig:
    """Temporal window and alignment settings."""

    # Study window relative to Time Zero (hours)
    study_window_start: int = -8760  # -365 days (lifetime history)
    study_window_end: int = 720      # +30 days

    # Seven temporal windows (hours relative to PE Time Zero)
    windows: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'lifetime_history': (-87600, -720),       # Before -30 days
        'remote_antecedent': (-720, -720),        # Placeholder - actually everything before provoking
        'provoking_window': (-720, 0),            # 1-30 days pre-PE
        'diagnostic_workup': (-24, 24),           # ±24h of PE
        'initial_treatment': (0, 72),             # 0-72h post-PE
        'escalation': (72, 720),                  # >72h during hospitalization
        'post_discharge': (720, 8760),            # After discharge
    })


TEMPORAL_CONFIG = TemporalConfig()


# =============================================================================
# CODE MAPPING CONFIGURATION
# =============================================================================

@dataclass
class MappingConfig:
    """Code mapping settings."""

    fuzzy_match_threshold: float = 0.85
    target_mapping_rate: float = 0.85

    # LLM settings for EPIC code classification
    llm_batch_size: int = 32
    llm_max_tokens: int = 256
    llm_temperature: float = 0.0


MAPPING_CONFIG = MappingConfig()


# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Embedding generation settings."""

    ontological_dim: int = 128
    semantic_dim: int = 768
    semantic_reduced_dim: int = 128
    temporal_sequence_dim: int = 128
    ccs_cooccurrence_dim: int = 128
    pe_cooccurrence_dim: int = 64
    complexity_dim: int = 16

    # Embedding weights
    weights: Dict[str, float] = field(default_factory=lambda: {
        'ontological': 1.0,
        'semantic': 0.8,
        'temporal_sequence': 0.9,
        'ccs_cooccurrence': 0.6,
        'pe_cooccurrence': 0.5,
        'complexity': 0.7,
    })


EMBEDDING_CONFIG = EmbeddingConfig()


# =============================================================================
# LAYER CONFIGURATION
# =============================================================================

@dataclass
class LayerConfig:
    """Layer-specific settings."""

    chunk_size: int = 1_000_000  # Rows per chunk for streaming


LAYER_CONFIG = LayerConfig()


# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Validation thresholds and targets."""

    # Layer 1
    target_records_loaded: int = 22_000_000
    target_patients_in_cohort: int = 8700
    target_pe_linkage_rate: float = 0.95

    # Layer 2 (Mapping)
    target_ccs_mapping_cpt: float = 0.95
    target_ccs_mapping_overall: float = 0.85
    target_llm_accuracy: float = 0.85

    # Layer 3 (PE Features)
    expected_cta_rate: Tuple[float, float] = (0.80, 0.95)
    expected_echo_rate: Tuple[float, float] = (0.50, 0.70)
    expected_intubation_rate: Tuple[float, float] = (0.05, 0.15)
    expected_ivc_filter_rate: Tuple[float, float] = (0.05, 0.15)
    expected_ecmo_rate: Tuple[float, float] = (0.00, 0.02)


VALIDATION_CONFIG = ValidationConfig()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def ensure_directories():
    """Create all required output directories."""
    for dir_path in [
        BRONZE_DIR,
        SILVER_DIR,
        GOLD_DIR / "ccs_indicators",
        GOLD_DIR / "pe_procedure_features",
        GOLD_DIR / "world_model_states",
        EMBEDDINGS_DIR,
        EXPORTS_DIR,
        VOCAB_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Module 6: Procedure Processing Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Module Root: {MODULE_ROOT}")
    print(f"Input Prc File: {PRC_FILE}")
    print(f"\nTemporal Windows:")
    for name, (start, end) in TEMPORAL_CONFIG.windows.items():
        print(f"  {name}: {start}h to {end}h")
    print(f"\nTarget CCS Mapping Rate: {MAPPING_CONFIG.target_mapping_rate:.0%}")
    print("=" * 60)
