"""
Module 4: Medication Processing Configuration
==============================================

Central configuration for the medication encoding pipeline.
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import yaml


# =============================================================================
# PATH CONFIGURATION
# =============================================================================

# Base paths
PROJECT_ROOT = Path("/home/moin/TDA_11_25")
MODULE_ROOT = PROJECT_ROOT / "module_04_medications"
DATA_DIR = PROJECT_ROOT / "Data"

# Module 1 outputs (for Time Zero reference)
MODULE_1_OUTPUT = PROJECT_ROOT / "module_1_core_infrastructure" / "outputs"
PATIENT_TIMELINES_PKL = MODULE_1_OUTPUT / "patient_timelines.pkl"

# Input data
MED_FILE = DATA_DIR / "Med.txt"
DEM_FILE = DATA_DIR / "Dem.txt"  # For patient weights

# RxNorm data
RXNORM_DIR = MODULE_ROOT / "data" / "rxnorm"
RXNORM_DB = RXNORM_DIR / "rxnorm.db"

# Output directories
BRONZE_DIR = MODULE_ROOT / "data" / "bronze"
SILVER_DIR = MODULE_ROOT / "data" / "silver"
GOLD_DIR = MODULE_ROOT / "data" / "gold"
EMBEDDINGS_DIR = MODULE_ROOT / "data" / "embeddings"
EXPORTS_DIR = MODULE_ROOT / "exports"

# Config files
CONFIG_DIR = MODULE_ROOT / "config"
THERAPEUTIC_CLASSES_YAML = CONFIG_DIR / "therapeutic_classes.yaml"
DOSE_PATTERNS_YAML = CONFIG_DIR / "dose_patterns.yaml"


# =============================================================================
# TEMPORAL CONFIGURATION
# =============================================================================

@dataclass
class TemporalConfig:
    """Temporal window and alignment settings."""

    # Study window relative to Time Zero (hours)
    study_window_start: int = -720  # -30 days
    study_window_end: int = 720     # +30 days

    # Clinical phase windows (hours relative to Time Zero)
    windows: Dict[str, Tuple[int, int]] = field(default_factory=lambda: {
        'baseline': (-72, 0),
        'acute': (0, 24),
        'subacute': (24, 72),
        'recovery': (72, 168),
    })

    # Temporal resolutions to generate
    resolutions: List[str] = field(default_factory=lambda: [
        'daily',    # Primary storage
        'hourly',   # Imputed for GRU-D alignment
        'window',   # Aggregated by clinical phase
    ])

    # Hourly imputation settings
    default_admin_hour: int = 9  # Default hour for unknown frequency


TEMPORAL_CONFIG = TemporalConfig()


# =============================================================================
# RXNORM MAPPING CONFIGURATION
# =============================================================================

@dataclass
class RxNormConfig:
    """RxNorm mapping settings."""

    # Mapping thresholds
    fuzzy_match_threshold: float = 0.85
    target_mapping_rate: float = 0.85

    # RxNorm term types to prefer (in order)
    preferred_tty: List[str] = field(default_factory=lambda: [
        'SCD',  # Semantic Clinical Drug
        'SBD',  # Semantic Branded Drug
        'SCDC', # Semantic Clinical Drug Component
        'SBDC', # Semantic Branded Drug Component
        'IN',   # Ingredient
        'MIN',  # Multiple Ingredients
        'PIN',  # Precise Ingredient
    ])

    # UMLS download URL
    umls_signup_url: str = "https://uts.nlm.nih.gov/uts/signup-login"
    rxnorm_download_url: str = "https://www.nlm.nih.gov/research/umls/rxnorm/docs/rxnormfiles.html"

    # SQLite settings
    sqlite_page_size: int = 4096
    sqlite_cache_size: int = 10000


RXNORM_CONFIG = RxNormConfig()


# =============================================================================
# LLM CONFIGURATION
# =============================================================================

@dataclass
class LLMConfig:
    """LLM settings for ambiguous dose parsing."""

    # Models to benchmark
    benchmark_models: List[Dict[str, str]] = field(default_factory=lambda: [
        {"name": "llama3-8b", "model_id": "meta-llama/Meta-Llama-3-8B-Instruct"},
        {"name": "mistral-7b", "model_id": "mistralai/Mistral-7B-Instruct-v0.2"},
        {"name": "phi3-mini", "model_id": "microsoft/Phi-3-mini-4k-instruct"},
        {"name": "gemma2-9b", "model_id": "google/gemma-2-9b-it"},
        {"name": "qwen2.5-7b", "model_id": "Qwen/Qwen2.5-7B-Instruct"},
        {"name": "qwen2.5-14b", "model_id": "Qwen/Qwen2.5-14B-Instruct"},
    ])

    # Benchmark settings
    benchmark_sample_size: int = 200
    min_acceptable_accuracy: float = 0.70
    min_acceptable_speed: float = 10.0  # strings/second

    # Inference settings
    max_tokens: int = 256
    temperature: float = 0.0  # Deterministic for extraction
    batch_size: int = 32


LLM_CONFIG = LLMConfig()


# =============================================================================
# EMBEDDING CONFIGURATION
# =============================================================================

@dataclass
class EmbeddingConfig:
    """Embedding generation settings."""

    # Semantic embeddings
    semantic_model: str = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    semantic_dim: int = 768

    # Ontological embeddings (Node2Vec)
    ontological_dim: int = 128
    node2vec_walk_length: int = 80
    node2vec_num_walks: int = 10
    node2vec_p: float = 1.0  # Return parameter
    node2vec_q: float = 0.5  # In-out parameter (BFS-biased)

    # Co-occurrence embeddings (Word2Vec)
    cooccurrence_dim: int = 128
    word2vec_window: int = 5
    word2vec_min_count: int = 20
    word2vec_epochs: int = 10

    # Pharmacokinetic features
    pk_dim: int = 10
    pk_features: List[str] = field(default_factory=lambda: [
        'half_life_hours',
        'onset_minutes',
        'peak_hours',
        'duration_hours',
        'bioavailability',
        'protein_binding',
        'volume_distribution',
        'clearance_rate',
        'therapeutic_index',
        'active_metabolites',
    ])

    # Hierarchical composite
    hierarchical_dim: int = 128


EMBEDDING_CONFIG = EmbeddingConfig()


# =============================================================================
# LAYER CONFIGURATION
# =============================================================================

@dataclass
class LayerConfig:
    """Layer-specific settings."""

    # Layer 1: Canonical records
    chunk_size: int = 1_000_000  # Rows per chunk for streaming

    # Layer 2: Therapeutic classes
    n_classes: int = 53

    # Layer 3: Individual medications
    prevalence_threshold: int = 20  # Min patients for inclusion
    always_include_classes: List[str] = field(default_factory=lambda: [
        'anticoagulants',
        'vasopressors',
        'thrombolytics',
    ])

    # Layer 5: Dose intensity
    standardization_methods: List[str] = field(default_factory=lambda: [
        'raw',
        'ddd_normalized',
        'weight_adjusted',
    ])


LAYER_CONFIG = LayerConfig()


# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

@dataclass
class ValidationConfig:
    """Validation thresholds and targets."""

    # Layer 1
    target_rxnorm_mapping_rate: float = 0.85
    target_dose_parsing_rate: float = 0.80

    # Layer 2
    target_anticoag_24h_rate: float = 0.90

    # Layer 3
    max_pairwise_correlation: float = 0.99

    # Layer 4
    similar_pair_min_similarity: float = 0.70
    dissimilar_pair_max_similarity: float = 0.40

    # Known similar/dissimilar pairs for validation
    similar_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('enoxaparin', 'dalteparin'),
        ('rivaroxaban', 'apixaban'),
        ('metoprolol', 'atenolol'),
        ('furosemide', 'bumetanide'),
    ])

    dissimilar_pairs: List[Tuple[str, str]] = field(default_factory=lambda: [
        ('enoxaparin', 'metoprolol'),
        ('heparin', 'omeprazole'),
        ('warfarin', 'albuterol'),
    ])


VALIDATION_CONFIG = ValidationConfig()


# =============================================================================
# EXPORT CONFIGURATION
# =============================================================================

@dataclass
class ExportConfig:
    """Method-specific export settings."""

    # GRU-D
    grud_n_hours: int = 168  # 7 days
    grud_n_features: int = 123  # 53 classes + 50 individual + 20 dose

    # GBTM
    gbtm_n_days: int = 7

    # XGBoost
    xgboost_embedding_pcs: int = 50
    xgboost_top_individual_meds: int = 100

    # World models
    world_model_action_dim: int = 128


EXPORT_CONFIG = ExportConfig()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_therapeutic_classes() -> Dict:
    """Load therapeutic class definitions from YAML."""
    with open(THERAPEUTIC_CLASSES_YAML, 'r') as f:
        return yaml.safe_load(f)


def load_dose_patterns() -> Dict:
    """Load dose parsing patterns from YAML."""
    with open(DOSE_PATTERNS_YAML, 'r') as f:
        return yaml.safe_load(f)


def ensure_directories():
    """Create all required output directories."""
    for dir_path in [
        BRONZE_DIR,
        SILVER_DIR,
        GOLD_DIR / "therapeutic_classes",
        GOLD_DIR / "individual_indicators",
        GOLD_DIR / "dose_intensity",
        EMBEDDINGS_DIR,
        EXPORTS_DIR,
        RXNORM_DIR,
    ]:
        dir_path.mkdir(parents=True, exist_ok=True)


def get_all_class_ids() -> List[str]:
    """Get list of all 53 therapeutic class IDs."""
    classes = load_therapeutic_classes()
    class_ids = []

    for category in classes.values():
        if isinstance(category, dict):
            for class_id in category.keys():
                if not class_id.startswith('_'):
                    class_ids.append(class_id)

    return class_ids


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("Module 4: Medication Processing Configuration")
    print("=" * 60)
    print(f"\nProject Root: {PROJECT_ROOT}")
    print(f"Module Root: {MODULE_ROOT}")
    print(f"Input Med File: {MED_FILE}")
    print(f"RxNorm DB: {RXNORM_DB}")
    print(f"\nTemporal Windows:")
    for name, (start, end) in TEMPORAL_CONFIG.windows.items():
        print(f"  {name}: {start}h to {end}h")
    print(f"\nTarget RxNorm Mapping Rate: {RXNORM_CONFIG.target_mapping_rate:.0%}")
    print(f"LLM Models to Benchmark: {len(LLM_CONFIG.benchmark_models)}")
    print(f"Embedding Types: 5")
    print(f"  - Semantic: {EMBEDDING_CONFIG.semantic_dim} dims")
    print(f"  - Ontological: {EMBEDDING_CONFIG.ontological_dim} dims")
    print(f"  - Co-occurrence: {EMBEDDING_CONFIG.cooccurrence_dim} dims")
    print(f"  - Pharmacokinetic: {EMBEDDING_CONFIG.pk_dim} dims")
    print(f"  - Hierarchical: {EMBEDDING_CONFIG.hierarchical_dim} dims")
    print(f"\nTherapeutic Classes: {LAYER_CONFIG.n_classes}")
    print(f"Individual Med Prevalence Threshold: {LAYER_CONFIG.prevalence_threshold} patients")
    print("=" * 60)
