"""
Procedure Embedding Generator
==============================

Layer 4: Generate multiple types of procedure embeddings:
1. Procedural Complexity Features (16 dims) - hand-crafted features
2. CCS Co-occurrence (128 dims) - Word2Vec on CCS category sequences
3. Placeholders for future: Ontological, Semantic, Temporal (random initialization)

Output: HDF5 file with vocabulary-level embeddings.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import h5py
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import (
    SILVER_DIR,
    EMBEDDINGS_DIR,
    EMBEDDING_CONFIG,
)


# =============================================================================
# PROCEDURAL COMPLEXITY FEATURES (16 dimensions)
# =============================================================================

# Hand-crafted complexity features for common PE-relevant procedures
# Format: [invasiveness(0-3), vte_risk, bleeding_risk, is_emergent, requires_or,
#          requires_icu, typical_duration_h, anesthesia_type(0-3), is_diagnostic,
#          is_therapeutic, is_monitoring, requires_contrast, radiation_exposure,
#          patient_positioning_required, requires_sedation, reversibility]

COMPLEXITY_FEATURES = {
    # Diagnostic imaging
    '71275': [1, 0.0, 0.1, 1, 0, 0, 0.5, 0, 1, 0, 0, 1, 1, 1, 0, 1],  # CTA chest
    '93306': [0, 0.0, 0.0, 1, 0, 0, 0.5, 0, 1, 0, 0, 0, 0, 0, 0, 1],  # Echo TTE
    '93312': [1, 0.0, 0.1, 0, 0, 0, 1.0, 1, 1, 0, 0, 0, 0, 1, 1, 1],  # Echo TEE

    # Respiratory support
    '31500': [2, 0.2, 0.2, 1, 0, 1, 0.25, 2, 0, 1, 0, 0, 0, 1, 1, 0],  # Intubation
    '94002': [0, 0.3, 0.0, 1, 0, 1, 24.0, 0, 0, 1, 1, 0, 0, 1, 0, 0],  # Vent management
    '31600': [3, 0.4, 0.4, 0, 1, 1, 2.0, 3, 0, 1, 0, 0, 0, 1, 1, 0],  # Tracheostomy

    # Vascular access
    '36555': [1, 0.2, 0.3, 1, 0, 0, 0.5, 1, 0, 1, 0, 0, 0, 1, 1, 1],  # Central line
    '36620': [1, 0.1, 0.3, 1, 0, 0, 0.25, 1, 0, 1, 1, 0, 0, 0, 1, 1],  # Arterial line
    '93503': [2, 0.3, 0.4, 0, 0, 1, 1.0, 2, 1, 0, 1, 1, 1, 1, 1, 1],  # PA catheter

    # IVC filter
    '37191': [2, 0.5, 0.4, 1, 0, 0, 1.0, 2, 0, 1, 0, 1, 1, 1, 1, 0],  # IVC filter placement
    '37193': [2, 0.3, 0.4, 0, 0, 0, 1.0, 2, 0, 1, 0, 1, 1, 1, 1, 1],  # IVC filter retrieval

    # Catheter-directed therapy
    '37211': [2, 0.4, 0.6, 1, 0, 1, 2.0, 2, 0, 1, 0, 1, 1, 1, 1, 0],  # CDT
    '37212': [2, 0.4, 0.6, 1, 0, 1, 2.0, 2, 0, 1, 0, 1, 1, 1, 1, 0],  # CDT

    # Transfusion
    '36430': [1, 0.1, 0.0, 1, 0, 0, 0.5, 0, 0, 1, 0, 0, 0, 0, 0, 1],  # Transfusion

    # ECMO
    '33946': [3, 0.8, 0.8, 1, 1, 1, 3.0, 3, 0, 1, 1, 0, 0, 1, 1, 0],  # ECMO cannulation
    '33947': [3, 0.8, 0.8, 1, 1, 1, 3.0, 3, 0, 1, 1, 0, 0, 1, 1, 0],  # ECMO cannulation

    # Resuscitation
    '92950': [2, 0.0, 0.2, 1, 0, 1, 0.5, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # CPR

    # Thoracic procedures
    '32551': [2, 0.3, 0.4, 1, 0, 0, 0.5, 1, 0, 1, 0, 0, 0, 1, 1, 1],  # Chest tube
    '32554': [1, 0.2, 0.3, 1, 0, 0, 0.25, 1, 1, 1, 0, 0, 0, 1, 1, 1],  # Thoracentesis
}


def get_complexity_embedding(
    ccs_category: Optional[str] = None,
    procedure_name: Optional[str] = None,
    cpt_code: Optional[str] = None
) -> np.ndarray:
    """
    Get procedural complexity feature embedding.

    Args:
        ccs_category: CCS procedure category
        procedure_name: Procedure name/description
        cpt_code: CPT code

    Returns:
        16-dimensional complexity feature vector (normalized to [0, 1])
    """
    # Try to find features by CPT code first
    if cpt_code and cpt_code in COMPLEXITY_FEATURES:
        features = np.array(COMPLEXITY_FEATURES[cpt_code], dtype=np.float32)
    else:
        # Default features for unknown procedures
        # [invasiveness, vte_risk, bleeding_risk, is_emergent, requires_or,
        #  requires_icu, duration_h, anesthesia, is_diagnostic, is_therapeutic,
        #  is_monitoring, requires_contrast, radiation, positioning, sedation, reversible]
        features = np.array([0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1], dtype=np.float32)

    # Normalize to [0, 1] range
    # Max values for each feature
    max_vals = np.array([3, 1, 1, 1, 1, 1, 24, 3, 1, 1, 1, 1, 1, 1, 1, 1], dtype=np.float32)
    features = np.clip(features / max_vals, 0, 1)

    return features


# =============================================================================
# CCS CO-OCCURRENCE EMBEDDINGS (Word2Vec)
# =============================================================================

def build_ccs_sequences(df: pd.DataFrame) -> List[List[str]]:
    """
    Build CCS category sequences per encounter for Word2Vec training.

    Args:
        df: Mapped procedures DataFrame with encounter_id and ccs_category

    Returns:
        List of CCS category sequences (one per encounter)
    """
    sequences = []

    for encounter_id, group in df.groupby('encounter_id'):
        # Sort by time
        group = group.sort_values('hours_from_pe')

        # Get sequence of CCS categories
        ccs_cats = group['ccs_category'].dropna().astype(str).tolist()

        if len(ccs_cats) >= 2:  # Need at least 2 for context
            sequences.append(ccs_cats)

    return sequences


def train_ccs_word2vec(
    sequences: List[List[str]],
    vector_size: int = None,
    window: int = 5,
    min_count: int = 2,
    epochs: int = 20
):
    """
    Train Word2Vec model on CCS category sequences.

    Args:
        sequences: List of CCS category sequences
        vector_size: Embedding dimension
        window: Context window size
        min_count: Minimum frequency threshold
        epochs: Training epochs

    Returns:
        Trained Word2Vec model or None if gensim unavailable
    """
    try:
        from gensim.models import Word2Vec
    except ImportError:
        print("Warning: gensim not installed. CCS co-occurrence embeddings unavailable.")
        return None

    if vector_size is None:
        vector_size = EMBEDDING_CONFIG.ccs_cooccurrence_dim

    if len(sequences) == 0:
        print("Warning: No sequences available for Word2Vec training.")
        return None

    # Adjust min_count based on vocabulary size
    # For small test datasets, use min_count=1
    vocab_size = len(set(word for seq in sequences for word in seq))
    effective_min_count = min(min_count, max(1, vocab_size // 10))

    model = Word2Vec(
        sentences=sequences,
        vector_size=vector_size,
        window=window,
        min_count=effective_min_count,
        epochs=epochs,
        workers=4,
        sg=1,  # Skip-gram
        seed=42,
    )

    return model


def get_ccs_embedding(
    model,
    ccs_category: str
) -> Optional[np.ndarray]:
    """
    Get CCS co-occurrence embedding from trained model.

    Args:
        model: Trained Word2Vec model
        ccs_category: CCS category code

    Returns:
        Embedding vector or None if not in vocabulary
    """
    if model is None:
        return None

    try:
        return model.wv[str(ccs_category)]
    except KeyError:
        return None


# =============================================================================
# PLACEHOLDER EMBEDDINGS (for future implementation)
# =============================================================================

def generate_ontological_placeholder(
    procedures: List[str],
    dim: int = 128
) -> Dict[str, np.ndarray]:
    """
    Generate placeholder ontological embeddings (random initialization).

    Future: Implement Node2Vec on SNOMED + CCS hierarchy graph.

    Args:
        procedures: List of procedure codes/categories
        dim: Embedding dimension

    Returns:
        Dictionary mapping procedure to embedding
    """
    np.random.seed(42)
    embeddings = {}

    for proc in procedures:
        # Initialize with small random values
        embeddings[proc] = np.random.randn(dim).astype(np.float32) * 0.1

    return embeddings


def generate_semantic_placeholder(
    procedures: List[Tuple[str, str]],
    dim: int = 128
) -> Dict[Tuple[str, str], np.ndarray]:
    """
    Generate placeholder semantic embeddings (random initialization).

    Future: Implement BioBERT embeddings on procedure descriptions.

    Args:
        procedures: List of (code, name) tuples
        dim: Embedding dimension

    Returns:
        Dictionary mapping (code, name) to embedding
    """
    np.random.seed(43)
    embeddings = {}

    for proc in procedures:
        embeddings[proc] = np.random.randn(dim).astype(np.float32) * 0.1

    return embeddings


def generate_temporal_placeholder(
    procedures: List[str],
    dim: int = 128
) -> Dict[str, np.ndarray]:
    """
    Generate placeholder temporal sequence embeddings (random initialization).

    Future: Implement Transformer on procedure ordering within encounters.

    Args:
        procedures: List of procedure codes/categories
        dim: Embedding dimension

    Returns:
        Dictionary mapping procedure to embedding
    """
    np.random.seed(44)
    embeddings = {}

    for proc in procedures:
        embeddings[proc] = np.random.randn(dim).astype(np.float32) * 0.1

    return embeddings


# =============================================================================
# HDF5 STORAGE
# =============================================================================

def save_vocabulary_embeddings_hdf5(
    embeddings: Dict[str, np.ndarray],
    output_path: Path,
    embedding_type: str,
    expected_dim: int
):
    """
    Save vocabulary-level embeddings to HDF5 file.

    Args:
        embeddings: Dict mapping code/category to embedding
        output_path: Output HDF5 path
        embedding_type: Name for this embedding type (e.g., 'complexity', 'ccs_cooccurrence')
        expected_dim: Expected embedding dimension
    """
    if len(embeddings) == 0:
        print(f"Warning: No embeddings to save for {embedding_type}")
        return

    # Convert to arrays
    codes = list(embeddings.keys())
    vectors = np.array([embeddings[code] for code in codes], dtype=np.float32)

    # Verify dimensions
    if vectors.shape[1] != expected_dim:
        print(f"Warning: Expected {expected_dim} dims, got {vectors.shape[1]} for {embedding_type}")

    # Open HDF5 file in append mode
    with h5py.File(output_path, 'a') as f:
        # Create vocabulary group if it doesn't exist
        if 'vocabulary' not in f:
            vocab_group = f.create_group('vocabulary')
        else:
            vocab_group = f['vocabulary']

        # Save embeddings
        if embedding_type in vocab_group:
            del vocab_group[embedding_type]

        vocab_group.create_dataset(
            embedding_type,
            data=vectors,
            dtype=np.float32,
            compression='gzip',
            compression_opts=4
        )

        # Save code-to-row mapping
        if 'code_to_row' not in vocab_group:
            # Create group for code mappings
            code_map_group = vocab_group.create_group('code_to_row')
        else:
            code_map_group = vocab_group['code_to_row']

        # Store mapping for this embedding type
        if embedding_type in code_map_group:
            del code_map_group[embedding_type]

        # Create subgroup for this embedding type's mapping
        if embedding_type in code_map_group:
            emb_map_group = code_map_group[embedding_type]
        else:
            emb_map_group = code_map_group.create_group(embedding_type)

        # Store codes and row indices separately to avoid dtype issues
        dt_string = h5py.special_dtype(vlen=str)
        emb_map_group.create_dataset('codes', data=np.array(codes, dtype=object), dtype=dt_string)
        emb_map_group.create_dataset('rows', data=np.arange(len(codes), dtype=np.int32))

        # Store metadata
        vocab_group[embedding_type].attrs['n_procedures'] = len(codes)
        vocab_group[embedding_type].attrs['embedding_dim'] = expected_dim


def load_vocabulary_embeddings_hdf5(
    input_path: Path,
    embedding_type: str
) -> Tuple[np.ndarray, Dict[str, int]]:
    """
    Load vocabulary embeddings from HDF5 file.

    Args:
        input_path: Input HDF5 path
        embedding_type: Name of embedding type to load

    Returns:
        Tuple of (embeddings array, code_to_row dict)
    """
    with h5py.File(input_path, 'r') as f:
        vectors = f[f'vocabulary/{embedding_type}'][:]

        # Load code mapping
        map_group = f[f'vocabulary/code_to_row/{embedding_type}']
        codes = [c.decode() if isinstance(c, bytes) else c for c in map_group['codes'][:]]
        rows = map_group['rows'][:]
        code_to_row = {str(code): int(row) for code, row in zip(codes, rows)}

    return vectors, code_to_row


# =============================================================================
# MAIN BUILDER
# =============================================================================

def build_layer4(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    test_mode: bool = False,
) -> Dict[str, Any]:
    """
    Build Layer 4 procedure embeddings.

    Args:
        input_path: Path to mapped_procedures.parquet (silver layer)
        output_path: Path for output HDF5
        test_mode: If True, process subset

    Returns:
        Dictionary with embedding statistics
    """
    print("=" * 60)
    print("Layer 4: Procedure Embeddings")
    print("=" * 60)

    # Load data
    if input_path is None:
        filename = "mapped_procedures_test.parquet" if test_mode else "mapped_procedures.parquet"
        input_path = SILVER_DIR / filename

    print(f"\n1. Loading mapped procedures: {input_path}")

    if not input_path.exists():
        print(f"Warning: Input file not found: {input_path}")
        # Create minimal empty DataFrame for testing
        df = pd.DataFrame(columns=[
            'empi', 'encounter_id', 'ccs_category', 'procedure_name',
            'code', 'hours_from_pe'
        ])
    else:
        df = pd.read_parquet(input_path)

    if test_mode and len(df) > 0:
        sample_empis = df['empi'].unique()[:50]
        df = df[df['empi'].isin(sample_empis)]

    print(f"   Records: {len(df):,}")

    # Prepare output
    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)
    if output_path is None:
        filename = "procedure_embeddings_test.h5" if test_mode else "procedure_embeddings.h5"
        output_path = EMBEDDINGS_DIR / filename

    # Remove existing file to start fresh
    if output_path.exists():
        output_path.unlink()

    results = {}

    # =========================================================================
    # 1. PROCEDURAL COMPLEXITY FEATURES (16 dims)
    # =========================================================================

    print("\n2. Generating procedural complexity embeddings...")

    if len(df) > 0:
        # Get unique CPT codes
        unique_codes = df[['code', 'ccs_category', 'procedure_name']].drop_duplicates()

        complexity_embeddings = {}
        for _, row in unique_codes.iterrows():
            code = str(row['code'])
            emb = get_complexity_embedding(
                ccs_category=row.get('ccs_category'),
                procedure_name=row.get('procedure_name'),
                cpt_code=code
            )
            complexity_embeddings[code] = emb

        print(f"   Generated {len(complexity_embeddings)} complexity embeddings")

        # Save to HDF5
        save_vocabulary_embeddings_hdf5(
            complexity_embeddings,
            output_path,
            'complexity',
            EMBEDDING_CONFIG.complexity_dim
        )

        results['complexity'] = len(complexity_embeddings)
    else:
        print("   No data available for complexity embeddings")
        results['complexity'] = 0

    # =========================================================================
    # 2. CCS CO-OCCURRENCE EMBEDDINGS (128 dims)
    # =========================================================================

    print("\n3. Training CCS co-occurrence embeddings (Word2Vec)...")

    if len(df) > 0 and 'encounter_id' in df.columns and 'ccs_category' in df.columns:
        # Build sequences
        sequences = build_ccs_sequences(df)
        print(f"   Built {len(sequences)} CCS category sequences")

        if len(sequences) > 0:
            # Train Word2Vec
            model = train_ccs_word2vec(sequences)

            if model is not None:
                # Extract embeddings for all CCS categories
                ccs_embeddings = {}
                for ccs_cat in df['ccs_category'].dropna().unique():
                    emb = get_ccs_embedding(model, str(ccs_cat))
                    if emb is not None:
                        ccs_embeddings[str(ccs_cat)] = emb

                print(f"   Generated {len(ccs_embeddings)} CCS co-occurrence embeddings")

                # Save to HDF5
                save_vocabulary_embeddings_hdf5(
                    ccs_embeddings,
                    output_path,
                    'ccs_cooccurrence',
                    EMBEDDING_CONFIG.ccs_cooccurrence_dim
                )

                results['ccs_cooccurrence'] = len(ccs_embeddings)
            else:
                print("   Word2Vec training failed (gensim not available)")
                results['ccs_cooccurrence'] = 0
        else:
            print("   No sequences available for Word2Vec training")
            results['ccs_cooccurrence'] = 0
    else:
        print("   No data available for CCS co-occurrence embeddings")
        results['ccs_cooccurrence'] = 0

    # =========================================================================
    # 3. PLACEHOLDER EMBEDDINGS (for future implementation)
    # =========================================================================

    print("\n4. Generating placeholder embeddings (ontological, semantic, temporal)...")

    if len(df) > 0:
        unique_ccs = df['ccs_category'].dropna().unique().tolist()

        # Ontological placeholder
        ont_embeddings = generate_ontological_placeholder(
            [str(c) for c in unique_ccs],
            dim=EMBEDDING_CONFIG.ontological_dim
        )
        save_vocabulary_embeddings_hdf5(
            ont_embeddings,
            output_path,
            'ontological',
            EMBEDDING_CONFIG.ontological_dim
        )
        results['ontological'] = len(ont_embeddings)

        # Temporal placeholder
        temp_embeddings = generate_temporal_placeholder(
            [str(c) for c in unique_ccs],
            dim=EMBEDDING_CONFIG.temporal_sequence_dim
        )
        save_vocabulary_embeddings_hdf5(
            temp_embeddings,
            output_path,
            'temporal_sequence',
            EMBEDDING_CONFIG.temporal_sequence_dim
        )
        results['temporal_sequence'] = len(temp_embeddings)

        print(f"   Generated {len(ont_embeddings)} placeholder embeddings per type")
    else:
        results['ontological'] = 0
        results['temporal_sequence'] = 0

    # =========================================================================
    # Summary
    # =========================================================================

    print("\n" + "=" * 60)
    print("Layer 4 Summary")
    print("=" * 60)
    print(f"   Complexity embeddings: {results.get('complexity', 0)}")
    print(f"   CCS co-occurrence embeddings: {results.get('ccs_cooccurrence', 0)}")
    print(f"   Ontological (placeholder): {results.get('ontological', 0)}")
    print(f"   Temporal (placeholder): {results.get('temporal_sequence', 0)}")
    print(f"\n   Output: {output_path}")
    if output_path.exists():
        print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")
    else:
        # Create empty HDF5 file with structure
        print("   Creating empty HDF5 structure (no data available)")
        with h5py.File(output_path, 'w') as f:
            f.create_group('vocabulary')
            f.create_group('vocabulary/code_to_row')
    print("\n" + "=" * 60)
    print("Layer 4 Complete!")
    print("=" * 60)

    return results


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build procedure embeddings")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--input', type=str, help='Input parquet file path')
    parser.add_argument('--output', type=str, help='Output HDF5 file path')
    args = parser.parse_args()

    input_path = Path(args.input) if args.input else None
    output_path = Path(args.output) if args.output else None

    build_layer4(
        input_path=input_path,
        output_path=output_path,
        test_mode=args.test
    )
