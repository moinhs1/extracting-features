"""Layer 2 Builder: Comorbidity indices and CCS categories."""
import pandas as pd
from pathlib import Path
from processing.charlson_calculator import calculate_charlson_batch
from processing.elixhauser_calculator import calculate_elixhauser_batch
from processing.ccs_mapper import CCSMapper


def build_layer2_comorbidity_scores(layer1_df: pd.DataFrame) -> pd.DataFrame:
    """Build Layer 2 comorbidity scores from Layer 1 data.

    Args:
        layer1_df: Layer 1 canonical diagnoses

    Returns:
        DataFrame with one row per patient, comorbidity scores (CCI + Elixhauser)
    """
    # Calculate Charlson
    cci_df = calculate_charlson_batch(layer1_df)

    # Calculate Elixhauser
    elix_df = calculate_elixhauser_batch(layer1_df)

    # Merge on EMPI
    result = cci_df.merge(elix_df, on='EMPI', how='outer')

    return result


def build_layer2_ccs_categories(layer1_df: pd.DataFrame, ccs_path: Path) -> pd.DataFrame:
    """Build CCS category assignments from Layer 1 data.

    Args:
        layer1_df: Layer 1 canonical diagnoses
        ccs_path: Path to CCS crosswalk CSV

    Returns:
        DataFrame with EMPI, ccs_category, ccs_description, diagnosis_count, is_preexisting
    """
    mapper = CCSMapper(ccs_path)

    results = []
    for empi, group in layer1_df.groupby('EMPI'):
        categories = mapper.categorize_patient_diagnoses(group)
        if len(categories) > 0:
            categories['EMPI'] = empi
            results.append(categories)

    if not results:
        return pd.DataFrame(columns=['EMPI', 'ccs_category', 'ccs_description',
                                      'diagnosis_count', 'is_preexisting'])

    result = pd.concat(results, ignore_index=True)
    # Reorder columns
    return result[['EMPI', 'ccs_category', 'ccs_description', 'diagnosis_count', 'is_preexisting']]


def save_layer2(scores_df: pd.DataFrame, ccs_df: pd.DataFrame, output_path: Path) -> None:
    """Save Layer 2 outputs to parquet.

    Args:
        scores_df: Comorbidity scores DataFrame
        ccs_df: CCS categories DataFrame (can be empty if CCS crosswalk not available)
        output_path: Output directory path
    """
    output_path.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(output_path / "comorbidity_scores.parquet", index=False)
    if not ccs_df.empty:
        ccs_df.to_parquet(output_path / "ccs_categories.parquet", index=False)
