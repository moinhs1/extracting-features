"""Layer 2 Builder: Comorbidity indices."""
import pandas as pd
from pathlib import Path
from processing.charlson_calculator import calculate_charlson_batch


def build_layer2_comorbidity_scores(layer1_df: pd.DataFrame) -> pd.DataFrame:
    """Build Layer 2 comorbidity scores from Layer 1 data.

    Args:
        layer1_df: Layer 1 canonical diagnoses

    Returns:
        DataFrame with one row per patient, comorbidity scores
    """
    # Calculate Charlson
    cci_df = calculate_charlson_batch(layer1_df)

    return cci_df


def save_layer2(df: pd.DataFrame, output_path: Path) -> None:
    """Save Layer 2 output to parquet.

    Args:
        df: Layer 2 DataFrame
        output_path: Output directory path
    """
    output_path.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path / "comorbidity_scores.parquet", index=False)
