"""Layer 3 Builder: PE-specific diagnosis features."""
from pathlib import Path
import pandas as pd
from processing.pe_feature_builder import build_pe_features_batch

# Layer 3 output schema
LAYER3_SCHEMA = {
    "EMPI": "str",
    # VTE History
    "prior_pe_ever": "bool",
    "prior_pe_months": "float64",
    "prior_pe_count": "int64",
    "prior_dvt_ever": "bool",
    "prior_dvt_months": "float64",
    "prior_vte_count": "int64",
    "is_recurrent_vte": "bool",
    # PE Characterization
    "pe_subtype": "str",
    "pe_bilateral": "bool",
    "pe_with_cor_pulmonale": "bool",
    "pe_high_risk_code": "bool",
    # Cancer
    "cancer_active": "bool",
    "cancer_site": "str",
    "cancer_metastatic": "bool",
    "cancer_recent_diagnosis": "bool",
    "cancer_on_chemotherapy": "bool",
    # Cardiovascular
    "heart_failure": "bool",
    "heart_failure_type": "str",
    "coronary_artery_disease": "bool",
    "atrial_fibrillation": "bool",
    "pulmonary_hypertension": "bool",
    "valvular_heart_disease": "bool",
    # Pulmonary
    "copd": "bool",
    "asthma": "bool",
    "interstitial_lung_disease": "bool",
    "home_oxygen": "bool",
    "prior_respiratory_failure": "bool",
    # Bleeding Risk
    "prior_major_bleeding": "bool",
    "prior_gi_bleeding": "bool",
    "prior_intracranial_hemorrhage": "bool",
    "active_peptic_ulcer": "bool",
    "thrombocytopenia": "bool",
    "coagulopathy": "bool",
    # Renal
    "ckd_stage": "int64",
    "ckd_dialysis": "bool",
    "aki_at_presentation": "bool",
    # Provoking Factors
    "recent_surgery": "bool",
    "recent_trauma": "bool",
    "immobilization": "bool",
    "pregnancy_related": "bool",
    "hormonal_therapy": "bool",
    "central_venous_catheter": "bool",
    "is_provoked_vte": "bool",
    # Complications
    "complication_aki": "bool",
    "complication_bleeding_any": "bool",
    "complication_bleeding_major": "bool",
    "complication_ich": "bool",
    "complication_respiratory_failure": "bool",
    "complication_cardiogenic_shock": "bool",
    "complication_cardiac_arrest": "bool",
    "complication_recurrent_vte": "bool",
    "complication_cteph": "bool",
}


def build_layer3(layer1_df: pd.DataFrame, output_dir: Path = None) -> pd.DataFrame:
    """Build Layer 3 PE-specific features from Layer 1 canonical records.

    Args:
        layer1_df: Layer 1 DataFrame with canonical diagnoses
        output_dir: Optional output directory for parquet file

    Returns:
        DataFrame with EMPI + all PE-specific features
    """
    print(f"Building Layer 3 for {layer1_df['EMPI'].nunique()} patients...")

    result = build_pe_features_batch(layer1_df)

    print(f"Layer 3 complete: {len(result)} patients, {len(result.columns)} features")

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "pe_diagnosis_features.parquet"
        result.to_parquet(output_path, index=False)
        print(f"Saved to {output_path}")

    return result
