"""Charlson Comorbidity Index calculator."""
from typing import Dict, List, Any
import json
import pandas as pd
from config.comorbidity_codes import CHARLSON_COMPONENTS

# Hierarchy rules: if both present, only count the more severe
CHARLSON_HIERARCHY = {
    "diabetes_uncomplicated": "diabetes_complicated",  # If complicated present, skip uncomplicated
    "mild_liver_disease": "moderate_severe_liver_disease",
    "malignancy": "metastatic_solid_tumor",
}


def code_matches_component(code: str, component: str, version: str) -> bool:
    """Check if ICD code matches a Charlson component.

    Args:
        code: ICD code (normalized)
        component: Charlson component name
        version: '9' or '10'

    Returns:
        True if code matches component
    """
    if component not in CHARLSON_COMPONENTS:
        return False

    comp_data = CHARLSON_COMPONENTS[component]
    code_list = comp_data.get(f"icd{version}", [])

    code = str(code).upper()
    for prefix in code_list:
        if code.startswith(prefix.upper()):
            return True

    return False


def calculate_charlson_for_patient(diagnoses: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Charlson Comorbidity Index for a single patient.

    Args:
        diagnoses: DataFrame with icd_code, icd_version, is_preexisting columns

    Returns:
        Dict with cci_score, cci_components, cci_component_count
    """
    # Filter to preexisting only
    preexisting = diagnoses[diagnoses["is_preexisting"] == True].copy()

    # Find which components are present
    components_present = set()

    for _, row in preexisting.iterrows():
        code = row["icd_code"]
        version = row["icd_version"]

        for component in CHARLSON_COMPONENTS:
            if code_matches_component(code, component, version):
                components_present.add(component)

    # Apply hierarchy rules
    final_components = set()
    for comp in components_present:
        # Check if this component is superseded by another
        superseded_by = CHARLSON_HIERARCHY.get(comp)
        if superseded_by and superseded_by in components_present:
            # Skip this one, the more severe is present
            continue
        final_components.add(comp)

    # Calculate score
    score = sum(
        CHARLSON_COMPONENTS[comp]["weight"]
        for comp in final_components
    )

    return {
        "cci_score": score,
        "cci_components": json.dumps(sorted(final_components)),
        "cci_component_count": len(final_components),
    }


def calculate_charlson_batch(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Calculate Charlson for multiple patients.

    Args:
        diagnoses: DataFrame with EMPI, icd_code, icd_version, is_preexisting

    Returns:
        DataFrame with EMPI, cci_score, cci_components, cci_component_count
    """
    results = []

    for empi, group in diagnoses.groupby("EMPI"):
        result = calculate_charlson_for_patient(group)
        result["EMPI"] = empi
        results.append(result)

    if not results:
        return pd.DataFrame(columns=["EMPI", "cci_score", "cci_components", "cci_component_count"])

    return pd.DataFrame(results)[["EMPI", "cci_score", "cci_components", "cci_component_count"]]
