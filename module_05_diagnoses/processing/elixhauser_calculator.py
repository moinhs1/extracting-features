"""Elixhauser Comorbidity Index calculator with van Walraven weights."""

from typing import Dict, List, Any
import json
import pandas as pd
from config.elixhauser_codes import ELIXHAUSER_COMPONENTS, ELIXHAUSER_HIERARCHY


def code_matches_component(code: str, component: str, version: str) -> bool:
    """Check if ICD code matches an Elixhauser component.

    Args:
        code: ICD code (normalized)
        component: Elixhauser component name
        version: '9' or '10'

    Returns:
        True if code matches component
    """
    if component not in ELIXHAUSER_COMPONENTS:
        return False

    comp_data = ELIXHAUSER_COMPONENTS[component]
    code_list = comp_data.get(f"icd{version}", [])

    code = str(code).upper()
    for prefix in code_list:
        if code.startswith(prefix.upper()):
            return True

    return False


def calculate_elixhauser_for_patient(diagnoses: pd.DataFrame) -> Dict[str, Any]:
    """Calculate Elixhauser Comorbidity Index for a single patient.

    Args:
        diagnoses: DataFrame with icd_code, icd_version, is_preexisting columns

    Returns:
        Dict with elixhauser_score, elixhauser_components, elixhauser_component_count
    """
    # Filter to preexisting only
    preexisting = diagnoses[diagnoses["is_preexisting"] == True].copy()

    # Find which components are present
    components_present = set()

    for _, row in preexisting.iterrows():
        code = row["icd_code"]
        version = row["icd_version"]

        for component in ELIXHAUSER_COMPONENTS:
            if code_matches_component(code, component, version):
                components_present.add(component)

    # Apply hierarchy rules
    final_components = set()
    for comp in components_present:
        # Check if this component is superseded by another
        superseded_by = ELIXHAUSER_HIERARCHY.get(comp)
        if superseded_by and superseded_by in components_present:
            # Skip this one, the more severe is present
            continue
        final_components.add(comp)

    # Calculate van Walraven score
    score = sum(
        ELIXHAUSER_COMPONENTS[comp]["weight"]
        for comp in final_components
    )

    return {
        "elixhauser_score": score,
        "elixhauser_components": json.dumps(sorted(final_components)),
        "elixhauser_component_count": len(final_components),
    }


def calculate_elixhauser_batch(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Calculate Elixhauser for multiple patients.

    Args:
        diagnoses: DataFrame with EMPI, icd_code, icd_version, is_preexisting

    Returns:
        DataFrame with EMPI, elixhauser_score, elixhauser_components, elixhauser_component_count
    """
    results = []

    for empi, group in diagnoses.groupby("EMPI"):
        result = calculate_elixhauser_for_patient(group)
        result["EMPI"] = empi
        results.append(result)

    if not results:
        return pd.DataFrame(columns=["EMPI", "elixhauser_score", "elixhauser_components",
                                      "elixhauser_component_count"])

    return pd.DataFrame(results)[["EMPI", "elixhauser_score", "elixhauser_components",
                                   "elixhauser_component_count"]]
