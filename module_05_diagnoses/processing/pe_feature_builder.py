"""PE (Pulmonary Embolism) feature extraction builder.

This module provides utilities for extracting PE-related features from diagnosis codes.
"""
from typing import Dict, Optional
import pandas as pd


def code_matches_category(code: str, category_codes: Dict[str, list], version: str) -> bool:
    """Check if ICD code matches a category definition.

    Uses prefix matching to determine if a given ICD code belongs to a category.
    For example, code "I26.99" matches category with prefix "I26".

    Args:
        code: ICD code (e.g., "I26.99")
        category_codes: Dict with "icd10" and "icd9" lists of prefixes
        version: "9" or "10" indicating ICD version

    Returns:
        True if code matches any prefix in the category for the given version

    Example:
        >>> category = {"icd10": ["I26"], "icd9": ["415"]}
        >>> code_matches_category("I26.99", category, "10")
        True
        >>> code_matches_category("415.19", category, "9")
        True
        >>> code_matches_category("I50.9", category, "10")
        False
    """
    # Get the appropriate code list for this ICD version
    code_list = category_codes.get(f"icd{version}", [])

    # Normalize code for comparison
    code = str(code).upper()

    # Check if code starts with any prefix in the category
    for prefix in code_list:
        if code.startswith(prefix.upper()):
            return True

    return False


# Temporal Filter Helpers

def get_preexisting_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Filter to diagnoses before PE (is_preexisting=True OR is_recent_antecedent=True).

    Args:
        diagnoses: DataFrame with temporal category flags

    Returns:
        Filtered DataFrame containing only preexisting or recent antecedent diagnoses
    """
    return diagnoses[
        (diagnoses["is_preexisting"] == True) |
        (diagnoses["is_recent_antecedent"] == True)
    ].copy()


def get_complication_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Filter to diagnoses after PE (is_complication=True).

    Args:
        diagnoses: DataFrame with temporal category flags

    Returns:
        Filtered DataFrame containing only complication diagnoses
    """
    return diagnoses[diagnoses["is_complication"] == True].copy()


def get_index_diagnoses(diagnoses: pd.DataFrame) -> pd.DataFrame:
    """Filter to diagnoses at PE presentation (is_index_concurrent=True).

    Args:
        diagnoses: DataFrame with temporal category flags

    Returns:
        Filtered DataFrame containing only index concurrent diagnoses
    """
    return diagnoses[diagnoses["is_index_concurrent"] == True].copy()


# Time-Based Helpers

def days_to_months(days: int) -> float:
    """Convert days to months (30.44 days/month average).

    Uses the average number of days per month (365.25 / 12 = 30.44).
    Handles negative days (before PE) by converting to positive months.

    Args:
        days: Number of days (can be negative for dates before PE)

    Returns:
        Number of months (always positive)

    Example:
        >>> days_to_months(365)
        11.99
        >>> days_to_months(-180)
        5.91
    """
    return abs(days) / 30.44


def get_most_recent_prior(diagnoses: pd.DataFrame, category_codes: dict) -> Optional[int]:
    """Get days_from_pe of most recent matching diagnosis before PE.

    Args:
        diagnoses: DataFrame with icd_code, icd_version, days_from_pe columns
        category_codes: Dict with "icd10" and "icd9" lists

    Returns:
        Negative int (days before PE) or None if no match

    Example:
        If a patient has PE codes at -365 and -30 days from index,
        returns -30 (most recent).
    """
    # Filter to only prior diagnoses (negative days_from_pe)
    prior = diagnoses[diagnoses["days_from_pe"] < 0].copy()
    if prior.empty:
        return None

    # Find matches using code_matches_category
    matches = prior[prior.apply(
        lambda row: code_matches_category(row["icd_code"], category_codes, row["icd_version"]),
        axis=1
    )]

    if matches.empty:
        return None

    # Most recent = closest to 0 (max of negative numbers)
    return int(matches["days_from_pe"].max())


# VTE History Feature Extraction

def extract_prior_pe_features(diagnoses: pd.DataFrame) -> dict:
    """Extract prior PE features from diagnoses before index PE.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "prior_pe_ever": bool,
            "prior_pe_months": float or None,
            "prior_pe_count": int,
        }
    """
    from config.icd_code_lists import VTE_CODES

    # Filter to prior diagnoses only (before index PE)
    prior = diagnoses[diagnoses["days_from_pe"] < 0].copy()

    # Find PE diagnoses
    pe_matches = prior[prior.apply(
        lambda row: code_matches_category(row["icd_code"], VTE_CODES["pe"], row["icd_version"]),
        axis=1
    )]

    prior_pe_ever = len(pe_matches) > 0
    prior_pe_count = len(pe_matches)

    prior_pe_months = None
    if prior_pe_ever:
        most_recent_days = pe_matches["days_from_pe"].max()  # max of negatives = most recent
        prior_pe_months = days_to_months(most_recent_days)

    return {
        "prior_pe_ever": prior_pe_ever,
        "prior_pe_months": prior_pe_months,
        "prior_pe_count": prior_pe_count,
    }


def extract_prior_dvt_features(diagnoses: pd.DataFrame) -> dict:
    """Extract prior DVT features from diagnoses before index PE.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "prior_dvt_ever": bool,
            "prior_dvt_months": float or None,
        }
    """
    from config.icd_code_lists import VTE_CODES

    prior = diagnoses[diagnoses["days_from_pe"] < 0].copy()

    # Combine DVT lower and upper
    dvt_matches = prior[prior.apply(
        lambda row: (
            code_matches_category(row["icd_code"], VTE_CODES["dvt_lower"], row["icd_version"]) or
            code_matches_category(row["icd_code"], VTE_CODES["dvt_upper"], row["icd_version"])
        ),
        axis=1
    )]

    prior_dvt_ever = len(dvt_matches) > 0
    prior_dvt_months = None
    if prior_dvt_ever:
        most_recent_days = dvt_matches["days_from_pe"].max()
        prior_dvt_months = days_to_months(most_recent_days)

    return {
        "prior_dvt_ever": prior_dvt_ever,
        "prior_dvt_months": prior_dvt_months,
    }


def extract_vte_history_features(diagnoses: pd.DataFrame) -> dict:
    """Extract all VTE history features.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "prior_pe_ever": bool,
            "prior_pe_months": float or None,
            "prior_pe_count": int,
            "prior_dvt_ever": bool,
            "prior_dvt_months": float or None,
            "prior_vte_count": int,
            "is_recurrent_vte": bool,
        }
    """
    pe_features = extract_prior_pe_features(diagnoses)
    dvt_features = extract_prior_dvt_features(diagnoses)

    # Count all VTE events
    from config.icd_code_lists import VTE_CODES
    prior = diagnoses[diagnoses["days_from_pe"] < 0].copy()
    vte_matches = prior[prior.apply(
        lambda row: (
            code_matches_category(row["icd_code"], VTE_CODES["pe"], row["icd_version"]) or
            code_matches_category(row["icd_code"], VTE_CODES["dvt_lower"], row["icd_version"]) or
            code_matches_category(row["icd_code"], VTE_CODES["dvt_upper"], row["icd_version"])
        ),
        axis=1
    )]

    return {
        **pe_features,
        **dvt_features,
        "prior_vte_count": len(vte_matches),
        "is_recurrent_vte": pe_features["prior_pe_ever"] or dvt_features["prior_dvt_ever"],
    }


def extract_pe_characterization(diagnoses: pd.DataFrame) -> dict:
    """Characterize the index PE from ICD codes.

    PE Subtype Logic (ICD-10-CM):
    - I26.92 → saddle
    - I26.93 → single subsegmental
    - I26.94 → multiple subsegmental
    - I26.99 → other (lobar)
    - I26.90 → unspecified
    - I26.0x → with cor pulmonale (high risk)

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe,
                   is_index_concurrent, is_pe_diagnosis

    Returns:
        {
            "pe_subtype": str,  # 'saddle', 'subsegmental', 'other', 'unspecified'
            "pe_bilateral": bool,
            "pe_with_cor_pulmonale": bool,
            "pe_high_risk_code": bool,
        }
    """
    # Get index PE diagnoses only
    index_dx = get_index_diagnoses(diagnoses)

    # Find PE codes at index
    pe_codes = index_dx[index_dx["is_pe_diagnosis"] == True]["icd_code"].tolist()

    # Determine subtype from codes
    pe_subtype = "unspecified"
    pe_bilateral = False
    pe_with_cor_pulmonale = False
    pe_high_risk_code = False

    for code in pe_codes:
        code_upper = str(code).upper()

        # Saddle PE
        if code_upper.startswith("I26.92"):
            pe_subtype = "saddle"
            pe_high_risk_code = True

        # Subsegmental
        elif code_upper.startswith("I26.93") or code_upper.startswith("I26.94"):
            if pe_subtype not in ["saddle"]:
                pe_subtype = "subsegmental"

        # Other (lobar assumed)
        elif code_upper.startswith("I26.99"):
            if pe_subtype not in ["saddle", "subsegmental"]:
                pe_subtype = "other"

        # With acute cor pulmonale = high risk
        if code_upper.startswith("I26.0"):
            pe_with_cor_pulmonale = True
            pe_high_risk_code = True

    return {
        "pe_subtype": pe_subtype,
        "pe_bilateral": pe_bilateral,  # Would need radiology data, keep as False
        "pe_with_cor_pulmonale": pe_with_cor_pulmonale,
        "pe_high_risk_code": pe_high_risk_code,
    }


# Cancer Feature Extraction

def extract_cancer_features(diagnoses: pd.DataFrame) -> dict:
    """Extract cancer-related features from preexisting diagnoses.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "cancer_active": bool,
            "cancer_site": str or None,  # 'lung', 'gi', 'gu', 'hematologic', 'breast', 'other'
            "cancer_metastatic": bool,
            "cancer_recent_diagnosis": bool,  # First cancer dx within 6 months of PE
            "cancer_on_chemotherapy": bool,
        }
    """
    from config.icd_code_lists import CANCER_CODES

    preexisting = get_preexisting_diagnoses(diagnoses)

    # Check for any cancer (priority order: solid tumors first, then hematologic)
    cancer_sites = ["lung", "gi", "gu", "breast", "hematologic"]
    cancer_active = False
    detected_site = None

    for site in cancer_sites:
        matches = preexisting[preexisting.apply(
            lambda row: code_matches_category(row["icd_code"], CANCER_CODES[site], row["icd_version"]),
            axis=1
        )]
        if len(matches) > 0:
            cancer_active = True
            detected_site = site
            break  # Take first match in priority order

    # Check metastatic
    metastatic_matches = preexisting[preexisting.apply(
        lambda row: code_matches_category(row["icd_code"], CANCER_CODES["metastatic"], row["icd_version"]),
        axis=1
    )]
    cancer_metastatic = len(metastatic_matches) > 0
    if cancer_metastatic and detected_site is None:
        detected_site = "other"
        cancer_active = True

    # Check recent diagnosis (within 6 months = 183 days)
    cancer_recent_diagnosis = False
    if cancer_active:
        all_cancer_dx = preexisting[preexisting.apply(
            lambda row: any(
                code_matches_category(row["icd_code"], CANCER_CODES[site], row["icd_version"])
                for site in cancer_sites + ["metastatic"]
            ),
            axis=1
        )]
        if len(all_cancer_dx) > 0:
            earliest_dx = all_cancer_dx["days_from_pe"].min()  # Most negative = first
            cancer_recent_diagnosis = abs(earliest_dx) <= 183

    # Check chemotherapy
    chemo_matches = preexisting[preexisting.apply(
        lambda row: code_matches_category(row["icd_code"], CANCER_CODES["chemotherapy"], row["icd_version"]),
        axis=1
    )]
    cancer_on_chemotherapy = len(chemo_matches) > 0

    return {
        "cancer_active": cancer_active,
        "cancer_site": detected_site,
        "cancer_metastatic": cancer_metastatic,
        "cancer_recent_diagnosis": cancer_recent_diagnosis,
        "cancer_on_chemotherapy": cancer_on_chemotherapy,
    }


# Cardiovascular Feature Extraction

def extract_cardiovascular_features(diagnoses: pd.DataFrame) -> dict:
    """Extract cardiovascular comorbidity features.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "heart_failure": bool,
            "heart_failure_type": str or None,  # 'HFrEF', 'HFpEF', 'unspecified'
            "coronary_artery_disease": bool,
            "atrial_fibrillation": bool,
            "pulmonary_hypertension": bool,
            "valvular_heart_disease": bool,
        }
    """
    from config.icd_code_lists import CARDIOVASCULAR_CODES

    preexisting = get_preexisting_diagnoses(diagnoses)

    def has_category(category):
        matches = preexisting[preexisting.apply(
            lambda row: code_matches_category(row["icd_code"], CARDIOVASCULAR_CODES[category], row["icd_version"]),
            axis=1
        )]
        return len(matches) > 0

    heart_failure = has_category("heart_failure")

    # Determine HF type
    heart_failure_type = None
    if heart_failure:
        if has_category("heart_failure_reduced"):
            heart_failure_type = "HFrEF"
        elif has_category("heart_failure_preserved"):
            heart_failure_type = "HFpEF"
        else:
            heart_failure_type = "unspecified"

    return {
        "heart_failure": heart_failure,
        "heart_failure_type": heart_failure_type,
        "coronary_artery_disease": has_category("coronary_artery_disease"),
        "atrial_fibrillation": has_category("atrial_fibrillation"),
        "pulmonary_hypertension": has_category("pulmonary_hypertension"),
        "valvular_heart_disease": has_category("valvular_heart_disease"),
    }


# Pulmonary Feature Extraction

def extract_pulmonary_features(diagnoses: pd.DataFrame) -> dict:
    """Extract pulmonary comorbidity features.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "copd": bool,
            "asthma": bool,
            "interstitial_lung_disease": bool,
            "home_oxygen": bool,
            "prior_respiratory_failure": bool,
        }
    """
    from config.icd_code_lists import PULMONARY_CODES

    preexisting = get_preexisting_diagnoses(diagnoses)

    def has_category(category):
        matches = preexisting[preexisting.apply(
            lambda row: code_matches_category(row["icd_code"], PULMONARY_CODES[category], row["icd_version"]),
            axis=1
        )]
        return len(matches) > 0

    return {
        "copd": has_category("copd"),
        "asthma": has_category("asthma"),
        "interstitial_lung_disease": has_category("interstitial_lung_disease"),
        "home_oxygen": has_category("home_oxygen"),
        "prior_respiratory_failure": has_category("respiratory_failure"),
    }


# Bleeding Risk Feature Extraction

def extract_bleeding_risk_features(diagnoses: pd.DataFrame) -> dict:
    """Extract bleeding risk features from preexisting diagnoses.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "prior_major_bleeding": bool,
            "prior_gi_bleeding": bool,
            "prior_intracranial_hemorrhage": bool,
            "active_peptic_ulcer": bool,
            "thrombocytopenia": bool,
            "coagulopathy": bool,
        }
    """
    from config.icd_code_lists import BLEEDING_CODES

    preexisting = get_preexisting_diagnoses(diagnoses)

    def has_category(category):
        matches = preexisting[preexisting.apply(
            lambda row: code_matches_category(row["icd_code"], BLEEDING_CODES[category], row["icd_version"]),
            axis=1
        )]
        return len(matches) > 0

    prior_gi_bleeding = has_category("gi_bleeding")
    prior_intracranial_hemorrhage = has_category("intracranial_hemorrhage")
    other_major_bleeding = has_category("other_major_bleeding")

    return {
        "prior_major_bleeding": prior_gi_bleeding or prior_intracranial_hemorrhage or other_major_bleeding,
        "prior_gi_bleeding": prior_gi_bleeding,
        "prior_intracranial_hemorrhage": prior_intracranial_hemorrhage,
        "active_peptic_ulcer": has_category("peptic_ulcer"),
        "thrombocytopenia": has_category("thrombocytopenia"),
        "coagulopathy": has_category("coagulopathy"),
    }


# Renal Feature Extraction

def extract_renal_features(diagnoses: pd.DataFrame) -> dict:
    """Extract renal function features.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe,
                   is_preexisting, is_index_concurrent

    Returns:
        {
            "ckd_stage": int,  # 0-5 (0 = no CKD)
            "ckd_dialysis": bool,
            "aki_at_presentation": bool,
        }
    """
    from config.icd_code_lists import RENAL_CODES

    preexisting = get_preexisting_diagnoses(diagnoses)
    index_dx = get_index_diagnoses(diagnoses)

    def has_category_in(df, category):
        if df.empty:
            return False
        matches = df[df.apply(
            lambda row: code_matches_category(row["icd_code"], RENAL_CODES[category], row["icd_version"]),
            axis=1
        )]
        return len(matches) > 0

    # Determine highest CKD stage
    ckd_stage = 0
    for stage in range(5, 0, -1):
        if has_category_in(preexisting, f"ckd_stage{stage}"):
            ckd_stage = stage
            break

    return {
        "ckd_stage": ckd_stage,
        "ckd_dialysis": has_category_in(preexisting, "dialysis"),
        "aki_at_presentation": has_category_in(index_dx, "aki"),
    }


# Provoking Factor Extraction

def extract_provoking_factors(diagnoses: pd.DataFrame) -> dict:
    """Extract VTE provoking factors from recent diagnoses.

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe

    Returns:
        {
            "recent_surgery": bool,
            "recent_trauma": bool,
            "immobilization": bool,
            "pregnancy_related": bool,
            "hormonal_therapy": bool,
            "central_venous_catheter": bool,
            "is_provoked_vte": bool,
        }
    """
    from config.icd_code_lists import PROVOKING_FACTOR_CODES

    # Use recent antecedent window (-30 to -1 days) for provoking factors
    # Excludes day 0 (index day) to avoid temporal leakage
    recent = diagnoses[
        (diagnoses["days_from_pe"] >= -30) &
        (diagnoses["days_from_pe"] < 0)
    ].copy()

    def has_category(category):
        if recent.empty:
            return False
        matches = recent[recent.apply(
            lambda row: code_matches_category(row["icd_code"], PROVOKING_FACTOR_CODES[category], row["icd_version"]),
            axis=1
        )]
        return len(matches) > 0

    recent_surgery = has_category("recent_surgery")
    recent_trauma = has_category("trauma")
    immobilization = has_category("immobilization")
    pregnancy_related = has_category("pregnancy")
    hormonal_therapy = has_category("hormonal_therapy")
    central_venous_catheter = has_category("central_venous_catheter")

    is_provoked = any([
        recent_surgery, recent_trauma, immobilization,
        pregnancy_related, hormonal_therapy, central_venous_catheter
    ])

    return {
        "recent_surgery": recent_surgery,
        "recent_trauma": recent_trauma,
        "immobilization": immobilization,
        "pregnancy_related": pregnancy_related,
        "hormonal_therapy": hormonal_therapy,
        "central_venous_catheter": central_venous_catheter,
        "is_provoked_vte": is_provoked,
    }


# Complication Feature Extraction

def extract_complication_features(diagnoses: pd.DataFrame) -> dict:
    """Extract post-PE complication features.

    Uses is_complication=True rows only (days_from_pe > 1).

    Args:
        diagnoses: Layer 1 DataFrame with icd_code, icd_version, days_from_pe,
                   is_complication

    Returns:
        {
            "complication_aki": bool,
            "complication_bleeding_any": bool,
            "complication_bleeding_major": bool,
            "complication_ich": bool,
            "complication_respiratory_failure": bool,
            "complication_cardiogenic_shock": bool,
            "complication_cardiac_arrest": bool,
            "complication_recurrent_vte": bool,
            "complication_cteph": bool,
        }
    """
    from config.icd_code_lists import COMPLICATION_CODES

    complications = get_complication_diagnoses(diagnoses)

    def has_category(category):
        if complications.empty:
            return False
        codes = COMPLICATION_CODES[category]
        matches = complications[complications.apply(
            lambda row: code_matches_category(row["icd_code"], codes, row["icd_version"]),
            axis=1
        )]
        return len(matches) > 0

    bleeding_gi = has_category("bleeding_major")  # GI bleeding
    bleeding_any = has_category("bleeding_any")
    ich = has_category("intracranial_hemorrhage")

    return {
        "complication_aki": has_category("aki"),
        "complication_bleeding_any": bleeding_any or ich,
        "complication_bleeding_major": bleeding_gi,
        "complication_ich": ich,
        "complication_respiratory_failure": has_category("respiratory_failure"),
        "complication_cardiogenic_shock": has_category("cardiogenic_shock"),
        "complication_cardiac_arrest": has_category("cardiac_arrest"),
        "complication_recurrent_vte": has_category("recurrent_vte"),
        "complication_cteph": has_category("cteph"),
    }
