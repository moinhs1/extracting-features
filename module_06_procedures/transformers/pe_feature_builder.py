"""
PE-Specific Procedure Feature Builder
=====================================

Layer 3: Builds PE-specific procedural features from mapped procedures.

Feature categories:
1. Lifetime History: Prior VTE procedures, surgical risk
2. Provoking Procedures: 1-30 days pre-PE surgeries
3. Diagnostic Workup: ±24h PE diagnostic testing
4. Initial Treatment: 0-72h reperfusion, support
5. Escalation: >72h complications, transfusions

All features preserve datetime information.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import (
    PE_PROCEDURE_CODES_YAML,
    SURGICAL_RISK_YAML,
    GOLD_DIR,
    SILVER_DIR,
)


# =============================================================================
# LOAD PE PROCEDURE CODE MAPPINGS
# =============================================================================

def load_pe_codes() -> Dict:
    """Load PE-specific procedure CPT codes from YAML."""
    with open(PE_PROCEDURE_CODES_YAML, 'r') as f:
        return yaml.safe_load(f)


def load_surgical_risk_config() -> Dict:
    """Load surgical risk classification from YAML."""
    with open(SURGICAL_RISK_YAML, 'r') as f:
        return yaml.safe_load(f)


PE_CODES = load_pe_codes()
SURGICAL_RISK = load_surgical_risk_config()


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _get_codes_for_category(category_path: str) -> List[str]:
    """
    Get CPT codes for a nested category.

    Args:
        category_path: Dot-notation path like 'diagnostic_imaging.cta_chest'

    Returns:
        List of CPT codes
    """
    parts = category_path.split('.')
    data = PE_CODES

    for part in parts:
        data = data.get(part, {})

    return data.get('cpt_codes', [])


def _is_code_in_category(code: str, category_path: str) -> bool:
    """Check if code is in a PE procedure category."""
    codes = _get_codes_for_category(category_path)
    return str(code) in codes


def _get_ccs_risk_level(ccs_category: str) -> str:
    """Get VTE risk level for a CCS category."""
    if pd.isna(ccs_category):
        return 'unknown'

    ccs_str = str(ccs_category)

    for risk_level, config in SURGICAL_RISK['risk_levels'].items():
        if ccs_str in config.get('ccs_categories', []):
            return risk_level

    return 'unknown'


# =============================================================================
# LIFETIME HISTORY FEATURES
# =============================================================================

def compute_lifetime_history_features(df: pd.DataFrame) -> Dict:
    """
    Compute lifetime procedure history features for a single patient.

    Args:
        df: Procedures for one patient with is_lifetime_history flag

    Returns:
        Dictionary of lifetime history features
    """
    lifetime = df[df['is_lifetime_history'] == True]

    features = {
        # Prior VTE procedures
        'prior_ivc_filter_ever': False,
        'prior_ivc_filter_still_present': False,
        'prior_thrombolysis_for_vte': False,
        'prior_cdt_for_vte': False,
        'prior_surgical_embolectomy': False,
        'prior_vte_procedure_count': 0,

        # Prior surgical risk
        'prior_major_surgery_ever': False,
        'prior_orthopedic_surgery_ever': False,
        'prior_joint_replacement': False,
        'prior_spine_surgery': False,
        'prior_cancer_surgery': False,
        'prior_cardiac_surgery': False,
        'lifetime_surgical_count': 0,
        'lifetime_surgical_risk_score': 0.0,

        # Chronic procedure markers
        'has_pacemaker_icd': False,
        'on_chronic_dialysis': False,
        'prior_amputation': False,
        'prior_organ_transplant': False,
        'chronic_procedure_burden': 0,
    }

    if len(lifetime) == 0:
        return features

    # IVC filter
    ivc_filter_placed = lifetime['code'].apply(
        lambda x: _is_code_in_category(x, 'ivc_filter.filter_placement')
    ).any()
    ivc_filter_retrieved = lifetime['code'].apply(
        lambda x: _is_code_in_category(x, 'ivc_filter.filter_retrieval')
    ).any()

    features['prior_ivc_filter_ever'] = ivc_filter_placed
    features['prior_ivc_filter_still_present'] = ivc_filter_placed and not ivc_filter_retrieved

    # CDT
    features['prior_cdt_for_vte'] = lifetime['code'].apply(
        lambda x: _is_code_in_category(x, 'reperfusion_therapy.catheter_directed_therapy')
    ).any()

    # Surgical embolectomy
    features['prior_surgical_embolectomy'] = lifetime['code'].apply(
        lambda x: _is_code_in_category(x, 'reperfusion_therapy.surgical_embolectomy')
    ).any()

    # Count VTE procedures
    vte_proc_count = 0
    if features['prior_ivc_filter_ever']:
        vte_proc_count += 1
    if features['prior_cdt_for_vte']:
        vte_proc_count += 1
    if features['prior_surgical_embolectomy']:
        vte_proc_count += 1
    features['prior_vte_procedure_count'] = vte_proc_count

    # Surgical risk assessment
    # Check CCS categories for surgical procedures
    if 'ccs_category' in lifetime.columns:
        ccs_categories = lifetime['ccs_category'].dropna().unique()

        # Count major surgeries (approximate by high-risk CCS categories)
        high_risk_categories = []
        very_high_risk_categories = []

        for risk_level in ['high', 'very_high']:
            high_risk_categories.extend(
                SURGICAL_RISK['risk_levels'][risk_level].get('ccs_categories', [])
            )

        very_high_risk_categories = SURGICAL_RISK['risk_levels']['very_high'].get('ccs_categories', [])

        surgical_count = sum(1 for cat in ccs_categories if str(cat) in high_risk_categories)
        features['lifetime_surgical_count'] = surgical_count
        features['prior_major_surgery_ever'] = surgical_count > 0

        # Specific surgery types
        # Orthopedic: CCS 153, 154 (hip/knee replacement)
        features['prior_joint_replacement'] = any(
            str(cat) in ['153', '154'] for cat in ccs_categories
        )
        features['prior_orthopedic_surgery_ever'] = features['prior_joint_replacement']

        # Spine: CCS 158, 166
        features['prior_spine_surgery'] = any(
            str(cat) in ['158', '166'] for cat in ccs_categories
        )

        # Cardiac: CCS 43, 44, 49
        features['prior_cardiac_surgery'] = any(
            str(cat) in ['43', '44', '49'] for cat in ccs_categories
        )

        # Risk score (0-1) based on highest risk surgery
        if any(str(cat) in very_high_risk_categories for cat in ccs_categories):
            features['lifetime_surgical_risk_score'] = 1.0
        elif surgical_count > 0:
            features['lifetime_surgical_risk_score'] = 0.6
        else:
            features['lifetime_surgical_risk_score'] = 0.0

    # Chronic dialysis (repeated dialysis procedures)
    dialysis_count = lifetime['code'].apply(
        lambda x: _is_code_in_category(x, 'dialysis.hemodialysis') or
                  _is_code_in_category(x, 'dialysis.crrt')
    ).sum()
    features['on_chronic_dialysis'] = dialysis_count >= 3

    features['chronic_procedure_burden'] = int(features['on_chronic_dialysis'])

    return features


# =============================================================================
# PROVOKING PROCEDURES
# =============================================================================

def compute_provoking_features(df: pd.DataFrame) -> Dict:
    """
    Compute provoking procedure features (1-30 days pre-PE).

    Args:
        df: Procedures for one patient with is_provoking_window flag

    Returns:
        Dictionary of provoking features
    """
    provoking = df[df['is_provoking_window'] == True]

    features = {
        'surgery_within_30_days': False,
        'surgery_type': None,
        'surgery_vte_risk_category': None,
        'days_from_surgery_to_pe': None,
        'orthopedic_surgery_within_30d': False,
        'cancer_surgery_within_30d': False,
        'neurosurgery_within_30d': False,
        'provoked_pe': False,
        'provocation_strength': 0.0,
    }

    if len(provoking) == 0:
        return features

    # Check for any surgical procedures
    if 'ccs_category' in provoking.columns:
        ccs_categories = provoking['ccs_category'].dropna().unique()

        # Get risk levels
        risk_levels = [_get_ccs_risk_level(cat) for cat in ccs_categories]

        # Check if any surgery present
        surgical_risk_categories = []
        for risk_level in ['very_high', 'high', 'moderate']:
            surgical_risk_categories.extend(
                SURGICAL_RISK['risk_levels'][risk_level].get('ccs_categories', [])
            )

        has_surgery = any(str(cat) in surgical_risk_categories for cat in ccs_categories)
        features['surgery_within_30_days'] = has_surgery

        if has_surgery:
            # Get highest risk level
            risk_order = ['very_high', 'high', 'moderate', 'low', 'minimal']
            for risk in risk_order:
                if risk in risk_levels:
                    features['surgery_vte_risk_category'] = risk
                    break

            # Days from surgery to PE (most recent surgery)
            if 'hours_from_pe' in provoking.columns:
                most_recent_surgery_hours = provoking['hours_from_pe'].max()
                features['days_from_surgery_to_pe'] = int(abs(most_recent_surgery_hours) / 24)

            # Specific surgery types
            features['orthopedic_surgery_within_30d'] = any(
                str(cat) in ['153', '154', '158', '166'] for cat in ccs_categories
            )

            # Cancer surgery - high/very high risk abdominal/thoracic
            features['cancer_surgery_within_30d'] = any(
                str(cat) in ['75', '78', '84'] for cat in ccs_categories
            )

            # Provocation assessment
            features['provoked_pe'] = True

            # Provocation strength based on risk
            strength_map = {
                'very_high': 1.0,
                'high': 0.8,
                'moderate': 0.6,
                'low': 0.3,
                'minimal': 0.1,
            }
            features['provocation_strength'] = strength_map.get(
                features['surgery_vte_risk_category'], 0.0
            )

    return features


# =============================================================================
# DIAGNOSTIC WORKUP FEATURES
# =============================================================================

def compute_diagnostic_features(df: pd.DataFrame) -> Dict:
    """
    Compute diagnostic workup features (±24h of PE).

    Args:
        df: Procedures for one patient with is_diagnostic_workup flag

    Returns:
        Dictionary of diagnostic features
    """
    diagnostic = df[df['is_diagnostic_workup'] == True]

    features = {
        # Imaging
        'cta_chest_performed': False,
        'cta_datetime': pd.NaT,
        'cta_hours_from_pe': None,
        'vq_scan_performed': False,
        'le_duplex_performed': False,
        'echo_performed': False,
        'echo_datetime': pd.NaT,
        'tte_vs_tee': None,
        'cardiac_cath_performed': False,
        'pulmonary_angiography_performed': False,

        # Workup metrics
        'diagnostic_workup_intensity': 0,
        'complete_pe_workup': False,
    }

    if len(diagnostic) == 0:
        return features

    # CTA chest
    cta_rows = diagnostic[diagnostic['code'].apply(
        lambda x: _is_code_in_category(x, 'diagnostic_imaging.cta_chest')
    )]
    if len(cta_rows) > 0:
        features['cta_chest_performed'] = True
        first_cta = cta_rows.iloc[0]
        if 'procedure_datetime' in first_cta:
            features['cta_datetime'] = first_cta['procedure_datetime']
        if 'hours_from_pe' in first_cta:
            features['cta_hours_from_pe'] = first_cta['hours_from_pe']

    # V/Q scan
    features['vq_scan_performed'] = diagnostic['code'].apply(
        lambda x: _is_code_in_category(x, 'diagnostic_imaging.vq_scan')
    ).any()

    # LE duplex
    features['le_duplex_performed'] = diagnostic['code'].apply(
        lambda x: _is_code_in_category(x, 'diagnostic_imaging.le_duplex')
    ).any()

    # Echo
    echo_tte_rows = diagnostic[diagnostic['code'].apply(
        lambda x: _is_code_in_category(x, 'diagnostic_imaging.echo_tte')
    )]
    echo_tee_rows = diagnostic[diagnostic['code'].apply(
        lambda x: _is_code_in_category(x, 'diagnostic_imaging.echo_tee')
    )]

    if len(echo_tte_rows) > 0:
        features['echo_performed'] = True
        features['tte_vs_tee'] = 'TTE'
        features['echo_datetime'] = echo_tte_rows.iloc[0].get('procedure_datetime', pd.NaT)
    elif len(echo_tee_rows) > 0:
        features['echo_performed'] = True
        features['tte_vs_tee'] = 'TEE'
        features['echo_datetime'] = echo_tee_rows.iloc[0].get('procedure_datetime', pd.NaT)

    # Pulmonary angiography
    features['pulmonary_angiography_performed'] = diagnostic['code'].apply(
        lambda x: _is_code_in_category(x, 'diagnostic_imaging.pulmonary_angiography')
    ).any()

    # Cardiac cath
    features['cardiac_cath_performed'] = diagnostic['code'].apply(
        lambda x: _is_code_in_category(x, 'cardiac_catheterization.right_heart_cath') or
                  _is_code_in_category(x, 'cardiac_catheterization.left_heart_cath')
    ).any()

    # Diagnostic workup intensity
    intensity = 0
    if features['cta_chest_performed']:
        intensity += 1
    if features['vq_scan_performed']:
        intensity += 1
    if features['le_duplex_performed']:
        intensity += 1
    if features['echo_performed']:
        intensity += 1
    if features['pulmonary_angiography_performed']:
        intensity += 1

    features['diagnostic_workup_intensity'] = intensity

    # Complete workup = CTA + Echo + LE Duplex
    features['complete_pe_workup'] = (
        features['cta_chest_performed'] and
        features['echo_performed'] and
        features['le_duplex_performed']
    )

    return features


# =============================================================================
# INITIAL TREATMENT FEATURES
# =============================================================================

def compute_initial_treatment_features(df: pd.DataFrame) -> Dict:
    """
    Compute initial treatment features (0-72h post-PE).

    Args:
        df: Procedures for one patient with is_initial_treatment flag

    Returns:
        Dictionary of initial treatment features
    """
    treatment = df[df['is_initial_treatment'] == True]

    features = {
        # Reperfusion
        'any_reperfusion_therapy': False,
        'catheter_directed_therapy': False,
        'cdt_datetime': pd.NaT,
        'cdt_hours_from_pe': None,
        'mechanical_thrombectomy': False,
        'surgical_embolectomy': False,

        # IVC filter
        'ivc_filter_placed': False,
        'ivc_filter_datetime': pd.NaT,
        'ivc_filter_hours_from_pe': None,

        # Vascular access
        'central_line_placed': False,
        'arterial_line_placed': False,
        'pa_catheter_placed': False,

        # Respiratory support
        'intubation_performed': False,
        'intubation_datetime': pd.NaT,
        'intubation_hours_from_pe': None,
        'hfnc_initiated': False,
        'nippv_initiated': False,

        # Circulatory support
        'ecmo_initiated': False,
        'ecmo_datetime': pd.NaT,
        'ecmo_hours_from_pe': None,
        'ecmo_type': None,
    }

    if len(treatment) == 0:
        return features

    # Catheter-directed therapy
    cdt_rows = treatment[treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'reperfusion_therapy.catheter_directed_therapy')
    )]
    if len(cdt_rows) > 0:
        features['catheter_directed_therapy'] = True
        features['any_reperfusion_therapy'] = True
        first_cdt = cdt_rows.iloc[0]
        if 'procedure_datetime' in first_cdt:
            features['cdt_datetime'] = first_cdt['procedure_datetime']
        if 'hours_from_pe' in first_cdt:
            features['cdt_hours_from_pe'] = first_cdt['hours_from_pe']

    # Mechanical thrombectomy
    if treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'reperfusion_therapy.mechanical_thrombectomy')
    ).any():
        features['mechanical_thrombectomy'] = True
        features['any_reperfusion_therapy'] = True

    # Surgical embolectomy
    if treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'reperfusion_therapy.surgical_embolectomy')
    ).any():
        features['surgical_embolectomy'] = True
        features['any_reperfusion_therapy'] = True

    # IVC filter
    ivc_rows = treatment[treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'ivc_filter.filter_placement')
    )]
    if len(ivc_rows) > 0:
        features['ivc_filter_placed'] = True
        first_ivc = ivc_rows.iloc[0]
        if 'procedure_datetime' in first_ivc:
            features['ivc_filter_datetime'] = first_ivc['procedure_datetime']
        if 'hours_from_pe' in first_ivc:
            features['ivc_filter_hours_from_pe'] = first_ivc['hours_from_pe']

    # Vascular access
    features['central_line_placed'] = treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'vascular_access.central_line')
    ).any()

    features['arterial_line_placed'] = treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'vascular_access.arterial_line')
    ).any()

    features['pa_catheter_placed'] = treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'vascular_access.pa_catheter')
    ).any()

    # Intubation
    intub_rows = treatment[treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'respiratory_support.intubation')
    )]
    if len(intub_rows) > 0:
        features['intubation_performed'] = True
        first_intub = intub_rows.iloc[0]
        if 'procedure_datetime' in first_intub:
            features['intubation_datetime'] = first_intub['procedure_datetime']
        if 'hours_from_pe' in first_intub:
            features['intubation_hours_from_pe'] = first_intub['hours_from_pe']

    # HFNC
    features['hfnc_initiated'] = treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'respiratory_support.hfnc')
    ).any()

    # NIPPV
    features['nippv_initiated'] = treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'respiratory_support.nippv')
    ).any()

    # ECMO
    ecmo_rows = treatment[treatment['code'].apply(
        lambda x: _is_code_in_category(x, 'circulatory_support.ecmo_va') or
                  _is_code_in_category(x, 'circulatory_support.ecmo_vv')
    )]
    if len(ecmo_rows) > 0:
        features['ecmo_initiated'] = True
        first_ecmo = ecmo_rows.iloc[0]
        if 'procedure_datetime' in first_ecmo:
            features['ecmo_datetime'] = first_ecmo['procedure_datetime']
        if 'hours_from_pe' in first_ecmo:
            features['ecmo_hours_from_pe'] = first_ecmo['hours_from_pe']
        # Note: Can't distinguish VA vs VV from CPT codes alone
        features['ecmo_type'] = 'unknown'

    return features


# =============================================================================
# ESCALATION/COMPLICATION FEATURES
# =============================================================================

def compute_escalation_features(df: pd.DataFrame) -> Dict:
    """
    Compute escalation/complication features (>72h post-PE).

    Args:
        df: Procedures for one patient with is_escalation flag

    Returns:
        Dictionary of escalation features
    """
    escalation = df[df['is_escalation'] == True]

    features = {
        # Respiratory escalation
        'delayed_intubation': False,
        'reintubation': False,
        'tracheostomy': False,

        # Bleeding complications
        'any_transfusion': False,
        'first_transfusion_datetime': pd.NaT,
        'rbc_units_total': 0,
        'massive_transfusion': False,
        'gi_endoscopy_for_bleeding': False,

        # Other complications
        'cardiac_arrest_post_pe': False,
        'cardiac_arrest_datetime': pd.NaT,
        'rrt_initiated': False,
    }

    if len(escalation) == 0:
        return features

    # Delayed intubation
    features['delayed_intubation'] = escalation['code'].apply(
        lambda x: _is_code_in_category(x, 'respiratory_support.intubation')
    ).any()

    # Reintubation (multiple intubation events in escalation window)
    intub_count = escalation['code'].apply(
        lambda x: _is_code_in_category(x, 'respiratory_support.intubation')
    ).sum()
    features['reintubation'] = intub_count > 1

    # Tracheostomy
    features['tracheostomy'] = escalation['code'].apply(
        lambda x: _is_code_in_category(x, 'respiratory_support.tracheostomy')
    ).any()

    # Transfusions
    transfusion_rows = escalation[escalation['code'].apply(
        lambda x: _is_code_in_category(x, 'transfusion.rbc_transfusion')
    )]

    if len(transfusion_rows) > 0:
        features['any_transfusion'] = True
        features['first_transfusion_datetime'] = transfusion_rows.iloc[0].get(
            'procedure_datetime', pd.NaT
        )

        # Sum RBC units (use quantity if available)
        if 'quantity' in transfusion_rows.columns:
            total_units = transfusion_rows['quantity'].fillna(1).sum()
            features['rbc_units_total'] = int(total_units)
            features['massive_transfusion'] = total_units >= 10

    # GI endoscopy for bleeding
    features['gi_endoscopy_for_bleeding'] = escalation['code'].apply(
        lambda x: _is_code_in_category(x, 'gi_bleeding_workup.egd') or
                  _is_code_in_category(x, 'gi_bleeding_workup.colonoscopy')
    ).any()

    # Cardiac arrest (CPR)
    cpr_rows = escalation[escalation['code'].apply(
        lambda x: _is_code_in_category(x, 'resuscitation.cpr')
    )]
    if len(cpr_rows) > 0:
        features['cardiac_arrest_post_pe'] = True
        features['cardiac_arrest_datetime'] = cpr_rows.iloc[0].get('procedure_datetime', pd.NaT)

    # RRT initiated
    features['rrt_initiated'] = escalation['code'].apply(
        lambda x: _is_code_in_category(x, 'dialysis.hemodialysis') or
                  _is_code_in_category(x, 'dialysis.crrt')
    ).any()

    return features


# =============================================================================
# PATIENT-LEVEL AGGREGATION
# =============================================================================

def build_patient_pe_features(patient_df: pd.DataFrame) -> Dict:
    """
    Build complete PE feature set for a single patient.

    Args:
        patient_df: All procedures for one patient

    Returns:
        Dictionary with all PE features
    """
    features = {'empi': patient_df['empi'].iloc[0]}

    # Compute each feature category
    features.update(compute_lifetime_history_features(patient_df))
    features.update(compute_provoking_features(patient_df))
    features.update(compute_diagnostic_features(patient_df))
    features.update(compute_initial_treatment_features(patient_df))
    features.update(compute_escalation_features(patient_df))

    return features


def build_pe_features_for_cohort(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build PE features for entire cohort.

    Args:
        df: Mapped procedures for all patients

    Returns:
        DataFrame with one row per patient, all PE features
    """
    print("=" * 60)
    print("Layer 3: PE-Specific Procedure Features")
    print("=" * 60)

    print(f"\n1. Processing {df['empi'].nunique()} patients...")

    all_features = []

    for empi, group in df.groupby('empi'):
        try:
            features = build_patient_pe_features(group)
            all_features.append(features)
        except Exception as e:
            print(f"   Warning: Failed to process patient {empi}: {e}")
            continue

    result = pd.DataFrame(all_features)

    print(f"\n2. Feature generation complete:")
    print(f"   Patients with features: {len(result)}")
    print(f"   Total features: {len(result.columns)}")

    # Summary statistics
    if len(result) > 0:
        print("\n3. Key feature prevalence:")
        print(f"   CTA performed: {result['cta_chest_performed'].sum()} ({result['cta_chest_performed'].mean():.1%})")
        print(f"   Echo performed: {result['echo_performed'].sum()} ({result['echo_performed'].mean():.1%})")
        print(f"   Intubation: {result['intubation_performed'].sum()} ({result['intubation_performed'].mean():.1%})")
        print(f"   CDT: {result['catheter_directed_therapy'].sum()} ({result['catheter_directed_therapy'].mean():.1%})")
        print(f"   IVC filter: {result['ivc_filter_placed'].sum()} ({result['ivc_filter_placed'].mean():.1%})")
        print(f"   ECMO: {result['ecmo_initiated'].sum()} ({result['ecmo_initiated'].mean():.1%})")
        print(f"   Provoked PE: {result['provoked_pe'].sum()} ({result['provoked_pe'].mean():.1%})")

    print("\n" + "=" * 60)
    print("Layer 3 Complete!")
    print("=" * 60)

    return result


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def run_pe_feature_builder(
    input_path: Optional[Path] = None,
    output_path: Optional[Path] = None,
    test_mode: bool = False,
) -> pd.DataFrame:
    """
    Main pipeline: mapped_procedures.parquet -> PE features.

    Args:
        input_path: Path to mapped procedures
        output_path: Path for output
        test_mode: If True, process subset

    Returns:
        DataFrame with PE features
    """
    # Load mapped procedures
    if input_path is None:
        filename = "mapped_procedures_test.parquet" if test_mode else "mapped_procedures.parquet"
        input_path = SILVER_DIR / filename

    print(f"Loading mapped procedures from: {input_path}")
    df = pd.read_parquet(input_path)

    if test_mode:
        # Process first 100 patients in test mode
        sample_empis = df['empi'].unique()[:100]
        df = df[df['empi'].isin(sample_empis)]

    # Build features
    result = build_pe_features_for_cohort(df)

    # Save output
    if output_path is None:
        output_dir = GOLD_DIR / "pe_procedure_features"
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = "pe_features_test.parquet" if test_mode else "pe_features.parquet"
        output_path = output_dir / filename

    print(f"\nSaving to: {output_path}")
    result.to_parquet(output_path, index=False)

    return result


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build PE-specific procedure features")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    args = parser.parse_args()

    run_pe_feature_builder(test_mode=args.test)
