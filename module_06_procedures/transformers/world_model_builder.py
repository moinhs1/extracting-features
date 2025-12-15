"""
World Model Procedure Representations (Layer 5)
===============================================

Dual representation of procedures:
1. **Actions** - Clinician decisions (discretion-weighted)
2. **State Updates** - Patient state changes

Outputs three components:
- Static state: One-time patient features (lifetime history, provocation)
- Dynamic state: Per-hour patient state (support level, cumulative burden)
- Action vectors: Per-hour discretion-weighted actions

For world model: Next_State = WorldModel(State_t, Action_t) + Stochastic_Events
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import (
    DISCRETION_WEIGHTS_YAML,
    SURGICAL_RISK_YAML,
    GOLD_DIR,
)


# =============================================================================
# DISCRETION WEIGHTS
# =============================================================================

_discretion_cache: Optional[Dict] = None


def load_discretion_weights(filepath: Path = None) -> Dict:
    """
    Load discretion weights from YAML configuration.

    Returns:
        Dictionary with discretion levels and procedure mappings
    """
    global _discretion_cache

    if _discretion_cache is None:
        filepath = filepath or DISCRETION_WEIGHTS_YAML

        with open(filepath, 'r') as f:
            _discretion_cache = yaml.safe_load(f)

    return _discretion_cache


def get_discretion_weight(procedure_code: str) -> float:
    """
    Get discretion weight for a procedure code.

    Args:
        procedure_code: CPT or other procedure code

    Returns:
        Discretion weight (0.0 = no discretion, 1.0 = full discretion)
    """
    config = load_discretion_weights()

    # Search through discretion levels
    discretion_levels = config.get('discretion_levels', {})

    for level_name, level_data in discretion_levels.items():
        procedures = level_data.get('procedures', [])

        for proc in procedures:
            cpt_codes = proc.get('cpt_codes', [])

            if procedure_code in cpt_codes:
                # Return specific weight if defined, otherwise level default
                if 'weight' in proc:
                    return proc['weight']
                elif 'weight' in level_data:
                    return level_data['weight']
                elif 'weight_range' in level_data:
                    # Return midpoint of range
                    weight_range = level_data['weight_range']
                    return (weight_range[0] + weight_range[1]) / 2

    # Default: moderate discretion for unknown procedures
    return 0.5


def _build_code_to_discretion_map() -> Dict[str, float]:
    """Build lookup dictionary of code -> discretion weight."""
    config = load_discretion_weights()
    code_map = {}

    discretion_levels = config.get('discretion_levels', {})

    for level_name, level_data in discretion_levels.items():
        procedures = level_data.get('procedures', [])

        for proc in procedures:
            cpt_codes = proc.get('cpt_codes', [])
            weight = proc.get('weight')

            if weight is None:
                if 'weight' in level_data:
                    weight = level_data['weight']
                elif 'weight_range' in level_data:
                    weight_range = level_data['weight_range']
                    weight = (weight_range[0] + weight_range[1]) / 2

            for code in cpt_codes:
                code_map[code] = weight

    return code_map


# =============================================================================
# ACTION VECTORS
# =============================================================================

def compute_action_vector(
    procedures_df: pd.DataFrame,
    hour: float,
    hour_window: float = 1.0
) -> Dict[str, Any]:
    """
    Compute action vector for a specific timestep.

    Actions are discretion-weighted procedures that can be counterfactually varied.

    Args:
        procedures_df: DataFrame with procedure records for single patient
        hour: Hour from PE to compute actions for
        hour_window: Hours around timestep to consider (default ±1h)

    Returns:
        Dictionary with action vector components
    """
    # Filter to procedures in hour window
    procs_at_hour = procedures_df[
        (procedures_df['hours_from_pe'] >= hour - hour_window) &
        (procedures_df['hours_from_pe'] < hour + hour_window)
    ].copy()

    # Initialize action vector
    action_vec = {
        # High-discretion PE therapeutic actions
        'thrombolysis_action': 0,
        'cdt_action': 0,
        'ivc_filter_action': 0,
        'surgical_embolectomy_action': 0,

        # Moderate-discretion support actions (weighted)
        'intubation_action': 0.0,
        'ecmo_action': 0.0,
        'central_access_action': 0.0,

        # Low-discretion actions
        'transfusion_action': 0.0,
        'dialysis_action': 0.0,

        # Aggregated metrics
        'num_therapeutic_procedures': 0,
        'max_invasiveness': 0,
        'escalation_indicator': 0,
    }

    if len(procs_at_hour) == 0:
        return action_vec

    # Map codes to actions
    code_to_discretion = _build_code_to_discretion_map()

    for _, proc in procs_at_hour.iterrows():
        code = str(proc['code'])
        discretion = code_to_discretion.get(code, 0.5)

        # High discretion procedures (weight = 1.0)
        if code in ['37211', '37212', '37213', '37214']:  # CDT
            action_vec['cdt_action'] = 1
            action_vec['num_therapeutic_procedures'] += 1

        elif code in ['37191']:  # IVC filter
            action_vec['ivc_filter_action'] = 1
            action_vec['num_therapeutic_procedures'] += 1

        elif code in ['33910', '33915', '33916']:  # Surgical embolectomy
            action_vec['surgical_embolectomy_action'] = 1
            action_vec['num_therapeutic_procedures'] += 1

        # Moderate discretion (weighted)
        elif code in ['31500']:  # Intubation
            action_vec['intubation_action'] = discretion
            action_vec['num_therapeutic_procedures'] += 1
            action_vec['max_invasiveness'] = max(action_vec['max_invasiveness'], 3)

        elif code in ['33946', '33947']:  # ECMO
            action_vec['ecmo_action'] = discretion
            action_vec['num_therapeutic_procedures'] += 1
            action_vec['max_invasiveness'] = max(action_vec['max_invasiveness'], 3)
            action_vec['escalation_indicator'] = 1

        elif code in ['36555', '36556', '36557', '36558']:  # Central line
            action_vec['central_access_action'] = discretion

        # Low discretion
        elif code in ['36430']:  # Transfusion
            action_vec['transfusion_action'] = discretion

        elif code in ['90935', '90937', '90945', '90947']:  # Dialysis
            action_vec['dialysis_action'] = discretion

        # CPR is excluded (discretion = 0, state update only)

    return action_vec


# =============================================================================
# STATE VECTORS
# =============================================================================

def compute_support_level(state: Dict[str, int]) -> int:
    """
    Compute ordinal support level (0-5).

    0: Room air, no support
    1: Supplemental O2
    2: HFNC or NIPPV
    3: Mechanical ventilation
    4: Mechanical ventilation + vasopressors
    5: ECMO

    Args:
        state: Dictionary with support indicators

    Returns:
        Ordinal support level
    """
    if state.get('on_ecmo', 0) == 1:
        return 5

    if state.get('on_mechanical_ventilation', 0) == 1:
        # Note: We don't have vasopressor data in procedures
        # So can't distinguish level 3 vs 4
        return 3

    # For levels 1-2, would need oxygen delivery data
    # Default to 0 (baseline)
    return 0


def compute_state_vector(
    procedures_df: pd.DataFrame,
    hour: float
) -> Dict[str, Any]:
    """
    Compute state vector for a specific timestep.

    State represents patient's current condition based on procedures.

    Args:
        procedures_df: DataFrame with procedure records for single patient
        hour: Hour from PE to compute state for

    Returns:
        Dictionary with state vector components
    """
    # Get all procedures up to this hour
    procs_up_to_hour = procedures_df[
        procedures_df['hours_from_pe'] <= hour
    ].copy()

    # Initialize state vector
    state_vec = {
        # Current support status (reversible)
        'on_mechanical_ventilation': 0,
        'on_ecmo': 0,
        'has_central_access': 0,

        # Support level ordinal
        'support_level': 0,

        # Complication markers (irreversible)
        'cardiac_arrest_occurred': 0,
        'major_bleeding_occurred': 0,
        'rrt_initiated': 0,

        # Cumulative burden
        'cumulative_rbc_units': 0,
        'cumulative_invasive_procedures': 0,
        'procedure_intensity_score': 0.0,
    }

    if len(procs_up_to_hour) == 0:
        return state_vec

    # Check for mechanical ventilation
    # Patient is on vent if intubated and not yet extubated
    # For simplicity, assume once intubated, on vent for duration
    intubation_codes = ['31500']
    if any(procs_up_to_hour['code'].isin(intubation_codes)):
        state_vec['on_mechanical_ventilation'] = 1

    # Check for ECMO
    ecmo_codes = ['33946', '33947']
    if any(procs_up_to_hour['code'].isin(ecmo_codes)):
        state_vec['on_ecmo'] = 1

    # Check for central access
    central_line_codes = ['36555', '36556', '36557', '36558']
    if any(procs_up_to_hour['code'].isin(central_line_codes)):
        state_vec['has_central_access'] = 1

    # Compute support level
    state_vec['support_level'] = compute_support_level(state_vec)

    # Check for cardiac arrest (CPR = irreversible marker)
    cpr_codes = ['92950']
    if any(procs_up_to_hour['code'].isin(cpr_codes)):
        state_vec['cardiac_arrest_occurred'] = 1

    # Check for major bleeding (transfusion as proxy)
    transfusion_codes = ['36430']
    transfusions = procs_up_to_hour[procs_up_to_hour['code'].isin(transfusion_codes)]
    if len(transfusions) >= 2:  # Multiple transfusions = major bleeding
        state_vec['major_bleeding_occurred'] = 1

    # Cumulative RBC units
    if len(transfusions) > 0:
        # Use quantity if available, otherwise count procedures
        if 'quantity' in transfusions.columns:
            state_vec['cumulative_rbc_units'] = int(transfusions['quantity'].fillna(1).sum())
        else:
            state_vec['cumulative_rbc_units'] = len(transfusions)

    # Check for RRT
    dialysis_codes = ['90935', '90937', '90945', '90947']
    if any(procs_up_to_hour['code'].isin(dialysis_codes)):
        state_vec['rrt_initiated'] = 1

    # Cumulative invasive procedures
    invasive_codes = intubation_codes + ecmo_codes + central_line_codes
    state_vec['cumulative_invasive_procedures'] = len(
        procs_up_to_hour[procs_up_to_hour['code'].isin(invasive_codes)]
    )

    # Procedure intensity score
    state_vec['procedure_intensity_score'] = compute_procedure_intensity_score(
        procs_up_to_hour.to_dict('records')
    )

    return state_vec


def compute_procedure_intensity_score(procedures: List[Dict]) -> float:
    """
    Compute aggregate procedure intensity/burden score (0-1).

    Combines:
    - Number of procedures
    - Invasiveness levels
    - Temporal clustering

    Args:
        procedures: List of procedure dictionaries

    Returns:
        Intensity score 0-1
    """
    if not procedures or len(procedures) == 0:
        return 0.0

    # Simple heuristic: count invasive procedures, normalize
    invasive_codes = {
        '31500': 3,  # Intubation
        '33946': 3,  # ECMO
        '33947': 3,
        '36555': 2,  # Central line
        '36556': 2,
        '36557': 2,
        '36558': 2,
        '37191': 2,  # IVC filter
        '37211': 3,  # CDT
        '37212': 3,
        '92950': 3,  # CPR
    }

    total_invasiveness = 0
    for proc in procedures:
        code = str(proc.get('code', ''))
        invasiveness = invasive_codes.get(code, 0)
        total_invasiveness += invasiveness

    # Normalize: ~5 highly invasive procedures = max intensity
    # This gives 3 procedures (invasiveness 3+3+2=8) a score of 8/15 = 0.53
    score = min(total_invasiveness / 15.0, 1.0)

    return score


# =============================================================================
# STATIC STATE
# =============================================================================

def build_static_state(
    procedures_df: pd.DataFrame,
    empi: str
) -> Dict[str, Any]:
    """
    Build static procedure state for a patient.

    Computed once per patient from lifetime history and provoking window.

    Args:
        procedures_df: DataFrame with procedure records for single patient
        empi: Patient EMPI

    Returns:
        Dictionary with static state features
    """
    static_state = {
        # Lifetime history
        'empi': empi,
        'prior_vte_procedures': 0,
        'prior_ivc_filter': 0,
        'prior_major_surgery': 0,
        'lifetime_surgical_risk_score': 0.0,

        # Provoking factors
        'provoked_pe': 0,
        'days_from_provoking_surgery': None,
        'provocation_strength': 0.0,

        # Chronic conditions
        'has_pacemaker': 0,
        'on_chronic_dialysis': 0,
    }

    # Lifetime history (before index admission)
    if 'is_lifetime_history' in procedures_df.columns:
        lifetime_procs = procedures_df[procedures_df['is_lifetime_history'] == True]
    else:
        # Fallback: procedures way in the past
        lifetime_procs = procedures_df[procedures_df['hours_from_pe'] < -720]

    # Prior VTE procedures
    vte_codes = ['37191', '37193', '37211', '37212', '37213', '37214',
                 '33910', '33915', '33916']
    prior_vte = lifetime_procs[lifetime_procs['code'].isin(vte_codes)]
    static_state['prior_vte_procedures'] = len(prior_vte)

    # Prior IVC filter
    ivc_filter_codes = ['37191']
    if any(lifetime_procs['code'].isin(ivc_filter_codes)):
        static_state['prior_ivc_filter'] = 1

    # Lifetime surgical risk
    # Simple heuristic: count major surgeries in lifetime
    major_surgery_ccs = ['43', '44', '49', '36', '75', '78', '84', '99',
                         '153', '154', '158']
    if 'ccs_category' in lifetime_procs.columns:
        major_surgeries = lifetime_procs[
            lifetime_procs['ccs_category'].isin(major_surgery_ccs)
        ]
        static_state['prior_major_surgery'] = int(len(major_surgeries) > 0)

        # Risk score: normalized count
        static_state['lifetime_surgical_risk_score'] = min(
            len(major_surgeries) / 5.0,
            1.0
        )

    # Provoking window (1-30 days pre-PE)
    if 'is_provoking_window' in procedures_df.columns:
        provoking_procs = procedures_df[procedures_df['is_provoking_window'] == True]
    else:
        provoking_procs = procedures_df[
            (procedures_df['hours_from_pe'] >= -720) &
            (procedures_df['hours_from_pe'] < 0)
        ]

    # Recent surgery
    if 'ccs_category' in provoking_procs.columns:
        recent_surgeries = provoking_procs[
            provoking_procs['ccs_category'].isin(major_surgery_ccs)
        ]

        if len(recent_surgeries) > 0:
            static_state['provoked_pe'] = 1

            # Days from most recent surgery
            most_recent = recent_surgeries.nsmallest(1, 'hours_from_pe')
            if len(most_recent) > 0:
                hours = abs(most_recent.iloc[0]['hours_from_pe'])
                static_state['days_from_provoking_surgery'] = int(hours / 24)

            # Provocation strength based on VTE risk and recency
            # Simpler version: count of high-risk procedures
            static_state['provocation_strength'] = min(len(recent_surgeries) / 3.0, 1.0)

    # Chronic dialysis
    dialysis_codes = ['90935', '90937', '90945', '90947']
    chronic_dialysis = lifetime_procs[lifetime_procs['code'].isin(dialysis_codes)]
    if len(chronic_dialysis) >= 3:  # Multiple dialysis = chronic
        static_state['on_chronic_dialysis'] = 1

    return static_state


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def build_world_model_features(
    procedures_df: pd.DataFrame,
    max_hours: int = 168,  # 7 days default
    output_dir: Optional[Path] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Build world model features: static state, dynamic state, and action vectors.

    Args:
        procedures_df: DataFrame with mapped procedure records
        max_hours: Maximum hours from PE to generate (default 168 = 7 days)
        output_dir: Output directory for parquet files (None = don't save)

    Returns:
        Dictionary with 'static_state', 'dynamic_state', 'action_vectors' DataFrames
    """
    print("=" * 60)
    print("Layer 5: World Model Procedure Representations")
    print("=" * 60)

    # Get unique patients
    empis = procedures_df['empi'].unique()
    print(f"\nPatients: {len(empis)}")

    # Build static state for each patient
    print("\n1. Building static procedure states...")
    static_states = []

    for empi in empis:
        patient_procs = procedures_df[procedures_df['empi'] == empi]
        static_state = build_static_state(patient_procs, empi)
        static_states.append(static_state)

    static_df = pd.DataFrame(static_states)
    print(f"   Static states: {len(static_df)} patients")

    # Build dynamic state and action vectors
    print(f"\n2. Building dynamic states and actions (0-{max_hours}h)...")

    dynamic_states = []
    action_vectors = []

    for empi in empis:
        patient_procs = procedures_df[procedures_df['empi'] == empi]

        # Generate hourly timesteps
        for hour in range(0, max_hours + 1):
            # Compute state at this hour
            state_vec = compute_state_vector(patient_procs, hour)
            state_vec['empi'] = empi
            state_vec['hour'] = hour
            dynamic_states.append(state_vec)

            # Compute actions at this hour
            action_vec = compute_action_vector(patient_procs, hour)
            action_vec['empi'] = empi
            action_vec['hour'] = hour
            action_vectors.append(action_vec)

    dynamic_df = pd.DataFrame(dynamic_states)
    action_df = pd.DataFrame(action_vectors)

    print(f"   Dynamic states: {len(dynamic_df)} patient-hours")
    print(f"   Action vectors: {len(action_df)} patient-hours")

    # Summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)

    print("\nStatic State:")
    print(f"  Prior IVC filters: {static_df['prior_ivc_filter'].sum()}")
    print(f"  Provoked PE: {static_df['provoked_pe'].sum()}")
    print(f"  Chronic dialysis: {static_df['on_chronic_dialysis'].sum()}")

    print("\nDynamic State (aggregated):")
    print(f"  Mechanical ventilation hours: {dynamic_df['on_mechanical_ventilation'].sum()}")
    print(f"  ECMO hours: {dynamic_df['on_ecmo'].sum()}")
    print(f"  Cardiac arrests: {dynamic_df['cardiac_arrest_occurred'].sum()}")

    print("\nActions (aggregated):")
    print(f"  CDT actions: {action_df['cdt_action'].sum()}")
    print(f"  IVC filter actions: {action_df['ivc_filter_action'].sum()}")
    print(f"  Intubation actions: {action_df['intubation_action'].sum():.1f}")

    # Save outputs
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        static_path = output_dir / "static_state.parquet"
        dynamic_path = output_dir / "dynamic_state.parquet"
        action_path = output_dir / "action_vectors.parquet"

        print(f"\n3. Saving outputs...")
        static_df.to_parquet(static_path, index=False)
        dynamic_df.to_parquet(dynamic_path, index=False)
        action_df.to_parquet(action_path, index=False)

        print(f"   Static state: {static_path}")
        print(f"   Dynamic state: {dynamic_path}")
        print(f"   Action vectors: {action_path}")

    print("\n" + "=" * 60)
    print("Layer 5 Complete!")
    print("=" * 60)

    return {
        'static_state': static_df,
        'dynamic_state': dynamic_df,
        'action_vectors': action_df,
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Build world model procedure representations (Layer 5)"
    )
    parser.add_argument(
        '--input',
        type=str,
        help='Path to mapped procedures parquet (default: silver/mapped_procedures.parquet)'
    )
    parser.add_argument(
        '--max-hours',
        type=int,
        default=168,
        help='Maximum hours from PE (default: 168 = 7 days)'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run in test mode with sample data'
    )

    args = parser.parse_args()

    # Load input data
    if args.input:
        input_path = Path(args.input)
    else:
        from config.procedure_config import SILVER_DIR
        filename = "mapped_procedures_test.parquet" if args.test else "mapped_procedures.parquet"
        input_path = SILVER_DIR / filename

    print(f"Loading procedures from: {input_path}")
    procedures_df = pd.read_parquet(input_path)

    if args.test:
        # Sample 100 patients for testing
        sample_empis = procedures_df['empi'].unique()[:100]
        procedures_df = procedures_df[procedures_df['empi'].isin(sample_empis)]

    # Build world model features
    output_dir = GOLD_DIR / "world_model_states"

    build_world_model_features(
        procedures_df,
        max_hours=args.max_hours,
        output_dir=output_dir,
    )
