#!/usr/bin/env python3
"""
Module 1: Core Infrastructure
==============================

Establishes Time Zero (PE diagnosis time), creates temporal windows,
and extracts all structured outcomes from CPT/ICD codes and medications.

Author: Generated with Claude Code
Date: 2025-11-02
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import pickle
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONSTANTS & CONFIGURATION
# ============================================================================

DATA_DIR = Path("/home/moin/TDA_11_1/Data")
OUTPUT_DIR = Path("/home/moin/TDA_11_1/module_1_core_infrastructure/outputs")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Temporal windows (hours relative to Time Zero)
TEMPORAL_WINDOWS = {
    'BASELINE': (-72, 0),
    'ACUTE': (0, 24),
    'SUBACUTE': (24, 72),
    'RECOVERY': (72, 168)
}

# CPT Codes for outcomes
CPT_CODES = {
    # ICU / Critical Care
    'icu_critical_care': ['99291', '99292'],

    # Intubation & Ventilation
    'intubation': ['31500'],
    'ventilation': ['94002', '94003', '94004'],
    'cpap': ['94660'],

    # Dialysis / Renal Replacement
    'hemodialysis': ['90935', '90937', '90940'],
    'crrt_peritoneal': ['90945', '90947'],

    # Advanced Interventions
    'ivc_filter': ['37191', '37192', '37193'],
    'catheter_directed': ['37187', '37188'],
    'surgical_embolectomy': ['33910', '33916'],
    'ecmo': ['33946', '33947', '33948', '33949', '33952', '33956',
             '33960', '33961', '33966', '33984'],
    'iabp': ['33967', '33968', '33970', '33971'],
    'vad': ['33975', '33976', '33977', '33978', '33979', '33980'],
    'transfusion': ['36430'],
    'cpr': ['92950'],
}

# ICD-10-PCS codes for vasopressor administration
VASOPRESSOR_PCS_CODES = ['00.17', '3E030XZ', '3E033XZ', '3E043XZ', '3E053XZ', '3E063XZ']

# ICD codes for bleeding events
ICD10_BLEEDING = {
    # Tier 1: Major/Fatal Bleeding
    'ich_subarachnoid': ['I60', 'I60.0', 'I60.1', 'I60.2', 'I60.3', 'I60.31', 'I60.32',
                          'I60.4', 'I60.5', 'I60.6', 'I60.7', 'I60.8', 'I60.9'],
    'ich_intracerebral': ['I61', 'I61.0', 'I61.1', 'I61.2', 'I61.3', 'I61.4',
                           'I61.5', 'I61.6', 'I61.8', 'I61.9'],
    'ich_other': ['I62', 'I62.0', 'I62.00', 'I62.01', 'I62.02', 'I62.03',
                   'I62.1', 'I62.9'],
    'gi_bleed': ['K92.0', 'K92.1', 'K92.2', 'I85.01'],
    'acute_blood_loss': ['D62'],

    # Tier 2: Clinically Significant
    'hematuria': ['R31.0'],
    'hemoptysis': ['R04.2'],
    'hemoperitoneum': ['K66.1'],
}

ICD9_BLEEDING = {
    'ich': ['430', '431', '432.0', '432.1', '432.9'],
    'gi_bleed': ['578.0', '578.1', '578.9', '456.0'],
    'acute_blood_loss': ['285.1'],
    'hematuria': ['599.70', '599.71'],
    'hemoptysis': ['786.3', '786.30'],
    'hemoperitoneum': ['568.81'],
}

# ICD codes for shock
ICD10_SHOCK = ['R57.0', 'R57.1', 'R57.9']
ICD9_SHOCK = ['785.50', '785.51', '785.59']

# Medication names for vasopressors/inotropes
VASOPRESSOR_NAMES = ['norepinephrine', 'epinephrine', 'vasopressin',
                     'dopamine', 'phenylephrine']
INOTROPE_NAMES = ['dobutamine', 'milrinone']


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class PatientTimeline:
    """Stores temporal information and outcomes for a patient."""
    patient_id: str
    time_zero: datetime
    window_start: datetime
    window_end: datetime
    phase_boundaries: Dict[str, datetime]
    encounter_info: Dict
    outcomes: Dict
    metadata: Dict

    def to_dict(self):
        """Convert to dictionary for serialization."""
        return asdict(self)


# ============================================================================
# DATA LOADING FUNCTIONS
# ============================================================================

def load_pe_cohort() -> pd.DataFrame:
    """Load PE cohort with Time Zero."""
    print("Loading PE cohort...")
    df = pd.read_csv(DATA_DIR / "PE_dataset_enhanced.csv", low_memory=False)

    # Parse Report_Date_Time as Time Zero
    df['time_zero'] = pd.to_datetime(df['Report_Date_Time'], errors='coerce')

    # Ensure EMPI is string for consistent merging
    df['EMPI'] = df['EMPI'].astype(str)

    print(f"  Loaded {len(df)} PE events")
    print(f"  Missing timestamps: {df['time_zero'].isna().sum()}")

    return df


def load_encounters() -> pd.DataFrame:
    """Load encounter data."""
    print("Loading encounters...")

    # Read file with pipe delimiter
    df = pd.read_csv(DATA_DIR / "FNR_20240409_091633_Enc.txt",
                     sep='|', low_memory=False)

    # Parse dates (note: columns are Admit_Date and Discharge_Date, not *_Date_Time)
    df['Admit_Date_Time'] = pd.to_datetime(df['Admit_Date'], errors='coerce')
    df['Discharge_Date_Time'] = pd.to_datetime(df['Discharge_Date'], errors='coerce')

    # Ensure EMPI is string for consistent merging
    df['EMPI'] = df['EMPI'].astype(str)

    print(f"  Loaded {len(df)} encounters")

    return df


def load_procedures() -> pd.DataFrame:
    """Load procedure data (Prc.txt)."""
    print("Loading procedures...")

    df = pd.read_csv(DATA_DIR / "FNR_20240409_091633_Prc.txt",
                     sep='|', low_memory=False)

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Ensure EMPI is string for consistent merging
    df['EMPI'] = df['EMPI'].astype(str)

    print(f"  Loaded {len(df)} procedures")

    return df


def load_diagnoses() -> pd.DataFrame:
    """Load diagnosis data (Dia.txt)."""
    print("Loading diagnoses...")

    df = pd.read_csv(DATA_DIR / "FNR_20240409_091633_Dia.txt",
                     sep='|', low_memory=False)

    # Parse dates
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')

    # Ensure EMPI is string for consistent merging
    df['EMPI'] = df['EMPI'].astype(str)

    print(f"  Loaded {len(df)} diagnoses")

    return df


def load_medications() -> pd.DataFrame:
    """Load medication data (Med.txt)."""
    print("Loading medications...")

    df = pd.read_csv(DATA_DIR / "FNR_20240409_091633_Med.txt",
                     sep='|', low_memory=False)

    # Parse dates
    df['Medication_Date'] = pd.to_datetime(df['Medication_Date'], errors='coerce')

    # Ensure EMPI is string for consistent merging
    df['EMPI'] = df['EMPI'].astype(str)

    print(f"  Loaded {len(df)} medication records")

    return df


def load_demographics() -> pd.DataFrame:
    """Load demographics data (Dem files)."""
    print("Loading demographics...")

    # Load both demographics files
    dem1 = pd.read_csv(DATA_DIR / "FNR_20240409_091633-1_Dem.txt",
                       sep='|', low_memory=False)
    dem2 = pd.read_csv(DATA_DIR / "FNR_20240409_091633-2_Dem.txt",
                       sep='|', low_memory=False)

    # Combine them
    df = pd.concat([dem1, dem2], ignore_index=True)

    # Remove duplicates (keep first occurrence)
    df = df.drop_duplicates(subset=['EMPI'], keep='first')

    # Parse Date_Of_Death
    df['Date_Of_Death'] = pd.to_datetime(df['Date_Of_Death'], errors='coerce')

    # Ensure EMPI is string for consistent merging
    df['EMPI'] = df['EMPI'].astype(str)

    print(f"  Loaded {len(df)} unique patients")

    return df


# ============================================================================
# TIME ZERO & TEMPORAL WINDOWS
# ============================================================================

def establish_time_zero(pe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Establish Time Zero for each patient (first PE diagnosis).

    Args:
        pe_df: PE cohort dataframe

    Returns:
        Deduplicated dataframe with one row per patient (first PE)
    """
    print("\nEstablishing Time Zero...")

    # Remove rows with missing time_zero
    pe_df = pe_df[pe_df['time_zero'].notna()].copy()

    # Quality filter: Remove very old data (before 2010)
    pe_df = pe_df[pe_df['time_zero'].dt.year >= 2010].copy()

    # For each patient, keep first PE diagnosis
    pe_df = pe_df.sort_values('time_zero').groupby('EMPI', as_index=False).first()

    print(f"  Total patients with first PE: {len(pe_df)}")
    print(f"  Date range: {pe_df['time_zero'].min()} to {pe_df['time_zero'].max()}")

    return pe_df


def link_encounters_to_patients(pe_df: pd.DataFrame, enc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Link encounters to PE patients using 4-tier matching strategy with fallback.

    Tier 1: Direct temporal overlap (wide window: 7d before to 30d after)
    Tier 2: Inpatient encounter containing PE date
    Tier 3: Closest inpatient encounter (±14 days)
    Tier 4: Fixed temporal window estimate

    Args:
        pe_df: PE cohort with time_zero
        enc_df: Encounter dataframe

    Returns:
        pe_df with encounter information and match quality metadata
    """
    print("\nLinking encounters to patients (4-tier strategy)...")

    # Initialize tracking columns
    pe_df['Encounter_number'] = None
    pe_df['Admit_Date_Time'] = pd.NaT
    pe_df['Discharge_Date_Time'] = pd.NaT
    pe_df['Inpatient_Outpatient'] = ''
    pe_df['encounter_match_method'] = ''
    pe_df['encounter_match_confidence'] = ''

    # Process each patient
    tier_counts = {'tier1': 0, 'tier2': 0, 'tier3': 0, 'tier4': 0}

    for idx, patient in pe_df.iterrows():
        empi = patient['EMPI']
        time_zero = patient['time_zero']

        # Get all encounters for this patient
        patient_encounters = enc_df[enc_df['EMPI'] == empi].copy()

        if len(patient_encounters) == 0:
            # No encounters at all - use Tier 4
            pe_df.at[idx, 'encounter_match_method'] = 'tier4'
            pe_df.at[idx, 'encounter_match_confidence'] = 'low'
            tier_counts['tier4'] += 1
            continue

        matched_encounter = None
        match_method = None

        # TIER 1: Direct temporal overlap (widened window: 7d before to 30d after)
        window_start = time_zero - pd.Timedelta(days=7)
        window_end = time_zero + pd.Timedelta(days=30)

        tier1_matches = patient_encounters[
            (patient_encounters['Admit_Date_Time'] <= window_end) &
            (patient_encounters['Discharge_Date_Time'] >= window_start)
        ].copy()

        if len(tier1_matches) > 0:
            # Prefer encounters closest to PE time
            tier1_matches['time_diff'] = abs(
                (tier1_matches['Admit_Date_Time'] - time_zero).dt.total_seconds()
            )
            matched_encounter = tier1_matches.sort_values('time_diff').iloc[0]
            match_method = 'tier1'
            tier_counts['tier1'] += 1

        # TIER 2: Inpatient encounter containing PE date (if Tier 1 failed)
        if matched_encounter is None:
            tier2_matches = patient_encounters[
                (patient_encounters['Inpatient_Outpatient'] == 'Inpatient') &
                (patient_encounters['Admit_Date_Time'] <= time_zero) &
                (patient_encounters['Discharge_Date_Time'] >= time_zero)
            ]

            if len(tier2_matches) > 0:
                # If multiple, take the one with earliest admission
                matched_encounter = tier2_matches.sort_values('Admit_Date_Time').iloc[0]
                match_method = 'tier2'
                tier_counts['tier2'] += 1

        # TIER 3: Closest inpatient encounter within ±14 days (if Tier 2 failed)
        if matched_encounter is None:
            tier3_window_start = time_zero - pd.Timedelta(days=14)
            tier3_window_end = time_zero + pd.Timedelta(days=14)

            tier3_matches = patient_encounters[
                (patient_encounters['Inpatient_Outpatient'] == 'Inpatient') &
                (patient_encounters['Admit_Date_Time'] >= tier3_window_start) &
                (patient_encounters['Admit_Date_Time'] <= tier3_window_end)
            ].copy()

            if len(tier3_matches) > 0:
                # Prefer encounters BEFORE PE date (PE diagnosed during hospitalization)
                tier3_matches['time_diff'] = (tier3_matches['Admit_Date_Time'] - time_zero).dt.total_seconds()
                tier3_matches['is_before'] = tier3_matches['time_diff'] < 0

                # Sort: before PE first, then by absolute time difference
                tier3_matches = tier3_matches.sort_values(
                    ['is_before', 'time_diff'],
                    ascending=[False, True]
                )
                matched_encounter = tier3_matches.iloc[0]
                match_method = 'tier3'
                tier_counts['tier3'] += 1

        # TIER 4: No match - will use fixed window
        if matched_encounter is None:
            match_method = 'tier4'
            tier_counts['tier4'] += 1

        # Store matched encounter data
        if matched_encounter is not None:
            pe_df.at[idx, 'Encounter_number'] = matched_encounter['Encounter_number']
            pe_df.at[idx, 'Admit_Date_Time'] = matched_encounter['Admit_Date_Time']
            pe_df.at[idx, 'Discharge_Date_Time'] = matched_encounter['Discharge_Date_Time']
            pe_df.at[idx, 'Inpatient_Outpatient'] = matched_encounter['Inpatient_Outpatient']

        pe_df.at[idx, 'encounter_match_method'] = match_method

        # Set confidence based on tier
        if match_method == 'tier1':
            pe_df.at[idx, 'encounter_match_confidence'] = 'high'
        elif match_method == 'tier2':
            pe_df.at[idx, 'encounter_match_confidence'] = 'high'
        elif match_method == 'tier3':
            pe_df.at[idx, 'encounter_match_confidence'] = 'medium'
        else:
            pe_df.at[idx, 'encounter_match_confidence'] = 'low'

    # Calculate median LOS for Tier 4 fallback
    matched_los = (
        pe_df[pe_df['encounter_match_method'] != 'tier4']['Discharge_Date_Time'] -
        pe_df[pe_df['encounter_match_method'] != 'tier4']['Admit_Date_Time']
    ).dt.total_seconds() / (3600 * 24)

    median_los = matched_los.median() if len(matched_los) > 0 else 7.0  # Default 7 days

    # Summary
    total_matched = tier_counts['tier1'] + tier_counts['tier2'] + tier_counts['tier3']
    print(f"  Tier 1 (temporal overlap): {tier_counts['tier1']} patients")
    print(f"  Tier 2 (inpatient containing PE): {tier_counts['tier2']} patients")
    print(f"  Tier 3 (closest inpatient): {tier_counts['tier3']} patients")
    print(f"  Tier 4 (no match - fixed window): {tier_counts['tier4']} patients")
    print(f"  Total matched encounters: {total_matched}/{len(pe_df)} ({100*total_matched/len(pe_df):.1f}%)")
    print(f"  Median LOS for matched patients: {median_los:.1f} days")

    return pe_df


def create_temporal_windows(pe_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create temporal windows for each patient.

    Args:
        pe_df: PE cohort with time_zero and encounter info

    Returns:
        pe_df with window_start, window_end, and phase boundaries
    """
    print("\nCreating temporal windows...")

    # Default windows: Time Zero ± fixed hours
    pe_df['default_start'] = pe_df['time_zero'] - pd.Timedelta(hours=72)
    pe_df['default_end'] = pe_df['time_zero'] + pd.Timedelta(hours=168)

    # Use encounter times if available, otherwise use defaults
    pe_df['window_start'] = pe_df['Admit_Date_Time'].fillna(pe_df['default_start'])
    pe_df['window_end'] = pe_df['Discharge_Date_Time'].fillna(pe_df['default_end'])

    # Ensure window_start is before time_zero and window_end is after
    pe_df['window_start'] = pe_df[['window_start', 'default_start']].min(axis=1)
    pe_df['window_end'] = pe_df[['window_end', 'default_end']].max(axis=1)

    # Create phase boundaries
    pe_df['phase_acute_end'] = pe_df['time_zero'] + pd.Timedelta(hours=24)
    pe_df['phase_subacute_end'] = pe_df['time_zero'] + pd.Timedelta(hours=72)
    pe_df['phase_recovery_end'] = pe_df['time_zero'] + pd.Timedelta(hours=168)

    print(f"  Created temporal windows for {len(pe_df)} patients")

    return pe_df


# ============================================================================
# OUTCOME EXTRACTION: MORTALITY
# ============================================================================

def extract_mortality(pe_df: pd.DataFrame, dem_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract mortality outcomes from demographics files.

    Args:
        pe_df: Patient dataframe with time_zero
        dem_df: Demographics dataframe with Vital_status and Date_Of_Death

    Returns:
        pe_df with mortality columns added
    """
    print("\nExtracting mortality outcomes...")

    # Initialize columns
    pe_df['death_flag'] = 0
    pe_df['date_of_death'] = pd.NaT
    pe_df['in_hospital_death'] = 0
    pe_df['mortality_30d'] = 0
    pe_df['mortality_90d'] = 0
    pe_df['mortality_1yr'] = 0
    pe_df['days_to_death'] = np.nan

    # Merge demographics data
    pe_df = pe_df.merge(
        dem_df[['EMPI', 'Vital_status', 'Date_Of_Death']],
        on='EMPI',
        how='left'
    )

    # Process each patient
    for idx, patient in pe_df.iterrows():
        date_of_death = patient.get('Date_Of_Death')
        vital_status = str(patient.get('Vital_status', '')).lower()
        time_zero = patient['time_zero']

        # Check if patient died
        is_deceased = (
            pd.notna(date_of_death) or
            'deceased' in vital_status or
            'dead' in vital_status
        )

        if is_deceased and pd.notna(date_of_death):
            # Patient died and we have death date
            pe_df.at[idx, 'death_flag'] = 1
            pe_df.at[idx, 'date_of_death'] = date_of_death

            # Calculate days to death
            days_to_death = (date_of_death - time_zero).total_seconds() / (3600 * 24)
            pe_df.at[idx, 'days_to_death'] = days_to_death

            # Mortality timeframes
            if days_to_death <= 30:
                pe_df.at[idx, 'mortality_30d'] = 1
            if days_to_death <= 90:
                pe_df.at[idx, 'mortality_90d'] = 1
            if days_to_death <= 365:
                pe_df.at[idx, 'mortality_1yr'] = 1

            # In-hospital mortality (requires encounter discharge date)
            discharge_date = patient.get('Discharge_Date_Time')
            if pd.notna(discharge_date):
                if date_of_death <= discharge_date:
                    pe_df.at[idx, 'in_hospital_death'] = 1

    # Summary statistics
    death_count = pe_df['death_flag'].sum()
    mort_30d = pe_df['mortality_30d'].sum()
    mort_90d = pe_df['mortality_90d'].sum()
    mort_1yr = pe_df['mortality_1yr'].sum()
    in_hosp = pe_df['in_hospital_death'].sum()

    print(f"  Total deaths: {death_count}/{len(pe_df)} ({100*death_count/len(pe_df):.1f}%)")
    print(f"  30-day mortality: {mort_30d}/{len(pe_df)} ({100*mort_30d/len(pe_df):.1f}%)")
    print(f"  90-day mortality: {mort_90d}/{len(pe_df)} ({100*mort_90d/len(pe_df):.1f}%)")
    print(f"  1-year mortality: {mort_1yr}/{len(pe_df)} ({100*mort_1yr/len(pe_df):.1f}%)")
    print(f"  In-hospital deaths: {in_hosp}/{len(pe_df)} ({100*in_hosp/len(pe_df):.1f}%)")

    return pe_df


# ============================================================================
# OUTCOME EXTRACTION: ICU ADMISSION
# ============================================================================

def extract_icu_admission(pe_df: pd.DataFrame, prc_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract ICU admission from CPT critical care codes and Clinic field.

    Args:
        pe_df: Patient dataframe with temporal windows
        prc_df: Procedures dataframe

    Returns:
        pe_df with ICU outcome columns added
    """
    print("\nExtracting ICU admission...")

    # Initialize columns
    pe_df['icu_admission'] = 0
    pe_df['time_to_icu_hours'] = np.nan
    pe_df['icu_los_days'] = np.nan
    pe_df['icu_type'] = ''
    pe_df['critical_care_minutes'] = 0

    # Filter procedures to critical care codes
    icu_codes = CPT_CODES['icu_critical_care']
    icu_procs = prc_df[prc_df['Code'].isin(icu_codes)].copy()

    print(f"  Found {len(icu_procs)} critical care procedures (CPT 99291/99292)")

    # Process each patient
    for idx, patient in pe_df.iterrows():
        empi = patient['EMPI']
        time_zero = patient['time_zero']
        window_start = patient['window_start']
        window_end = patient['window_end']

        # Get ICU procedures for this patient in temporal window
        patient_icu = icu_procs[
            (icu_procs['EMPI'] == empi) &
            (icu_procs['Date'] >= window_start) &
            (icu_procs['Date'] <= window_end)
        ].copy()

        if len(patient_icu) > 0:
            pe_df.at[idx, 'icu_admission'] = 1

            # Time to ICU
            first_icu = patient_icu['Date'].min()
            time_to_icu = (first_icu - time_zero).total_seconds() / 3600
            pe_df.at[idx, 'time_to_icu_hours'] = time_to_icu

            # ICU length of stay
            last_icu = patient_icu['Date'].max()
            icu_los = (last_icu - first_icu).total_seconds() / (3600 * 24)
            pe_df.at[idx, 'icu_los_days'] = icu_los

            # Critical care minutes (99291 = 30-74 min, 99292 = +30 min)
            count_99291 = (patient_icu['Code'] == '99291').sum()
            count_99292 = (patient_icu['Code'] == '99292').sum()
            total_minutes = (count_99291 * 52) + (count_99292 * 30)  # Use midpoint for 99291
            pe_df.at[idx, 'critical_care_minutes'] = total_minutes

            # ICU type from Clinic field
            if 'Clinic' in patient_icu.columns:
                clinics = patient_icu['Clinic'].dropna().str.upper()
                if any('MICU' in c for c in clinics):
                    pe_df.at[idx, 'icu_type'] = 'Medical'
                elif any('SICU' in c for c in clinics):
                    pe_df.at[idx, 'icu_type'] = 'Surgical'
                elif any('CCU' in c or 'CARDIAC' in c for c in clinics):
                    pe_df.at[idx, 'icu_type'] = 'Cardiac'
                elif any('NEURO' in c for c in clinics):
                    pe_df.at[idx, 'icu_type'] = 'Neuro'
                elif any('ICU' in c for c in clinics):
                    pe_df.at[idx, 'icu_type'] = 'Other'

    icu_count = pe_df['icu_admission'].sum()
    print(f"  ICU admissions: {icu_count}/{len(pe_df)} ({100*icu_count/len(pe_df):.1f}%)")

    return pe_df


# ============================================================================
# OUTCOME EXTRACTION: VENTILATION & INTUBATION
# ============================================================================

def extract_ventilation(pe_df: pd.DataFrame, prc_df: pd.DataFrame) -> pd.DataFrame:
    """Extract intubation and mechanical ventilation outcomes."""
    print("\nExtracting ventilation/intubation...")

    # Initialize columns
    pe_df['intubation_flag'] = 0
    pe_df['time_to_intubation_hours'] = np.nan
    pe_df['ventilation_flag'] = 0
    pe_df['ventilation_days'] = np.nan
    pe_df['cpap_only'] = 0

    # Get relevant codes
    intubation_codes = CPT_CODES['intubation']
    ventilation_codes = CPT_CODES['ventilation']
    cpap_codes = CPT_CODES['cpap']

    # Filter procedures
    vent_procs = prc_df[
        prc_df['Code'].isin(intubation_codes + ventilation_codes + cpap_codes)
    ].copy()

    print(f"  Found {len(vent_procs)} ventilation-related procedures")

    # Process each patient
    for idx, patient in pe_df.iterrows():
        empi = patient['EMPI']
        time_zero = patient['time_zero']
        window_start = patient['window_start']
        window_end = patient['window_end']

        patient_vent = vent_procs[
            (vent_procs['EMPI'] == empi) &
            (vent_procs['Date'] >= window_start) &
            (vent_procs['Date'] <= window_end)
        ].copy()

        if len(patient_vent) > 0:
            # Intubation
            intubations = patient_vent[patient_vent['Code'].isin(intubation_codes)]
            if len(intubations) > 0:
                pe_df.at[idx, 'intubation_flag'] = 1
                first_intubation = intubations['Date'].min()
                time_to_intub = (first_intubation - time_zero).total_seconds() / 3600
                pe_df.at[idx, 'time_to_intubation_hours'] = time_to_intub

            # Mechanical ventilation
            ventilations = patient_vent[patient_vent['Code'].isin(ventilation_codes)]
            if len(ventilations) > 0:
                pe_df.at[idx, 'ventilation_flag'] = 1
                first_vent = ventilations['Date'].min()
                last_vent = ventilations['Date'].max()
                vent_days = (last_vent - first_vent).total_seconds() / (3600 * 24)
                pe_df.at[idx, 'ventilation_days'] = vent_days

            # CPAP only (no intubation or mechanical vent)
            cpap = patient_vent[patient_vent['Code'].isin(cpap_codes)]
            if len(cpap) > 0 and pe_df.at[idx, 'intubation_flag'] == 0 and pe_df.at[idx, 'ventilation_flag'] == 0:
                pe_df.at[idx, 'cpap_only'] = 1

    intub_count = pe_df['intubation_flag'].sum()
    vent_count = pe_df['ventilation_flag'].sum()
    print(f"  Intubations: {intub_count}/{len(pe_df)} ({100*intub_count/len(pe_df):.1f}%)")
    print(f"  Mechanical ventilation: {vent_count}/{len(pe_df)} ({100*vent_count/len(pe_df):.1f}%)")

    return pe_df


# ============================================================================
# OUTCOME EXTRACTION: DIALYSIS
# ============================================================================

def extract_dialysis(pe_df: pd.DataFrame, prc_df: pd.DataFrame) -> pd.DataFrame:
    """Extract dialysis/CRRT outcomes."""
    print("\nExtracting dialysis/CRRT...")

    # Initialize columns
    pe_df['dialysis_flag'] = 0
    pe_df['dialysis_type'] = ''
    pe_df['time_to_dialysis_hours'] = np.nan
    pe_df['dialysis_sessions_count'] = 0

    # Get dialysis codes
    hd_codes = CPT_CODES['hemodialysis']
    crrt_codes = CPT_CODES['crrt_peritoneal']
    all_dialysis_codes = hd_codes + crrt_codes

    dialysis_procs = prc_df[prc_df['Code'].isin(all_dialysis_codes)].copy()

    print(f"  Found {len(dialysis_procs)} dialysis procedures")

    # Process each patient
    for idx, patient in pe_df.iterrows():
        empi = patient['EMPI']
        time_zero = patient['time_zero']
        window_start = patient['window_start']
        window_end = patient['window_end']

        patient_dialysis = dialysis_procs[
            (dialysis_procs['EMPI'] == empi) &
            (dialysis_procs['Date'] >= window_start) &
            (dialysis_procs['Date'] <= window_end)
        ].copy()

        if len(patient_dialysis) > 0:
            pe_df.at[idx, 'dialysis_flag'] = 1
            pe_df.at[idx, 'dialysis_sessions_count'] = len(patient_dialysis)

            # Time to dialysis
            first_dialysis = patient_dialysis['Date'].min()
            time_to_dial = (first_dialysis - time_zero).total_seconds() / 3600
            pe_df.at[idx, 'time_to_dialysis_hours'] = time_to_dial

            # Dialysis type
            has_hd = patient_dialysis['Code'].isin(hd_codes).any()
            has_crrt = patient_dialysis['Code'].isin(crrt_codes).any()

            if has_hd and has_crrt:
                pe_df.at[idx, 'dialysis_type'] = 'Both'
            elif has_crrt:
                pe_df.at[idx, 'dialysis_type'] = 'CRRT/PD'
            elif has_hd:
                pe_df.at[idx, 'dialysis_type'] = 'HD'

    dialysis_count = pe_df['dialysis_flag'].sum()
    print(f"  Dialysis: {dialysis_count}/{len(pe_df)} ({100*dialysis_count/len(pe_df):.1f}%)")

    return pe_df


# ============================================================================
# OUTCOME EXTRACTION: ADVANCED INTERVENTIONS
# ============================================================================

def extract_advanced_interventions(pe_df: pd.DataFrame, prc_df: pd.DataFrame) -> pd.DataFrame:
    """Extract advanced interventions (ECMO, IABP, VAD, IVC filter, etc.)."""
    print("\nExtracting advanced interventions...")

    # Initialize columns for each intervention
    interventions = {
        'ivc_filter': CPT_CODES['ivc_filter'],
        'catheter_directed_therapy': CPT_CODES['catheter_directed'],
        'surgical_embolectomy': CPT_CODES['surgical_embolectomy'],
        'ecmo': CPT_CODES['ecmo'],
        'iabp': CPT_CODES['iabp'],
        'vad': CPT_CODES['vad'],
        'transfusion': CPT_CODES['transfusion'],
        'cpr': CPT_CODES['cpr'],
    }

    for intervention_name in interventions.keys():
        pe_df[f'{intervention_name}_flag'] = 0
        pe_df[f'time_to_{intervention_name}_hours'] = np.nan

    # Process each intervention type
    for intervention_name, codes in interventions.items():
        intervention_procs = prc_df[prc_df['Code'].isin(codes)].copy()

        if len(intervention_procs) > 0:
            print(f"  Found {len(intervention_procs)} {intervention_name} procedures")

            for idx, patient in pe_df.iterrows():
                empi = patient['EMPI']
                time_zero = patient['time_zero']
                window_start = patient['window_start']
                window_end = patient['window_end']

                patient_intervention = intervention_procs[
                    (intervention_procs['EMPI'] == empi) &
                    (intervention_procs['Date'] >= window_start) &
                    (intervention_procs['Date'] <= window_end)
                ].copy()

                if len(patient_intervention) > 0:
                    pe_df.at[idx, f'{intervention_name}_flag'] = 1
                    first_intervention = patient_intervention['Date'].min()
                    time_to_intervention = (first_intervention - time_zero).total_seconds() / 3600
                    pe_df.at[idx, f'time_to_{intervention_name}_hours'] = time_to_intervention

    # Print summary
    for intervention_name in interventions.keys():
        count = pe_df[f'{intervention_name}_flag'].sum()
        if count > 0:
            print(f"    {intervention_name}: {count} patients")

    return pe_df


# ============================================================================
# OUTCOME EXTRACTION: VASOPRESSORS & INOTROPES
# ============================================================================

def extract_vasopressors_inotropes(pe_df: pd.DataFrame, prc_df: pd.DataFrame, med_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract vasopressor and inotrope use from both procedures and medications.
    """
    print("\nExtracting vasopressors/inotropes...")

    # Initialize columns
    pe_df['vasopressor_flag'] = 0
    pe_df['vasopressor_list'] = ''
    pe_df['time_to_vasopressor_hours'] = np.nan
    pe_df['inotrope_flag'] = 0
    pe_df['inotrope_list'] = ''
    pe_df['time_to_inotrope_hours'] = np.nan

    # Source 1: Procedure codes (ICD-10-PCS vasopressor administration)
    vasopressor_procs = prc_df[prc_df['Code'].isin(VASOPRESSOR_PCS_CODES)].copy()
    print(f"  Found {len(vasopressor_procs)} vasopressor procedure codes")

    # Source 2: Medications
    # Create case-insensitive patterns
    vasopressor_pattern = '|'.join(VASOPRESSOR_NAMES)
    inotrope_pattern = '|'.join(INOTROPE_NAMES)

    vasopressor_meds = med_df[
        med_df['Medication'].str.contains(vasopressor_pattern, case=False, na=False)
    ].copy()

    inotrope_meds = med_df[
        med_df['Medication'].str.contains(inotrope_pattern, case=False, na=False)
    ].copy()

    print(f"  Found {len(vasopressor_meds)} vasopressor medication records")
    print(f"  Found {len(inotrope_meds)} inotrope medication records")

    # Process each patient
    for idx, patient in pe_df.iterrows():
        empi = patient['EMPI']
        time_zero = patient['time_zero']
        window_start = patient['window_start']
        window_end = patient['window_end']

        # Check vasopressor procedures
        patient_vaso_proc = vasopressor_procs[
            (vasopressor_procs['EMPI'] == empi) &
            (vasopressor_procs['Date'] >= window_start) &
            (vasopressor_procs['Date'] <= window_end)
        ]

        # Check vasopressor medications
        patient_vaso_med = vasopressor_meds[
            (vasopressor_meds['EMPI'] == empi) &
            (vasopressor_meds['Medication_Date'] >= window_start) &
            (vasopressor_meds['Medication_Date'] <= window_end)
        ]

        # Check inotrope medications
        patient_inotrope = inotrope_meds[
            (inotrope_meds['EMPI'] == empi) &
            (inotrope_meds['Medication_Date'] >= window_start) &
            (inotrope_meds['Medication_Date'] <= window_end)
        ]

        # Vasopressors
        if len(patient_vaso_proc) > 0 or len(patient_vaso_med) > 0:
            pe_df.at[idx, 'vasopressor_flag'] = 1

            # Get earliest date from either source
            dates = []
            if len(patient_vaso_proc) > 0:
                dates.append(patient_vaso_proc['Date'].min())
            if len(patient_vaso_med) > 0:
                dates.append(patient_vaso_med['Medication_Date'].min())

            if dates:
                first_vaso = min(dates)
                time_to_vaso = (first_vaso - time_zero).total_seconds() / 3600
                pe_df.at[idx, 'time_to_vasopressor_hours'] = time_to_vaso

            # Extract specific vasopressor names
            if len(patient_vaso_med) > 0:
                vaso_names = []
                for vname in VASOPRESSOR_NAMES:
                    if patient_vaso_med['Medication'].str.contains(vname, case=False).any():
                        vaso_names.append(vname)
                pe_df.at[idx, 'vasopressor_list'] = ';'.join(vaso_names)

        # Inotropes
        if len(patient_inotrope) > 0:
            pe_df.at[idx, 'inotrope_flag'] = 1

            first_inotrope = patient_inotrope['Medication_Date'].min()
            time_to_inotrope = (first_inotrope - time_zero).total_seconds() / 3600
            pe_df.at[idx, 'time_to_inotrope_hours'] = time_to_inotrope

            # Extract specific inotrope names
            inotrope_names = []
            for iname in INOTROPE_NAMES:
                if patient_inotrope['Medication'].str.contains(iname, case=False).any():
                    inotrope_names.append(iname)
            pe_df.at[idx, 'inotrope_list'] = ';'.join(inotrope_names)

    vaso_count = pe_df['vasopressor_flag'].sum()
    inotrope_count = pe_df['inotrope_flag'].sum()
    print(f"  Vasopressors: {vaso_count}/{len(pe_df)} ({100*vaso_count/len(pe_df):.1f}%)")
    print(f"  Inotropes: {inotrope_count}/{len(pe_df)} ({100*inotrope_count/len(pe_df):.1f}%)")

    return pe_df


# ============================================================================
# OUTCOME EXTRACTION: BLEEDING
# ============================================================================

def extract_bleeding(pe_df: pd.DataFrame, dia_df: pd.DataFrame) -> pd.DataFrame:
    """Extract bleeding outcomes from ICD codes."""
    print("\nExtracting bleeding events...")

    # Initialize columns
    pe_df['major_bleeding_flag'] = 0
    pe_df['bleeding_type'] = ''
    pe_df['ich_flag'] = 0
    pe_df['gi_bleed_flag'] = 0
    pe_df['acute_blood_loss_flag'] = 0
    pe_df['clinically_significant_bleeding'] = 0
    pe_df['time_to_bleeding_hours'] = np.nan
    pe_df['hemoptysis_flag'] = 0

    # Flatten ICD-10 bleeding codes
    ich_codes = (ICD10_BLEEDING['ich_subarachnoid'] +
                 ICD10_BLEEDING['ich_intracerebral'] +
                 ICD10_BLEEDING['ich_other'])
    gi_codes = ICD10_BLEEDING['gi_bleed']
    blood_loss_codes = ICD10_BLEEDING['acute_blood_loss']
    hematuria_codes = ICD10_BLEEDING['hematuria']
    hemoptysis_codes = ICD10_BLEEDING['hemoptysis']

    # Add ICD-9 codes
    ich_codes += ICD9_BLEEDING['ich']
    gi_codes += ICD9_BLEEDING['gi_bleed']
    blood_loss_codes += ICD9_BLEEDING['acute_blood_loss']
    hematuria_codes += ICD9_BLEEDING['hematuria']
    hemoptysis_codes += ICD9_BLEEDING['hemoptysis']

    # Filter to bleeding diagnoses
    all_bleeding_codes = ich_codes + gi_codes + blood_loss_codes + hematuria_codes + hemoptysis_codes

    # Create patterns for partial matching (since ICD codes may have more digits)
    def matches_code_pattern(code_series, code_list):
        """Check if diagnosis codes start with any code in the list."""
        pattern = '|'.join([f'^{code}' for code in code_list])
        return code_series.str.contains(pattern, case=False, na=False, regex=True)

    bleeding_dx = dia_df[matches_code_pattern(dia_df['Code'].astype(str), all_bleeding_codes)].copy()

    print(f"  Found {len(bleeding_dx)} bleeding diagnosis records")

    # Process each patient
    for idx, patient in pe_df.iterrows():
        empi = patient['EMPI']
        time_zero = patient['time_zero']
        window_start = patient['window_start']
        window_end = patient['window_end']

        patient_bleeding = bleeding_dx[
            (bleeding_dx['EMPI'] == empi) &
            (bleeding_dx['Date'] >= window_start) &
            (bleeding_dx['Date'] <= window_end)
        ].copy()

        if len(patient_bleeding) > 0:
            # Classify bleeding type
            has_ich = matches_code_pattern(patient_bleeding['Code'].astype(str), ich_codes).any()
            has_gi = matches_code_pattern(patient_bleeding['Code'].astype(str), gi_codes).any()
            has_blood_loss = matches_code_pattern(patient_bleeding['Code'].astype(str), blood_loss_codes).any()
            has_hematuria = matches_code_pattern(patient_bleeding['Code'].astype(str), hematuria_codes).any()
            has_hemoptysis = matches_code_pattern(patient_bleeding['Code'].astype(str), hemoptysis_codes).any()

            # Tier 1: Major bleeding
            if has_ich or has_gi or has_blood_loss:
                pe_df.at[idx, 'major_bleeding_flag'] = 1

                bleeding_types = []
                if has_ich:
                    pe_df.at[idx, 'ich_flag'] = 1
                    bleeding_types.append('ICH')
                if has_gi:
                    pe_df.at[idx, 'gi_bleed_flag'] = 1
                    bleeding_types.append('GI')
                if has_blood_loss:
                    pe_df.at[idx, 'acute_blood_loss_flag'] = 1
                    bleeding_types.append('Acute_Blood_Loss')

                pe_df.at[idx, 'bleeding_type'] = ';'.join(bleeding_types)

                # Time to bleeding
                first_bleeding = patient_bleeding['Date'].min()
                time_to_bleeding = (first_bleeding - time_zero).total_seconds() / 3600
                pe_df.at[idx, 'time_to_bleeding_hours'] = time_to_bleeding

            # Tier 2: Clinically significant
            elif has_hematuria:
                pe_df.at[idx, 'clinically_significant_bleeding'] = 1
                pe_df.at[idx, 'bleeding_type'] = 'Hematuria'

            # Hemoptysis: only if >24h from PE (otherwise likely PE symptom)
            if has_hemoptysis:
                hemoptysis_dates = patient_bleeding[matches_code_pattern(
                    patient_bleeding['Code'].astype(str), hemoptysis_codes
                )]['Date']

                for hdate in hemoptysis_dates:
                    hours_from_pe = (hdate - time_zero).total_seconds() / 3600
                    if hours_from_pe > 24:
                        pe_df.at[idx, 'hemoptysis_flag'] = 1
                        pe_df.at[idx, 'clinically_significant_bleeding'] = 1
                        break

    major_bleeding_count = pe_df['major_bleeding_flag'].sum()
    ich_count = pe_df['ich_flag'].sum()
    gi_count = pe_df['gi_bleed_flag'].sum()

    print(f"  Major bleeding: {major_bleeding_count}/{len(pe_df)} ({100*major_bleeding_count/len(pe_df):.1f}%)")
    print(f"    ICH: {ich_count}")
    print(f"    GI bleed: {gi_count}")

    return pe_df


# ============================================================================
# OUTCOME EXTRACTION: READMISSIONS & SHOCK
# ============================================================================

def extract_readmissions_shock(pe_df: pd.DataFrame, enc_df: pd.DataFrame, dia_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract readmission, healthcare utilization, and shock outcomes.

    Fixed to properly distinguish inpatient readmissions from outpatient visits.
    """
    print("\nExtracting readmissions, healthcare utilization, and shock...")

    # Initialize readmission columns (inpatient-only)
    pe_df['readmission_30day'] = 0  # Legacy column for compatibility
    pe_df['readmission_30d_inpatient'] = 0
    pe_df['readmission_30d_count'] = 0
    pe_df['days_to_first_readmission'] = np.nan

    # Initialize healthcare utilization columns
    pe_df['ed_visits_30d'] = 0
    pe_df['days_to_first_ed_visit'] = np.nan
    pe_df['cardiology_visits_30d'] = 0
    pe_df['pulmonary_visits_30d'] = 0
    pe_df['total_outpatient_visits_30d'] = 0

    # Initialize shock columns
    pe_df['shock_flag'] = 0
    pe_df['time_to_shock_hours'] = np.nan

    # Process each patient
    for idx, patient in pe_df.iterrows():
        empi = patient['EMPI']
        discharge_date = patient['window_end']

        if pd.notna(discharge_date):
            readmit_window_end = discharge_date + pd.Timedelta(days=30)

            # Get all encounters within 30 days post-discharge
            all_encounters = enc_df[
                (enc_df['EMPI'] == empi) &
                (enc_df['Admit_Date_Time'] > discharge_date) &
                (enc_df['Admit_Date_Time'] <= readmit_window_end)
            ].copy()

            if len(all_encounters) > 0:
                # 1. INPATIENT READMISSIONS
                inpatient_readmits = all_encounters[
                    all_encounters['Inpatient_Outpatient'] == 'Inpatient'
                ]

                if len(inpatient_readmits) > 0:
                    pe_df.at[idx, 'readmission_30d_inpatient'] = 1
                    pe_df.at[idx, 'readmission_30day'] = 1  # Legacy compatibility
                    pe_df.at[idx, 'readmission_30d_count'] = len(inpatient_readmits)

                    first_readmit = inpatient_readmits['Admit_Date_Time'].min()
                    days_to_readmit = (first_readmit - discharge_date).total_seconds() / (3600 * 24)
                    pe_df.at[idx, 'days_to_first_readmission'] = days_to_readmit

                # 2. EMERGENCY DEPARTMENT VISITS (not resulting in admission)
                ed_visits = all_encounters[
                    all_encounters['Inpatient_Outpatient'].str.contains('Emergency', case=False, na=False)
                ]

                if len(ed_visits) > 0:
                    pe_df.at[idx, 'ed_visits_30d'] = len(ed_visits)
                    first_ed = ed_visits['Admit_Date_Time'].min()
                    days_to_ed = (first_ed - discharge_date).total_seconds() / (3600 * 24)
                    pe_df.at[idx, 'days_to_first_ed_visit'] = days_to_ed

                # 3. SPECIALTY OUTPATIENT VISITS
                outpatient_visits = all_encounters[
                    all_encounters['Inpatient_Outpatient'].str.contains('Outpatient', case=False, na=False)
                ]

                if len(outpatient_visits) > 0:
                    pe_df.at[idx, 'total_outpatient_visits_30d'] = len(outpatient_visits)

                    # Cardiology visits
                    if 'Clinic_Name' in outpatient_visits.columns:
                        cardio_visits = outpatient_visits[
                            outpatient_visits['Clinic_Name'].str.contains(
                                'CARDIO|CARD |HEART', case=False, na=False, regex=True
                            )
                        ]
                        pe_df.at[idx, 'cardiology_visits_30d'] = len(cardio_visits)

                    # Pulmonary visits
                    if 'Clinic_Name' in outpatient_visits.columns:
                        pulm_visits = outpatient_visits[
                            outpatient_visits['Clinic_Name'].str.contains(
                                'PULM|LUNG|RESPIR', case=False, na=False, regex=True
                            )
                        ]
                        pe_df.at[idx, 'pulmonary_visits_30d'] = len(pulm_visits)

    # SHOCK EXTRACTION (unchanged)
    shock_codes = ICD10_SHOCK + ICD9_SHOCK
    shock_pattern = '|'.join([f'^{code}' for code in shock_codes])
    shock_dx = dia_df[dia_df['Code'].astype(str).str.contains(shock_pattern, case=False, na=False, regex=True)].copy()

    print(f"  Found {len(shock_dx)} shock diagnosis records")

    for idx, patient in pe_df.iterrows():
        empi = patient['EMPI']
        time_zero = patient['time_zero']
        window_start = patient['window_start']
        window_end = patient['window_end']

        patient_shock = shock_dx[
            (shock_dx['EMPI'] == empi) &
            (shock_dx['Date'] >= window_start) &
            (shock_dx['Date'] <= window_end)
        ]

        if len(patient_shock) > 0:
            pe_df.at[idx, 'shock_flag'] = 1
            first_shock = patient_shock['Date'].min()
            time_to_shock = (first_shock - time_zero).total_seconds() / 3600
            pe_df.at[idx, 'time_to_shock_hours'] = time_to_shock

    # Summary statistics
    inpatient_readmit_count = pe_df['readmission_30d_inpatient'].sum()
    ed_visit_count = pe_df['ed_visits_30d'].sum()
    outpatient_count = pe_df['total_outpatient_visits_30d'].sum()
    shock_count = pe_df['shock_flag'].sum()

    print(f"  30-day inpatient readmissions: {inpatient_readmit_count}/{len(pe_df)} ({100*inpatient_readmit_count/len(pe_df):.1f}%)")
    print(f"  30-day ED visits: {ed_visit_count} visits across {(pe_df['ed_visits_30d'] > 0).sum()} patients")
    print(f"  30-day outpatient visits: {outpatient_count} visits across {(pe_df['total_outpatient_visits_30d'] > 0).sum()} patients")
    print(f"  Shock: {shock_count}/{len(pe_df)} ({100*shock_count/len(pe_df):.1f}%)")

    return pe_df


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main(test_mode=False, test_n_patients=100):
    """
    Main execution function.

    Args:
        test_mode: If True, process only a subset of patients for testing
        test_n_patients: Number of patients to process in test mode
    """
    print("="*80)
    print("MODULE 1: CORE INFRASTRUCTURE")
    if test_mode:
        print(f"*** TEST MODE: Processing first {test_n_patients} patients ***")
    print("="*80)

    # 1. Load data
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    pe_df = load_pe_cohort()
    enc_df = load_encounters()
    prc_df = load_procedures()
    dia_df = load_diagnoses()
    med_df = load_medications()
    dem_df = load_demographics()

    # 2. Establish Time Zero
    print("\n" + "="*80)
    print("STEP 2: ESTABLISHING TIME ZERO & TEMPORAL WINDOWS")
    print("="*80)

    pe_df = establish_time_zero(pe_df)

    # Test mode: process only subset
    if test_mode:
        print(f"\n*** Limiting to first {test_n_patients} patients for testing ***")
        pe_df = pe_df.head(test_n_patients).copy()

        # Filter other dataframes to only include these patients
        test_empis = set(pe_df['EMPI'].unique())
        enc_df = enc_df[enc_df['EMPI'].isin(test_empis)].copy()
        prc_df = prc_df[prc_df['EMPI'].isin(test_empis)].copy()
        dia_df = dia_df[dia_df['EMPI'].isin(test_empis)].copy()
        med_df = med_df[med_df['EMPI'].isin(test_empis)].copy()
        dem_df = dem_df[dem_df['EMPI'].isin(test_empis)].copy()

        print(f"  Filtered to {len(enc_df)} encounters")
        print(f"  Filtered to {len(prc_df)} procedures")
        print(f"  Filtered to {len(dia_df)} diagnoses")
        print(f"  Filtered to {len(med_df)} medications")
        print(f"  Filtered to {len(dem_df)} demographics")

    pe_df = link_encounters_to_patients(pe_df, enc_df)
    pe_df = create_temporal_windows(pe_df)

    # 3. Extract outcomes
    print("\n" + "="*80)
    print("STEP 3: EXTRACTING OUTCOMES")
    print("="*80)

    pe_df = extract_mortality(pe_df, dem_df)
    pe_df = extract_icu_admission(pe_df, prc_df)
    pe_df = extract_ventilation(pe_df, prc_df)
    pe_df = extract_dialysis(pe_df, prc_df)
    pe_df = extract_advanced_interventions(pe_df, prc_df)
    pe_df = extract_vasopressors_inotropes(pe_df, prc_df, med_df)
    pe_df = extract_bleeding(pe_df, dia_df)
    pe_df = extract_readmissions_shock(pe_df, enc_df, dia_df)

    # 4. Save outcomes CSV
    print("\n" + "="*80)
    print("STEP 4: SAVING OUTPUTS")
    print("="*80)

    output_filename = "outcomes_test.csv" if test_mode else "outcomes.csv"
    outcomes_file = OUTPUT_DIR / output_filename
    pe_df.to_csv(outcomes_file, index=False)
    print(f"  Saved outcomes to: {outcomes_file}")
    print(f"  Total patients: {len(pe_df)}")
    print(f"  Total columns: {len(pe_df.columns)}")

    print("\n" + "="*80)
    print("MODULE 1 COMPLETE!")
    print("="*80)

    return pe_df


if __name__ == "__main__":
    import sys

    # Check for test mode flag
    test_mode = '--test' in sys.argv or '-t' in sys.argv

    # Get number of test patients if specified
    test_n = 100
    for arg in sys.argv:
        if arg.startswith('--n='):
            test_n = int(arg.split('=')[1])

    main(test_mode=test_mode, test_n_patients=test_n)
