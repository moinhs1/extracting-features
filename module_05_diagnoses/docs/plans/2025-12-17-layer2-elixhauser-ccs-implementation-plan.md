# Layer 2 Elixhauser + CCS Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Extend Layer 2 with Elixhauser Comorbidity Index (van Walraven weights) and CCS categories.

**Architecture:** `elixhauser_calculator.py` mirrors existing Charlson pattern. `ccs_mapper.py` handles ICD→CCS mapping. Updated `layer2_builder.py` produces extended `comorbidity_scores.parquet` and new `ccs_categories.parquet`.

**Tech Stack:** Python, pandas, pytest

**Prerequisites:**
- CCS crosswalk files downloaded to `data/vocabularies/ccs/`

---

## Task 0: Download CCS Crosswalk Files (Manual)

**This task requires manual user action before code implementation begins.**

### Step 0.1: Create directory structure

```bash
mkdir -p data/vocabularies/ccs
```

### Step 0.2: Download CCS for ICD-9

1. Go to https://hcup-us.ahrq.gov/toolssoftware/ccs/ccs.jsp
2. Download "Single Level CCS" tool
3. Extract the diagnosis crosswalk file
4. Save as `data/vocabularies/ccs/ccs_icd9_crosswalk.csv`

### Step 0.3: Download CCSR for ICD-10 (maps to CCS)

1. Go to https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp
2. Download the CCSR reference file (CSV)
3. We'll map CCSR categories back to single-level CCS
4. Save as `data/vocabularies/ccs/ccs_icd10_crosswalk.csv`

**Alternative:** Create simplified crosswalk manually for the most common codes (faster for MVP).

---

## Task 1: Elixhauser Code Definitions

**Files:**
- Create: `module_05_diagnoses/config/elixhauser_codes.py`

**Step 1: Create Elixhauser code definitions**

```python
"""ICD code definitions for Elixhauser Comorbidity Index with van Walraven weights."""

# 31 Elixhauser components with ICD-9 and ICD-10 codes
# Weights from van Walraven et al. 2009
ELIXHAUSER_COMPONENTS = {
    "congestive_heart_failure": {
        "weight": 7,
        "icd10": ["I09.9", "I11.0", "I13.0", "I13.2", "I25.5", "I42.0", "I42.5",
                  "I42.6", "I42.7", "I42.8", "I42.9", "I43", "I50", "P29.0"],
        "icd9": ["398.91", "402.01", "402.11", "402.91", "404.01", "404.03",
                 "404.11", "404.13", "404.91", "404.93", "425.4", "425.5",
                 "425.6", "425.7", "425.8", "425.9", "428"],
    },
    "cardiac_arrhythmias": {
        "weight": 5,
        "icd10": ["I44.1", "I44.2", "I44.3", "I45.6", "I45.9", "I47", "I48", "I49",
                  "R00.0", "R00.1", "R00.8", "T82.1", "Z45.0", "Z95.0"],
        "icd9": ["426.0", "426.13", "426.7", "426.9", "426.10", "426.12", "427.0",
                 "427.1", "427.2", "427.31", "427.60", "427.9", "785.0",
                 "996.01", "996.04", "V45.0", "V53.3"],
    },
    "valvular_disease": {
        "weight": -1,
        "icd10": ["A52.0", "I05", "I06", "I07", "I08", "I09.1", "I09.8", "I34",
                  "I35", "I36", "I37", "I38", "I39", "Q23.0", "Q23.1", "Q23.2",
                  "Q23.3", "Z95.2", "Z95.3", "Z95.4"],
        "icd9": ["093.2", "394", "395", "396", "397", "424", "746.3", "746.4",
                 "746.5", "746.6", "V42.2", "V43.3"],
    },
    "pulmonary_circulation_disorders": {
        "weight": 4,
        "icd10": ["I26", "I27", "I28.0", "I28.8", "I28.9"],
        "icd9": ["415.0", "415.1", "416", "417.0", "417.8", "417.9"],
    },
    "peripheral_vascular_disorders": {
        "weight": 2,
        "icd10": ["I70", "I71", "I73.1", "I73.8", "I73.9", "I77.1", "I79.0",
                  "I79.2", "K55.1", "K55.8", "K55.9", "Z95.8", "Z95.9"],
        "icd9": ["093.0", "437.3", "440", "441", "443.1", "443.2", "443.8",
                 "443.9", "447.1", "557.1", "557.9", "V43.4"],
    },
    "hypertension_uncomplicated": {
        "weight": 0,
        "icd10": ["I10"],
        "icd9": ["401.1", "401.9"],
    },
    "hypertension_complicated": {
        "weight": 0,
        "icd10": ["I11", "I12", "I13", "I15"],
        "icd9": ["401.0", "402", "403", "404", "405"],
    },
    "paralysis": {
        "weight": 7,
        "icd10": ["G04.1", "G11.4", "G80.1", "G80.2", "G81", "G82", "G83.0",
                  "G83.1", "G83.2", "G83.3", "G83.4", "G83.9"],
        "icd9": ["334.1", "342", "343", "344.0", "344.1", "344.2", "344.3",
                 "344.4", "344.5", "344.6", "344.9"],
    },
    "other_neurological_disorders": {
        "weight": 6,
        "icd10": ["G10", "G11", "G12", "G13", "G20", "G21", "G22", "G25.4",
                  "G25.5", "G31.2", "G31.8", "G31.9", "G32", "G35", "G36",
                  "G37", "G40", "G41", "G93.1", "G93.4", "R47.0", "R56"],
        "icd9": ["331.9", "332.0", "332.1", "333.4", "333.5", "333.92", "334",
                 "335", "336.2", "340", "341", "345", "348.1", "348.3", "780.3",
                 "784.3"],
    },
    "chronic_pulmonary_disease": {
        "weight": 3,
        "icd10": ["I27.8", "I27.9", "J40", "J41", "J42", "J43", "J44", "J45",
                  "J46", "J47", "J60", "J61", "J62", "J63", "J64", "J65", "J66",
                  "J67", "J68.4", "J70.1", "J70.3"],
        "icd9": ["416.8", "416.9", "490", "491", "492", "493", "494", "495",
                 "496", "500", "501", "502", "503", "504", "505", "506.4",
                 "508.1", "508.8"],
    },
    "diabetes_uncomplicated": {
        "weight": 0,
        "icd10": ["E10.0", "E10.1", "E10.9", "E11.0", "E11.1", "E11.9", "E12.0",
                  "E12.1", "E12.9", "E13.0", "E13.1", "E13.9", "E14.0", "E14.1",
                  "E14.9"],
        "icd9": ["250.0", "250.1", "250.2", "250.3"],
    },
    "diabetes_complicated": {
        "weight": 0,
        "icd10": ["E10.2", "E10.3", "E10.4", "E10.5", "E10.6", "E10.7", "E10.8",
                  "E11.2", "E11.3", "E11.4", "E11.5", "E11.6", "E11.7", "E11.8",
                  "E12.2", "E12.3", "E12.4", "E12.5", "E12.6", "E12.7", "E12.8",
                  "E13.2", "E13.3", "E13.4", "E13.5", "E13.6", "E13.7", "E13.8",
                  "E14.2", "E14.3", "E14.4", "E14.5", "E14.6", "E14.7", "E14.8"],
        "icd9": ["250.4", "250.5", "250.6", "250.7", "250.8", "250.9"],
    },
    "hypothyroidism": {
        "weight": 0,
        "icd10": ["E00", "E01", "E02", "E03", "E89.0"],
        "icd9": ["240.9", "243", "244", "246.1", "246.8"],
    },
    "renal_failure": {
        "weight": 5,
        "icd10": ["I12.0", "I13.1", "N18", "N19", "N25.0", "Z49.0", "Z49.1",
                  "Z49.2", "Z94.0", "Z99.2"],
        "icd9": ["403.01", "403.11", "403.91", "404.02", "404.03", "404.12",
                 "404.13", "404.92", "404.93", "585", "586", "588.0", "V42.0",
                 "V45.1", "V56"],
    },
    "liver_disease": {
        "weight": 11,
        "icd10": ["B18", "I85", "I86.4", "I98.2", "K70", "K71.1", "K71.3",
                  "K71.4", "K71.5", "K71.7", "K72", "K73", "K74", "K76.0",
                  "K76.2", "K76.3", "K76.4", "K76.5", "K76.6", "K76.7", "K76.8",
                  "K76.9", "Z94.4"],
        "icd9": ["070.22", "070.23", "070.32", "070.33", "070.44", "070.54",
                 "070.6", "070.9", "456.0", "456.1", "456.2", "571", "572.2",
                 "572.3", "572.4", "572.8", "V42.7"],
    },
    "peptic_ulcer_disease": {
        "weight": 0,
        "icd10": ["K25.7", "K25.9", "K26.7", "K26.9", "K27.7", "K27.9", "K28.7",
                  "K28.9"],
        "icd9": ["531.7", "531.9", "532.7", "532.9", "533.7", "533.9", "534.7",
                 "534.9"],
    },
    "aids_hiv": {
        "weight": 0,
        "icd10": ["B20", "B21", "B22", "B24"],
        "icd9": ["042", "043", "044"],
    },
    "lymphoma": {
        "weight": 9,
        "icd10": ["C81", "C82", "C83", "C84", "C85", "C88", "C96", "C90.0",
                  "C90.2"],
        "icd9": ["200", "201", "202", "203.0", "238.6"],
    },
    "metastatic_cancer": {
        "weight": 12,
        "icd10": ["C77", "C78", "C79", "C80"],
        "icd9": ["196", "197", "198", "199.0", "199.1"],
    },
    "solid_tumor_without_metastasis": {
        "weight": 4,
        "icd10": ["C00", "C01", "C02", "C03", "C04", "C05", "C06", "C07", "C08",
                  "C09", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17",
                  "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25", "C26",
                  "C30", "C31", "C32", "C33", "C34", "C37", "C38", "C39", "C40",
                  "C41", "C43", "C45", "C46", "C47", "C48", "C49", "C50", "C51",
                  "C52", "C53", "C54", "C55", "C56", "C57", "C58", "C60", "C61",
                  "C62", "C63", "C64", "C65", "C66", "C67", "C68", "C69", "C70",
                  "C71", "C72", "C73", "C74", "C75", "C76", "C97"],
        "icd9": ["140", "141", "142", "143", "144", "145", "146", "147", "148",
                 "149", "150", "151", "152", "153", "154", "155", "156", "157",
                 "158", "159", "160", "161", "162", "163", "164", "165", "170",
                 "171", "172", "174", "175", "176", "179", "180", "181", "182",
                 "183", "184", "185", "186", "187", "188", "189", "190", "191",
                 "192", "193", "194", "195"],
    },
    "rheumatoid_arthritis": {
        "weight": 0,
        "icd10": ["L94.0", "L94.1", "L94.3", "M05", "M06", "M31.5", "M32", "M33",
                  "M34", "M35.1", "M35.3", "M36.0"],
        "icd9": ["446.5", "710.0", "710.1", "710.2", "710.3", "710.4", "714.0",
                 "714.1", "714.2", "714.8", "725"],
    },
    "coagulopathy": {
        "weight": 3,
        "icd10": ["D65", "D66", "D67", "D68", "D69.1", "D69.3", "D69.4", "D69.5",
                  "D69.6"],
        "icd9": ["286", "287.1", "287.3", "287.4", "287.5"],
    },
    "obesity": {
        "weight": -4,
        "icd10": ["E66"],
        "icd9": ["278.0"],
    },
    "weight_loss": {
        "weight": 6,
        "icd10": ["E40", "E41", "E42", "E43", "E44", "E45", "E46", "R63.4", "R64"],
        "icd9": ["260", "261", "262", "263", "783.2", "799.4"],
    },
    "fluid_electrolyte_disorders": {
        "weight": 5,
        "icd10": ["E22.2", "E86", "E87"],
        "icd9": ["253.6", "276"],
    },
    "blood_loss_anemia": {
        "weight": -2,
        "icd10": ["D50.0"],
        "icd9": ["280.0"],
    },
    "deficiency_anemia": {
        "weight": -2,
        "icd10": ["D50.8", "D50.9", "D51", "D52", "D53"],
        "icd9": ["280.1", "280.8", "280.9", "281"],
    },
    "alcohol_abuse": {
        "weight": 0,
        "icd10": ["F10", "E52", "G62.1", "I42.6", "K29.2", "K70.0", "K70.3",
                  "K70.9", "T51", "Z50.2", "Z71.4", "Z72.1"],
        "icd9": ["265.2", "291.1", "291.2", "291.3", "291.5", "291.8", "291.9",
                 "303.0", "303.9", "305.0", "357.5", "425.5", "535.3", "571.0",
                 "571.1", "571.2", "571.3", "980", "V11.3"],
    },
    "drug_abuse": {
        "weight": -7,
        "icd10": ["F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19", "Z71.5",
                  "Z72.2"],
        "icd9": ["292", "304", "305.2", "305.3", "305.4", "305.5", "305.6",
                 "305.7", "305.8", "305.9", "V65.42"],
    },
    "psychoses": {
        "weight": 0,
        "icd10": ["F20", "F22", "F23", "F24", "F25", "F28", "F29", "F30.2",
                  "F31.2", "F31.5"],
        "icd9": ["293.8", "295", "296.04", "296.14", "296.44", "296.54", "297",
                 "298"],
    },
    "depression": {
        "weight": -3,
        "icd10": ["F20.4", "F31.3", "F31.4", "F31.5", "F32", "F33", "F34.1",
                  "F41.2", "F43.2"],
        "icd9": ["296.2", "296.3", "296.5", "300.4", "309", "311"],
    },
}

# Hierarchy rules: if both present, only count the more severe
ELIXHAUSER_HIERARCHY = {
    "diabetes_uncomplicated": "diabetes_complicated",
    "hypertension_uncomplicated": "hypertension_complicated",
    "solid_tumor_without_metastasis": "metastatic_cancer",
}
```

**Step 2: Verify syntax**

Run: `cd module_05_diagnoses && python -c "from config.elixhauser_codes import ELIXHAUSER_COMPONENTS, ELIXHAUSER_HIERARCHY; print(len(ELIXHAUSER_COMPONENTS), 'components')"`
Expected: `31 components`

**Step 3: Commit**

```bash
git add module_05_diagnoses/config/elixhauser_codes.py
git commit -m "feat(module05): add Elixhauser ICD codes with van Walraven weights"
```

---

## Task 2: Elixhauser Calculator - Tests

**Files:**
- Create: `module_05_diagnoses/tests/test_elixhauser_calculator.py`

**Step 1: Write tests for Elixhauser calculator**

```python
"""Tests for Elixhauser Comorbidity Index calculator."""

import pytest
import pandas as pd
from processing.elixhauser_calculator import (
    code_matches_component,
    calculate_elixhauser_for_patient,
    calculate_elixhauser_batch
)


class TestCodeMatching:
    def test_exact_match_icd10(self):
        assert code_matches_component("I50.9", "congestive_heart_failure", "10") == True

    def test_prefix_match_icd10(self):
        assert code_matches_component("I50.23", "congestive_heart_failure", "10") == True

    def test_no_match(self):
        assert code_matches_component("J44.1", "congestive_heart_failure", "10") == False

    def test_icd9_match(self):
        assert code_matches_component("428.0", "congestive_heart_failure", "9") == True


class TestElixhauserCalculation:
    @pytest.fixture
    def sample_diagnoses(self):
        return pd.DataFrame({
            'icd_code': ['I50.9', 'I48.0', 'E66.0', 'J44.1'],
            'icd_version': ['10', '10', '10', '10'],
            'is_preexisting': [True, True, True, True],
        })

    def test_calculates_score(self, sample_diagnoses):
        result = calculate_elixhauser_for_patient(sample_diagnoses)
        # CHF (7) + Arrhythmia (5) + Obesity (-4) + COPD (3) = 11
        assert result['elixhauser_score'] == 11

    def test_counts_components(self, sample_diagnoses):
        result = calculate_elixhauser_for_patient(sample_diagnoses)
        assert result['elixhauser_component_count'] == 4

    def test_hierarchy_diabetes(self):
        diagnoses = pd.DataFrame({
            'icd_code': ['E11.9', 'E11.5'],  # Uncomplicated and complicated
            'icd_version': ['10', '10'],
            'is_preexisting': [True, True],
        })
        result = calculate_elixhauser_for_patient(diagnoses)
        # Should only count complicated (weight 0), not uncomplicated
        assert result['elixhauser_component_count'] == 1
        assert 'diabetes_complicated' in result['elixhauser_components']

    def test_hierarchy_cancer(self):
        diagnoses = pd.DataFrame({
            'icd_code': ['C34.9', 'C78.0'],  # Solid tumor and metastatic
            'icd_version': ['10', '10'],
            'is_preexisting': [True, True],
        })
        result = calculate_elixhauser_for_patient(diagnoses)
        # Should only count metastatic (12), not solid tumor (4)
        assert result['elixhauser_score'] == 12


class TestBatchCalculation:
    def test_batch_returns_all_patients(self):
        diagnoses = pd.DataFrame({
            'EMPI': ['P1', 'P1', 'P2'],
            'icd_code': ['I50.9', 'J44.1', 'E66.0'],
            'icd_version': ['10', '10', '10'],
            'is_preexisting': [True, True, True],
        })
        result = calculate_elixhauser_batch(diagnoses)
        assert len(result) == 2
        assert set(result['EMPI']) == {'P1', 'P2'}
```

**Step 2: Run tests to verify they fail**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_elixhauser_calculator.py -v`
Expected: FAIL with `ModuleNotFoundError`

---

## Task 3: Elixhauser Calculator - Implementation

**Files:**
- Create: `module_05_diagnoses/processing/elixhauser_calculator.py`

**Step 1: Implement Elixhauser calculator**

```python
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
```

**Step 2: Run tests to verify they pass**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_elixhauser_calculator.py -v`
Expected: All tests pass (8 passed)

**Step 3: Commit**

```bash
git add module_05_diagnoses/processing/elixhauser_calculator.py module_05_diagnoses/tests/test_elixhauser_calculator.py
git commit -m "feat(module05): add Elixhauser calculator with van Walraven weights"
```

---

## Task 4: CCS Mapper - Simplified Crosswalk

**Files:**
- Create: `data/vocabularies/ccs/ccs_crosswalk.csv`

**Step 1: Create simplified CCS crosswalk for common PE-related codes**

For MVP, create a simplified crosswalk with the most common codes. Full AHRQ crosswalk can be added later.

```csv
icd_code,icd_version,ccs_category,ccs_description
I26,10,103,Pulmonary heart disease
I26.0,10,103,Pulmonary heart disease
I26.9,10,103,Pulmonary heart disease
I50,10,108,Congestive heart failure
I48,10,106,Cardiac dysrhythmias
I21,10,100,Acute myocardial infarction
I25,10,101,Coronary atherosclerosis
J44,10,127,Chronic obstructive pulmonary disease and bronchiectasis
J45,10,128,Asthma
E11,10,50,Diabetes mellitus with complications
E10,10,49,Diabetes mellitus without complication
C34,10,19,Cancer of bronchus; lung
C50,10,24,Cancer of breast
N18,10,158,Chronic kidney disease
I10,10,98,Essential hypertension
I82,10,118,Phlebitis; thrombophlebitis and thromboembolism
K92,10,153,Gastrointestinal hemorrhage
D69,10,62,Coagulation and hemorrhagic disorders
I60,10,109,Acute cerebrovascular disease
I61,10,109,Acute cerebrovascular disease
I62,10,109,Acute cerebrovascular disease
N17,10,157,Acute and unspecified renal failure
410,9,100,Acute myocardial infarction
428,9,108,Congestive heart failure
415,9,103,Pulmonary heart disease
427,9,106,Cardiac dysrhythmias
491,9,127,Chronic obstructive pulmonary disease and bronchiectasis
492,9,127,Chronic obstructive pulmonary disease and bronchiectasis
493,9,128,Asthma
250,9,49,Diabetes mellitus without complication
162,9,19,Cancer of bronchus; lung
585,9,158,Chronic kidney disease
401,9,98,Essential hypertension
453,9,118,Phlebitis; thrombophlebitis and thromboembolism
578,9,153,Gastrointestinal hemorrhage
286,9,62,Coagulation and hemorrhagic disorders
430,9,109,Acute cerebrovascular disease
431,9,109,Acute cerebrovascular disease
584,9,157,Acute and unspecified renal failure
```

**Step 2: Save file**

```bash
mkdir -p data/vocabularies/ccs
# Save above content to data/vocabularies/ccs/ccs_crosswalk.csv
```

**Step 3: Commit**

```bash
git add data/vocabularies/ccs/ccs_crosswalk.csv
git commit -m "data: add simplified CCS crosswalk for common codes"
```

---

## Task 5: CCS Mapper - Tests

**Files:**
- Create: `module_05_diagnoses/tests/test_ccs_mapper.py`

**Step 1: Write tests for CCS mapper**

```python
"""Tests for CCS mapper."""

import pytest
import pandas as pd
from pathlib import Path
from processing.ccs_mapper import CCSMapper


@pytest.fixture
def ccs_mapper(tmp_path):
    """Create CCS mapper with test crosswalk."""
    crosswalk = pd.DataFrame({
        'icd_code': ['I26', 'I26.0', 'I50', 'J44', '428', '415'],
        'icd_version': ['10', '10', '10', '10', '9', '9'],
        'ccs_category': [103, 103, 108, 127, 108, 103],
        'ccs_description': ['Pulmonary heart disease', 'Pulmonary heart disease',
                           'CHF', 'COPD', 'CHF', 'Pulmonary heart disease'],
    })
    crosswalk_path = tmp_path / "ccs_crosswalk.csv"
    crosswalk.to_csv(crosswalk_path, index=False)
    return CCSMapper(crosswalk_path)


class TestCCSMapper:
    def test_exact_match(self, ccs_mapper):
        category = ccs_mapper.get_ccs_category("I50", "10")
        assert category == 108

    def test_prefix_match(self, ccs_mapper):
        # I26.99 should match I26
        category = ccs_mapper.get_ccs_category("I26.99", "10")
        assert category == 103

    def test_icd9_match(self, ccs_mapper):
        category = ccs_mapper.get_ccs_category("428.0", "9")
        assert category == 108

    def test_no_match_returns_none(self, ccs_mapper):
        category = ccs_mapper.get_ccs_category("INVALID", "10")
        assert category is None

    def test_get_description(self, ccs_mapper):
        desc = ccs_mapper.get_ccs_description(108)
        assert desc == "CHF"


class TestPatientCategorization:
    def test_categorize_patient(self, ccs_mapper):
        diagnoses = pd.DataFrame({
            'icd_code': ['I26.99', 'I50.9', 'J44.1'],
            'icd_version': ['10', '10', '10'],
            'is_preexisting': [True, True, True],
        })
        result = ccs_mapper.categorize_patient_diagnoses(diagnoses)

        # Should have 3 unique categories
        assert len(result) == 3
        assert 103 in result['ccs_category'].values  # PE
        assert 108 in result['ccs_category'].values  # CHF
        assert 127 in result['ccs_category'].values  # COPD
```

**Step 2: Run tests to verify they fail**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_ccs_mapper.py -v`
Expected: FAIL with `ModuleNotFoundError`

---

## Task 6: CCS Mapper - Implementation

**Files:**
- Create: `module_05_diagnoses/processing/ccs_mapper.py`

**Step 1: Implement CCS mapper**

```python
"""CCS (Clinical Classifications Software) mapper."""

import pandas as pd
from pathlib import Path
from typing import Optional, Dict


class CCSMapper:
    """Map ICD codes to CCS categories."""

    def __init__(self, crosswalk_path: Path):
        """
        Load CCS crosswalk file.

        Args:
            crosswalk_path: Path to CSV with icd_code, icd_version, ccs_category, ccs_description
        """
        self.crosswalk = pd.read_csv(crosswalk_path)
        self._build_lookups()

    def _build_lookups(self):
        """Build efficient lookup dictionaries."""
        self._icd10_to_ccs = {}
        self._icd9_to_ccs = {}
        self._ccs_descriptions = {}

        for _, row in self.crosswalk.iterrows():
            code = str(row['icd_code']).upper()
            version = str(row['icd_version'])
            category = int(row['ccs_category'])
            description = row['ccs_description']

            if version == '10':
                self._icd10_to_ccs[code] = category
            else:
                self._icd9_to_ccs[code] = category

            self._ccs_descriptions[category] = description

    def get_ccs_category(self, icd_code: str, version: str) -> Optional[int]:
        """
        Map ICD code to CCS category.

        Args:
            icd_code: ICD-9 or ICD-10 code
            version: '9' or '10'

        Returns:
            CCS category number or None if not found
        """
        lookup = self._icd10_to_ccs if version == '10' else self._icd9_to_ccs
        code = str(icd_code).upper()

        # Try exact match first
        if code in lookup:
            return lookup[code]

        # Try prefix matching (progressively shorter)
        for length in range(len(code) - 1, 2, -1):
            prefix = code[:length]
            if prefix in lookup:
                return lookup[prefix]

        return None

    def get_ccs_description(self, category: int) -> str:
        """Get description for CCS category."""
        return self._ccs_descriptions.get(category, f"CCS Category {category}")

    def categorize_patient_diagnoses(self, diagnoses: pd.DataFrame) -> pd.DataFrame:
        """
        Categorize all diagnoses for a patient into CCS categories.

        Args:
            diagnoses: DataFrame with icd_code, icd_version, is_preexisting

        Returns:
            DataFrame with ccs_category, ccs_description, diagnosis_count, is_preexisting
        """
        category_counts = {}
        category_preexisting = {}

        for _, row in diagnoses.iterrows():
            category = self.get_ccs_category(row['icd_code'], row['icd_version'])
            if category is None:
                continue

            if category not in category_counts:
                category_counts[category] = 0
                category_preexisting[category] = True

            category_counts[category] += 1
            # If any diagnosis in category is not preexisting, mark as not preexisting
            if not row.get('is_preexisting', True):
                category_preexisting[category] = False

        results = []
        for category, count in category_counts.items():
            results.append({
                'ccs_category': category,
                'ccs_description': self.get_ccs_description(category),
                'diagnosis_count': count,
                'is_preexisting': category_preexisting[category],
            })

        return pd.DataFrame(results)
```

**Step 2: Run tests to verify they pass**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_ccs_mapper.py -v`
Expected: All tests pass (6 passed)

**Step 3: Commit**

```bash
git add module_05_diagnoses/processing/ccs_mapper.py module_05_diagnoses/tests/test_ccs_mapper.py
git commit -m "feat(module05): add CCS mapper for ICD to CCS category mapping"
```

---

## Task 7: Update Layer 2 Builder

**Files:**
- Modify: `module_05_diagnoses/processing/layer2_builder.py`
- Modify: `module_05_diagnoses/tests/test_layer2_builder.py` (if exists)

**Step 1: Update layer2_builder.py**

```python
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
        ccs_df: CCS categories DataFrame
        output_path: Output directory path
    """
    output_path.mkdir(parents=True, exist_ok=True)
    scores_df.to_parquet(output_path / "comorbidity_scores.parquet", index=False)
    ccs_df.to_parquet(output_path / "ccs_categories.parquet", index=False)
```

**Step 2: Run existing tests**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/ -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add module_05_diagnoses/processing/layer2_builder.py
git commit -m "feat(module05): extend Layer 2 builder with Elixhauser and CCS"
```

---

## Task 8: Update build_layers.py Pipeline

**Files:**
- Modify: `module_05_diagnoses/build_layers.py`

**Step 1: Update pipeline to include CCS**

Add CCS path configuration and update Layer 2 build function to generate both outputs.

**Step 2: Test pipeline**

Run: `cd module_05_diagnoses && python build_layers.py --test --n=100`
Expected: Layer 2 now produces both files

**Step 3: Verify outputs**

Run: `ls -la module_05_diagnoses/outputs/layer2/`
Expected: Both `comorbidity_scores.parquet` and `ccs_categories.parquet`

**Step 4: Commit**

```bash
git add module_05_diagnoses/build_layers.py
git commit -m "feat(module05): integrate Elixhauser and CCS into pipeline"
```

---

## Task 9: Integration Tests

**Files:**
- Create: `module_05_diagnoses/tests/test_layer2_integration.py`

**Step 1: Write integration tests**

```python
"""Integration tests for Layer 2 with Elixhauser and CCS."""

import pytest
import pandas as pd
from processing.layer2_builder import build_layer2_comorbidity_scores, build_layer2_ccs_categories
from pathlib import Path


@pytest.fixture
def sample_layer1():
    """Sample Layer 1 data for testing."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P1', 'P2', 'P2'],
        'icd_code': ['I26.99', 'I50.9', 'J44.1', 'E66.0', 'C34.9'],
        'icd_version': ['10', '10', '10', '10', '10'],
        'is_preexisting': [True, True, True, True, True],
    })


class TestComorbidityScoresIntegration:
    def test_produces_both_indices(self, sample_layer1):
        result = build_layer2_comorbidity_scores(sample_layer1)

        assert 'cci_score' in result.columns
        assert 'elixhauser_score' in result.columns
        assert len(result) == 2  # 2 patients

    def test_scores_reasonable(self, sample_layer1):
        result = build_layer2_comorbidity_scores(sample_layer1)

        # P1: CHF, COPD in both indices
        p1 = result[result['EMPI'] == 'P1'].iloc[0]
        assert p1['cci_score'] >= 2  # CHF + COPD
        assert p1['elixhauser_score'] >= 10  # CHF(7) + COPD(3)


class TestCCSIntegration:
    def test_produces_categories(self, sample_layer1, tmp_path):
        # Create minimal crosswalk
        crosswalk = pd.DataFrame({
            'icd_code': ['I26', 'I50', 'J44', 'E66', 'C34'],
            'icd_version': ['10', '10', '10', '10', '10'],
            'ccs_category': [103, 108, 127, 58, 19],
            'ccs_description': ['PE', 'CHF', 'COPD', 'Obesity', 'Lung cancer'],
        })
        crosswalk_path = tmp_path / "ccs_crosswalk.csv"
        crosswalk.to_csv(crosswalk_path, index=False)

        result = build_layer2_ccs_categories(sample_layer1, crosswalk_path)

        assert len(result) > 0
        assert 'EMPI' in result.columns
        assert 'ccs_category' in result.columns
```

**Step 2: Run integration tests**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/test_layer2_integration.py -v`
Expected: All tests pass

**Step 3: Commit**

```bash
git add module_05_diagnoses/tests/test_layer2_integration.py
git commit -m "test(module05): add Layer 2 integration tests for Elixhauser and CCS"
```

---

## Task 10: Final Verification

**Step 1: Run all tests**

Run: `cd module_05_diagnoses && PYTHONPATH=. pytest tests/ -v`
Expected: All tests pass

**Step 2: Run full pipeline**

Run: `cd module_05_diagnoses && python build_layers.py --test --n=100`
Expected: All layers build successfully

**Step 3: Verify Layer 2 outputs**

Run:
```bash
cd module_05_diagnoses && python -c "
import pandas as pd
scores = pd.read_parquet('outputs/layer2/comorbidity_scores.parquet')
print('Comorbidity Scores:')
print(scores.columns.tolist())
print(f'Patients: {len(scores)}')
print(f'CCI mean: {scores.cci_score.mean():.1f}')
print(f'Elixhauser mean: {scores.elixhauser_score.mean():.1f}')
"
```

Run:
```bash
cd module_05_diagnoses && python -c "
import pandas as pd
ccs = pd.read_parquet('outputs/layer2/ccs_categories.parquet')
print('CCS Categories:')
print(ccs.columns.tolist())
print(f'Total rows: {len(ccs)}')
print(f'Unique patients: {ccs.EMPI.nunique()}')
print(f'Unique categories: {ccs.ccs_category.nunique()}')
"
```

**Step 4: Final commit**

```bash
git add -A
git commit -m "feat(module05): complete Layer 2 extension with Elixhauser and CCS

Layer 2 now includes:
- Charlson Comorbidity Index (existing)
- Elixhauser Index with van Walraven weights (31 categories)
- CCS categories (Clinical Classifications Software)

Outputs:
- comorbidity_scores.parquet (CCI + Elixhauser)
- ccs_categories.parquet (long format)

Tests: X passed"
```

---

## Summary

| Task | Description | Tests |
|------|-------------|-------|
| 0 | Download CCS crosswalks (manual) | - |
| 1 | Elixhauser code definitions | - |
| 2 | Elixhauser calculator tests | 8 |
| 3 | Elixhauser calculator implementation | - |
| 4 | CCS crosswalk file | - |
| 5 | CCS mapper tests | 6 |
| 6 | CCS mapper implementation | - |
| 7 | Update Layer 2 builder | - |
| 8 | Update pipeline | - |
| 9 | Integration tests | 3 |
| 10 | Final verification | - |

**Total: ~17 new tests**
