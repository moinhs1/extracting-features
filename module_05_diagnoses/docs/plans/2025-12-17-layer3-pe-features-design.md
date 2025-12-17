# Layer 3: PE-Specific Diagnosis Features Design

**Date:** 2025-12-17
**Status:** Approved
**Phase:** Module 05 Phase 2

---

## Overview

Extract ~50 diagnosis-based features directly relevant to PE trajectory analysis from Layer 1 canonical diagnoses.

### Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Code storage | Python dict | Consistent with Phase 1 `comorbidity_codes.py` |
| Code organization | Single file, grouped dicts | Easy to review, maintain |
| Feature builder | Single class, grouped methods | Mirrors Layer1/Layer2 builders |
| Temporal logic | Hardcoded per method | Explicit, easy to understand |
| Output format | Wide DataFrame | Consistent with Layer 2, ML-ready |
| Missing values | Boolean (False = no code) | Standard EHR/claims convention |
| Code sources | CCSR + custom PE-specific | Authoritative + clinically relevant |

---

## File Structure

```
module_05_diagnoses/
├── config/
│   ├── comorbidity_codes.py       # Existing (Phase 1)
│   └── pe_feature_codes.py        # NEW: ICD codes for 9 feature groups
├── processing/
│   ├── layer1_builder.py          # Existing
│   ├── layer2_builder.py          # Existing
│   └── pe_feature_builder.py      # NEW: PEFeatureBuilder class
├── tests/
│   ├── test_layer1_builder.py     # Existing
│   └── test_pe_feature_builder.py # NEW: ~60 tests
├── outputs/
│   ├── layer1/
│   ├── layer2/
│   └── layer3/
│       └── pe_diagnosis_features.parquet  # NEW: Wide format output
└── build_layers.py                # UPDATE: Add Layer 3
```

---

## Feature Groups

### Summary Table

| Group | Features | Temporal Window | Count |
|-------|----------|-----------------|-------|
| 1. VTE History | prior_pe_ever, prior_pe_months, prior_pe_count, prior_dvt_ever, prior_dvt_months, prior_vte_count, is_recurrent_vte | `days < -30` | 7 |
| 2. PE Index | pe_subtype, pe_bilateral, pe_with_cor_pulmonale, pe_high_risk_code | `abs(days) <= 1` | 4 |
| 3. Cancer | cancer_active, cancer_site, cancer_metastatic, cancer_recent_diagnosis | `days < 0` | 4 |
| 4. Cardiovascular | heart_failure, hf_type, cad, atrial_fibrillation, pulmonary_hypertension, valvular_disease | `days < 0` | 6 |
| 5. Pulmonary | copd, asthma, ild, home_oxygen, prior_resp_failure | `days < 0` | 5 |
| 6. Bleeding Risk | prior_major_bleed, prior_gi_bleed, prior_ich, active_pud, thrombocytopenia, coagulopathy | `days < 0` | 6 |
| 7. Renal | ckd_stage, ckd_dialysis, aki_at_presentation | CKD: `days < 0`, AKI: `abs(days) <= 1` | 3 |
| 8. Provoking Factors | recent_surgery, recent_trauma, immobilization, pregnancy_related, hormonal_therapy, cvc, is_provoked_vte | `-30 <= days < 0` | 7 |
| 9. Complications | complication_aki, complication_bleeding_any, complication_bleeding_major, complication_ich, complication_resp_failure, complication_shock, complication_arrest, complication_recurrent_vte, complication_cteph | `days > 1` | 9 |
| **Total** | | | **~51** |

### Feature Definitions

#### Group 1: VTE History

| Feature | Type | Definition |
|---------|------|------------|
| prior_pe_ever | bool | Any PE diagnosis >30 days before index |
| prior_pe_months | float | Months since most recent prior PE (null if none) |
| prior_pe_count | int | Number of prior PE events |
| prior_dvt_ever | bool | Any DVT diagnosis before index |
| prior_dvt_months | float | Months since most recent DVT |
| prior_vte_count | int | Total prior VTE events (PE + DVT) |
| is_recurrent_vte | bool | prior_pe_ever OR prior_dvt_ever |

#### Group 2: PE Index Characterization

| Feature | Type | Definition |
|---------|------|------------|
| pe_subtype | str | 'saddle', 'lobar', 'segmental', 'subsegmental', 'unspecified' |
| pe_bilateral | bool | Bilateral PE code present |
| pe_with_cor_pulmonale | bool | I26.0x (acute cor pulmonale) vs I26.9x |
| pe_high_risk_code | bool | Saddle or with cor pulmonale |

#### Group 3: Cancer

| Feature | Type | Definition |
|---------|------|------------|
| cancer_active | bool | Any malignancy diagnosis before PE |
| cancer_site | str | 'lung', 'colorectal', 'breast', 'prostate', 'hematologic', 'other', null |
| cancer_metastatic | bool | Metastatic disease codes (C77-C80) |
| cancer_recent_diagnosis | bool | Cancer diagnosed within 6 months of PE |

#### Group 4: Cardiovascular

| Feature | Type | Definition |
|---------|------|------------|
| heart_failure | bool | Any HF diagnosis before PE |
| hf_type | str | 'HFrEF', 'HFpEF', 'unspecified', null |
| cad | bool | CAD/prior MI codes |
| atrial_fibrillation | bool | AF/AFL codes |
| pulmonary_hypertension | bool | Pre-existing PH codes |
| valvular_disease | bool | Significant valve disease |

#### Group 5: Pulmonary

| Feature | Type | Definition |
|---------|------|------------|
| copd | bool | COPD diagnosis |
| asthma | bool | Asthma diagnosis |
| ild | bool | Interstitial lung disease |
| home_oxygen | bool | Chronic oxygen use (Z99.81) |
| prior_resp_failure | bool | Prior respiratory failure |

#### Group 6: Bleeding Risk

| Feature | Type | Definition |
|---------|------|------------|
| prior_major_bleed | bool | Major bleeding history |
| prior_gi_bleed | bool | GI bleeding history |
| prior_ich | bool | Intracranial hemorrhage history |
| active_pud | bool | Active peptic ulcer disease |
| thrombocytopenia | bool | Low platelet diagnosis |
| coagulopathy | bool | Coagulation disorder |

#### Group 7: Renal

| Feature | Type | Definition |
|---------|------|------------|
| ckd_stage | int | CKD stage 0-5 (0 = no CKD) |
| ckd_dialysis | bool | Dialysis-dependent |
| aki_at_presentation | bool | AKI at PE index |

#### Group 8: Provoking Factors

| Feature | Type | Definition |
|---------|------|------------|
| recent_surgery | bool | Surgery within 30 days before PE |
| recent_trauma | bool | Major trauma codes |
| immobilization | bool | Immobility/bedrest codes |
| pregnancy_related | bool | Pregnancy/postpartum codes |
| hormonal_therapy | bool | OCP/HRT codes |
| cvc | bool | Central venous catheter codes |
| is_provoked_vte | bool | Any provoking factor present |

#### Group 9: Complications

| Feature | Type | Definition |
|---------|------|------------|
| complication_aki | bool | AKI after PE (days > 1) |
| complication_bleeding_any | bool | Any bleeding post-PE |
| complication_bleeding_major | bool | Major bleeding post-PE |
| complication_ich | bool | ICH post-PE |
| complication_resp_failure | bool | Respiratory failure post-PE |
| complication_shock | bool | Cardiogenic shock post-PE |
| complication_arrest | bool | Cardiac arrest post-PE |
| complication_recurrent_vte | bool | Recurrent VTE during follow-up |
| complication_cteph | bool | CTEPH development |

---

## ICD Code Structure

### File: `config/pe_feature_codes.py`

```python
"""ICD code definitions for PE-specific features."""

# Group 1: VTE History
VTE_CODES = {
    "pe": {
        "icd10": ["I26"],
        "icd9": ["415.1"],
    },
    "dvt_lower_extremity": {
        "icd10": ["I82.40", "I82.41", "I82.42", "I82.43", "I82.44", "I82.49",
                  "I82.4Y", "I82.4Z"],
        "icd9": ["453.4", "453.40", "453.41", "453.42"],
    },
    "dvt_upper_extremity": {
        "icd10": ["I82.60", "I82.61", "I82.62", "I82.A1", "I82.B1"],
        "icd9": ["453.82", "453.83"],
    },
}

# Group 2: PE Index Characterization
PE_SUBTYPE_CODES = {
    "saddle": {
        "icd10": ["I26.02"],
        "icd9": [],
    },
    "with_cor_pulmonale": {
        "icd10": ["I26.0", "I26.01", "I26.02", "I26.09"],
        "icd9": ["415.0"],
    },
    "without_cor_pulmonale": {
        "icd10": ["I26.9", "I26.90", "I26.92", "I26.93", "I26.94", "I26.99"],
        "icd9": ["415.11", "415.19"],
    },
}

# Group 3: Cancer (CCSR-aligned categories)
CANCER_CODES = {
    "lung": {
        "icd10": ["C34"],
        "icd9": ["162"],
    },
    "colorectal": {
        "icd10": ["C18", "C19", "C20", "C21"],
        "icd9": ["153", "154"],
    },
    "breast": {
        "icd10": ["C50"],
        "icd9": ["174", "175"],
    },
    "prostate": {
        "icd10": ["C61"],
        "icd9": ["185"],
    },
    "pancreas": {
        "icd10": ["C25"],
        "icd9": ["157"],
    },
    "gastric": {
        "icd10": ["C16"],
        "icd9": ["151"],
    },
    "ovarian": {
        "icd10": ["C56"],
        "icd9": ["183"],
    },
    "renal": {
        "icd10": ["C64", "C65"],
        "icd9": ["189.0", "189.1"],
    },
    "bladder": {
        "icd10": ["C67"],
        "icd9": ["188"],
    },
    "hematologic": {
        "icd10": ["C81", "C82", "C83", "C84", "C85", "C86",
                  "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96"],
        "icd9": ["200", "201", "202", "203", "204", "205", "206", "207", "208"],
    },
    "brain": {
        "icd10": ["C71"],
        "icd9": ["191"],
    },
    "metastatic": {
        "icd10": ["C77", "C78", "C79", "C80"],
        "icd9": ["196", "197", "198", "199"],
    },
}

# Group 4: Cardiovascular
CARDIOVASCULAR_CODES = {
    "heart_failure": {
        "icd10": ["I50"],
        "icd9": ["428"],
    },
    "hf_reduced_ef": {
        "icd10": ["I50.2", "I50.20", "I50.21", "I50.22", "I50.23"],
        "icd9": ["428.2"],
    },
    "hf_preserved_ef": {
        "icd10": ["I50.3", "I50.30", "I50.31", "I50.32", "I50.33"],
        "icd9": ["428.3"],
    },
    "cad": {
        "icd10": ["I25", "I21", "I22"],
        "icd9": ["414", "410", "412"],
    },
    "atrial_fibrillation": {
        "icd10": ["I48"],
        "icd9": ["427.31", "427.32"],
    },
    "pulmonary_hypertension": {
        "icd10": ["I27.0", "I27.2", "I27.20", "I27.21", "I27.22", "I27.23", "I27.24", "I27.29"],
        "icd9": ["416.0", "416.8"],
    },
    "valvular_disease": {
        "icd10": ["I05", "I06", "I07", "I08", "I34", "I35", "I36", "I37"],
        "icd9": ["394", "395", "396", "397", "424"],
    },
}

# Group 5: Pulmonary
PULMONARY_CODES = {
    "copd": {
        "icd10": ["J44"],
        "icd9": ["491", "492", "496"],
    },
    "asthma": {
        "icd10": ["J45", "J46"],
        "icd9": ["493"],
    },
    "ild": {
        "icd10": ["J84"],
        "icd9": ["516"],
    },
    "home_oxygen": {
        "icd10": ["Z99.81"],
        "icd9": ["V46.2"],
    },
    "respiratory_failure": {
        "icd10": ["J96"],
        "icd9": ["518.81", "518.82", "518.83", "518.84"],
    },
}

# Group 6: Bleeding Risk
BLEEDING_CODES = {
    "gi_bleeding": {
        "icd10": ["K92.0", "K92.1", "K92.2", "K25.0", "K25.4", "K26.0", "K26.4",
                  "K27.0", "K27.4", "K28.0", "K28.4", "K62.5"],
        "icd9": ["578", "531.0", "532.0", "533.0", "534.0"],
    },
    "intracranial_hemorrhage": {
        "icd10": ["I60", "I61", "I62"],
        "icd9": ["430", "431", "432"],
    },
    "major_bleeding_other": {
        "icd10": ["D62", "R58", "N02", "R31"],
        "icd9": ["285.1", "459.0", "599.7"],
    },
    "peptic_ulcer_active": {
        "icd10": ["K25", "K26", "K27", "K28"],
        "icd9": ["531", "532", "533", "534"],
    },
    "thrombocytopenia": {
        "icd10": ["D69.4", "D69.5", "D69.6"],
        "icd9": ["287.3", "287.4", "287.5"],
    },
    "coagulopathy": {
        "icd10": ["D68"],
        "icd9": ["286"],
    },
}

# Group 7: Renal
RENAL_CODES = {
    "ckd_stage1": {"icd10": ["N18.1"], "icd9": ["585.1"]},
    "ckd_stage2": {"icd10": ["N18.2"], "icd9": ["585.2"]},
    "ckd_stage3": {"icd10": ["N18.3", "N18.30", "N18.31", "N18.32"], "icd9": ["585.3"]},
    "ckd_stage4": {"icd10": ["N18.4"], "icd9": ["585.4"]},
    "ckd_stage5": {"icd10": ["N18.5", "N18.6"], "icd9": ["585.5", "585.6"]},
    "dialysis": {
        "icd10": ["Z99.2", "Z49"],
        "icd9": ["V45.1", "V56"],
    },
    "aki": {
        "icd10": ["N17"],
        "icd9": ["584"],
    },
}

# Group 8: Provoking Factors
PROVOKING_CODES = {
    "surgery": {
        "icd10": ["Z96", "Z98"],  # Post-procedural states
        "icd9": ["V45"],
    },
    "trauma": {
        "icd10": ["S", "T0", "T1"],  # Injury codes (prefix)
        "icd9": ["8", "9"],  # 800-999 range
    },
    "immobilization": {
        "icd10": ["Z74.0", "Z74.1", "R26.3"],
        "icd9": ["V49.84"],
    },
    "pregnancy": {
        "icd10": ["O"],  # All pregnancy codes
        "icd9": ["V22", "V23", "V27", "63", "64", "65", "66", "67", "68", "69"],
    },
    "hormonal_therapy": {
        "icd10": ["Z79.3", "Z79.890"],
        "icd9": ["V25.4", "V07.4"],
    },
    "central_venous_catheter": {
        "icd10": ["Z45.2", "T80.211", "T80.212", "T80.218", "T80.219"],
        "icd9": ["V45.81", "999.31", "999.32"],
    },
}

# Group 9: Complications (reuse codes from above, different temporal window)
COMPLICATION_CODES = {
    "aki": RENAL_CODES["aki"],
    "bleeding_gi": BLEEDING_CODES["gi_bleeding"],
    "bleeding_ich": BLEEDING_CODES["intracranial_hemorrhage"],
    "bleeding_other": BLEEDING_CODES["major_bleeding_other"],
    "respiratory_failure": PULMONARY_CODES["respiratory_failure"],
    "cardiogenic_shock": {
        "icd10": ["R57.0"],
        "icd9": ["785.51"],
    },
    "cardiac_arrest": {
        "icd10": ["I46"],
        "icd9": ["427.5"],
    },
    "recurrent_pe": VTE_CODES["pe"],
    "recurrent_dvt": VTE_CODES["dvt_lower_extremity"],
    "cteph": {
        "icd10": ["I27.24", "I27.29"],
        "icd9": ["416.8"],
    },
}
```

---

## PEFeatureBuilder Class

### File: `processing/pe_feature_builder.py`

```python
"""Build PE-specific diagnosis features from Layer 1 canonical diagnoses."""

import pandas as pd
import numpy as np
from typing import Optional
from config.pe_feature_codes import (
    VTE_CODES, PE_SUBTYPE_CODES, CANCER_CODES, CARDIOVASCULAR_CODES,
    PULMONARY_CODES, BLEEDING_CODES, RENAL_CODES, PROVOKING_CODES,
    COMPLICATION_CODES
)


class PEFeatureBuilder:
    """Build PE-specific diagnosis features from Layer 1 canonical diagnoses."""

    def __init__(self, canonical_diagnoses: pd.DataFrame):
        """
        Initialize with Layer 1 canonical diagnoses.

        Args:
            canonical_diagnoses: DataFrame with columns:
                EMPI, icd_code, icd_version, days_from_pe, ...
        """
        self.diagnoses = canonical_diagnoses
        self.empis = canonical_diagnoses['EMPI'].unique()

    def _match_codes(self, df: pd.DataFrame, code_dict: dict,
                     version: Optional[str] = None) -> pd.DataFrame:
        """
        Filter diagnoses matching any code in dict (prefix matching).

        Args:
            df: Diagnoses DataFrame to filter
            code_dict: Dict with 'icd10' and 'icd9' code lists
            version: Optional filter for specific ICD version

        Returns:
            Filtered DataFrame with matching diagnoses
        """
        masks = []

        for icd_ver, codes in [('10', code_dict.get('icd10', [])),
                                ('9', code_dict.get('icd9', []))]:
            if version and icd_ver != version:
                continue
            for code in codes:
                mask = (df['icd_version'] == icd_ver) & \
                       (df['icd_code'].str.startswith(code))
                masks.append(mask)

        if not masks:
            return df.iloc[0:0]  # Empty DataFrame with same columns

        combined_mask = masks[0]
        for m in masks[1:]:
            combined_mask = combined_mask | m

        return df[combined_mask]

    def _match_any_category(self, df: pd.DataFrame,
                            code_categories: dict) -> pd.DataFrame:
        """Match diagnoses against any category in a code group."""
        results = []
        for category, codes in code_categories.items():
            matched = self._match_codes(df, codes)
            if not matched.empty:
                matched = matched.copy()
                matched['matched_category'] = category
                results.append(matched)

        if not results:
            return df.iloc[0:0]
        return pd.concat(results, ignore_index=True)

    def build_vte_history_features(self) -> pd.DataFrame:
        """Extract VTE history features (prior PE/DVT)."""
        prior = self.diagnoses[self.diagnoses['days_from_pe'] < -30]

        results = []
        for empi in self.empis:
            patient_prior = prior[prior['EMPI'] == empi]

            # Prior PE
            prior_pe = self._match_codes(patient_prior, VTE_CODES['pe'])
            prior_pe_ever = len(prior_pe) > 0
            prior_pe_count = len(prior_pe)
            prior_pe_months = None
            if prior_pe_ever:
                most_recent = prior_pe['days_from_pe'].max()
                prior_pe_months = abs(most_recent) / 30.44

            # Prior DVT
            dvt_codes = {
                'icd10': VTE_CODES['dvt_lower_extremity']['icd10'] +
                         VTE_CODES['dvt_upper_extremity']['icd10'],
                'icd9': VTE_CODES['dvt_lower_extremity']['icd9'] +
                        VTE_CODES['dvt_upper_extremity']['icd9']
            }
            prior_dvt = self._match_codes(patient_prior, dvt_codes)
            prior_dvt_ever = len(prior_dvt) > 0
            prior_dvt_months = None
            if prior_dvt_ever:
                most_recent = prior_dvt['days_from_pe'].max()
                prior_dvt_months = abs(most_recent) / 30.44

            results.append({
                'EMPI': empi,
                'prior_pe_ever': prior_pe_ever,
                'prior_pe_months': prior_pe_months,
                'prior_pe_count': prior_pe_count,
                'prior_dvt_ever': prior_dvt_ever,
                'prior_dvt_months': prior_dvt_months,
                'prior_vte_count': prior_pe_count + len(prior_dvt),
                'is_recurrent_vte': prior_pe_ever or prior_dvt_ever,
            })

        return pd.DataFrame(results).set_index('EMPI')

    # Additional methods follow same pattern...
    def build_pe_index_features(self) -> pd.DataFrame: ...
    def build_cancer_features(self) -> pd.DataFrame: ...
    def build_cardiovascular_features(self) -> pd.DataFrame: ...
    def build_pulmonary_features(self) -> pd.DataFrame: ...
    def build_bleeding_risk_features(self) -> pd.DataFrame: ...
    def build_renal_features(self) -> pd.DataFrame: ...
    def build_provoking_factors(self) -> pd.DataFrame: ...
    def build_complication_features(self) -> pd.DataFrame: ...

    def build_all_features(self) -> pd.DataFrame:
        """Merge all feature groups into single wide DataFrame."""
        dfs = [
            self.build_vte_history_features(),
            self.build_pe_index_features(),
            self.build_cancer_features(),
            self.build_cardiovascular_features(),
            self.build_pulmonary_features(),
            self.build_bleeding_risk_features(),
            self.build_renal_features(),
            self.build_provoking_factors(),
            self.build_complication_features(),
        ]

        result = dfs[0]
        for df in dfs[1:]:
            result = result.join(df, how='outer')

        # Fill boolean columns with False, numeric with appropriate defaults
        bool_cols = result.select_dtypes(include=['bool']).columns
        result[bool_cols] = result[bool_cols].fillna(False)

        return result.reset_index()
```

---

## Testing Strategy

### File: `tests/test_pe_feature_builder.py`

**Structure:**
- One test class per feature group
- Synthetic test data with known expected outcomes
- ~5-10 tests per group (~60 tests total)

**Example Tests:**

```python
import pytest
import pandas as pd
from processing.pe_feature_builder import PEFeatureBuilder

@pytest.fixture
def sample_diagnoses():
    """Synthetic diagnoses with known expected outcomes."""
    return pd.DataFrame({
        'EMPI': ['P1', 'P1', 'P1', 'P2', 'P2', 'P3'],
        'icd_code': ['I26.99', 'I26.92', 'I50.9', 'I26.0', 'C34.9', 'I26.9'],
        'icd_version': ['10', '10', '10', '10', '10', '10'],
        'days_from_pe': [-60, 0, -10, 0, -90, 0],
    })


class TestVTEHistoryFeatures:
    def test_prior_pe_detected(self, sample_diagnoses):
        builder = PEFeatureBuilder(sample_diagnoses)
        features = builder.build_vte_history_features()
        assert features.loc['P1', 'prior_pe_ever'] == True
        assert features.loc['P2', 'prior_pe_ever'] == False

    def test_prior_pe_months_calculated(self, sample_diagnoses):
        builder = PEFeatureBuilder(sample_diagnoses)
        features = builder.build_vte_history_features()
        assert features.loc['P1', 'prior_pe_months'] == pytest.approx(1.97, rel=0.1)

    def test_no_prior_pe_returns_null_months(self, sample_diagnoses):
        builder = PEFeatureBuilder(sample_diagnoses)
        features = builder.build_vte_history_features()
        assert pd.isna(features.loc['P2', 'prior_pe_months'])


class TestCancerFeatures:
    def test_active_cancer_detected(self, sample_diagnoses):
        builder = PEFeatureBuilder(sample_diagnoses)
        features = builder.build_cancer_features()
        assert features.loc['P2', 'cancer_active'] == True
        assert features.loc['P2', 'cancer_site'] == 'lung'

    def test_no_cancer_returns_false(self, sample_diagnoses):
        builder = PEFeatureBuilder(sample_diagnoses)
        features = builder.build_cancer_features()
        assert features.loc['P1', 'cancer_active'] == False
```

---

## Pipeline Integration

### Update `build_layers.py`

```python
def build_layer3(layer1_path: Path, output_dir: Path) -> pd.DataFrame:
    """Build Layer 3: PE-specific diagnosis features."""
    from processing.pe_feature_builder import PEFeatureBuilder

    canonical = pd.read_parquet(layer1_path)
    builder = PEFeatureBuilder(canonical)
    features = builder.build_all_features()

    output_path = output_dir / "layer3" / "pe_diagnosis_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(output_path, index=False)

    print(f"Layer 3: {len(features)} patients, {len(features.columns)} features")
    return features
```

---

## Implementation Order

| Order | Component | Description |
|-------|-----------|-------------|
| 1 | `pe_feature_codes.py` (VTE only) | Foundation, verify code matching |
| 2 | `PEFeatureBuilder.__init__` + `_match_codes` | Core utilities |
| 3 | `build_vte_history_features` + tests | Most PE-specific, establishes pattern |
| 4 | `build_pe_index_features` + tests | PE characterization |
| 5 | `build_cancer_features` + tests | High clinical importance |
| 6 | `build_cardiovascular_features` + tests | Common comorbidities |
| 7 | `build_pulmonary_features` + tests | Common comorbidities |
| 8 | `build_bleeding_risk_features` + tests | Treatment decisions |
| 9 | `build_renal_features` + tests | Simple, few features |
| 10 | `build_provoking_factors` + tests | Different temporal window |
| 11 | `build_complication_features` + tests | Post-PE window |
| 12 | `build_all_features` + integration | Final assembly |
| 13 | Pipeline integration | Update `build_layers.py` |

---

## Output Schema

### `layer3/pe_diagnosis_features.parquet`

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier |
| prior_pe_ever | bool | Prior PE >30 days before index |
| prior_pe_months | float | Months since prior PE |
| prior_pe_count | int | Count of prior PE events |
| prior_dvt_ever | bool | Prior DVT |
| prior_dvt_months | float | Months since prior DVT |
| prior_vte_count | int | Total prior VTE |
| is_recurrent_vte | bool | Any prior VTE |
| pe_subtype | str | Saddle/lobar/etc |
| pe_bilateral | bool | Bilateral PE |
| pe_with_cor_pulmonale | bool | Acute cor pulmonale |
| pe_high_risk_code | bool | High-risk PE code |
| cancer_active | bool | Active malignancy |
| cancer_site | str | Cancer site category |
| cancer_metastatic | bool | Metastatic disease |
| cancer_recent_diagnosis | bool | Cancer within 6mo |
| heart_failure | bool | Heart failure |
| hf_type | str | HFrEF/HFpEF/unspecified |
| cad | bool | Coronary artery disease |
| atrial_fibrillation | bool | AF/AFL |
| pulmonary_hypertension | bool | Pre-existing PH |
| valvular_disease | bool | Valve disease |
| copd | bool | COPD |
| asthma | bool | Asthma |
| ild | bool | Interstitial lung disease |
| home_oxygen | bool | Chronic O2 use |
| prior_resp_failure | bool | Prior resp failure |
| prior_major_bleed | bool | Major bleeding hx |
| prior_gi_bleed | bool | GI bleeding hx |
| prior_ich | bool | ICH history |
| active_pud | bool | Active peptic ulcer |
| thrombocytopenia | bool | Low platelets |
| coagulopathy | bool | Coagulation disorder |
| ckd_stage | int | CKD stage 0-5 |
| ckd_dialysis | bool | On dialysis |
| aki_at_presentation | bool | AKI at PE |
| recent_surgery | bool | Surgery <30d |
| recent_trauma | bool | Major trauma |
| immobilization | bool | Immobility |
| pregnancy_related | bool | Pregnancy/postpartum |
| hormonal_therapy | bool | OCP/HRT |
| cvc | bool | Central line |
| is_provoked_vte | bool | Any provoking factor |
| complication_aki | bool | AKI post-PE |
| complication_bleeding_any | bool | Any bleeding |
| complication_bleeding_major | bool | Major bleeding |
| complication_ich | bool | ICH post-PE |
| complication_resp_failure | bool | Resp failure |
| complication_shock | bool | Cardiogenic shock |
| complication_arrest | bool | Cardiac arrest |
| complication_recurrent_vte | bool | Recurrent VTE |
| complication_cteph | bool | CTEPH |

---

## References

- [Quan et al. 2005](https://pubmed.ncbi.nlm.nih.gov/16224307/) - Charlson/Elixhauser ICD coding algorithms
- [AHRQ CCSR](https://hcup-us.ahrq.gov/toolssoftware/ccsr/dxccsr.jsp) - Clinical Classifications Software Refined
- [CCMDB Wiki](https://ccmdb.kuality.ca/index.php?title=Charlson_Comorbidities_in_ICD10_codes) - ICD-10 Charlson reference

---

**Document Version:** 1.0
**Author:** Brainstorming session 2025-12-17
