"""Tests for PE feature builder - starting with generic code matcher."""
import pytest
import pandas as pd
from processing.pe_feature_builder import (
    code_matches_category,
    get_preexisting_diagnoses,
    get_complication_diagnoses,
    get_index_diagnoses,
    days_to_months,
    get_most_recent_prior,
    extract_prior_pe_features,
    extract_prior_dvt_features,
    extract_vte_history_features,
    extract_pe_characterization,
)
from config.icd_code_lists import (
    VTE_CODES,
    CANCER_CODES,
    CARDIOVASCULAR_CODES,
    PULMONARY_CODES,
    BLEEDING_CODES,
    RENAL_CODES,
    PROVOKING_FACTOR_CODES,
    COMPLICATION_CODES,
)


class TestCodeMatcher:
    """Tests for the generic ICD code matcher."""

    def test_exact_match(self):
        """Test exact code match within category prefixes."""
        category_codes = {
            "icd10": ["I26.0", "I26.9"],
            "icd9": []
        }
        assert code_matches_category("I26.0", category_codes, "10") is True

    def test_prefix_match(self):
        """Test prefix matching - longer code matches shorter prefix."""
        category_codes = {
            "icd10": ["I26"],
            "icd9": []
        }
        assert code_matches_category("I26.99", category_codes, "10") is True

    def test_no_match(self):
        """Test that non-matching codes return False."""
        category_codes = {
            "icd10": ["I26"],
            "icd9": []
        }
        assert code_matches_category("I50.9", category_codes, "10") is False

    def test_icd9_match(self):
        """Test ICD-9 code matching with prefix."""
        category_codes = {
            "icd10": ["I26"],
            "icd9": ["415"]
        }
        assert code_matches_category("415.19", category_codes, "9") is True


class TestVTECodes:
    """Tests for VTE ICD code lists."""

    def test_pe_icd10_codes_exist(self):
        """Verify PE ICD-10 codes are defined."""
        assert "pe" in VTE_CODES
        assert "icd10" in VTE_CODES["pe"]
        assert len(VTE_CODES["pe"]["icd10"]) > 0

    def test_pe_code_matches(self):
        """Test that I26.99 matches PE category."""
        assert code_matches_category("I26.99", VTE_CODES["pe"], "10") is True

    def test_dvt_lower_matches(self):
        """Test that I82.401 matches lower DVT category."""
        assert code_matches_category("I82.401", VTE_CODES["dvt_lower"], "10") is True

    def test_non_vte_no_match(self):
        """Test that I50.9 (heart failure) does NOT match any VTE codes."""
        assert code_matches_category("I50.9", VTE_CODES["pe"], "10") is False
        assert code_matches_category("I50.9", VTE_CODES["dvt_lower"], "10") is False
        assert code_matches_category("I50.9", VTE_CODES["dvt_upper"], "10") is False


class TestICDCodeLists:
    """Tests for all ICD code list structures and basic matching."""

    # CANCER_CODES tests
    def test_cancer_codes_structure(self):
        """Verify CANCER_CODES has expected categories and structure."""
        expected_categories = ["lung", "gi", "gu", "hematologic", "breast", "metastatic", "chemotherapy"]
        for category in expected_categories:
            assert category in CANCER_CODES
            assert "icd10" in CANCER_CODES[category]
            assert "icd9" in CANCER_CODES[category]

    def test_cancer_lung_match(self):
        """Test that C34.90 (lung cancer) matches lung category."""
        assert code_matches_category("C34.90", CANCER_CODES["lung"], "10") is True

    def test_cancer_gi_match(self):
        """Test that C18.9 (colon cancer) matches GI category."""
        assert code_matches_category("C18.9", CANCER_CODES["gi"], "10") is True

    def test_cancer_metastatic_match(self):
        """Test that C79.51 (bone metastasis) matches metastatic category."""
        assert code_matches_category("C79.51", CANCER_CODES["metastatic"], "10") is True

    def test_cancer_chemotherapy_match(self):
        """Test that Z51.11 (chemotherapy) matches chemotherapy category."""
        assert code_matches_category("Z51.11", CANCER_CODES["chemotherapy"], "10") is True

    # CARDIOVASCULAR_CODES tests
    def test_cardiovascular_codes_structure(self):
        """Verify CARDIOVASCULAR_CODES has expected categories."""
        expected_categories = ["heart_failure", "heart_failure_reduced", "heart_failure_preserved",
                               "coronary_artery_disease", "atrial_fibrillation", "pulmonary_hypertension",
                               "valvular_heart_disease"]
        for category in expected_categories:
            assert category in CARDIOVASCULAR_CODES
            assert "icd10" in CARDIOVASCULAR_CODES[category]
            assert "icd9" in CARDIOVASCULAR_CODES[category]

    def test_cv_heart_failure_match(self):
        """Test that I50.9 (heart failure) matches heart_failure category."""
        assert code_matches_category("I50.9", CARDIOVASCULAR_CODES["heart_failure"], "10") is True

    def test_cv_afib_match(self):
        """Test that I48.91 (atrial fibrillation) matches afib category."""
        assert code_matches_category("I48.91", CARDIOVASCULAR_CODES["atrial_fibrillation"], "10") is True

    def test_cv_cad_match(self):
        """Test that I25.10 (CAD) matches coronary_artery_disease category."""
        assert code_matches_category("I25.10", CARDIOVASCULAR_CODES["coronary_artery_disease"], "10") is True

    # PULMONARY_CODES tests
    def test_pulmonary_codes_structure(self):
        """Verify PULMONARY_CODES has expected categories."""
        expected_categories = ["copd", "asthma", "interstitial_lung_disease", "home_oxygen", "respiratory_failure"]
        for category in expected_categories:
            assert category in PULMONARY_CODES
            assert "icd10" in PULMONARY_CODES[category]
            assert "icd9" in PULMONARY_CODES[category]

    def test_pulm_copd_match(self):
        """Test that J44.1 (COPD) matches copd category."""
        assert code_matches_category("J44.1", PULMONARY_CODES["copd"], "10") is True

    def test_pulm_asthma_match(self):
        """Test that J45.909 (asthma) matches asthma category."""
        assert code_matches_category("J45.909", PULMONARY_CODES["asthma"], "10") is True

    def test_pulm_resp_failure_match(self):
        """Test that J96.90 (respiratory failure) matches respiratory_failure category."""
        assert code_matches_category("J96.90", PULMONARY_CODES["respiratory_failure"], "10") is True

    # BLEEDING_CODES tests
    def test_bleeding_codes_structure(self):
        """Verify BLEEDING_CODES has expected categories."""
        expected_categories = ["gi_bleeding", "intracranial_hemorrhage", "other_major_bleeding",
                               "peptic_ulcer", "thrombocytopenia", "coagulopathy"]
        for category in expected_categories:
            assert category in BLEEDING_CODES
            assert "icd10" in BLEEDING_CODES[category]
            assert "icd9" in BLEEDING_CODES[category]

    def test_bleeding_gi_match(self):
        """Test that K92.2 (GI bleeding) matches gi_bleeding category."""
        assert code_matches_category("K92.2", BLEEDING_CODES["gi_bleeding"], "10") is True

    def test_bleeding_ich_match(self):
        """Test that I61.9 (intracerebral hemorrhage) matches intracranial_hemorrhage category."""
        assert code_matches_category("I61.9", BLEEDING_CODES["intracranial_hemorrhage"], "10") is True

    def test_bleeding_coagulopathy_match(self):
        """Test that D68.9 (coagulopathy) matches coagulopathy category."""
        assert code_matches_category("D68.9", BLEEDING_CODES["coagulopathy"], "10") is True

    # RENAL_CODES tests
    def test_renal_codes_structure(self):
        """Verify RENAL_CODES has expected categories."""
        expected_categories = ["ckd_stage1", "ckd_stage2", "ckd_stage3", "ckd_stage4", "ckd_stage5",
                               "dialysis", "aki"]
        for category in expected_categories:
            assert category in RENAL_CODES
            assert "icd10" in RENAL_CODES[category]
            assert "icd9" in RENAL_CODES[category]

    def test_renal_ckd3_match(self):
        """Test that N18.31 (CKD stage 3a) matches ckd_stage3 category."""
        assert code_matches_category("N18.31", RENAL_CODES["ckd_stage3"], "10") is True

    def test_renal_dialysis_match(self):
        """Test that Z99.2 (dialysis) matches dialysis category."""
        assert code_matches_category("Z99.2", RENAL_CODES["dialysis"], "10") is True

    def test_renal_aki_match(self):
        """Test that N17.9 (acute kidney injury) matches aki category."""
        assert code_matches_category("N17.9", RENAL_CODES["aki"], "10") is True

    # PROVOKING_FACTOR_CODES tests
    def test_provoking_factor_codes_structure(self):
        """Verify PROVOKING_FACTOR_CODES has expected categories."""
        expected_categories = ["recent_surgery", "trauma", "immobilization", "pregnancy",
                               "hormonal_therapy", "central_venous_catheter"]
        for category in expected_categories:
            assert category in PROVOKING_FACTOR_CODES
            assert "icd10" in PROVOKING_FACTOR_CODES[category]
            assert "icd9" in PROVOKING_FACTOR_CODES[category]

    def test_provoking_surgery_match(self):
        """Test that Z98.89 (post-surgical) matches recent_surgery category."""
        assert code_matches_category("Z98.89", PROVOKING_FACTOR_CODES["recent_surgery"], "10") is True

    def test_provoking_trauma_match(self):
        """Test that S82.001A (tibial fracture) matches trauma category."""
        assert code_matches_category("S82.001A", PROVOKING_FACTOR_CODES["trauma"], "10") is True

    def test_provoking_pregnancy_match(self):
        """Test that O99.89 (pregnancy complication) matches pregnancy category."""
        assert code_matches_category("O99.89", PROVOKING_FACTOR_CODES["pregnancy"], "10") is True

    # COMPLICATION_CODES tests
    def test_complication_codes_structure(self):
        """Verify COMPLICATION_CODES has expected categories."""
        expected_categories = ["aki", "bleeding_any", "bleeding_major", "intracranial_hemorrhage",
                               "respiratory_failure", "cardiogenic_shock", "cardiac_arrest",
                               "recurrent_vte", "cteph"]
        for category in expected_categories:
            assert category in COMPLICATION_CODES
            assert "icd10" in COMPLICATION_CODES[category]
            assert "icd9" in COMPLICATION_CODES[category]

    def test_complication_shock_match(self):
        """Test that R57.0 (cardiogenic shock) matches cardiogenic_shock category."""
        assert code_matches_category("R57.0", COMPLICATION_CODES["cardiogenic_shock"], "10") is True

    def test_complication_cardiac_arrest_match(self):
        """Test that I46.9 (cardiac arrest) matches cardiac_arrest category."""
        assert code_matches_category("I46.9", COMPLICATION_CODES["cardiac_arrest"], "10") is True

    def test_complication_cteph_match(self):
        """Test that I27.24 (CTEPH) matches cteph category."""
        assert code_matches_category("I27.24", COMPLICATION_CODES["cteph"], "10") is True


class TestTemporalFilters:
    """Tests for temporal filter helper functions."""

    def test_preexisting_filter(self):
        """Returns only preexisting or recent antecedent rows."""
        df = pd.DataFrame({
            "is_preexisting": [True, False, False],
            "is_recent_antecedent": [False, True, False],
            "is_complication": [False, False, True],
            "is_index_concurrent": [False, False, False],
        })
        result = get_preexisting_diagnoses(df)
        assert len(result) == 2

    def test_complication_filter(self):
        """Returns only complication rows."""
        df = pd.DataFrame({
            "is_preexisting": [True, False],
            "is_recent_antecedent": [False, False],
            "is_complication": [False, True],
            "is_index_concurrent": [False, False],
        })
        result = get_complication_diagnoses(df)
        assert len(result) == 1

    def test_index_filter(self):
        """Returns only index concurrent rows."""
        df = pd.DataFrame({
            "is_preexisting": [False, False],
            "is_recent_antecedent": [False, False],
            "is_complication": [False, False],
            "is_index_concurrent": [True, False],
        })
        result = get_index_diagnoses(df)
        assert len(result) == 1


class TestTimeHelpers:
    """Tests for time-based helper functions."""

    def test_days_to_months_positive(self):
        """Converts 365 days to approximately 12 months."""
        result = days_to_months(365)
        assert 11.9 < result < 12.1

    def test_days_to_months_negative(self):
        """Handles negative days (before PE)."""
        result = days_to_months(-180)
        assert 5.8 < result < 6.0

    def test_get_most_recent_prior_found(self):
        """Finds most recent prior matching diagnosis."""
        df = pd.DataFrame({
            "icd_code": ["I26.0", "I26.9"],
            "icd_version": ["10", "10"],
            "days_from_pe": [-365, -30],  # -30 is more recent
        })
        result = get_most_recent_prior(df, VTE_CODES["pe"])
        assert result == -30

    def test_get_most_recent_prior_none(self):
        """Returns None when no matches."""
        df = pd.DataFrame({
            "icd_code": ["I50.9"],  # Not PE
            "icd_version": ["10"],
            "days_from_pe": [-30],
        })
        result = get_most_recent_prior(df, VTE_CODES["pe"])
        assert result is None


class TestVTEHistoryFeatures:
    """Tests for VTE history feature extraction."""

    def test_no_prior_pe(self):
        """Patient with no prior PE."""
        df = pd.DataFrame({
            "icd_code": ["I50.9"],
            "icd_version": ["10"],
            "days_from_pe": [-30],
            "is_pe_diagnosis": [False],
        })
        result = extract_prior_pe_features(df)
        assert result["prior_pe_ever"] == False
        assert result["prior_pe_months"] is None
        assert result["prior_pe_count"] == 0

    def test_single_prior_pe(self):
        """One PE 6 months ago."""
        df = pd.DataFrame({
            "icd_code": ["I26.99"],
            "icd_version": ["10"],
            "days_from_pe": [-183],  # ~6 months
            "is_pe_diagnosis": [True],
        })
        result = extract_prior_pe_features(df)
        assert result["prior_pe_ever"] == True
        assert 5.9 < result["prior_pe_months"] < 6.1
        assert result["prior_pe_count"] == 1

    def test_multiple_prior_pe(self):
        """Three prior PEs."""
        df = pd.DataFrame({
            "icd_code": ["I26.0", "I26.9", "I26.99"],
            "icd_version": ["10", "10", "10"],
            "days_from_pe": [-365, -180, -30],
            "is_pe_diagnosis": [True, True, True],
        })
        result = extract_prior_pe_features(df)
        assert result["prior_pe_count"] == 3
        assert result["prior_pe_months"] < 1.5  # Most recent was 30 days ago

    def test_prior_dvt(self):
        """Prior DVT detection."""
        df = pd.DataFrame({
            "icd_code": ["I82.401"],
            "icd_version": ["10"],
            "days_from_pe": [-90],
            "is_pe_diagnosis": [False],
        })
        result = extract_prior_dvt_features(df)
        assert result["prior_dvt_ever"] == True
        assert result["prior_dvt_months"] is not None

    def test_recurrent_vte_pe_only(self):
        """Prior PE only → is_recurrent_vte=True."""
        df = pd.DataFrame({
            "icd_code": ["I26.99"],
            "icd_version": ["10"],
            "days_from_pe": [-180],
            "is_pe_diagnosis": [True],
        })
        result = extract_vte_history_features(df)
        assert result["is_recurrent_vte"] == True

    def test_recurrent_vte_dvt_only(self):
        """Prior DVT only → is_recurrent_vte=True."""
        df = pd.DataFrame({
            "icd_code": ["I82.401"],
            "icd_version": ["10"],
            "days_from_pe": [-90],
            "is_pe_diagnosis": [False],
        })
        result = extract_vte_history_features(df)
        assert result["is_recurrent_vte"] == True

    def test_no_recurrence(self):
        """No prior VTE → is_recurrent_vte=False."""
        df = pd.DataFrame({
            "icd_code": ["I50.9"],
            "icd_version": ["10"],
            "days_from_pe": [-30],
            "is_pe_diagnosis": [False],
        })
        result = extract_vte_history_features(df)
        assert result["is_recurrent_vte"] == False


class TestPECharacterization:
    """Tests for PE index characterization."""

    def test_saddle_pe(self):
        """Saddle PE detected from I26.92."""
        df = pd.DataFrame({
            "icd_code": ["I26.92"],
            "icd_version": ["10"],
            "days_from_pe": [0],
            "is_pe_diagnosis": [True],
            "is_index_concurrent": [True],
        })
        result = extract_pe_characterization(df)
        assert result["pe_subtype"] == "saddle"
        assert result["pe_high_risk_code"] == True

    def test_subsegmental_pe(self):
        """Subsegmental PE detected."""
        df = pd.DataFrame({
            "icd_code": ["I26.93"],
            "icd_version": ["10"],
            "days_from_pe": [0],
            "is_pe_diagnosis": [True],
            "is_index_concurrent": [True],
        })
        result = extract_pe_characterization(df)
        assert result["pe_subtype"] == "subsegmental"

    def test_cor_pulmonale(self):
        """PE with acute cor pulmonale."""
        df = pd.DataFrame({
            "icd_code": ["I26.02"],
            "icd_version": ["10"],
            "days_from_pe": [0],
            "is_pe_diagnosis": [True],
            "is_index_concurrent": [True],
        })
        result = extract_pe_characterization(df)
        assert result["pe_with_cor_pulmonale"] == True
        assert result["pe_high_risk_code"] == True
