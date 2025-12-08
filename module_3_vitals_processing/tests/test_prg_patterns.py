"""Tests for prg_patterns module."""
import pytest
import re


class TestPrgSectionPatterns:
    """Test Prg section pattern definitions."""

    def test_section_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        assert isinstance(PRG_SECTION_PATTERNS, dict)
        assert len(PRG_SECTION_PATTERNS) >= 10

    def test_physical_exam_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['physical_exam']
        assert re.search(pattern, "Physical Exam: BP 120/80", re.IGNORECASE)
        assert re.search(pattern, "Physical Examination: normal", re.IGNORECASE)

    def test_vitals_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['vitals']
        assert re.search(pattern, "Vitals: HR 72", re.IGNORECASE)
        assert re.search(pattern, "Vital: T 98.6", re.IGNORECASE)

    def test_on_exam_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['on_exam']
        assert re.search(pattern, "ON EXAM: Vital Signs BP 120/80", re.IGNORECASE)

    def test_objective_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SECTION_PATTERNS
        pattern, _ = PRG_SECTION_PATTERNS['objective']
        assert re.search(pattern, "Objective: Physical Exam", re.IGNORECASE)


class TestPrgSkipPatterns:
    """Test Prg skip section patterns."""

    def test_skip_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        assert isinstance(PRG_SKIP_PATTERNS, list)
        assert len(PRG_SKIP_PATTERNS) >= 10

    def test_allergies_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Allergies: atenolol - fatigue, HR 50"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_medications_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Medications: lisinopril 10mg daily"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_past_medical_history_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Past Medical History: hypertension, diabetes"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_family_history_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Family History: father with MI"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched

    def test_reactions_pattern_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SKIP_PATTERNS
        text = "Reactions: hives, swelling"
        matched = any(re.search(p, text, re.IGNORECASE) for p in PRG_SKIP_PATTERNS)
        assert matched


class TestPrgVitalsPatterns:
    """Test Prg-specific vitals patterns."""

    def test_bp_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_BP_PATTERNS
        assert isinstance(PRG_BP_PATTERNS, list)
        assert len(PRG_BP_PATTERNS) >= 2

    def test_bp_spelled_out_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_BP_PATTERNS
        text = "Blood pressure 130/85"
        for pattern, _ in PRG_BP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                assert match.group(1) == '130'
                assert match.group(2) == '85'
                break
        else:
            pytest.fail("No BP pattern matched 'Blood pressure 130/85'")

    def test_hr_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_HR_PATTERNS
        assert isinstance(PRG_HR_PATTERNS, list)
        assert len(PRG_HR_PATTERNS) >= 2

    def test_hr_p_format_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_HR_PATTERNS
        text = "BP 120/80, P 72"
        for pattern, _ in PRG_HR_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                assert match.group(1) == '72'
                break
        else:
            pytest.fail("No HR pattern matched 'P 72'")

    def test_spo2_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SPO2_PATTERNS
        assert isinstance(PRG_SPO2_PATTERNS, list)

    def test_spo2_o2_sat_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_SPO2_PATTERNS
        text = "O2 sat 97"
        for pattern, _ in PRG_SPO2_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                assert match.group(1) == '97'
                break
        else:
            pytest.fail("No SpO2 pattern matched 'O2 sat 97'")

    def test_rr_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_RR_PATTERNS
        assert isinstance(PRG_RR_PATTERNS, list)

    def test_rr_resp_format_matches(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_RR_PATTERNS
        text = "Resp: 18"
        for pattern, _ in PRG_RR_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                assert match.group(1) == '18'
                break
        else:
            pytest.fail("No RR pattern matched 'Resp: 18'")


class TestTempMethodPatterns:
    """Test temperature method extraction patterns."""

    def test_temp_method_patterns_exist(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_TEMP_PATTERNS
        assert isinstance(PRG_TEMP_PATTERNS, list)
        assert len(PRG_TEMP_PATTERNS) >= 2

    def test_temp_method_map_exists(self):
        from module_3_vitals_processing.extractors.prg_patterns import TEMP_METHOD_MAP
        assert isinstance(TEMP_METHOD_MAP, dict)
        assert 'oral' in TEMP_METHOD_MAP
        assert 'temporal' in TEMP_METHOD_MAP
        assert 'rectal' in TEMP_METHOD_MAP

    def test_temp_with_oral_method(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_TEMP_PATTERNS
        text = "Temp 36.8 °C (98.2 °F) (Oral)"
        for pattern, _ in PRG_TEMP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.lastindex >= 3:
                assert 'oral' in match.group(3).lower()
                break

    def test_temp_with_temporal_method(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_TEMP_PATTERNS
        text = "Temp 36.2 °C (97.1 °F) (Temporal)"
        for pattern, _ in PRG_TEMP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match and match.lastindex >= 3:
                assert 'temporal' in match.group(3).lower()
                break

    def test_temp_src_format(self):
        from module_3_vitals_processing.extractors.prg_patterns import PRG_TEMP_PATTERNS
        text = "Temp(Src) 36.7 °C (98 °F) (Oral)"
        matched = False
        for pattern, _ in PRG_TEMP_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                matched = True
                break
        assert matched, "Temp(Src) format should match"


class TestPrgConfig:
    """Test Prg configuration in vitals_config."""

    def test_prg_columns_defined(self):
        from module_3_vitals_processing.config.vitals_config import PRG_COLUMNS
        assert isinstance(PRG_COLUMNS, list)
        assert 'EMPI' in PRG_COLUMNS
        assert 'Report_Text' in PRG_COLUMNS

    def test_prg_input_path_defined(self):
        from module_3_vitals_processing.config.vitals_config import PRG_INPUT_PATH
        assert 'Prg.txt' in str(PRG_INPUT_PATH)

    def test_prg_output_path_defined(self):
        from module_3_vitals_processing.config.vitals_config import PRG_OUTPUT_PATH
        assert 'prg_vitals_raw.parquet' in str(PRG_OUTPUT_PATH)
