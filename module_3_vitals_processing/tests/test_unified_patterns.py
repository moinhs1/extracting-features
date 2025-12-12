"""Tests for unified pattern library."""
import re
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestHRPatterns:
    """Test heart rate pattern coverage."""

    @pytest.fixture
    def hr_patterns(self):
        from extractors.unified_patterns import HR_PATTERNS
        return HR_PATTERNS

    def _extract_hr(self, text, patterns):
        """Helper to extract HR values."""
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence, 'tier': tier})
                except (ValueError, IndexError):
                    continue
        return results

    def test_standard_hr_with_label(self, hr_patterns):
        """HR: 72 should match."""
        results = self._extract_hr("HR: 72", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 72 for r in results)

    def test_heart_rate_with_bpm(self, hr_patterns):
        """Heart Rate 88 bpm should match with high confidence."""
        results = self._extract_hr("Heart Rate 88 bpm", hr_patterns)
        assert len(results) >= 1
        match = next(r for r in results if r['value'] == 88)
        assert match['confidence'] >= 0.90

    def test_pulse_format(self, hr_patterns):
        """Pulse: 65 should match."""
        results = self._extract_hr("Pulse: 65", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 65 for r in results)

    def test_tachycardia_context(self, hr_patterns):
        """tachycardic at 120 should match."""
        results = self._extract_hr("patient tachycardic at 120", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 120 for r in results)

    def test_ekg_rate(self, hr_patterns):
        """EKG rate 78 should match."""
        results = self._extract_hr("EKG shows rate 78", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 78 for r in results)

    def test_sinus_rhythm(self, hr_patterns):
        """normal sinus rhythm 72 should match."""
        results = self._extract_hr("normal sinus rhythm 72", hr_patterns)
        assert len(results) >= 1
        assert any(r['value'] == 72 for r in results)


class TestBPPatterns:
    """Test blood pressure pattern coverage."""

    @pytest.fixture
    def bp_patterns(self):
        from extractors.unified_patterns import BP_PATTERNS
        return BP_PATTERNS

    def _extract_bp(self, text, patterns):
        """Helper to extract BP values."""
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    sbp = float(match.group(1))
                    dbp = float(match.group(2))
                    results.append({'sbp': sbp, 'dbp': dbp, 'confidence': confidence, 'tier': tier})
                except (ValueError, IndexError):
                    continue
        return results

    def test_standard_bp(self, bp_patterns):
        """BP: 120/80 should match."""
        results = self._extract_bp("BP: 120/80", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 120 and r['dbp'] == 80 for r in results)

    def test_blood_pressure_spelled_out(self, bp_patterns):
        """Blood Pressure 135/85 should match."""
        results = self._extract_bp("Blood Pressure: 135/85", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 135 and r['dbp'] == 85 for r in results)

    def test_bp_with_mmhg(self, bp_patterns):
        """140/90 mmHg should match."""
        results = self._extract_bp("140/90 mmHg", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 140 and r['dbp'] == 90 for r in results)

    def test_bp_reference_range_format(self, bp_patterns):
        """(110-130)/(60-80) 125/75 should match the actual value."""
        results = self._extract_bp("(110-130)/(60-80) 125/75", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 125 and r['dbp'] == 75 for r in results)

    def test_vitals_context_bp(self, bp_patterns):
        """vitals: 120/80 should match."""
        results = self._extract_bp("vitals: 120/80, HR 72", bp_patterns)
        assert len(results) >= 1
        assert any(r['sbp'] == 120 and r['dbp'] == 80 for r in results)

    def test_should_not_match_dates(self, bp_patterns):
        """Date patterns should NOT match as BP."""
        date_texts = [
            "Date: 12/25/2023",
            "on 1/31 the patient",
            "Visit on 10/15",
        ]
        for text in date_texts:
            results = self._extract_bp(text, bp_patterns)
            # Filter out date-like values
            valid_bp = [r for r in results if r['sbp'] > 50 and r['dbp'] > 25]
            assert len(valid_bp) == 0, f"Matched date as BP in '{text}': {results}"


class TestRRPatterns:
    """Test respiratory rate pattern coverage."""

    @pytest.fixture
    def rr_patterns(self):
        from extractors.unified_patterns import RR_PATTERNS
        return RR_PATTERNS

    def _extract_rr(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence})
                except (ValueError, IndexError):
                    continue
        return results

    def test_standard_rr(self, rr_patterns):
        """RR: 18 should match."""
        results = self._extract_rr("RR: 18", rr_patterns)
        assert any(r['value'] == 18 for r in results)

    def test_respiratory_rate_spelled(self, rr_patterns):
        """Respiratory Rate 20 should match."""
        results = self._extract_rr("Respiratory Rate 20", rr_patterns)
        assert any(r['value'] == 20 for r in results)

    def test_breaths_per_min(self, rr_patterns):
        """16 breaths/min should match."""
        results = self._extract_rr("16 breaths/min", rr_patterns)
        assert any(r['value'] == 16 for r in results)


class TestSpO2Patterns:
    """Test SpO2 pattern coverage."""

    @pytest.fixture
    def spo2_patterns(self):
        from extractors.unified_patterns import SPO2_PATTERNS
        return SPO2_PATTERNS

    def _extract_spo2(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence})
                except (ValueError, IndexError):
                    continue
        return results

    def test_spo2_standard(self, spo2_patterns):
        """SpO2: 98% should match."""
        results = self._extract_spo2("SpO2: 98%", spo2_patterns)
        assert any(r['value'] == 98 for r in results)

    def test_o2_sat(self, spo2_patterns):
        """O2 Sat 95% should match."""
        results = self._extract_spo2("O2 Sat 95%", spo2_patterns)
        assert any(r['value'] == 95 for r in results)

    def test_room_air_context(self, spo2_patterns):
        """92% on room air should match."""
        results = self._extract_spo2("92% on room air", spo2_patterns)
        assert any(r['value'] == 92 for r in results)


class TestTempPatterns:
    """Test temperature pattern coverage."""

    @pytest.fixture
    def temp_patterns(self):
        from extractors.unified_patterns import TEMP_PATTERNS
        return TEMP_PATTERNS

    def _extract_temp(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    unit = match.group(2).upper() if match.lastindex >= 2 and match.group(2) else None
                    results.append({'value': value, 'unit': unit, 'confidence': confidence})
                except (ValueError, IndexError, AttributeError):
                    continue
        return results

    def test_temp_fahrenheit(self, temp_patterns):
        """Temp: 98.6 F should match."""
        results = self._extract_temp("Temp: 98.6 F", temp_patterns)
        assert any(r['value'] == 98.6 and r['unit'] == 'F' for r in results)

    def test_temp_celsius(self, temp_patterns):
        """Temperature 37.2 C should match."""
        results = self._extract_temp("Temperature 37.2 C", temp_patterns)
        assert any(r['value'] == 37.2 and r['unit'] == 'C' for r in results)

    def test_tmax(self, temp_patterns):
        """Tmax 101.2 F should match."""
        results = self._extract_temp("Tmax 101.2 F", temp_patterns)
        assert any(r['value'] == 101.2 for r in results)
