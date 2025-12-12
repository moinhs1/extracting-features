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
