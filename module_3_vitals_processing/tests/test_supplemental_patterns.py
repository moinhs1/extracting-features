"""Tests for supplemental vital patterns (O2, BMI)."""
import re
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestO2FlowPatterns:
    """Test O2 flow rate pattern coverage."""

    @pytest.fixture
    def o2_flow_patterns(self):
        from extractors.unified_patterns import O2_FLOW_PATTERNS
        return O2_FLOW_PATTERNS

    def _extract(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence})
                except (ValueError, IndexError):
                    continue
        return results

    def test_liters_nc(self, o2_flow_patterns):
        """2L NC should match."""
        results = self._extract("on 2L NC", o2_flow_patterns)
        assert any(r['value'] == 2 for r in results)

    def test_liters_per_minute(self, o2_flow_patterns):
        """4 L/min should match."""
        results = self._extract("4 L/min via nasal cannula", o2_flow_patterns)
        assert any(r['value'] == 4 for r in results)

    def test_high_flow(self, o2_flow_patterns):
        """40L HFNC should match."""
        results = self._extract("40L HFNC", o2_flow_patterns)
        assert any(r['value'] == 40 for r in results)


class TestO2DevicePatterns:
    """Test O2 device pattern coverage."""

    @pytest.fixture
    def o2_device_patterns(self):
        from extractors.unified_patterns import O2_DEVICE_PATTERNS
        return O2_DEVICE_PATTERNS

    def _extract(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                results.append({'device': match.group(0), 'confidence': confidence})
        return results

    def test_nasal_cannula(self, o2_device_patterns):
        """nasal cannula should match."""
        results = self._extract("on nasal cannula", o2_device_patterns)
        assert len(results) >= 1

    def test_room_air(self, o2_device_patterns):
        """room air should match."""
        results = self._extract("on room air", o2_device_patterns)
        assert len(results) >= 1

    def test_high_flow(self, o2_device_patterns):
        """HFNC should match."""
        results = self._extract("on HFNC", o2_device_patterns)
        assert len(results) >= 1


class TestBMIPatterns:
    """Test BMI pattern coverage."""

    @pytest.fixture
    def bmi_patterns(self):
        from extractors.unified_patterns import BMI_PATTERNS
        return BMI_PATTERNS

    def _extract(self, text, patterns):
        results = []
        for pattern, confidence, tier in patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                try:
                    value = float(match.group(1))
                    results.append({'value': value, 'confidence': confidence})
                except (ValueError, IndexError):
                    continue
        return results

    def test_bmi_standard(self, bmi_patterns):
        """BMI: 24.5 should match."""
        results = self._extract("BMI: 24.5", bmi_patterns)
        assert any(r['value'] == 24.5 for r in results)

    def test_bmi_with_units(self, bmi_patterns):
        """BMI 28.3 kg/m2 should match."""
        results = self._extract("BMI 28.3 kg/m2", bmi_patterns)
        assert any(r['value'] == 28.3 for r in results)
