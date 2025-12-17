# tests/test_unit_conversion.py
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestWeightConversion:
    def test_lbs_to_kg(self):
        from utils.unit_conversion import lbs_to_kg
        assert lbs_to_kg(220) == pytest.approx(99.79, rel=0.01)

    def test_kg_to_lbs(self):
        from utils.unit_conversion import kg_to_lbs
        assert kg_to_lbs(100) == pytest.approx(220.46, rel=0.01)


class TestHeightConversion:
    def test_inches_to_cm(self):
        from utils.unit_conversion import inches_to_cm
        assert inches_to_cm(70) == pytest.approx(177.8, rel=0.01)

    def test_cm_to_m(self):
        from utils.unit_conversion import cm_to_m
        assert cm_to_m(175) == 1.75


class TestBMICalculation:
    def test_calculate_bmi(self):
        from utils.unit_conversion import calculate_bmi
        # 80kg, 1.75m -> BMI = 80 / 1.75^2 = 26.12
        assert calculate_bmi(80, 1.75) == pytest.approx(26.12, rel=0.01)

    def test_bmi_with_zero_height_returns_none(self):
        from utils.unit_conversion import calculate_bmi
        assert calculate_bmi(80, 0) is None


class TestBSACalculation:
    def test_bsa_dubois(self):
        from utils.unit_conversion import calculate_bsa_dubois
        # 70kg, 170cm -> ~1.81 m²
        assert calculate_bsa_dubois(70, 170) == pytest.approx(1.81, rel=0.02)

    def test_bsa_mosteller(self):
        from utils.unit_conversion import calculate_bsa_mosteller
        # 70kg, 170cm -> ~1.82 m²
        assert calculate_bsa_mosteller(70, 170) == pytest.approx(1.82, rel=0.02)
