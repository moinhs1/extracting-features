"""
Tests for unit conversion system.
"""

import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent / 'module_2_laboratory_processing'))

from unit_converter import UnitConverter


class TestUnitConverter:
    """Test suite for UnitConverter."""

    def test_glucose_conversion_mmol_to_mgdl(self):
        """Test glucose conversion from mmol/L to mg/dL."""
        converter = UnitConverter()

        # 5.5 mmol/L = 99.099 mg/dL
        converted, target_unit, success = converter.convert_value(
            value=5.5,
            test_component='glucose',
            source_unit='mmol/L'
        )

        assert success is True
        assert target_unit == 'mg/dL'
        assert abs(converted - 99.099) < 0.01

    def test_glucose_no_conversion_needed(self):
        """Test glucose when already in mg/dL."""
        converter = UnitConverter()

        converted, target_unit, success = converter.convert_value(
            value=100.0,
            test_component='glucose',
            source_unit='mg/dL'
        )

        assert success is True
        assert target_unit == 'mg/dL'
        assert converted == 100.0

    def test_unknown_component(self):
        """Test conversion for unknown test component."""
        converter = UnitConverter()

        converted, target_unit, success = converter.convert_value(
            value=5.0,
            test_component='unknown_test',
            source_unit='unknown_unit'
        )

        assert success is False
        assert converted == 5.0
        assert target_unit == 'unknown_unit'

    def test_creatinine_conversion(self):
        """Test creatinine conversion from µmol/L to mg/dL."""
        converter = UnitConverter()

        # 88.4 µmol/L = 1.0 mg/dL
        converted, target_unit, success = converter.convert_value(
            value=88.4,
            test_component='creatinine',
            source_unit='µmol/L'
        )

        assert success is True
        assert target_unit == 'mg/dL'
        assert abs(converted - 0.999) < 0.01

    def test_cholesterol_conversion(self):
        """Test cholesterol conversion from mmol/L to mg/dL."""
        converter = UnitConverter()

        # 5.0 mmol/L = 193.35 mg/dL
        converted, target_unit, success = converter.convert_value(
            value=5.0,
            test_component='cholesterol',
            source_unit='mmol/L'
        )

        assert success is True
        assert target_unit == 'mg/dL'
        assert abs(converted - 193.35) < 0.01
