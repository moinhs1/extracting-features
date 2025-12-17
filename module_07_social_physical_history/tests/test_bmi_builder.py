# tests/test_bmi_builder.py
import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBMIClassification:
    """Test BMI category classification."""

    def test_underweight(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(17.0) == 'underweight'

    def test_normal(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(22.0) == 'normal'

    def test_overweight(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(27.0) == 'overweight'

    def test_obese_1(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(32.0) == 'obese_1'

    def test_obese_2(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(37.0) == 'obese_2'

    def test_obese_3(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(42.0) == 'obese_3'

    def test_none_returns_unknown(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_bmi(None) == 'unknown'


class TestTrendClassification:
    """Test trend direction classification."""

    def test_increasing(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_trend(10.0) == 'increasing'

    def test_decreasing(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_trend(-10.0) == 'decreasing'

    def test_stable(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_trend(2.0) == 'stable'

    def test_none_returns_unknown(self):
        from transformers.bmi_builder import BMIBuilder
        builder = BMIBuilder(pd.DataFrame(), {})
        assert builder.classify_trend(None) == 'unknown'
