# tests/test_config.py
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPathConfiguration:
    """Test path configuration."""

    def test_project_root_exists(self):
        """Project root path exists."""
        from config.social_physical_config import PROJECT_ROOT
        assert PROJECT_ROOT.exists()

    def test_module_root_exists(self):
        """Module root path is defined correctly."""
        from config.social_physical_config import MODULE_ROOT
        assert MODULE_ROOT.name == "module_07_social_physical_history"

    def test_phy_file_defined(self):
        """Phy.txt input file path is defined."""
        from config.social_physical_config import PHY_FILE
        assert PHY_FILE.name == "Phy.txt"


class TestStalenessThresholds:
    """Test staleness thresholds are defined."""

    def test_bmi_staleness(self):
        """BMI staleness is 180 days."""
        from config.social_physical_config import STALENESS_THRESHOLDS
        assert STALENESS_THRESHOLDS['bmi'] == 180

    def test_weight_staleness(self):
        """Weight staleness is 90 days."""
        from config.social_physical_config import STALENESS_THRESHOLDS
        assert STALENESS_THRESHOLDS['weight'] == 90

    def test_ivdu_never_expires(self):
        """IVDU staleness is infinity (never expires)."""
        from config.social_physical_config import STALENESS_THRESHOLDS
        assert STALENESS_THRESHOLDS['ivdu'] == float('inf')
