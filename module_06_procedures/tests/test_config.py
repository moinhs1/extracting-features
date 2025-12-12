"""Tests for procedure configuration."""

import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestPathConfiguration:
    """Test path configuration."""

    def test_project_root_exists(self):
        """Project root path exists."""
        from config.procedure_config import PROJECT_ROOT
        assert PROJECT_ROOT.exists()

    def test_module_root_exists(self):
        """Module root path is defined correctly."""
        from config.procedure_config import MODULE_ROOT
        assert MODULE_ROOT.name == "module_06_procedures"

    def test_prc_file_defined(self):
        """Prc.txt input file path is defined."""
        from config.procedure_config import PRC_FILE
        assert PRC_FILE.name == "Prc.txt"


class TestTemporalConfiguration:
    """Test temporal window configuration."""

    def test_temporal_windows_defined(self):
        """Seven temporal windows are defined."""
        from config.procedure_config import TEMPORAL_CONFIG
        assert len(TEMPORAL_CONFIG.windows) == 7

    def test_provoking_window_range(self):
        """Provoking window is -720h to 0h."""
        from config.procedure_config import TEMPORAL_CONFIG
        window = TEMPORAL_CONFIG.windows['provoking_window']
        assert window == (-720, 0)

    def test_diagnostic_workup_window(self):
        """Diagnostic workup is ±24h."""
        from config.procedure_config import TEMPORAL_CONFIG
        window = TEMPORAL_CONFIG.windows['diagnostic_workup']
        assert window == (-24, 24)


class TestMappingConfiguration:
    """Test code mapping configuration."""

    def test_fuzzy_match_threshold(self):
        """Fuzzy match threshold is 0.85."""
        from config.procedure_config import MAPPING_CONFIG
        assert MAPPING_CONFIG.fuzzy_match_threshold == 0.85

    def test_target_mapping_rate(self):
        """Target CCS mapping rate is 85%."""
        from config.procedure_config import MAPPING_CONFIG
        assert MAPPING_CONFIG.target_mapping_rate == 0.85
