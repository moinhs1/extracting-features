"""Tests for temporal_aligner module."""
import pytest
from datetime import datetime, timedelta
from processing.temporal_aligner import calculate_hours_from_pe


class TestHoursFromPE:
    """Tests for PE-relative time calculation."""

    def test_exact_pe_time_is_zero(self):
        """Timestamp at PE index time is hour 0."""
        pe_time = datetime(2023, 6, 15, 10, 30, 0)
        vital_time = datetime(2023, 6, 15, 10, 30, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == 0.0

    def test_one_hour_after_pe(self):
        """One hour after PE is +1.0."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = datetime(2023, 6, 15, 11, 0, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == 1.0

    def test_one_hour_before_pe(self):
        """One hour before PE is -1.0."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = datetime(2023, 6, 15, 9, 0, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == -1.0

    def test_30_minutes_is_half_hour(self):
        """30 minutes is 0.5 hours."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = datetime(2023, 6, 15, 10, 30, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == 0.5

    def test_24_hours_before_pe(self):
        """24 hours before PE is -24.0."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = datetime(2023, 6, 14, 10, 0, 0)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == -24.0

    def test_720_hours_after_pe(self):
        """30 days after PE is +720.0."""
        pe_time = datetime(2023, 6, 15, 10, 0, 0)
        vital_time = pe_time + timedelta(days=30)
        result = calculate_hours_from_pe(vital_time, pe_time)
        assert result == 720.0


class TestWindowFiltering:
    """Tests for temporal window filtering."""

    def test_within_window(self):
        """Value at hour 0 is within default window."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(0.0) is True

    def test_at_window_start(self):
        """Value at -24h is within window (inclusive)."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(-24.0) is True

    def test_at_window_end(self):
        """Value at +720h is within window (inclusive)."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(720.0) is True

    def test_before_window(self):
        """Value at -25h is outside window."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(-25.0) is False

    def test_after_window(self):
        """Value at +721h is outside window."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(721.0) is False

    def test_custom_window(self):
        """Custom window boundaries work."""
        from processing.temporal_aligner import is_within_window
        assert is_within_window(5.0, min_hours=-10, max_hours=10) is True
        assert is_within_window(15.0, min_hours=-10, max_hours=10) is False


class TestHourBucket:
    """Tests for hour bucket assignment."""

    def test_hour_zero_bucket(self):
        """Hours 0.0 to 0.99 go in bucket 0."""
        from processing.temporal_aligner import assign_hour_bucket
        assert assign_hour_bucket(0.0) == 0
        assert assign_hour_bucket(0.5) == 0
        assert assign_hour_bucket(0.99) == 0

    def test_hour_one_bucket(self):
        """Hours 1.0 to 1.99 go in bucket 1."""
        from processing.temporal_aligner import assign_hour_bucket
        assert assign_hour_bucket(1.0) == 1
        assert assign_hour_bucket(1.5) == 1

    def test_negative_hour_bucket(self):
        """Negative hours floor correctly."""
        from processing.temporal_aligner import assign_hour_bucket
        assert assign_hour_bucket(-0.5) == -1
        assert assign_hour_bucket(-1.0) == -1
        assert assign_hour_bucket(-1.5) == -2
        assert assign_hour_bucket(-24.0) == -24

    def test_max_hour_bucket(self):
        """Hour 720 is in bucket 720."""
        from processing.temporal_aligner import assign_hour_bucket
        assert assign_hour_bucket(720.0) == 720
        assert assign_hour_bucket(720.5) == 720
