"""Tests for temporal classification relative to PE index."""
import pytest
from datetime import datetime, timedelta
from processing.temporal_classifier import (
    calculate_days_from_pe,
    classify_temporal_category,
    get_temporal_flags,
)


class TestDaysFromPE:
    """Tests for PE-relative day calculation."""

    def test_same_day_is_zero(self):
        """Same day is 0 days from PE."""
        pe_date = datetime(2023, 6, 15)
        dx_date = datetime(2023, 6, 15)
        assert calculate_days_from_pe(dx_date, pe_date) == 0

    def test_one_day_after(self):
        """One day after PE is +1."""
        pe_date = datetime(2023, 6, 15)
        dx_date = datetime(2023, 6, 16)
        assert calculate_days_from_pe(dx_date, pe_date) == 1

    def test_one_day_before(self):
        """One day before PE is -1."""
        pe_date = datetime(2023, 6, 15)
        dx_date = datetime(2023, 6, 14)
        assert calculate_days_from_pe(dx_date, pe_date) == -1

    def test_30_days_before(self):
        """30 days before PE is -30."""
        pe_date = datetime(2023, 6, 15)
        dx_date = pe_date - timedelta(days=30)
        assert calculate_days_from_pe(dx_date, pe_date) == -30


class TestTemporalCategory:
    """Tests for temporal category assignment."""

    def test_preexisting_remote(self):
        """More than 30 days before PE is preexisting_remote."""
        assert classify_temporal_category(-31) == "preexisting_remote"
        assert classify_temporal_category(-365) == "preexisting_remote"

    def test_preexisting_recent(self):
        """8-30 days before PE is preexisting_recent."""
        assert classify_temporal_category(-30) == "preexisting_recent"
        assert classify_temporal_category(-8) == "preexisting_recent"

    def test_antecedent(self):
        """1-7 days before PE is antecedent."""
        assert classify_temporal_category(-7) == "antecedent"
        assert classify_temporal_category(-1) == "antecedent"

    def test_index_concurrent(self):
        """Day 0 and 1 are index_concurrent."""
        assert classify_temporal_category(0) == "index_concurrent"
        assert classify_temporal_category(1) == "index_concurrent"

    def test_early_complication(self):
        """Days 2-7 after PE are early_complication."""
        assert classify_temporal_category(2) == "early_complication"
        assert classify_temporal_category(7) == "early_complication"

    def test_late_complication(self):
        """Days 8-30 after PE are late_complication."""
        assert classify_temporal_category(8) == "late_complication"
        assert classify_temporal_category(30) == "late_complication"

    def test_follow_up(self):
        """More than 30 days after PE is follow_up."""
        assert classify_temporal_category(31) == "follow_up"
        assert classify_temporal_category(365) == "follow_up"


class TestTemporalFlags:
    """Tests for temporal flag generation."""

    def test_preexisting_flag(self):
        """is_preexisting is True for days < -30."""
        flags = get_temporal_flags(-31)
        assert flags["is_preexisting"] is True
        assert flags["is_complication"] is False

    def test_complication_flag(self):
        """is_complication is True for days > 1."""
        flags = get_temporal_flags(5)
        assert flags["is_complication"] is True
        assert flags["is_preexisting"] is False

    def test_index_concurrent_flag(self):
        """is_index_concurrent is True for days 0-1."""
        flags = get_temporal_flags(0)
        assert flags["is_index_concurrent"] is True
        flags = get_temporal_flags(1)
        assert flags["is_index_concurrent"] is True
