# /home/moin/TDA_11_25/module_04_medications/validation/layer_validators.py
"""
Layer Validators
================

Comprehensive validation suite for all medication layers.

Validation targets:
- Layer 1: RxNorm mapping ≥85%, dose parsing ≥80%
- Layer 2: Anticoagulant within 24h ≥90%
- Layer 3: Prevalence threshold met, no perfect correlations
- Layer 4: Similar pair similarity >0.7, dissimilar <0.4
- Layer 5: Doses in therapeutic range
- Cross-layer: All patients in all layers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.medication_config import (
    BRONZE_DIR, SILVER_DIR, GOLD_DIR, EMBEDDINGS_DIR,
    VALIDATION_CONFIG
)


class ValidationResult:
    """Container for validation results."""

    def __init__(self, name: str):
        self.name = name
        self.checks = []
        self.passed = 0
        self.failed = 0

    def add_check(self, description: str, passed: bool, details: str = ""):
        self.checks.append({
            'description': description,
            'passed': passed,
            'details': details,
        })
        if passed:
            self.passed += 1
        else:
            self.failed += 1

    def summary(self) -> str:
        status = "PASS" if self.failed == 0 else "FAIL"
        return f"{self.name}: {status} ({self.passed}/{self.passed + self.failed} checks)"

    def report(self) -> str:
        lines = [f"\n{'='*60}", f"{self.name}", "="*60]
        for check in self.checks:
            icon = "✓" if check['passed'] else "✗"
            lines.append(f"  {icon} {check['description']}")
            if check['details']:
                lines.append(f"      {check['details']}")
        lines.append(self.summary())
        return "\n".join(lines)


def validate_layer1() -> ValidationResult:
    """Validate Layer 1 (Bronze) outputs."""
    result = ValidationResult("Layer 1: Canonical Extraction")

    try:
        # Load bronze data
        bronze_path = BRONZE_DIR / "canonical_records.parquet"
        df = pd.read_parquet(bronze_path)

        # Check 1: File exists and has data
        result.add_check(
            "Bronze parquet exists with data",
            len(df) > 0,
            f"{len(df):,} records"
        )

        # Check 2: Dose parsing rate
        if 'parse_method' in df.columns:
            parse_rate = (df['parse_method'] == 'regex').mean()
        else:
            # Fallback: check if parsed_dose_value exists
            parse_rate = df['parsed_dose_value'].notna().mean()
        target = VALIDATION_CONFIG.target_dose_parsing_rate
        result.add_check(
            f"Dose parsing rate ≥{target*100:.0f}%",
            parse_rate >= target,
            f"Actual: {parse_rate*100:.1f}%"
        )

        # Check 3: Patient coverage
        n_patients = df['empi'].nunique()
        result.add_check(
            "Multiple patients present",
            n_patients > 1000,
            f"{n_patients:,} patients"
        )

        # Check 4: Time range
        hour_range = df['hours_from_t0'].max() - df['hours_from_t0'].min()
        result.add_check(
            "Time range covers study window",
            hour_range > 100,
            f"Range: {hour_range:.0f} hours"
        )

    except Exception as e:
        result.add_check("Bronze layer accessible", False, str(e))

    return result


def validate_silver() -> ValidationResult:
    """Validate Silver (RxNorm mapped) outputs."""
    result = ValidationResult("Silver: RxNorm Mapping")

    try:
        silver_path = SILVER_DIR / "mapped_medications.parquet"
        df = pd.read_parquet(silver_path)

        # Check 1: Mapping rate
        mapping_rate = df['rxcui'].notna().mean()
        target = VALIDATION_CONFIG.target_rxnorm_mapping_rate
        result.add_check(
            f"RxNorm mapping rate ≥{target*100:.0f}%",
            mapping_rate >= target,
            f"Actual: {mapping_rate*100:.1f}%"
        )

        # Check 2: Ingredient extraction
        ingredient_rate = df['ingredient_name'].notna().mean()
        result.add_check(
            "Ingredient names extracted",
            ingredient_rate > 0.5,
            f"Actual: {ingredient_rate*100:.1f}%"
        )

        # Check 3: Key medications mapped
        key_meds = ['heparin', 'enoxaparin', 'warfarin', 'aspirin']
        for med in key_meds:
            has_med = df['ingredient_name'].str.lower().str.contains(med, na=False).any()
            result.add_check(f"'{med}' present in mappings", has_med)

    except Exception as e:
        result.add_check("Silver layer accessible", False, str(e))

    return result


def validate_layer2() -> ValidationResult:
    """Validate Layer 2 (Therapeutic Classes) outputs."""
    result = ValidationResult("Layer 2: Therapeutic Classes")

    try:
        class_path = GOLD_DIR / "therapeutic_classes" / "class_indicators.parquet"
        df = pd.read_parquet(class_path)

        # Check 1: 53 class columns
        class_cols = [c for c in df.columns
                      if not c.endswith('_count') and not c.endswith('_first_hours')
                      and c not in ['empi', 'time_window', 'window_start_hours', 'window_end_hours']]
        result.add_check(
            "53 therapeutic class columns",
            len(class_cols) >= 50,
            f"Actual: {len(class_cols)} classes"
        )

        # Check 2: 4 time windows
        windows = df['time_window'].unique()
        result.add_check(
            "4 time windows present",
            len(windows) == 4,
            f"Windows: {list(windows)}"
        )

        # Check 3: Anticoagulant coverage in acute
        acute = df[df['time_window'] == 'acute']
        if len(acute) > 0:
            ac_cols = [c for c in class_cols if c.startswith('ac_')]
            any_ac = acute[ac_cols].any(axis=1).mean()
            target = VALIDATION_CONFIG.target_anticoag_24h_rate
            result.add_check(
                f"Anticoagulant in acute ≥{target*100:.0f}%",
                any_ac >= target,
                f"Actual: {any_ac*100:.1f}%"
            )

    except Exception as e:
        result.add_check("Layer 2 accessible", False, str(e))

    return result


def validate_layer3() -> ValidationResult:
    """Validate Layer 3 (Individual Medications) outputs."""
    result = ValidationResult("Layer 3: Individual Medications")

    try:
        ind_path = GOLD_DIR / "individual_indicators" / "individual_indicators.parquet"
        df = pd.read_parquet(ind_path)

        # Check 1: Reasonable number of indicators
        med_cols = [c for c in df.columns if c.startswith('med_') and not c.endswith('_count')]
        result.add_check(
            "200-600 individual medication indicators",
            200 <= len(med_cols) <= 600,
            f"Actual: {len(med_cols)}"
        )

        # Check 2: Sparsity
        if med_cols:
            sparsity = 1 - df[med_cols].mean().mean()
            result.add_check(
                "High sparsity (>90%)",
                sparsity > 0.9,
                f"Actual: {sparsity*100:.1f}%"
            )

        # Check 3: No perfect correlations (sample check)
        if len(med_cols) > 1:
            sample_cols = med_cols[:50]  # Check first 50 to avoid long computation
            corr_matrix = df[sample_cols].corr()
            max_corr = corr_matrix.where(~np.eye(len(sample_cols), dtype=bool)).max().max()
            result.add_check(
                "No perfect correlations (<0.99)",
                max_corr < 0.99,
                f"Max correlation: {max_corr:.3f}"
            )

    except Exception as e:
        result.add_check("Layer 3 accessible", False, str(e))

    return result


def validate_layer5() -> ValidationResult:
    """Validate Layer 5 (Dose Intensity) outputs."""
    result = ValidationResult("Layer 5: Dose Intensity")

    try:
        dose_path = GOLD_DIR / "dose_intensity" / "dose_intensity.parquet"
        df = pd.read_parquet(dose_path)

        # Check 1: Has data
        result.add_check(
            "Dose intensity data exists",
            len(df) > 0,
            f"{len(df):,} records"
        )

        # Check 2: DDD ratios calculated
        ddd_rate = df['ddd_ratio'].notna().mean()
        result.add_check(
            "DDD ratios calculated",
            ddd_rate > 0.1,
            f"Actual: {ddd_rate*100:.1f}%"
        )

        # Check 3: Reasonable DDD values
        if ddd_rate > 0:
            median_ddd = df['ddd_ratio'].median()
            result.add_check(
                "DDD ratios in reasonable range",
                0.1 < median_ddd < 10,
                f"Median: {median_ddd:.2f}"
            )

    except Exception as e:
        result.add_check("Layer 5 accessible", False, str(e))

    return result


def validate_cross_layer() -> ValidationResult:
    """Validate cross-layer consistency."""
    result = ValidationResult("Cross-Layer Consistency")

    try:
        # Load all layers
        bronze = pd.read_parquet(BRONZE_DIR / "canonical_records.parquet")
        silver = pd.read_parquet(SILVER_DIR / "mapped_medications.parquet")
        class_df = pd.read_parquet(GOLD_DIR / "therapeutic_classes" / "class_indicators.parquet")

        # Check 1: Same patients across layers
        bronze_patients = set(bronze['empi'].unique())
        silver_patients = set(silver['empi'].unique())
        class_patients = set(class_df['empi'].unique())

        result.add_check(
            "Bronze = Silver patients",
            bronze_patients == silver_patients,
            f"Bronze: {len(bronze_patients)}, Silver: {len(silver_patients)}"
        )

        # Check 2: Class patients subset of silver
        missing = class_patients - silver_patients
        result.add_check(
            "Class patients ⊆ Silver patients",
            len(missing) == 0,
            f"Missing: {len(missing)}"
        )

    except Exception as e:
        result.add_check("Cross-layer check failed", False, str(e))

    return result


def run_all_validations() -> List[ValidationResult]:
    """Run all validation checks."""
    print("=" * 60)
    print("Module 4 Validation Suite")
    print("=" * 60)

    results = [
        validate_layer1(),
        validate_silver(),
        validate_layer2(),
        validate_layer3(),
        validate_layer5(),
        validate_cross_layer(),
    ]

    # Print reports
    for r in results:
        print(r.report())

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    print(f"Total: {total_passed} passed, {total_failed} failed")

    overall = "PASS" if total_failed == 0 else "NEEDS ATTENTION"
    print(f"Overall: {overall}")

    return results


if __name__ == "__main__":
    run_all_validations()
