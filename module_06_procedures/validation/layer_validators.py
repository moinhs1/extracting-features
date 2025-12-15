"""
Layer Validators
================

Comprehensive validation suite for all procedure layers.

Validation targets:
- Layer 1: Records loaded (22M), patients in cohort (~8,700), PE linkage >95%
- Layer 2: CCS mapping rate for CPT >95%, overall >85%
- Layer 3: CTA 80-95%, Echo 50-70%, Intubation 5-15%, IVC filter 5-15%, ECMO <2%
- Cross-layer: All patients present in all layers, timestamps aligned
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from config.procedure_config import (
    BRONZE_DIR, SILVER_DIR, GOLD_DIR,
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
        """Add a validation check result."""
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
        """Get summary string."""
        status = "PASS" if self.failed == 0 else "FAIL"
        return f"{self.name}: {status} ({self.passed}/{self.passed + self.failed} checks)"

    def report(self) -> str:
        """Get full report string."""
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
        bronze_path = BRONZE_DIR / "canonical_procedures.parquet"
        df = pd.read_parquet(bronze_path)

        # Check 1: File exists and has data
        result.add_check(
            "Bronze parquet exists with data",
            len(df) > 0,
            f"{len(df):,} records"
        )

        # Check 2: Target records loaded
        target_records = VALIDATION_CONFIG.target_records_loaded
        result.add_check(
            f"Records loaded target ({target_records/1e6:.0f}M)",
            len(df) >= target_records * 0.8,  # Allow 80% of target
            f"Actual: {len(df):,} records"
        )

        # Check 3: Patient coverage
        n_patients = df['empi'].nunique()
        target_patients = VALIDATION_CONFIG.target_patients_in_cohort
        result.add_check(
            f"Patients in cohort (target: {target_patients})",
            n_patients >= target_patients * 0.9,  # Allow 90% of target
            f"Actual: {n_patients:,} patients"
        )

        # Check 4: Code type distribution
        if 'code_type' in df.columns:
            code_dist = df['code_type'].value_counts(normalize=True)
            cpt_pct = code_dist.get('CPT', 0)
            result.add_check(
                "CPT codes are majority (>50%)",
                cpt_pct > 0.5,
                f"CPT: {cpt_pct*100:.1f}%"
            )

        # Check 5: PE time linkage
        if 'hours_from_pe' in df.columns:
            pe_linked = df['hours_from_pe'].notna().mean()
            target = VALIDATION_CONFIG.target_pe_linkage_rate
            result.add_check(
                f"PE time linkage ≥{target*100:.0f}%",
                pe_linked >= target,
                f"Actual: {pe_linked*100:.1f}%"
            )

        # Check 6: Temporal flags exist
        temporal_flags = [
            'is_lifetime_history', 'is_provoking_window',
            'is_diagnostic_workup', 'is_initial_treatment'
        ]
        has_flags = all(flag in df.columns for flag in temporal_flags)
        result.add_check(
            "Temporal flags computed",
            has_flags,
            f"Found: {sum(flag in df.columns for flag in temporal_flags)}/7"
        )

    except FileNotFoundError:
        result.add_check("Bronze layer accessible", False, "File not found")
    except Exception as e:
        result.add_check("Bronze layer accessible", False, str(e))

    return result


def validate_silver() -> ValidationResult:
    """Validate Silver (Mapped) outputs."""
    result = ValidationResult("Silver: Code Mapping")

    try:
        silver_path = SILVER_DIR / "mapped_procedures.parquet"
        df = pd.read_parquet(silver_path)

        # Check 1: Has data
        result.add_check(
            "Silver parquet exists with data",
            len(df) > 0,
            f"{len(df):,} records"
        )

        # Check 2: Overall CCS mapping rate
        if 'ccs_category' in df.columns:
            mapping_rate = df['ccs_category'].notna().mean()
            target = VALIDATION_CONFIG.target_ccs_mapping_overall
            result.add_check(
                f"Overall CCS mapping rate ≥{target*100:.0f}%",
                mapping_rate >= target,
                f"Actual: {mapping_rate*100:.1f}%"
            )

        # Check 3: CPT-specific mapping rate
        if 'code_type' in df.columns and 'ccs_category' in df.columns:
            cpt_df = df[df['code_type'] == 'CPT']
            if len(cpt_df) > 0:
                cpt_mapping = cpt_df['ccs_category'].notna().mean()
                target = VALIDATION_CONFIG.target_ccs_mapping_cpt
                result.add_check(
                    f"CPT CCS mapping rate ≥{target*100:.0f}%",
                    cpt_mapping >= target,
                    f"Actual: {cpt_mapping*100:.1f}%"
                )

        # Check 4: Mapping methods distributed
        if 'mapping_method' in df.columns:
            methods = df['mapping_method'].value_counts()
            result.add_check(
                "Multiple mapping methods used",
                len(methods) > 1,
                f"Methods: {list(methods.index)}"
            )

    except FileNotFoundError:
        result.add_check("Silver layer accessible", False, "File not found")
    except Exception as e:
        result.add_check("Silver layer accessible", False, str(e))

    return result


def validate_layer2() -> ValidationResult:
    """Validate Layer 2 (CCS Indicators) outputs."""
    result = ValidationResult("Layer 2: CCS Indicators")

    try:
        ccs_path = GOLD_DIR / "ccs_indicators" / "ccs_indicators.parquet"
        df = pd.read_parquet(ccs_path)

        # Check 1: Has data
        result.add_check(
            "CCS indicators exist",
            len(df) > 0,
            f"{len(df):,} records"
        )

        # Check 2: CCS columns present
        ccs_cols = [c for c in df.columns if c.startswith('ccs_')]
        result.add_check(
            "CCS category columns exist (>50)",
            len(ccs_cols) > 50,
            f"Actual: {len(ccs_cols)} CCS columns"
        )

        # Check 3: Temporal categories present
        if 'temporal_category' in df.columns:
            categories = df['temporal_category'].unique()
            result.add_check(
                "Multiple temporal categories present",
                len(categories) >= 4,
                f"Categories: {list(categories)}"
            )

        # Check 4: Patient count reasonable
        if 'empi' in df.columns:
            n_patients = df['empi'].nunique()
            result.add_check(
                "Patient count reasonable (>1000)",
                n_patients > 1000,
                f"Patients: {n_patients:,}"
            )

    except FileNotFoundError:
        result.add_check("Layer 2 accessible", False, "File not found")
    except Exception as e:
        result.add_check("Layer 2 accessible", False, str(e))

    return result


def validate_layer3() -> ValidationResult:
    """Validate Layer 3 (PE Features) outputs."""
    result = ValidationResult("Layer 3: PE Features")

    try:
        pe_path = GOLD_DIR / "pe_procedure_features" / "pe_features.parquet"
        df = pd.read_parquet(pe_path)

        # Check 1: Has data
        result.add_check(
            "PE features exist",
            len(df) > 0,
            f"{len(df):,} patients"
        )

        # Check 2: CTA performed rate
        if 'cta_performed' in df.columns:
            cta_rate = df['cta_performed'].mean()
            target_low, target_high = VALIDATION_CONFIG.expected_cta_rate
            result.add_check(
                f"CTA rate in expected range ({target_low*100:.0f}-{target_high*100:.0f}%)",
                target_low <= cta_rate <= target_high,
                f"Actual: {cta_rate*100:.1f}%"
            )

        # Check 3: Echo performed rate
        if 'echo_performed' in df.columns:
            echo_rate = df['echo_performed'].mean()
            target_low, target_high = VALIDATION_CONFIG.expected_echo_rate
            result.add_check(
                f"Echo rate in expected range ({target_low*100:.0f}-{target_high*100:.0f}%)",
                target_low <= echo_rate <= target_high,
                f"Actual: {echo_rate*100:.1f}%"
            )

        # Check 4: Intubation rate
        if 'intubation_performed' in df.columns:
            intub_rate = df['intubation_performed'].mean()
            target_low, target_high = VALIDATION_CONFIG.expected_intubation_rate
            result.add_check(
                f"Intubation rate in expected range ({target_low*100:.0f}-{target_high*100:.0f}%)",
                target_low <= intub_rate <= target_high,
                f"Actual: {intub_rate*100:.1f}%"
            )

        # Check 5: IVC filter rate
        if 'ivc_filter_placed' in df.columns:
            filter_rate = df['ivc_filter_placed'].mean()
            target_low, target_high = VALIDATION_CONFIG.expected_ivc_filter_rate
            result.add_check(
                f"IVC filter rate in expected range ({target_low*100:.0f}-{target_high*100:.0f}%)",
                target_low <= filter_rate <= target_high,
                f"Actual: {filter_rate*100:.1f}%"
            )

        # Check 6: ECMO rate
        if 'ecmo_initiated' in df.columns:
            ecmo_rate = df['ecmo_initiated'].mean()
            target_low, target_high = VALIDATION_CONFIG.expected_ecmo_rate
            result.add_check(
                f"ECMO rate in expected range ({target_low*100:.1f}-{target_high*100:.0f}%)",
                target_low <= ecmo_rate <= target_high,
                f"Actual: {ecmo_rate*100:.2f}%"
            )

    except FileNotFoundError:
        result.add_check("Layer 3 accessible", False, "File not found")
    except Exception as e:
        result.add_check("Layer 3 accessible", False, str(e))

    return result


def validate_cross_layer(
    layer1_df=None, layer2_df=None, layer3_df=None
) -> ValidationResult:
    """Validate cross-layer consistency."""
    result = ValidationResult("Cross-Layer Consistency")

    try:
        # Load all layers if not provided
        if layer1_df is None:
            bronze_path = BRONZE_DIR / "canonical_procedures.parquet"
            layer1_df = pd.read_parquet(bronze_path)

        if layer2_df is None:
            silver_path = SILVER_DIR / "mapped_procedures.parquet"
            layer2_df = pd.read_parquet(silver_path)

        if layer3_df is None:
            ccs_path = GOLD_DIR / "ccs_indicators" / "ccs_indicators.parquet"
            layer3_df = pd.read_parquet(ccs_path)

        # Check 1: Same patients in bronze and silver
        bronze_patients = set(layer1_df['empi'].unique())
        silver_patients = set(layer2_df['empi'].unique())

        result.add_check(
            "Bronze and Silver have same patients",
            bronze_patients == silver_patients,
            f"Bronze: {len(bronze_patients):,}, Silver: {len(silver_patients):,}"
        )

        # Check 2: CCS patients subset of silver
        ccs_patients = set(layer3_df['empi'].unique())
        missing = ccs_patients - silver_patients
        result.add_check(
            "All CCS patients present in Silver",
            len(missing) == 0,
            f"Missing: {len(missing)}, Total CCS: {len(ccs_patients):,}"
        )

        # Check 3: Timestamps aligned (all use same PE Time Zero)
        if 'hours_from_pe' in layer1_df.columns and 'hours_from_pe' in layer2_df.columns:
            # Sample check - pick a patient and verify time alignment
            sample_empi = list(bronze_patients)[0] if bronze_patients else None
            if sample_empi:
                bronze_times = layer1_df[layer1_df['empi'] == sample_empi]['hours_from_pe']
                silver_times = layer2_df[layer2_df['empi'] == sample_empi]['hours_from_pe']
                times_aligned = bronze_times.equals(silver_times)
                result.add_check(
                    "Timestamps aligned (sample check)",
                    times_aligned,
                    f"Sample patient: {sample_empi}"
                )

    except FileNotFoundError as e:
        result.add_check("Cross-layer check failed", False, f"File not found: {e}")
    except Exception as e:
        result.add_check("Cross-layer check failed", False, str(e))

    return result


def run_full_validation() -> List[ValidationResult]:
    """Run all validation checks."""
    print("=" * 60)
    print("Module 6 Validation Suite")
    print("=" * 60)

    results = [
        validate_layer1(),
        validate_silver(),
        validate_layer2(),
        validate_layer3(),
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
    run_full_validation()
