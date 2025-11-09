#!/usr/bin/env python3
"""
Analyze lab coverage across patients (v2 - fixed version).

This script:
1. Loads lab features CSV
2. Calculates overall lab coverage (% patients with at least one lab)
3. Breaks down coverage by individual lab tests
4. Shows top 20 labs by patient count
"""

import pandas as pd
import numpy as np
from pathlib import Path


def analyze_lab_coverage(features_file):
    """Analyze lab coverage across patients."""

    print("="*80)
    print("LAB COVERAGE ANALYSIS")
    print("="*80)

    # Load lab features
    print(f"\nLoading {features_file}...")
    df = pd.read_csv(features_file)
    total_patients = len(df)
    print(f"  Total patients: {total_patients}")

    # Extract unique lab tests from column names
    # Columns are formatted as: {test_name}_{phase}_{feature}
    columns = df.columns.tolist()
    columns = [c for c in columns if c != 'patient_id']

    # Extract test names
    test_names = set()
    phases = ['BASELINE', 'ACUTE', 'SUBACUTE', 'RECOVERY']

    for col in columns:
        for phase in phases:
            if f'_{phase}_' in col:
                test_name = col.split(f'_{phase}_')[0]
                test_names.add(test_name)
                break

    test_names = sorted(test_names)
    print(f"  Total unique lab tests: {len(test_names)}")

    # Calculate overall coverage (patients with at least one lab measurement)
    # Check if ANY feature column has a non-null value
    patients_with_labs = 0
    for idx, row in df.iterrows():
        has_lab = False
        for col in columns:
            if pd.notna(row[col]) and row[col] != '' and row[col] != 0:
                # Found a non-null, non-zero value
                has_lab = True
                break
        if has_lab:
            patients_with_labs += 1

    overall_coverage_pct = (patients_with_labs / total_patients) * 100

    print("\n" + "="*80)
    print("OVERALL LAB COVERAGE")
    print("="*80)
    print(f"  Patients with at least one lab: {patients_with_labs}/{total_patients} ({overall_coverage_pct:.1f}%)")
    print(f"  Patients with no labs: {total_patients - patients_with_labs}/{total_patients} ({100 - overall_coverage_pct:.1f}%)")

    # Calculate coverage for each individual lab test
    print("\n" + "="*80)
    print("CALCULATING COVERAGE BY LAB TEST")
    print("="*80)

    lab_coverage = []

    for test in test_names:
        # Count patients with at least one non-null value for this test
        patients_with_test = 0
        total_measurements = 0

        for idx, row in df.iterrows():
            has_measurement = False
            patient_measurements = 0

            # Check all feature columns for this test
            for phase in phases:
                # Look for _first column (always exists if patient has data)
                first_col = f"{test}_{phase}_first"
                count_col = f"{test}_{phase}_count"

                if first_col in df.columns:
                    first_val = row[first_col]
                    if pd.notna(first_val) and first_val != '':
                        has_measurement = True

                if count_col in df.columns:
                    count = row[count_col]
                    if pd.notna(count) and count > 0:
                        patient_measurements += int(count)

            if has_measurement:
                patients_with_test += 1
                total_measurements += patient_measurements

        coverage_pct = (patients_with_test / total_patients) * 100

        lab_coverage.append({
            'test_name': test,
            'patient_count': patients_with_test,
            'coverage_pct': coverage_pct,
            'total_measurements': total_measurements,
            'avg_measurements_per_patient': total_measurements / patients_with_test if patients_with_test > 0 else 0
        })

    # Convert to DataFrame and sort by patient count
    coverage_df = pd.DataFrame(lab_coverage)
    coverage_df = coverage_df.sort_values('patient_count', ascending=False)

    # Display top 20 labs
    print("\n" + "="*80)
    print("TOP 20 LABS BY PATIENT COVERAGE")
    print("="*80)
    print()
    print(f"{'Rank':<6} {'Lab Test':<50} {'Patients':<12} {'Coverage':<12} {'Measurements':<15} {'Avg/Patient':<12}")
    print("-" * 115)

    for rank, (idx, row) in enumerate(coverage_df.head(20).iterrows(), 1):
        test_name = row['test_name']
        patient_count = int(row['patient_count'])
        coverage_pct = row['coverage_pct']
        total_measurements = int(row['total_measurements'])
        avg_measurements = row['avg_measurements_per_patient']

        # Truncate test name if too long
        if len(test_name) > 48:
            test_name = test_name[:45] + '...'

        print(f"{rank:<6} {test_name:<50} {patient_count:<12} {coverage_pct:>6.1f}%     {total_measurements:<15} {avg_measurements:>6.1f}")

    # Show tests with 100% coverage
    full_coverage = coverage_df[coverage_df['coverage_pct'] == 100.0]
    if len(full_coverage) > 0:
        print("\n" + "="*80)
        print(f"LABS WITH 100% PATIENT COVERAGE ({len(full_coverage)} tests)")
        print("="*80)
        for idx, row in full_coverage.iterrows():
            print(f"  - {row['test_name']}: {int(row['total_measurements'])} measurements ({row['avg_measurements_per_patient']:.1f} avg per patient)")

    # Show tests with >50% coverage
    high_coverage = coverage_df[coverage_df['coverage_pct'] >= 50.0]
    if len(high_coverage) > 0:
        print("\n" + "="*80)
        print(f"LABS WITH >50% COVERAGE ({len(high_coverage)} tests)")
        print("="*80)
        for idx, row in high_coverage.iterrows():
            print(f"  - {row['test_name']}: {int(row['patient_count'])} patients ({row['coverage_pct']:.1f}%)")

    # Show tests with <10% coverage (rare tests)
    rare_tests = coverage_df[coverage_df['coverage_pct'] < 10.0]
    if len(rare_tests) > 0:
        print("\n" + "="*80)
        print(f"RARE LABS (<10% COVERAGE) - {len(rare_tests)} tests")
        print("="*80)
        for idx, row in rare_tests.head(10).iterrows():
            print(f"  - {row['test_name']}: {int(row['patient_count'])} patients ({row['coverage_pct']:.1f}%)")
        if len(rare_tests) > 10:
            print(f"  ... and {len(rare_tests) - 10} more")

    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print(f"  Total lab tests: {len(coverage_df)}")
    print(f"  Tests with 100% coverage: {len(full_coverage)}")
    print(f"  Tests with >90% coverage: {len(coverage_df[coverage_df['coverage_pct'] >= 90.0])}")
    print(f"  Tests with >50% coverage: {len(coverage_df[coverage_df['coverage_pct'] >= 50.0])}")
    print(f"  Tests with <10% coverage: {len(rare_tests)}")
    print(f"  Median coverage: {coverage_df['coverage_pct'].median():.1f}%")
    print(f"  Mean coverage: {coverage_df['coverage_pct'].mean():.1f}%")

    # Save detailed coverage report
    output_file = features_file.parent / f"{features_file.stem}_coverage_report_v2.csv"
    coverage_df.to_csv(output_file, index=False)
    print(f"\n✓ Detailed coverage report saved to: {output_file}")

    return coverage_df


def main():
    # Path to lab features file
    features_file = Path('outputs/test_n10_lab_features.csv')

    if not features_file.exists():
        print(f"ERROR: Lab features file not found: {features_file}")
        print("Please run Phase 2 first: python module_02_laboratory_processing.py --phase2 --test --n=10")
        return

    # Analyze coverage
    coverage_df = analyze_lab_coverage(features_file)


if __name__ == '__main__':
    main()
