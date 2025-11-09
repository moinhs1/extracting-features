#!/usr/bin/env python3
"""
Analyze lab coverage for full cohort (~3,565 patients).

This script:
1. Loads lab sequences HDF5 file
2. Calculates overall lab coverage
3. Identifies top labs by patient count
4. Generates comprehensive coverage statistics
"""

import h5py
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime


def analyze_full_cohort_coverage(hdf5_file, outcomes_file):
    """Analyze lab coverage for full cohort."""

    print("="*80)
    print("LAB COVERAGE ANALYSIS - FULL COHORT")
    print("="*80)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Load outcomes to get total patient count
    print("Loading patient outcomes...")
    outcomes_df = pd.read_csv(outcomes_file)
    total_patients = len(outcomes_df)
    print(f"  Total patients in cohort: {total_patients}")
    print()

    # Open HDF5 file
    print(f"Loading lab sequences from {hdf5_file}...")
    with h5py.File(hdf5_file, 'r') as f:
        sequences_grp = f['sequences']
        patients_in_hdf5 = len(sequences_grp.keys())

        print(f"  Patients in HDF5 file: {patients_in_hdf5}")

        # Count patients with labs and collect test coverage
        patients_with_labs = 0
        test_coverage = {}

        print("  Scanning patient lab data...")

        for patient_id in sequences_grp.keys():
            patient_grp = sequences_grp[patient_id]

            if len(patient_grp.keys()) > 0:
                patients_with_labs += 1

                # Count tests for this patient
                for test_name in patient_grp.keys():
                    test_grp = patient_grp[test_name]

                    # Handle nested structure (e.g., erythrocyte/blood)
                    if 'values' in test_grp:
                        # Leaf node - actual test
                        if test_name not in test_coverage:
                            test_coverage[test_name] = {'patients': set(), 'measurements': 0}
                        test_coverage[test_name]['patients'].add(patient_id)
                        test_coverage[test_name]['measurements'] += len(test_grp['values'])
                    else:
                        # Nested group - need to go deeper
                        for subtest_name in test_grp.keys():
                            full_name = f'{test_name}/{subtest_name}'
                            subtest_grp = test_grp[subtest_name]
                            if 'values' in subtest_grp:
                                if full_name not in test_coverage:
                                    test_coverage[full_name] = {'patients': set(), 'measurements': 0}
                                test_coverage[full_name]['patients'].add(patient_id)
                                test_coverage[full_name]['measurements'] += len(subtest_grp['values'])

        print(f"  ✓ Scanned {patients_in_hdf5} patients")
        print()

        # Calculate coverage
        coverage_pct = (patients_with_labs / total_patients) * 100

        print("="*80)
        print("OVERALL LAB COVERAGE")
        print("="*80)
        print(f"  Patients with labs: {patients_with_labs:,}/{total_patients:,} ({coverage_pct:.1f}%)")
        print(f"  Patients without labs: {total_patients - patients_with_labs:,}/{total_patients:,} ({100 - coverage_pct:.1f}%)")
        print()

        # Create sorted list of tests by patient count
        test_list = []
        for test_name, data in test_coverage.items():
            test_list.append({
                'test_name': test_name,
                'patient_count': len(data['patients']),
                'coverage_pct': (len(data['patients']) / total_patients) * 100,
                'total_measurements': data['measurements'],
                'avg_per_patient': data['measurements'] / len(data['patients'])
            })

        test_list = sorted(test_list, key=lambda x: (x['patient_count'], x['total_measurements']), reverse=True)

        # Display top 20 labs
        print("="*80)
        print("TOP 20 LABS BY PATIENT COVERAGE")
        print("="*80)
        print()
        print(f"{'Rank':<6} {'Lab Test':<50} {'Patients':<12} {'Coverage':<12} {'Measurements':<15} {'Avg/Patient':<12}")
        print("-" * 115)

        for rank, test in enumerate(test_list[:20], 1):
            test_name = test['test_name']
            if len(test_name) > 48:
                test_name = test_name[:45] + '...'

            print(f"{rank:<6} {test_name:<50} {test['patient_count']:<12,} {test['coverage_pct']:>6.1f}%     {test['total_measurements']:<15,} {test['avg_per_patient']:>6.1f}")

        # Summary by coverage brackets
        print()
        print("="*80)
        print("SUMMARY BY COVERAGE BRACKETS")
        print("="*80)

        cov_100 = [t for t in test_list if t['coverage_pct'] == 100.0]
        cov_90 = [t for t in test_list if 90.0 <= t['coverage_pct'] < 100.0]
        cov_80 = [t for t in test_list if 80.0 <= t['coverage_pct'] < 90.0]
        cov_50 = [t for t in test_list if 50.0 <= t['coverage_pct'] < 80.0]
        cov_25 = [t for t in test_list if 25.0 <= t['coverage_pct'] < 50.0]
        cov_10 = [t for t in test_list if 10.0 <= t['coverage_pct'] < 25.0]
        cov_low = [t for t in test_list if t['coverage_pct'] < 10.0]

        print(f"  100% coverage: {len(cov_100)} tests")
        if cov_100:
            for t in cov_100:
                print(f"    - {t['test_name']}: {t['total_measurements']:,} measurements")

        print(f"  90-99% coverage: {len(cov_90)} tests")
        if cov_90:
            for t in cov_90[:5]:
                print(f"    - {t['test_name']}: {t['patient_count']:,} patients ({t['coverage_pct']:.1f}%)")
            if len(cov_90) > 5:
                print(f"    ... and {len(cov_90) - 5} more")

        print(f"  80-89% coverage: {len(cov_80)} tests")
        print(f"  50-79% coverage: {len(cov_50)} tests")
        print(f"  25-49% coverage: {len(cov_25)} tests")
        print(f"  10-24% coverage: {len(cov_10)} tests")
        print(f"  <10% coverage: {len(cov_low)} tests")
        print()

        # Overall statistics
        total_measurements = sum(t['total_measurements'] for t in test_list)
        avg_tests_per_patient = len(test_coverage) / patients_with_labs if patients_with_labs > 0 else 0

        print("="*80)
        print("OVERALL STATISTICS")
        print("="*80)
        print(f"  Total unique tests: {len(test_list):,}")
        print(f"  Total measurements: {total_measurements:,}")
        print(f"  Avg measurements per patient (with labs): {total_measurements / patients_with_labs:.1f}")
        print(f"  Avg unique tests per patient (with labs): {avg_tests_per_patient:.1f}")
        print(f"  Median test coverage: {np.median([t['coverage_pct'] for t in test_list]):.1f}%")
        print(f"  Mean test coverage: {np.mean([t['coverage_pct'] for t in test_list]):.1f}%")
        print()

        # Save detailed report
        coverage_df = pd.DataFrame(test_list)
        output_file = Path('outputs/full_lab_coverage_report.csv')
        coverage_df.to_csv(output_file, index=False)
        print(f"✓ Detailed coverage report saved to: {output_file}")

        return coverage_df, {
            'total_patients': total_patients,
            'patients_with_labs': patients_with_labs,
            'coverage_pct': coverage_pct,
            'total_tests': len(test_list),
            'total_measurements': total_measurements
        }


def main():
    # Paths
    hdf5_file = Path('outputs/full_lab_sequences.h5')
    outcomes_file = Path('../module_1_core_infrastructure/outputs/outcomes.csv')

    if not hdf5_file.exists():
        print(f"ERROR: Lab sequences file not found: {hdf5_file}")
        print("Please run Phase 2 first: python module_02_laboratory_processing.py --phase2")
        return

    if not outcomes_file.exists():
        print(f"ERROR: Outcomes file not found: {outcomes_file}")
        print("Please run Module 1 first")
        return

    # Analyze coverage
    coverage_df, stats = analyze_full_cohort_coverage(hdf5_file, outcomes_file)

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE!")
    print("="*80)
    print(f"  {stats['patients_with_labs']:,}/{stats['total_patients']:,} patients ({stats['coverage_pct']:.1f}%) have lab data")
    print(f"  {stats['total_tests']:,} unique lab tests")
    print(f"  {stats['total_measurements']:,} total measurements")
    print()


if __name__ == '__main__':
    main()
