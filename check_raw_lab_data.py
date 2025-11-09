#!/usr/bin/env python3
"""Check raw lab data in patient timelines."""

import sys
import os

# Change to module 1 directory to import PatientTimeline
os.chdir('module_1_core_infrastructure')
sys.path.insert(0, '.')

import pickle
import pandas as pd
from pathlib import Path

# Import PatientTimeline class
from module_01_core_infrastructure import PatientTimeline

# Load patient timelines
timeline_file = Path('outputs/patient_timelines_test.pkl')

print("Loading patient timelines...")
with open(timeline_file, 'rb') as f:
    timelines = pickle.load(f)

print(f"Total patients: {len(timelines)}")
print()

# Check raw lab data
patient_lab_counts = []
for pt in timelines:
    # Check if patient has labs attribute
    if hasattr(pt, 'labs') and pt.labs is not None and len(pt.labs) > 0:
        patient_lab_counts.append((pt.patient_id, len(pt.labs)))

print("="*80)
print("RAW LAB DATA IN PATIENT TIMELINES")
print("="*80)
print(f"Patients with raw lab data: {len(patient_lab_counts)}/{len(timelines)} ({len(patient_lab_counts)/len(timelines)*100:.1f}%)")
print()

if patient_lab_counts:
    print("Lab measurement counts per patient:")
    total_labs = 0
    for patient_id, count in sorted(patient_lab_counts, key=lambda x: x[1], reverse=True):
        print(f"  Patient {patient_id}: {count} lab measurements")
        total_labs += count

    print()
    print(f"Total lab measurements: {total_labs}")
    print(f"Average per patient (with labs): {total_labs / len(patient_lab_counts):.1f}")
else:
    print("No patients have lab data!")
    print()
    print("This suggests Module 1 did not load lab data.")
    print("Check if Data/FNR_20240409_091633_Lab.txt exists and has data for these patients.")
