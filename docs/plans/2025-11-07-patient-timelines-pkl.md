# Patient Timelines Pickle File Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Add patient_timelines.pkl creation to Module 1 to enable fast temporal lookups in downstream modules (Module 2+).

**Architecture:** Convert outcomes DataFrame to dictionary of PatientTimeline objects keyed by EMPI, serialize to pickle file. PatientTimeline class already exists (line 111), just need conversion and save logic.

**Tech Stack:** Python 3.12, pandas, pickle, dataclasses

---

## Task 1: Create DataFrame to PatientTimeline Conversion Function

**Files:**
- Modify: `/home/moin/TDA_11_1/module_1_core_infrastructure/module_01_core_infrastructure.py:1175-1190` (insert before main())

**Step 1: Add function to convert DataFrame to PatientTimeline objects**

Insert this function at line 1175 (before `def main()`):

```python
def create_patient_timelines(pe_df: pd.DataFrame) -> Dict[str, PatientTimeline]:
    """
    Convert outcomes DataFrame to dictionary of PatientTimeline objects.

    Args:
        pe_df: DataFrame with patient outcomes and temporal windows

    Returns:
        Dictionary mapping EMPI -> PatientTimeline object
    """
    print("\nCreating PatientTimeline objects...")

    timelines = {}

    for idx, row in pe_df.iterrows():
        empi = row['EMPI']

        # Build phase boundaries
        phase_boundaries = {}
        time_zero = row['time_zero']
        for phase, (start_h, end_h) in TEMPORAL_WINDOWS.items():
            phase_boundaries[f"{phase}_start"] = time_zero + pd.Timedelta(hours=start_h)
            phase_boundaries[f"{phase}_end"] = time_zero + pd.Timedelta(hours=end_h)

        # Build encounter info
        encounter_info = {
            'encounter_number': row.get('Encounter_number'),
            'encounter_match_method': row.get('encounter_match_method'),
            'encounter_match_confidence': row.get('encounter_match_confidence'),
            'hospital_los_days': row.get('hospital_los_days'),
        }

        # Build outcomes dict (all columns not in the core temporal/encounter fields)
        core_fields = {'EMPI', 'time_zero', 'window_start', 'window_end',
                      'Encounter_number', 'encounter_match_method',
                      'encounter_match_confidence', 'hospital_los_days'}

        outcomes = {}
        for col in pe_df.columns:
            if col not in core_fields:
                value = row[col]
                # Convert NaT and NaN to None for JSON serialization
                if pd.isna(value):
                    outcomes[col] = None
                elif isinstance(value, (pd.Timestamp, datetime)):
                    outcomes[col] = value
                else:
                    outcomes[col] = value

        # Build metadata
        metadata = {
            'created_timestamp': datetime.now(),
            'module_version': '2.0',
            'has_encounter_match': pd.notna(row.get('Encounter_number')),
        }

        # Create PatientTimeline object
        timeline = PatientTimeline(
            patient_id=empi,
            time_zero=time_zero,
            window_start=row['window_start'],
            window_end=row['window_end'],
            phase_boundaries=phase_boundaries,
            encounter_info=encounter_info,
            outcomes=outcomes,
            metadata=metadata
        )

        timelines[empi] = timeline

    print(f"  Created {len(timelines)} PatientTimeline objects")

    return timelines
```

**Expected output:** Function added before main(), ready to be called

---

## Task 2: Add Pickle Save Function Call to main()

**Files:**
- Modify: `/home/moin/TDA_11_1/module_1_core_infrastructure/module_01_core_infrastructure.py:1260-1276`

**Step 1: Add patient timelines creation and save after CSV save**

Locate the CSV save section (around line 1267) and add this code after it:

```python
    # 4. Save outcomes CSV
    print("\n" + "="*80)
    print("STEP 4: SAVING OUTPUTS")
    print("="*80)

    output_filename = "outcomes_test.csv" if test_mode else "outcomes.csv"
    outcomes_file = OUTPUT_DIR / output_filename
    pe_df.to_csv(outcomes_file, index=False)
    print(f"  Saved outcomes to: {outcomes_file}")
    print(f"  Total patients: {len(pe_df)}")
    print(f"  Total columns: {len(pe_df.columns)}")

    # NEW CODE STARTS HERE
    # 5. Create and save patient timelines
    timelines = create_patient_timelines(pe_df)

    pkl_filename = "patient_timelines_test.pkl" if test_mode else "patient_timelines.pkl"
    pkl_file = OUTPUT_DIR / pkl_filename

    with open(pkl_file, 'wb') as f:
        pickle.dump(timelines, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  Saved patient timelines to: {pkl_file}")
    print(f"  Total timeline objects: {len(timelines)}")
    print(f"  File size: {pkl_file.stat().st_size / (1024*1024):.2f} MB")
    # NEW CODE ENDS HERE

    print("\n" + "="*80)
    print("MODULE 1 COMPLETE!")
    print("="*80)

    return pe_df
```

**Expected output:** main() now creates both CSV and pkl files

---

## Task 3: Update STEP 4 Header to Include Pickle Output

**Files:**
- Modify: `/home/moin/TDA_11_1/module_1_core_infrastructure/module_01_core_infrastructure.py:1261-1263`

**Step 1: Update print statement to reflect both outputs**

Change:
```python
    print("\n" + "="*80)
    print("STEP 4: SAVING OUTPUTS")
    print("="*80)
```

To:
```python
    print("\n" + "="*80)
    print("STEP 4: SAVING OUTPUTS (CSV + PKL)")
    print("="*80)
```

**Expected output:** Header indicates both output formats

---

## Task 4: Test the Implementation

**Files:**
- Run: Module 1 in test mode

**Step 1: Run Module 1 with test flag**

```bash
cd /home/moin/TDA_11_1/module_1_core_infrastructure
python module_01_core_infrastructure.py --test --n=10
```

**Expected output:**
```
================================================================================
STEP 4: SAVING OUTPUTS (CSV + PKL)
================================================================================
  Saved outcomes to: .../outcomes_test.csv
  Total patients: 10
  Total columns: 113

Creating PatientTimeline objects...
  Created 10 PatientTimeline objects
  Saved patient timelines to: .../patient_timelines_test.pkl
  Total timeline objects: 10
  File size: 0.XX MB
```

**Step 2: Verify pickle file was created**

```bash
ls -lh /home/moin/TDA_11_1/module_1_core_infrastructure/outputs/patient_timelines_test.pkl
```

**Expected output:** File exists with reasonable size (few KB for 10 patients)

**Step 3: Test loading the pickle file**

```bash
python3 -c "
import pickle
from pathlib import Path

pkl_file = Path('/home/moin/TDA_11_1/module_1_core_infrastructure/outputs/patient_timelines_test.pkl')
with open(pkl_file, 'rb') as f:
    timelines = pickle.load(f)

print(f'Loaded {len(timelines)} patient timelines')
print(f'Sample EMPI: {list(timelines.keys())[0]}')

# Check first timeline structure
first_timeline = list(timelines.values())[0]
print(f'Patient ID: {first_timeline.patient_id}')
print(f'Time Zero: {first_timeline.time_zero}')
print(f'Window: {first_timeline.window_start} to {first_timeline.window_end}')
print(f'Phases: {list(first_timeline.phase_boundaries.keys())}')
print(f'Outcomes count: {len(first_timeline.outcomes)}')
"
```

**Expected output:**
```
Loaded 10 patient timelines
Sample EMPI: <some_empi>
Patient ID: <same_empi>
Time Zero: 2020-XX-XX XX:XX:XX
Window: 2020-XX-XX XX:XX:XX to 2020-XX-XX XX:XX:XX
Phases: ['BASELINE_start', 'BASELINE_end', 'ACUTE_start', ...]
Outcomes count: ~100
```

---

## Task 5: Update README Documentation

**Files:**
- Modify: `/home/moin/TDA_11_1/module_1_core_infrastructure/README.md:430-440`

**Step 1: Update "Outputs" section to mark pkl as completed**

Locate the section about `patient_timelines.pkl` (around line 371) and update status:

Change:
```markdown
### 2. patient_timelines.pkl (TODO)
**Location:** `module_1_core_infrastructure/outputs/patient_timelines.pkl`
```

To:
```markdown
### 2. patient_timelines.pkl ✅
**Location:** `module_1_core_infrastructure/outputs/patient_timelines.pkl`
```

And update the description to remove "TODO" language.

**Expected output:** README reflects pkl file is now implemented

---

## Task 6: Update "Completed in V2" Section

**Files:**
- Modify: `/home/moin/TDA_11_1/module_1_core_infrastructure/README.md:612-618`

**Step 1: Add patient timelines to completed list**

Locate "Completed in V2" section and add:

```markdown
### Completed in V2 (2025-11-02)

✅ **Mortality extraction** - Fully implemented with demographics files
✅ **4-tier encounter matching** - Achieves 100% match rate
✅ **Inpatient-only readmissions** - Proper separation from outpatient visits
✅ **Healthcare utilization tracking** - 8 new metrics added
✅ **Test mode** - Enables rapid development/testing
✅ **Patient timeline objects** - PatientTimeline pkl for downstream modules
```

**Expected output:** Documentation shows pkl creation is complete

---

## Task 7: Commit Changes

**Files:**
- All modified files

**Step 1: Stage changes**

```bash
git add module_1_core_infrastructure/module_01_core_infrastructure.py
git add module_1_core_infrastructure/README.md
git add module_1_core_infrastructure/outputs/patient_timelines_test.pkl
```

**Step 2: Commit with descriptive message**

```bash
git commit -m "$(cat <<'EOF'
feat(module1): add patient_timelines.pkl creation

Add PatientTimeline object creation and pickle serialization to enable
fast temporal lookups in downstream modules (Module 2+).

Changes:
- Add create_patient_timelines() function to convert DataFrame to dict
- Save timelines as pickle file (patient_timelines.pkl)
- Update main() to create both CSV and PKL outputs
- Test with 10 patients: creates ~0.XX MB pkl file
- Update README to mark pkl creation as completed

Ready for Module 2 dependency.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

**Expected output:** Commit created successfully

---

## Task 8: Optional - Run Full Cohort

**Files:**
- Run: Module 1 in production mode

**Step 1: Run full cohort (if desired)**

```bash
cd /home/moin/TDA_11_1/module_1_core_infrastructure
python module_01_core_infrastructure.py
```

**Expected runtime:** ~2-3 hours
**Expected output:** `patient_timelines.pkl` with 3,565 patient objects (~10-50 MB)

**Note:** This step is optional - can proceed to Module 2 with test data first.

---

## Validation Checklist

After completing all tasks, verify:

- ✅ `create_patient_timelines()` function exists before `main()`
- ✅ `main()` calls `create_patient_timelines()` after CSV save
- ✅ Test run creates `patient_timelines_test.pkl`
- ✅ Pickle file can be loaded and contains PatientTimeline objects
- ✅ Each PatientTimeline has: patient_id, time_zero, windows, phases, outcomes
- ✅ README updated to show pkl creation is complete
- ✅ Changes committed to git
- ✅ Ready for Module 2 to use `patient_timelines.pkl` as input

---

## Total Time Estimate

- Task 1-3: Code changes - **5 minutes**
- Task 4: Testing - **5 minutes**
- Task 5-6: Documentation - **3 minutes**
- Task 7: Commit - **2 minutes**
- Task 8: Full cohort (optional) - **2-3 hours**

**Total (without full cohort):** ~15 minutes
**Total (with full cohort):** ~2-3 hours

---

## Next Steps After Completion

Once patient_timelines.pkl is created:

1. **Proceed to Module 2 design** - Use brainstorming skill to design lab processing
2. **Module 2 can now:** Load pkl file for fast temporal window lookups
3. **No more CSV parsing** in downstream modules - pkl is faster and structured

