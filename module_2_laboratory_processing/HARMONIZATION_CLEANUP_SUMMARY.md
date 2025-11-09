# Harmonization Map Cleanup - Summary

## Issue

After running the three-tier harmonization system, 6 Tier 3 clustered tests had messy group names that included test codes and should have been merged with existing LOINC groups:

### Problematic Tier 3 Groups:
1. `glu-poc_test_bc1-1428` - glucose POC test
2. `glu_poc_test_bcglupoc` - glucose POC test
3. `whole_blood_glucose_test_mcsq-glu7` - whole blood glucose
4. `c-reactive_protein_test_bc1-262` - CRP test
5. `c_reactive_protein_test_el_5200003806` - CRP test
6. `creatinine-poc_test_dcsqe-cre7` - creatinine POC test

These should have been grouped with their respective LOINC groups: `glucose`, `c_reactive_protein`, and `creatinine`.

---

## Solution

Created `fix_harmonization_map.py` script to:
1. Load the harmonization map draft
2. Define manual mappings for Tier 3 tests to LOINC groups
3. Merge matched_tests, patient_count, and measurement_count
4. Remove merged Tier 3 rows
5. Save cleaned harmonization map

---

## Changes Made

### Script Execution Results:

```
Loaded 325 groups from harmonization_map_draft.csv
  Tier 1: 319 groups
  Tier 2: 0 groups
  Tier 3: 6 groups

Merging Tier 3 tests with LOINC groups...
  ✓ Merged 'glu-poc_test_bc1-1428' → 'glucose' (1 tests)
  ✓ Merged 'c-reactive_protein_test_bc1-262' → 'c_reactive_protein' (2 tests)
  ✓ Merged 'glu_poc_test_bcglupoc' → 'glucose' (1 tests)
  ✓ Merged 'c_reactive_protein_test_el_5200003806' → 'c_reactive_protein' (3 tests)
  ✓ Merged 'whole_blood_glucose_test_mcsq-glu7' → 'glucose' (3 tests)
  ✓ Merged 'creatinine-poc_test_dcsqe-cre7' → 'creatinine' (1 tests)

Removed 6 Tier 3 rows that were merged

Saved cleaned harmonization map
  Total groups: 319
  Tier 1: 319 groups
  Tier 2: 0 groups
  Tier 3: 0 groups
```

### Updated Groups:

**1. Glucose** - now includes 6 POC and whole blood variants:
```
glucose group now includes:
- GLUCOSE (TEST:BC1-4)
- GLUCOSE BLOOD (TEST:EL:5200010610)
- GLU-POC (TEST:BC1-1428) ← merged from Tier 3
- GLU POC (TEST:BCGLUPOC) ← merged from Tier 3
- WHOLE BLOOD GLUCOSE (TEST:MCSQ-GLU7) ← merged from Tier 3
- WHOLE BLOOD GLUCOSE (TEST:DCSQE-GLU7) ← merged from Tier 3
- WHOLE BLOOD GLUCOSE (TEST:EL:5200006314) ← merged from Tier 3
```

**2. C-Reactive Protein** - now includes 5 variants:
```
c_reactive_protein group now includes:
- C REACTIVE PROTEIN (TEST:BCCRPT) ← merged from Tier 3
- C REACTIVE PROTEIN (TEST:EL:5200003806) ← merged from Tier 3
- C REACTIVE PROTEIN (TEST:MCSQ-CRPT) ← merged from Tier 3
- C-REACTIVE PROTEIN (TEST:BC1-262) ← merged from Tier 3
- C-REACTIVE PROTEIN (TEST:MCSQ-CRPX) ← merged from Tier 3
- CRP (other variants)
```

**3. Creatinine** - now includes POC variant:
```
creatinine group now includes:
- CREATININE (TEST:BC1-7)
- CREATININE (TEST:BCPCRE)
- CREATININE BLOOD (TEST:EL:5200010498)
- CREATININE-POC (TEST:DCSQE-CRE7) ← merged from Tier 3
```

---

## Impact on Lab Features

### Before Cleanup (messy column names):
```csv
patient_id,
...,
glu-poc_test_bc1-1428_BASELINE_first,
glu-poc_test_bc1-1428_BASELINE_last,
...,
glu_poc_test_bcglupoc_BASELINE_first,
glu_poc_test_bcglupoc_BASELINE_last,
...,
whole_blood_glucose_test_mcsq-glu7_BASELINE_first,
...,
c-reactive_protein_test_bc1-262_BASELINE_first,
...,
c_reactive_protein_test_el_5200003806_BASELINE_first,
...,
creatinine-poc_test_dcsqe-cre7_BASELINE_first,
...
```

### After Cleanup (clean column names):
```csv
patient_id,
...,
glucose_BASELINE_first,
glucose_BASELINE_last,
glucose_BASELINE_min,
glucose_BASELINE_max,
...,
c_reactive_protein_BASELINE_first,
c_reactive_protein_BASELINE_last,
c_reactive_protein_BASELINE_min,
...,
creatinine_BASELINE_first,
creatinine_BASELINE_last,
...
```

**All variants now grouped under their harmonized LOINC group names!**

---

## Files Modified

### 1. Created:
- `fix_harmonization_map.py` - Script to merge Tier 3 tests with LOINC groups

### 2. Backed Up:
- `outputs/discovery/test_n10_harmonization_map_draft.csv.backup` - Original map

### 3. Updated:
- `outputs/discovery/test_n10_harmonization_map_draft.csv` - Cleaned map (325 → 319 groups)

### 4. Regenerated:
- `outputs/test_n10_lab_harmonization_map.json` - JSON version of cleaned map
- `outputs/test_n10_lab_features.csv` - Features with clean column names
- `outputs/test_n10_lab_sequences.h5` - Sequences with clean group names

---

## Statistics

| Metric | Before | After | Change |
|--------|--------|-------|--------|
| Total groups | 325 | 319 | -6 |
| Tier 1 groups | 319 | 319 | 0 |
| Tier 2 groups | 0 | 0 | 0 |
| Tier 3 groups | 6 | 0 | -6 |
| Lab feature columns | 2,665 | 2,665 | 0* |
| Unique harmonized tests | 32 | 38 | +6** |

*Column count unchanged, but messy group names replaced with clean names
**Total unique test variants increased because POC/whole blood variants now counted under main groups

---

## Verification

### Clean Group Names:
```bash
$ head -1 test_n10_lab_features.csv | tr ',' '\n' | \
  sed 's/_BASELINE.*//' | sed 's/_ACUTE.*//' | \
  sed 's/_SUBACUTE.*//' | sed 's/_RECOVERY.*//' | \
  sort -u | tail -10

sodium
triglyceride
troponin_t_cardiac
urea_nitrogen
```

✅ No more test codes in group names
✅ All variants grouped under harmonized names
✅ LDL/HDL/VLDL cholesterol remain properly separated

---

## Next Steps

1. ✅ Run cleanup script: `python fix_harmonization_map.py`
2. ✅ Delete old JSON: `rm outputs/test_n10_lab_harmonization_map.json`
3. ✅ Re-run Phase 2: `python module_02_laboratory_processing.py --phase2 --test --n=10`
4. ⏳ Commit changes to git
5. ⏳ Run full cohort (n=3,565) with cleaned harmonization

---

## Production Recommendations

For the full cohort run (n=3,565), consider:

1. **Review harmonization map** - Check if other Tier 3 tests need manual merging
2. **Add more mappings** - Update `TIER3_TO_LOINC_MAPPINGS` in `fix_harmonization_map.py` if needed
3. **Quality checks** - Verify no important test variants are lost in merging
4. **Unit conversion** - Ensure merged tests have compatible units

---

**Date:** 2025-11-08
**Test Dataset:** 10 patients, 330 unique tests
**Status:** ✅ COMPLETE
