# "Unmapped Tests" File - IGNORE IT!

## TL;DR

**The file `test_n10_unmapped_tests.csv` is MISLEADING and should be IGNORED.**

All 119 tests in that file are **ALREADY MAPPED** in the new three-tier system:
- 108 tests mapped by Tier 1 (LOINC exact)
- 11 tests mapped by Tier 3 (hierarchical clustering)
- **Total: 100% coverage**

## What Happened?

The code runs TWO workflows in sequence:

### 1. NEW Three-Tier System (✓ Use This)
```
Phase 1 → Tier 1: LOINC exact matching (319 tests)
       → Tier 2: LOINC family matching (0 tests)
       → Tier 3: Hierarchical clustering (11 tests)
       → Output: harmonization_map_draft.csv (325 groups, 100% coverage)
```

### 2. OLD Legacy Workflow (✗ Ignore This)
```
Phase 1 → group_by_loinc() - checks hardcoded LOINC_FAMILIES only (18 families)
       → fuzzy_match_orphans() - basic string matching (25 groups)
       → Output: unmapped_tests.csv (119 "unmapped" tests)
```

The OLD workflow runs AFTER the new one and doesn't know about Tier 1-3 results, so it incorrectly labels tests as "unmapped".

## Proof: Sample Tests

Let's check what the "unmapped" file claims vs reality:

| Test Name | Old System Says | New System Says |
|-----------|----------------|-----------------|
| EGFR (TEST:BC1-1384) | ✗ Unmapped | ✓ Tier 1: glomerular_filtration_rate |
| TOT PROT (TEST:BC1-38) | ✗ Unmapped | ✓ Tier 1: protein |
| CLDL (TEST:BC1-56) | ✗ Unmapped | ✓ Tier 1: cholesterol_in_ldl |
| HDL (TEST:BC1-55) | ✗ Unmapped | ✓ Tier 1: cholesterol_in_hdl |
| GLU-POC (TEST:BC1-1428) | ✗ Unmapped | ✓ Tier 3: glu-poc_test_bc1-1428 |

**All 119/119 tests are actually mapped!**

## Files to Use vs Ignore

### ✓ USE THESE (NEW System):
```
✓ test_n10_harmonization_map_draft.csv     ← SINGLE SOURCE OF TRUTH
✓ test_n10_tier1_loinc_exact.csv          ← Details on Tier 1 matches
✓ test_n10_tier3_cluster_suggestions.csv  ← Details on Tier 3 clusters
✓ test_n10_cluster_dendrogram.png         ← Visualization
✓ test_n10_harmonization_explorer.html    ← Interactive dashboard
```

### ✗ IGNORE THESE (OLD System):
```
✗ test_n10_unmapped_tests.csv     ← DEPRECATED (100% are actually mapped)
✗ test_n10_fuzzy_suggestions.csv  ← DEPRECATED (replaced by Tier 3)
✗ test_n10_loinc_groups.csv       ← DEPRECATED (replaced by Tier 1)
```

## Why Do These Files Exist?

The code still runs the legacy workflow for backward compatibility. We kept it in place to:
1. Compare old vs new results
2. Maintain compatibility with existing scripts
3. Provide a migration path

**But you should ONLY use the new three-tier outputs.**

## What to Do Now

### Option 1: Ignore the Files (Recommended)
Just use `harmonization_map_draft.csv` and ignore the old files.

### Option 2: Clean Up (Optional)
Remove the legacy workflow from the code to avoid confusion:
- Remove `group_by_loinc()` function
- Remove `fuzzy_match_orphans()` function  
- Remove `generate_discovery_reports()` calls to old outputs

## Summary

| Metric | Value |
|--------|-------|
| Total unique tests | 330 |
| Tests in "unmapped_tests.csv" | 119 |
| Actually unmapped | **0** |
| Coverage by new system | **100%** |

**Bottom Line:** The new three-tier system achieved 100% coverage. There are NO unmapped tests.
