# MODULE 1: Core Infrastructure & Temporal Reference System

## ✅ STATUS: COMPLETE (Version 2.0)

**Implementation Date:** 2025-11-02
**Last Updated:** 2025-11-07
**Status:** ✅ Complete and tested
**Documentation:** See `module_1_core_infrastructure/README.md` for full implementation details

### Quick Summary
- ✅ Time Zero establishment (PE diagnosis timestamp)
- ✅ 4-tier encounter matching (100% match rate)
- ✅ Comprehensive outcome extraction (113 columns)
- ✅ Mortality extraction from demographics
- ✅ Inpatient-only readmission logic
- ✅ **PatientTimeline objects** saved to `patient_timelines.pkl`
- ✅ Test mode support (`--test --n=X`)

### Implementation Location
- **Code:** `/home/moin/TDA_11_1/module_1_core_infrastructure/module_01_core_infrastructure.py`
- **Outputs:** `/home/moin/TDA_11_1/module_1_core_infrastructure/outputs/`
- **Documentation:** `/home/moin/TDA_11_1/module_1_core_infrastructure/README.md`

### Test Results (10 patients)
- Encounter matching: 100% (10/10)
- Outcomes extracted: 113 columns
- patient_timelines_test.pkl: 59KB
- All temporal windows validated

---

## Overview (Original Specification)

**Purpose:** Establish temporal reference system with PE diagnosis as Time Zero, create patient cohort with quality control, extract outcomes data.

**Priority:** CRITICAL - Foundation for all other modules

**Dependencies:** None (first module)

**Actual Runtime:** 3-5 minutes for test (10 patients), ~2-3 hours for full cohort (3,565 patients)

---

## Input Requirements

### Required Files

```
1. PE Cohort Dataset
   Path: /home/moin/Research/PE_dataset_combined_proper.csv
   Format: CSV
   Size: ~10-50 MB
   
   Required Columns:
   ✓ EMPI (patient identifier)
   ✓ Report_Date_Time (CT scan time - our Time Zero)
   
   Optional But Recommended:
   ○ Admission_Date_Time (or similar column)
   ○ Discharge_Date_Time (or similar column)
   ○ Any outcomes columns (mortality, complications, etc.)
```

### Configuration Needed

```yaml
temporal_windows:
  baseline_start: -72  # hours before PE
  baseline_end: 0
  acute_end: 24
  subacute_end: 72
  recovery_end: 168

output_directory: /home/moin/Research/TDA/updated_analysis_8_14

random_seed: 42
```

---

## Critical Questions to Answer First

### ✅ MUST ANSWER BEFORE PROCEEDING:

**Q1: Admission Time Column**
- [ ] Do you have admission time in your PE dataset?
- [ ] If yes, what is the exact column name? _________________
- [ ] Format: datetime? string? _________________

**Q2: Discharge Time Column**
- [ ] Do you have discharge time in your PE dataset?
- [ ] If yes, what is the exact column name? _________________
- [ ] Format: datetime? string? _________________

**Q3: Outcomes Data**
- [ ] Do you have outcomes data available?
- [ ] If yes, where is it?
  - [ ] In PE dataset (which columns?)
  - [ ] Separate file (path: _________________)
  - [ ] Database query needed
- [ ] Which outcomes are available?
  - [ ] In-hospital mortality
  - [ ] 30-day mortality
  - [ ] 90-day mortality
  - [ ] ICU readmission
  - [ ] Recurrent PE
  - [ ] Bleeding complications
  - [ ] Other: _________________

**Q4: Outcome Derivation**
- [ ] If outcomes not directly available, should we derive from:
  - [ ] Discharge disposition codes (expired = mortality)
  - [ ] Procedure codes (intubation, vasopressors as deterioration proxies)
  - [ ] ICD codes for complications
  - [ ] Other approach: _________________

---

## What This Module Does

### Step 1: Load and Validate PE Cohort

```
Actions:
1. Load PE dataset from CSV
2. Check for required columns (EMPI, Report_Date_Time)
3. Log dataset shape and basic info
4. Identify available optional columns
5. Report missing columns if any

Quality Checks:
- File exists and readable
- Required columns present
- Patient IDs are unique
- No completely empty rows
```

### Step 2: Parse Temporal Reference Points

```
For Each Patient:
1. Parse PE diagnosis time (Report_Date_Time)
   - Convert to datetime
   - Handle errors gracefully
   - Flag invalid/missing times
   
2. Parse admission time (if available)
   - Convert to datetime
   - Handle missing values
   
3. Parse discharge time (if available)
   - Convert to datetime
   - Handle missing values
   
Quality Checks:
- Valid datetime format
- PE time not in distant past/future
- Admission before PE (if both present)
- PE before discharge (if both present)
- No impossible time gaps
```

### Step 3: Create Patient Timeline Objects

```
For Each Patient, Create Timeline With:

Core Fields:
- patient_id (EMPI)
- pe_diagnosis_time (Time Zero)
- admission_time (optional)
- discharge_time (optional)

Derived Fields:
- diagnostic_delay_hours = PE_time - Admission_time
- length_of_stay_hours = Discharge - Admission

Methods:
- hours_from_pe(timestamp) → float
- hours_from_admission(timestamp) → float
- is_in_window(timestamp, window_type) → bool
- classify_timepoint(hours_from_pe) → window_enum
```

### Step 4: Calculate Cohort Statistics

```
Temporal Statistics:
- Diagnostic delay (PE - Admission)
  * Mean, median, std, min, max, quartiles
  * Distribution plot
  * Outliers (delays > 7 days?)

- Length of stay (Discharge - Admission)
  * Mean, median, std, min, max, quartiles
  * Distribution plot

- Data completeness
  * % patients with valid PE time
  * % patients with admission time
  * % patients with discharge time

Quality Flags:
- Patients with missing PE time
- Patients with suspicious time gaps
- Patients with data quality issues
```

### Step 5: Extract/Prepare Outcomes

```
If Outcomes Directly Available:
1. Load outcome columns from dataset
2. Validate outcome values
3. Create time-to-event if needed
4. Handle censoring

If Outcomes Need Derivation:
1. Load procedure codes / disposition codes
2. Map codes to outcomes:
   - Disposition = "Expired" → mortality
   - Intubation code → respiratory failure
   - Vasopressor codes → shock
   - Bleeding ICD codes → complications
3. Calculate time-to-event from PE
4. Create binary outcome flags

Output Format:
- patient_id
- outcome_type (mortality, deterioration, etc.)
- outcome_occurred (binary 0/1)
- time_to_event_hours (from PE diagnosis)
- censoring_status
```

### Step 6: Apply Quality Filters

```
Inclusion Criteria:
✓ Valid PE diagnosis time present
✓ PE diagnosis within study period (if defined)
✓ Patient has some follow-up data

Exclusion Criteria:
✗ Missing PE diagnosis time (cannot establish Time Zero)
✗ Impossible temporal values
✗ Data quality flags (if severe)

Create Filtered Cohort:
- Apply filters
- Document exclusions
- Report final cohort size
```

### Step 7: Define Temporal Windows

```
For Analysis, Define Standard Windows:

BASELINE: -72h to 0h
- Pre-PE measurements
- Establish patient's baseline
- Compare "before" vs "after" PE

ACUTE: 0h to 24h
- Most critical period
- Hourly resolution recommended
- Where most deterioration occurs
- Treatment response window

SUBACUTE: 24h to 72h
- Stabilization vs continued deterioration
- Risk reclassification window
- Treatment optimization

RECOVERY: 72h to 168h (1 week)
- Long-term trajectory
- Discharge readiness assessment
- Late complications

EXTENDED: >168h
- Extended follow-up
- Long-term outcomes
- Readmissions
```

---

## Outputs

### 1. patient_timelines.pkl (Primary Output)

```
Format: Python pickle (dict)
Size: ~10-50 MB for 3,657 patients

Structure:
{
    'patient_12345': PatientTimeline(
        patient_id='12345',
        admission_time=Timestamp('2024-01-15 08:30:00'),
        pe_diagnosis_time=Timestamp('2024-01-15 14:20:00'),  # Time Zero
        discharge_time=Timestamp('2024-01-20 10:00:00'),
        diagnostic_delay_hours=5.83,
        length_of_stay_hours=121.5
    ),
    'patient_12346': PatientTimeline(...),
    ...
}

Usage by Other Modules:
- All modules load this to get Time Zero reference
- Calculate hours_from_pe for temporal alignment
- Filter measurements by temporal windows
```

### 2. cohort_metadata.json

```
Format: JSON
Size: ~1-5 MB

Contents:
{
    "cohort_statistics": {
        "total_patients": 3657,
        "patients_with_valid_pe_time": 3654,
        "patients_with_admission_time": 3621,
        "patients_with_discharge_time": 3598,
        "final_cohort_size": 3654
    },
    
    "temporal_statistics": {
        "diagnostic_delay": {
            "n": 3621,
            "mean": 28.4,
            "median": 12.5,
            "std": 36.2,
            "min": 0.5,
            "max": 168.3,
            "q25": 4.2,
            "q75": 36.8
        },
        "length_of_stay": {
            "n": 3598,
            "mean": 156.7,
            "median": 120.5,
            ...
        }
    },
    
    "temporal_window_definitions": {
        "baseline": {"start": -72, "end": 0},
        "acute": {"start": 0, "end": 24},
        "subacute": {"start": 24, "end": 72},
        "recovery": {"start": 72, "end": 168}
    },
    
    "exclusion_summary": {
        "missing_pe_time": 3,
        "invalid_temporal_values": 0,
        "total_excluded": 3
    },
    
    "data_quality": {
        "pe_time_coverage": 0.9992,
        "admission_time_coverage": 0.9902,
        "discharge_time_coverage": 0.9839
    }
}
```

### 3. cohort_filtered.csv

```
Format: CSV
Size: ~5-10 MB

Columns:
- All original columns from PE dataset
- PLUS derived temporal features:
  * diagnostic_delay_hours
  * length_of_stay_hours
  * has_admission_time (binary flag)
  * has_discharge_time (binary flag)
  * temporal_quality_flag (0=good, 1=minor issues, 2=major issues)
  * included_in_analysis (binary)
  * exclusion_reason (if excluded)

One row per patient, only patients meeting inclusion criteria
```

### 4. outcomes.csv

```
Format: CSV
Size: ~500 KB - 2 MB

Structure Option A (Wide Format):
patient_id, in_hospital_mortality, time_to_death_hours, 
deterioration_72h, time_to_deterioration_hours,
icu_readmission, time_to_readmission_hours,
bleeding_complication, recurrent_pe, ...

Structure Option B (Long Format):
patient_id, outcome_type, occurred (0/1), 
time_to_event_hours, censored (0/1)

One row per patient (wide) or one row per patient per outcome (long)
```

### 5. Quality Control Outputs

```
temporal_validation_report.txt
- Temporal consistency checks
- Identified issues
- Resolution recommendations

diagnostic_delay_distribution.png
- Histogram of delays
- Box plot
- Outlier identification

length_of_stay_distribution.png
- LOS histogram
- By outcome if available

cohort_flowchart.txt / .png
- CONSORT-style flowchart
- Inclusion/exclusion at each step
```

---

## Validation Checks

### Automated Validation

```python
def validate_module_1_outputs():
    """
    Checks to run after Module 1 completes
    """
    
    checks = {
        'patient_timelines_exists': os.path.exists('patient_timelines.pkl'),
        'cohort_size_reasonable': 3000 < len(timelines) < 4000,
        'pe_time_coverage_high': coverage > 0.95,
        'no_negative_delays': min(delays) >= 0,  # admission before PE
        'no_negative_los': min(los) >= 0,
        'temporal_windows_defined': all windows present,
        'outcomes_available': outcomes file exists and not empty
    }
    
    return all(checks.values()), checks
```

### Manual Review Checklist

- [ ] Review `cohort_metadata.json` - statistics look reasonable?
- [ ] Check diagnostic delay distribution - any extreme outliers?
- [ ] Verify PE diagnosis times are in expected date range
- [ ] Confirm admission times are before PE times (when both present)
- [ ] Review exclusion reasons - appropriate?
- [ ] Validate outcome data - coverage and quality OK?
- [ ] Check for any suspicious patterns in temporal data

---

## Success Criteria

### ✅ Module 1 Complete When:

1. **Core Outputs Present:**
   - [ ] patient_timelines.pkl created
   - [ ] cohort_metadata.json created
   - [ ] cohort_filtered.csv created
   - [ ] outcomes.csv created (if outcomes available)

2. **Data Quality Metrics:**
   - [ ] >95% patients have valid PE diagnosis time
   - [ ] Temporal consistency checks passed
   - [ ] No critical data quality issues

3. **Statistics Reasonable:**
   - [ ] Diagnostic delay: mean 12-48 hours (typical)
   - [ ] Length of stay: mean 5-7 days for PE (typical)
   - [ ] No impossible values (negative times, future dates)

4. **Outcomes Prepared:**
   - [ ] Outcome data extracted or derivation strategy defined
   - [ ] At least one target variable available

5. **Documentation Complete:**
   - [ ] Cohort flowchart created
   - [ ] Quality report generated
   - [ ] Statistics logged

---

## Common Issues & Solutions

### Issue 1: Missing PE Diagnosis Time

**Problem:** Some patients don't have Report_Date_Time

**Solutions:**
1. Check if time is in different column
2. Try to derive from other dates (admission + average delay)
3. Exclude patients without PE time (cannot establish Time Zero)
4. Document exclusions clearly

### Issue 2: Admission Time After PE Time

**Problem:** Some patients have admission after PE diagnosis

**Possible Explanations:**
- PE diagnosed in emergency department, then admitted
- PE diagnosed at outside facility, transferred in
- Data entry error

**Solutions:**
1. If few cases (<5%), set admission = PE time
2. If many cases, investigate data source
3. May indicate ED vs inpatient cohort distinction

### Issue 3: No Outcomes Data

**Problem:** Outcomes not readily available

**Solutions:**
1. Use discharge disposition as mortality proxy
2. Derive deterioration from procedure codes:
   - Intubation codes → respiratory failure
   - Vasopressor codes → shock
   - ICU transfer codes → deterioration
3. Plan prospective outcome collection
4. Use intermediate outcomes (treatment escalation)

### Issue 4: Extreme Diagnostic Delays

**Problem:** Some patients have PE diagnosed days/weeks after admission

**Considerations:**
- May be incidental PE findings
- May represent delayed diagnosis
- May affect trajectory analysis (different clinical scenario)

**Solutions:**
1. Flag patients with delay >72 hours
2. Stratify analysis by diagnostic delay
3. Consider separate analysis for early vs delayed diagnosis

### Issue 5: Very Short Length of Stay

**Problem:** Some patients discharged within 24-48 hours

**Considerations:**
- May be low-risk patients (good!)
- May have incomplete data for trajectory analysis
- May represent transfers

**Solutions:**
1. Flag short-stay patients
2. Ensure at least baseline + acute window data
3. May need different analysis approach for low-risk

---

## Integration with Other Modules

### Downstream Module Dependencies

**Module 2 (Labs):**
- Loads patient_timelines.pkl
- Uses hours_from_pe() to calculate temporal alignment
- Filters labs to relevant temporal windows

**Module 3 (Vitals):**
- Loads patient_timelines.pkl
- Aligns vitals to Time Zero
- Creates hourly buckets relative to PE

**Module 4 (Medications):**
- Loads patient_timelines.pkl
- Calculates time to first anticoagulation from PE
- Tracks medication timing relative to Time Zero

**Module 5 (Clinical NLP):**
- Loads patient_timelines.pkl
- Aligns note timestamps to PE diagnosis
- Tracks symptom mentions over time

**All Downstream Modules:**
- Use temporal_bounds to classify measurements by window
- Use outcomes.csv as target variables
- Reference cohort_filtered.csv for patient list

---

## Testing Strategy

### Unit Tests

```python
def test_timeline_creation():
    """Test PatientTimeline object creation"""
    # Test with all fields
    # Test with missing optional fields
    # Test derived field calculation
    
def test_hours_from_pe():
    """Test time delta calculation"""
    # Test positive hours (after PE)
    # Test negative hours (before PE)
    # Test edge cases
    
def test_window_classification():
    """Test temporal window assignment"""
    # Test baseline window
    # Test acute window
    # Test boundary conditions
```

### Integration Tests

```python
def test_full_pipeline():
    """Test complete Module 1 execution"""
    # Run with small test dataset (100 patients)
    # Verify all outputs created
    # Check output formats
    # Validate statistics
```

### Validation Tests

```python
def test_temporal_consistency():
    """Validate temporal relationships"""
    # Admission before PE
    # PE before discharge
    # No future dates
    # Reasonable time spans
```

---

## Implementation Notes

### Performance Considerations

```
Expected Performance:
- Load PE dataset: <5 seconds
- Parse timestamps: ~1 second per 1000 patients
- Create timelines: <1 second per 1000 patients
- Calculate statistics: <5 seconds
- Save outputs: <10 seconds

Total: ~1-2 minutes for 3,657 patients

Memory Usage:
- PE dataset: ~10-50 MB
- Timeline objects: ~10-50 MB
- Peak: ~100-200 MB
```

### Optimization Tips

1. Use pandas vectorized operations for datetime parsing
2. Avoid loops where possible
3. Use pickle for fast timeline object storage
4. Generate plots only if requested (can be slow)

### Error Handling

```python
# Graceful handling of missing data
try:
    admission_time = pd.to_datetime(row['Admission_Time'])
except:
    admission_time = None  # Store as None, not NaT

# Validate before calculation
if admission_time and pe_time:
    delay = (pe_time - admission_time).total_seconds() / 3600
else:
    delay = None

# Log warnings, don't crash
if delay and delay < 0:
    logger.warning(f"Patient {patient_id}: admission after PE ({delay:.1f}h)")
```

---

## Next Steps After Module 1

### Immediate Actions

1. **Review outputs:**
   - Open cohort_metadata.json
   - Check statistics are reasonable
   - Review any QC warnings

2. **Validate temporal references:**
   - Plot diagnostic delay distribution
   - Check for outliers
   - Confirm Time Zero makes clinical sense

3. **Confirm outcomes strategy:**
   - If outcomes extracted, validate completeness
   - If need derivation, plan Module 9
   - Document target variables for analysis

### Prepare for Module 2

1. **Answer questions for Module 2:**
   - Confirm lab file accessible
   - Review PE-specific lab thresholds
   - Decide on QC stringency

2. **Plan computational resources:**
   - 16 GB lab file processing
   - Chunking strategy
   - Parallel processing if available

3. **Set up monitoring:**
   - Log file watching
   - Progress tracking
   - Memory monitoring

---

## Questions Before Implementation

### Configuration Questions

- [ ] Output directory confirmed: `/home/moin/Research/TDA/updated_analysis_8_14`
- [ ] Temporal windows acceptable (-72 to +168h)?
- [ ] Need different window definitions?

### Data Questions

- [ ] Admission column name: _______________
- [ ] Discharge column name: _______________
- [ ] Outcomes available? Where? _______________
- [ ] Derive outcomes from codes? Yes/No _______________

### Analysis Questions

- [ ] Study period constraints? (e.g., only 2020-2023?)
- [ ] Minimum follow-up required? (e.g., 24h data?)
- [ ] Specific exclusion criteria? _______________

---

## Ready to Implement?

Once you answer the questions above, I can write the Module 1 code that will:

1. Load your PE cohort
2. Extract temporal references with PE as Time Zero
3. Create patient timeline objects
4. Calculate cohort statistics
5. Extract/prepare outcomes
6. Generate all outputs and QC reports

**Provide your answers to proceed with implementation!**
