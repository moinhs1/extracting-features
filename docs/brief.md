# Session Brief: TDA Project - Module 3 Vitals Processing
*Last Updated: 2025-12-02*

---

## Active TODO List

**Module 3 Vitals Processing - Submodules 3.1 & 3.2 COMPLETE:**
- [x] Submodule 3.1: Phy Extractor (structured vitals) - 39 tests passing
- [x] Submodule 3.2: Hnp NLP Extractor (H&P notes) - 74 tests passing
- [ ] Task: Run full Phy extraction (USER TO RUN MANUALLY)
- [ ] Task: Run full Hnp extraction (USER TO RUN MANUALLY)

**Remaining Module 3 Submodules:**
- [ ] Submodule 3.3: Prg NLP Extractor (Progress notes - 8.7M notes)
- [ ] Submodule 3.4: Vitals Harmonizer (merge 3 sources)
- [ ] Submodule 3.5: Unit Converter & QC Filter
- [ ] Submodule 3.6-3.10: Temporal alignment & feature engineering

**Future Modules:**
- [ ] Module 4: Medications Processing
- [ ] Module 5: Diagnoses/Procedures Processing
- [ ] Module 6: Temporal Alignment
- [ ] Module 7: Trajectory Feature Engineering

---

## Current Session Progress (Dec 2, 2025)

### Module 3 Submodule 3.2: Hnp NLP Extractor - COMPLETE

**Goal:** Extract vital signs from Hnp.txt (2.3GB, 136,950 H&P notes) using NLP/regex.

**Implementation Summary:**
- Complete TDD implementation with 14 tasks
- 10 core functions + CLI entry point
- 74 unit tests + 1 integration test (all passing)
- 16 git commits made this session
- Final code review: 9.5/10 - APPROVED FOR PRODUCTION

**Files Created:**

| File | Purpose |
|------|---------|
| `extractors/hnp_patterns.py` | 29 regex patterns, section patterns, negation patterns, valid ranges |
| `extractors/hnp_extractor.py` | Main extractor (662 lines, 10 functions + CLI) |
| `tests/test_hnp_patterns.py` | 4 pattern tests |
| `tests/test_hnp_extractor.py` | 70 extractor tests |
| `docs/plans/2025-12-02-submodule-3-2-hnp-extractor-design.md` | Design document |
| `docs/plans/2025-12-02-submodule-3-2-hnp-extractor.md` | Implementation plan |

**Functions Implemented:**

1. `identify_sections(text)` - Find Exam/Vitals/ED sections
2. `check_negation(text, position, window)` - Context window negation detection
3. `extract_heart_rate(text)` - HR extraction with 4 patterns
4. `extract_blood_pressure(text)` - BP extraction, auto-swap if SBP<DBP
5. `extract_respiratory_rate(text)` - RR extraction
6. `extract_spo2(text)` - SpO2 extraction
7. `extract_temperature(text)` - Temp extraction with unit auto-detection
8. `extract_timestamp(text, section, report_dt)` - Explicit or estimated timestamps
9. `process_hnp_row(row)` - Process single note → list of vital records
10. `extract_hnp_vitals(input, output, workers, chunk)` - Parallel processing

**Test Results:**
```
113 tests passing total (74 hnp + 39 phy):
- TestPatternsExist: 4 tests
- TestIdentifySections: 6 tests
- TestCheckNegation: 6 tests
- TestExtractHeartRate: 9 tests
- TestExtractBloodPressure: 9 tests
- TestExtractRespiratoryRate: 7 tests
- TestExtractSpO2: 9 tests
- TestExtractTemperature: 10 tests
- TestExtractTimestamp: 6 tests
- TestProcessHnpRow: 5 tests
- TestExtractHnpVitals: 2 tests
- TestIntegration: 1 test
```

**Integration Test Output (1000 notes):**
```
Extracted 11,909 vital records:
  SBP       2,550 (21.4%)
  DBP       2,550 (21.4%)
  HR        2,251 (18.9%)
  TEMP      2,179 (18.3%)
  RR        1,263 (10.6%)
  SPO2      1,116 (9.4%)
```

---

## Key Decisions & Architecture

### Decision 1: Hybrid Extraction Approach (Submodule 3.2)

- **Section-first:** Extract from Exam/Vitals/ED Course sections
- **Full-text fallback:** Use stricter patterns if no sections found
- **Rationale:** Maximizes coverage while minimizing false positives

### Decision 2: Keep All Vitals Per Note

- Extract ALL occurrences (ED vitals + Exam vitals)
- Tag each with `extraction_context` (ed_course, exam, vitals, other)
- Allows downstream harmonizer to select best value

### Decision 3: Timestamp Estimation

| Context | Offset from Report_Date_Time |
|---------|------------------------------|
| ED/Triage | -6 hours |
| Exam/Vitals | -1 hour |
| Current | 0 hours |
| Unknown | -2 hours |

### Decision 4: Pattern-Based Confidence Scoring

| Confidence | Pattern Type |
|------------|--------------|
| 1.0 | Explicit label ("Heart Rate: 88") |
| 0.9-0.95 | Standard abbreviation ("HR 71") |
| 0.8-0.85 | Value with context/units |
| 0.6-0.7 | Ambiguous/fallback patterns |

### Decision 5: Context Window Negation Detection

- Check ±50 characters around each match
- 8 negation patterns: "not obtained", "unable to", "refused", etc.
- Prevents false positives like "BP not obtained"

### Decision 6: 6-Layer Information Preservation (from ARCHITECTURE.md)

1. **Layer 1:** Raw source-specific values
2. **Layer 2:** Hierarchical merged values (Prg > Hnp > Phy priority)
3. **Layer 3:** Conflict detection & quality metrics
4. **Layer 4:** Temporal precision tracking
5. **Layer 5:** Temporal consistency validation
6. **Layer 6:** Encounter pattern features

---

## Technical Details

### Hnp.txt File Format

```
EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text
```

10 pipe-delimited columns. Report_Text contains free-text H&P notes.

### HNP Output Schema

Parquet file with 13 columns:
- EMPI (str)
- timestamp (datetime) - extracted or estimated
- timestamp_source (str) - 'explicit' or 'estimated'
- timestamp_offset_hours (float)
- vital_type (str): HR, SBP, DBP, RR, SPO2, TEMP
- value (float)
- units (str) - original units
- source (str): 'hnp'
- extraction_context (str): ed_course, exam, vitals, other
- confidence (float): 0.6-1.0
- is_flagged_abnormal (bool) - True if (!) found
- report_number (str)
- report_date_time (datetime)

### Regex Pattern Highlights

**Encoding awareness:** Handles `?C`/`?F` (broken Unicode for degree symbols)
**Abnormal flags:** Detects `(!)` markers in clinical text
**Range notation:** Extracts current value from `[62-72] 72` patterns
**BP auto-swap:** Corrects transposed values (70/120 → 120/70)
**Temp unit detection:** Value >50 = Fahrenheit, ≤50 = Celsius

### CLI Usage

```bash
# Phy extractor (structured)
python3 -m module_3_vitals_processing.extractors.phy_extractor

# Hnp extractor (H&P notes)
python3 -m module_3_vitals_processing.extractors.hnp_extractor

# With custom options
python3 -m module_3_vitals_processing.extractors.hnp_extractor \
  -i /path/to/Hnp.txt \
  -o /path/to/output.parquet \
  -w 4 \
  -c 10000
```

---

## Important Context

### Commits Made This Session (16 for Submodule 3.2)

```
d977ca6 docs(module3): add submodule 3.2 hnp extractor design
c873fb4 docs(module3): add submodule 3.2 hnp extractor implementation plan
c9613af feat(module3): add hnp extractor patterns module
ec6e302 feat(module3): add section identification for hnp extractor
3aed41e feat(module3): add negation detection for hnp extractor
fb0550d feat(module3): add heart rate extraction for hnp extractor
83018d7 feat(module3): add blood pressure extraction for hnp extractor
02c2129 feat(module3): add respiratory rate extraction for hnp extractor
1edec9e feat(module3): add SpO2 extraction for hnp extractor
094b9ca feat(module3): add temperature extraction for hnp extractor
64f51de feat(module3): add timestamp extraction for hnp extractor
65f7b8f feat(module3): add row processor for hnp extractor
fe2ce18 feat(module3): add main extraction function with parallel processing
d409890 feat(module3): add CLI entry point for hnp extractor
97d2e71 test(module3): add integration test for hnp extractor
74b38f7 feat(module3): update config with hnp extractor paths
```

### Valid Ranges for Vital Signs

| Vital | Range | Rationale |
|-------|-------|-----------|
| HR | 20-300 bpm | Covers bradycardia to severe tachycardia |
| SBP | 40-350 mmHg | Extreme hypotension to crisis |
| DBP | 20-250 mmHg | Clinical range |
| RR | 4-80 breaths/min | Apnea to extreme tachypnea |
| SPO2 | 40-100% | Severe hypoxia to normal |
| TEMP_C | 25-45°C | Hypothermia to hyperthermia |
| TEMP_F | 77-113°F | Equivalent Fahrenheit |

### Branch Status

- **Branch:** main (24 commits ahead of origin/main)
- **Tests:** 113 passing
- **Ready to push:** Yes

---

## Unfinished Tasks & Next Steps

### Immediate: Run Full Extractions (User Tasks)

```bash
cd /home/moin/TDA_11_25

# Run Phy extraction (structured vitals)
python3 -m module_3_vitals_processing.extractors.phy_extractor

# Run Hnp extraction (H&P notes)
python3 -m module_3_vitals_processing.extractors.hnp_extractor
```

**Expected outputs:**
- `outputs/discovery/phy_vitals_raw.parquet` (~8-9M records)
- `outputs/discovery/hnp_vitals_raw.parquet` (~1.6M records estimated)

### Next Submodules

1. **Submodule 3.3: Prg NLP Extractor**
   - Parse 8.7M progress notes (29.7GB Prg.txt)
   - Similar patterns to Hnp extractor
   - Additional: range handling ("HR 70-85"), narrative parsing ("afebrile", "tachycardic")
   - Higher priority in merge hierarchy (Prg > Hnp > Phy)

2. **Submodule 3.4: Vitals Harmonizer**
   - Merge all three sources
   - Apply hierarchical priority
   - Detect conflicts between sources
   - Create Layer 2 (merged values with provenance)

3. **Submodule 3.5: Unit Converter & QC Filter**
   - Standardize units (Fahrenheit → Celsius)
   - Apply physiological QC
   - Add clinical flags (tachycardia, hypoxemia, etc.)

### Pattern for Next Submodules

Follow same TDD approach:
1. Brainstorm design with `superpowers:brainstorming`
2. Create plan with `/superpowers:write-plan`
3. Execute with `superpowers:subagent-driven-development`
4. Review with `superpowers:code-reviewer`
5. Complete with `superpowers:finishing-a-development-branch`

---

## Related Resources

### Key Files

**Module 3:**
- `extractors/phy_extractor.py` - Structured vitals extractor (265 lines)
- `extractors/hnp_extractor.py` - H&P notes extractor (662 lines)
- `extractors/hnp_patterns.py` - 29 regex patterns (88 lines)
- `config/vitals_config.py` - Paths and constants
- `tests/test_phy_extractor.py` - 39 tests
- `tests/test_hnp_extractor.py` - 70 tests
- `docs/plans/2025-12-02-submodule-3-2-hnp-extractor-design.md` - Design doc
- `docs/plans/2025-12-02-submodule-3-2-hnp-extractor.md` - Implementation plan

**Dependencies (Module 1 outputs):**
- `module_1_core_infrastructure/outputs/patient_timelines.pkl`
- `module_1_core_infrastructure/outputs/outcomes.csv`

### Data Files

| File | Size | Rows | Content |
|------|------|------|---------|
| `Data/Phy.txt` | 2.7GB | ~33M | Structured flowsheet vitals |
| `Data/Hnp.txt` | 2.3GB | 136,950 | H&P admission notes |
| `Data/Prg.txt` | 29.7GB | ~8.7M | Progress notes (next) |

---

## Previous Module Status

### Module 1: Core Infrastructure - COMPLETE

- **Cohort:** 8,713 Gemma PE-positive patients
- **Output:** `patient_timelines.pkl`, `outcomes.csv`
- **Encounter matching:** 99.5% Tier 1
- **30-day mortality:** 11.2%

### Module 2: Laboratory Processing - COMPLETE

- **Patient coverage:** 100% (3,565 patients with labs)
- **Total measurements:** 7.6 million
- **Harmonized groups:** 48 lab tests
- **Note:** Needs rerun on expanded 8,713 cohort

---

**END OF BRIEF**

*This brief preserves context for Module 3 Submodules 3.1 and 3.2 implementation. When starting a new session, reference with `@docs/brief.md` to restore context.*

*Current Status: Submodule 3.2 COMPLETE - 113 tests passing, both CLIs ready, user to run full extractions.*
