# Module 3: Vitals Processing - Implementation Roadmap

**Timeline:** 10 weeks (with 2-week buffer for validation refinement)
**Start Date:** TBD
**Team Size:** 1-3 developers (can parallelize Phase 1)

---

## Overview

This roadmap breaks Module 3 implementation into 10 weekly sprints, with clear deliverables, checkpoints, and success criteria for each week.

### Key Milestones

| Week | Milestone | Deliverable |
|------|-----------|-------------|
| 1 | Foundation - Structured Data | Phy.txt extraction + harmonization working |
| 3 | NLP Extraction - H&P | Admission vitals extraction complete |
| 5 | NLP Extraction - Progress Notes | All three sources extracted |
| 7 | Integration Complete | Final feature set with provenance |
| 10 | Validation Complete | Publication-ready with ≥90% accuracy |

---

## Week 1: Foundation - Structured Data Extraction

### Goals
- Extract vitals from structured Phy.txt
- Build harmonization framework
- Establish code structure and testing patterns

### Tasks

#### Day 1: Project Setup
- [ ] Create virtual environment
- [ ] Install dependencies (`requirements.txt`)
- [ ] Set up logging configuration (`utils/logging_utils.py`)
- [ ] Create test dataset (sample 1000 patients from Phy.txt)
- [ ] Set up pytest configuration

**Deliverable:** Working development environment

#### Days 2-3: Submodule 3.1 (Phy Extractor)
- [ ] Implement `extractors/phy_extractor.py`
  - [ ] `load_phy_vitals()` - Load and filter
  - [ ] `parse_combined_bp()` - Parse "124/68"
  - [ ] `map_phy_concepts_to_canonical()` - Concept mapping
- [ ] Write unit tests (`tests/test_phy_extractor.py`)
- [ ] Run on test dataset
- [ ] Verify output schema

**Deliverable:** `phy_vitals_raw.parquet` with correct schema

**Success Criteria:**
- All vital sign concepts correctly identified
- BP values correctly parsed
- Output schema matches specification
- Tests pass

#### Days 4-5: Submodule 3.4 (Harmonizer)
- [ ] Implement `processing/harmonizer.py`
  - [ ] `create_harmonization_map()` - Build mapping
  - [ ] `consolidate_bp_measurements()` - Handle BP formats
  - [ ] `deduplicate_vitals()` - Remove duplicates
- [ ] Create `config/harmonization_map.json`
- [ ] Write unit tests (`tests/test_harmonizer.py`)
- [ ] Run on Phy.txt output

**Deliverable:** `harmonized_vitals.parquet` with canonical names

**Success Criteria:**
- All vital variants mapped to canonical names
- BP correctly split into SBP/DBP rows
- No duplicate measurements
- Coverage report shows expected vital counts

#### Week 1 Checkpoint
**Review Questions:**
- Are all Phy.txt vitals extracting correctly?
- Is the harmonization map complete?
- Are tests passing reliably?
- Is the code structure scalable for NLP modules?

**Blockers to Address:**
- Missing vital sign concepts in Phy.txt
- Unexpected data formats
- Performance issues with 33M rows

---

## Week 2: Foundation - QC & Parallel NLP Start

### Goals
- Complete QC and unit conversion for structured data
- Begin H&P NLP pattern development
- Validate end-to-end structured pipeline

### Tasks

#### Days 1-2: Submodule 3.5 (Unit Converter)
- [ ] Implement `processing/unit_converter.py`
  - [ ] `detect_units()` - Auto-detect from ranges
  - [ ] `convert_to_standard_units()` - F→C, lbs→kg, etc.
  - [ ] Handle edge cases (missing units, ambiguous)
- [ ] Create `config/qc_thresholds.json`
- [ ] Write unit tests (`tests/test_unit_converter.py`)

**Deliverable:** Unit conversion working with validation

#### Day 3: Submodule 3.5 (QC Filter)
- [ ] Implement `processing/qc_filter.py`
  - [ ] `apply_physiological_qc()` - Filter implausible
  - [ ] `create_clinical_flags()` - Tachycardia, hypoxemia, etc.
- [ ] Write unit tests (`tests/test_qc_filter.py`)
- [ ] Generate `qc_report.csv` with rejection statistics

**Deliverable:** `qc_vitals.parquet` with cleaned data

**Success Criteria:**
- <1% of values rejected as out-of-range
- Clinical flags correctly applied
- QC report shows sensible rejection reasons

#### Days 4-5: Submodule 3.2 (H&P Patterns - Initial)
- [ ] Create `extractors/patterns.py`
  - [ ] Define HR_PATTERNS (5-10 regex variants)
  - [ ] Define BP_PATTERNS (handle "120/80", "BP: 120/80 mmHg")
  - [ ] Define TEMP_PATTERNS (handle °C/°F)
  - [ ] Define RR_PATTERNS
  - [ ] Define SPO2_PATTERNS
- [ ] Implement basic `extractors/hnp_nlp_extractor.py`
  - [ ] `extract_vitals_patterns()` - Apply patterns
  - [ ] Test on 10 sample H&P notes manually

**Deliverable:** Initial regex patterns tested on samples

**Success Criteria:**
- Patterns correctly extract vitals from sample notes
- No false positives on sample review

#### Week 2 Checkpoint
**Review Questions:**
- Is QC working correctly on structured data?
- Are unit conversions accurate?
- Do initial H&P regex patterns look promising?
- Is performance acceptable (~30min for full Phy.txt)?

**Blockers to Address:**
- Unexpected unit formats
- QC thresholds too strict/lenient
- Regex patterns too complex/too simple

---

## Week 3: NLP Extraction - H&P Notes

### Goals
- Complete H&P NLP extraction with context awareness
- Implement negation handling
- Achieve initial validation on sample

### Tasks

#### Days 1-2: Negation Handler
- [ ] Implement `extractors/negation_handler.py`
  - [ ] Define NEGATION_PHRASES
  - [ ] `detect_negation()` - Check context around match
  - [ ] Handle long-distance negation
- [ ] Write unit tests for negation patterns
- [ ] Test on sample notes with negation

**Deliverable:** Negation detection working

#### Days 3-4: Complete Submodule 3.2
- [ ] Implement full `extractors/hnp_nlp_extractor.py`
  - [ ] `identify_vitals_section()` - Find vital signs section
  - [ ] `validate_extraction_context()` - Full context checks
  - [ ] Handle ranges ("120-130/70-80")
  - [ ] Confidence scoring
- [ ] Write comprehensive tests (`tests/test_hnp_extractor.py`)
- [ ] Run on full Hnp.csv (257K notes)

**Deliverable:** `hnp_vitals_raw.parquet` from full dataset

**Success Criteria:**
- Completes in <2 hours
- ~80% of notes with vitals extracted
- Sample manual review shows ~85-90% accuracy

#### Day 5: Initial Validation
- [ ] Cross-validate H&P extractions with Phy.txt
  - [ ] Match on patient + date (±1 hour)
  - [ ] Calculate clinical agreement
  - [ ] Identify high-discrepancy cases
- [ ] Manually review 20 high-discrepancy cases
- [ ] Refine patterns based on errors

**Deliverable:** Initial cross-validation report

**Success Criteria:**
- ≥80% clinical agreement (will improve to ≥90% later)
- Error patterns identified for refinement

#### Week 3 Checkpoint
**Review Questions:**
- Is H&P extraction accuracy acceptable?
- Are negation patterns catching false positives?
- Is performance acceptable for 257K notes?
- What are the most common error types?

**Blockers to Address:**
- Low accuracy on certain vital types
- Performance bottlenecks
- Complex negation patterns missed

---

## Week 4: NLP Extraction - Progress Notes (Part 1)

### Goals
- Implement progress notes extraction framework
- Handle range and narrative patterns
- Begin full-scale extraction

### Tasks

#### Days 1-2: Range & Narrative Patterns
- [ ] Extend `extractors/patterns.py`
  - [ ] RANGE_PATTERNS for "HR 70-85"
  - [ ] NARRATIVE_MAP for "tachycardic", "afebrile", etc.
- [ ] Implement `extractors/prg_nlp_extractor.py`
  - [ ] `parse_vital_range()` - Return (min, mean, max)
  - [ ] `extract_from_narrative()` - Map descriptive to values
  - [ ] Handle uncertainty ("approximately", "~")
- [ ] Test on sample progress notes

**Deliverable:** Range and narrative extraction working

#### Days 3-5: Begin Full Extraction
- [ ] Implement chunked processing for 8.7M notes
  - [ ] Use Dask or chunked pandas
  - [ ] Progress bar and logging
  - [ ] Checkpoint every 1M notes
- [ ] Run extraction on first 2M notes
- [ ] Monitor performance and memory usage

**Deliverable:** Partial `prg_vitals_raw.parquet` (2M notes)

**Success Criteria:**
- Extraction rate: ~20K-50K notes/hour
- Memory usage stable
- Checkpoint/resume working

#### Week 4 Checkpoint
**Review Questions:**
- Is range extraction working correctly?
- Are narrative extractions reasonable?
- Is performance on track for <8 hour full extraction?
- Are there unexpected data format issues?

**Blockers to Address:**
- Performance too slow
- Memory issues with large notes
- Unexpected text formats

---

## Week 5: NLP Extraction - Progress Notes (Part 2)

### Goals
- Complete progress notes extraction
- Refine patterns based on errors
- Validate all three sources

### Tasks

#### Days 1-3: Complete Full Extraction
- [ ] Continue extraction on remaining 6.7M notes
- [ ] Handle errors gracefully (log and continue)
- [ ] Generate extraction report:
  - [ ] Coverage by vital type
  - [ ] Confidence distribution
  - [ ] Extraction rate by note year

**Deliverable:** Complete `prg_vitals_raw.parquet` (8.7M notes)

#### Days 4-5: Pattern Refinement
- [ ] Cross-validate Prg.csv extractions with Phy.txt
- [ ] Manually review 30 notes:
  - [ ] 10 high-discrepancy
  - [ ] 10 with ranges
  - [ ] 10 with narratives
- [ ] Refine patterns based on findings
- [ ] Re-run extraction on error cases

**Deliverable:** Refined patterns, updated outputs

**Success Criteria:**
- All 8.7M notes processed
- ~35% of notes with vitals extracted
- Initial cross-validation ≥80% agreement
- Error taxonomy documented

#### Week 5 Checkpoint
**Review Questions:**
- Are all three sources (Phy, HNP, Prg) extracted?
- What is overall data completeness by vital type?
- Which vitals have lowest extraction accuracy?
- Are there systematic errors to address?

**Blockers to Address:**
- Low accuracy for specific vitals (e.g., RR)
- Large extraction time
- Unexpected note formats

---

## Week 6: Integration - Temporal Alignment

### Goals
- Merge all three sources with temporal alignment
- Implement hierarchical merge strategy
- Extract admission vitals separately

### Tasks

#### Days 1-2: Temporal Grid Construction
- [ ] Implement `processing/temporal_aligner.py`
  - [ ] `create_temporal_grid()` - Phase-specific grids
  - [ ] Load patient_timelines.pkl from Module 1
  - [ ] Create temporal phases per patient
- [ ] Implement `utils/temporal_utils.py`
  - [ ] Date/time conversion helpers
  - [ ] Grid binning functions
- [ ] Write unit tests for temporal logic

**Deliverable:** Temporal grid framework working

#### Days 3-4: Multi-Source Merging
- [ ] Implement hierarchical merge
  - [ ] `merge_multi_source_vitals()` - Prg > Hnp > Phy
  - [ ] Track source attribution
  - [ ] Handle time windows (±1 hour for matching)
- [ ] Implement admission vitals extraction
  - [ ] `extract_admission_vitals()` - First H&P after index
- [ ] Generate Layer 1-2 HDF5 structure

**Deliverable:** `aligned_vitals_raw.h5` with merged data

#### Day 5: Validation & Debugging
- [ ] Verify HDF5 structure correct
- [ ] Sample 10 patients, manually inspect timelines
- [ ] Check source attribution correct
- [ ] Verify admission vitals extraction

**Success Criteria:**
- All three sources merged correctly
- Hierarchical priority respected
- Admission vitals identified
- HDF5 readable and well-structured

#### Week 6 Checkpoint
**Review Questions:**
- Are temporal bins correct for all phases?
- Is hierarchical merge working as intended?
- Are conflicts between sources being preserved?
- Is HDF5 storage efficient?

**Blockers to Address:**
- Temporal alignment bugs
- HDF5 structure issues
- Performance issues with large patient counts

---

## Week 7: Integration - Provenance & Features

### Goals
- Calculate quality metrics (Layers 3-6)
- Engineer temporal features
- Generate final feature set

### Tasks

#### Days 1-2: Submodule 3.7 (Provenance)
- [ ] Implement `processing/provenance_calculator.py`
  - [ ] `detect_conflicts()` - Flag disagreements
  - [ ] `calculate_completeness_metrics()` - Coverage scores
  - [ ] `calculate_time_deltas()` - Temporal precision
  - [ ] `validate_temporal_consistency()` - Rate of change
  - [ ] `detect_outliers()` - Modified Z-score
  - [ ] `classify_encounter_pattern()` - Categorical
- [ ] Write unit tests (`tests/test_provenance.py`)
- [ ] Generate `vitals_with_provenance.h5`

**Deliverable:** Full quality metrics calculated

#### Days 3-5: Submodule 3.8 (Feature Engineering)
- [ ] Implement `processing/feature_engineer.py`
  - [ ] `aggregate_by_temporal_phase()` - Stats per phase
  - [ ] `calculate_trajectory_features()` - Slopes, volatility
  - [ ] `calculate_clinical_composites()` - Shock index, MAP, PP
  - [ ] `aggregate_clinical_flags()` - Proportion abnormal
  - [ ] `format_admission_features()` - ADMISSION_* features
  - [ ] `forward_fill_with_decay()` - Imputation
- [ ] Write unit tests (`tests/test_feature_engineer.py`)
- [ ] Generate `vitals_features_final.h5`

**Deliverable:** Final feature set ready for modeling

**Success Criteria:**
- All quality metrics calculated correctly
- Features match specification
- Sample patient features look clinically reasonable
- HDF5 structure correct

#### Week 7 Checkpoint
**Review Questions:**
- Do quality metrics make sense?
- Are features correctly calculated?
- Are there any missing feature types?
- Is data ready for modeling?

**Blockers to Address:**
- Feature calculation bugs
- Performance issues
- Missing clinical composite formulas

---

## Week 8: Validation - Tier 1 & Tier 2 Preparation

### Goals
- Complete cross-validation with structured data
- Generate stratified sample for manual review
- Begin manual annotation

### Tasks

#### Days 1-3: Tier 1 Cross-Validation
- [ ] Implement `validation/cross_validator.py`
  - [ ] Match note extractions to Phy.txt
  - [ ] Calculate clinical agreement per vital
  - [ ] Calculate Pearson correlation
  - [ ] Calculate MAE, MAPE
  - [ ] Generate Bland-Altman plots
- [ ] Run full cross-validation
- [ ] Generate report with metrics per vital type
- [ ] Identify vitals meeting ≥90% target

**Deliverable:** `cross_validation_results.json` + plots

**Success Criteria:**
- Most vitals achieve ≥85% clinical agreement
- High-discrepancy cases identified
- Clear error patterns documented

#### Days 4-5: Tier 2 Sample Generation
- [ ] Implement `validation/manual_review_sampler.py`
  - [ ] Stratified sampling (200 notes):
    - [ ] 50 high-discrepancy from cross-val
    - [ ] 75 no structured match
    - [ ] 40 critical values
    - [ ] 35 edge cases
- [ ] Export to `manual_review_cases.csv`
- [ ] Set up REDCap annotation project (or similar)
- [ ] Train two independent reviewers
  - [ ] Create annotation codebook
  - [ ] Practice on 10 consensus cases

**Deliverable:** `manual_review_cases.csv` ready for annotation

**Success Criteria:**
- Sample is representative
- Annotation interface working
- Reviewers trained and calibrated

#### Week 8 Checkpoint
**Review Questions:**
- Which vitals need pattern refinement?
- Is cross-validation coverage sufficient (~30-40%)?
- Are manual review strata appropriate?
- Are reviewers ready to begin annotation?

**Blockers to Address:**
- Low cross-validation agreement for certain vitals
- Insufficient high-discrepancy cases
- Reviewer training issues

---

## Week 9: Validation - Manual Review & Pattern Refinement

### Goals
- Complete dual independent annotation (200 notes)
- Calculate inter-rater reliability
- Refine patterns based on error taxonomy

### Tasks

#### Days 1-4: Dual Annotation
- [ ] Reviewer 1: Annotate all 200 notes independently
- [ ] Reviewer 2: Annotate all 200 notes independently
- [ ] Track time per note (~10 min target)
- [ ] Export annotations to CSV

**Deliverable:** `manual_review_annotations.csv` from both reviewers

**Time Estimate:** 2 reviewers × 200 notes × 10 min = ~33 hours total (16-17 hours each)

#### Day 5: IRR Analysis & Error Taxonomy
- [ ] Calculate Cohen's kappa (binary correct/incorrect)
- [ ] Calculate ICC (continuous values)
- [ ] Adjudicate disagreements between reviewers
- [ ] Categorize errors by type:
  - [ ] False negatives (missed)
  - [ ] False positives (hallucinated)
  - [ ] Wrong value
  - [ ] Wrong context
  - [ ] Unit errors
  - [ ] Decimal errors
  - [ ] Sys/dia confusion
- [ ] Prioritize error types by frequency
- [ ] Identify pattern refinements needed

**Deliverable:** Error taxonomy with prioritized fixes

**Success Criteria:**
- κ ≥ 0.80 (good agreement)
- ICC ≥ 0.90 (excellent value agreement)
- Overall accuracy ≥85% (will improve with pattern fixes)
- Clear action items for pattern refinement

#### Week 9 Checkpoint
**Review Questions:**
- Is IRR acceptable?
- What are the most common error types?
- Which patterns need urgent fixes?
- Will pattern fixes achieve ≥90% target?

**Blockers to Address:**
- Low IRR (need adjudication and codebook refinement)
- Accuracy below 85% (may need medical NER)
- Specific vital types with low accuracy

---

## Week 10: Validation - Final Tier & Report Generation

### Goals
- Implement Tier 3 & 4 validation
- Refine patterns based on manual review
- Generate comprehensive validation report
- Complete orchestrator and testing

### Tasks

#### Days 1-2: Pattern Refinement & Re-extraction
- [ ] Implement pattern fixes from error taxonomy
- [ ] Re-run extraction on error cases
- [ ] Re-validate on sample
- [ ] Iterate until accuracy ≥90%

**Deliverable:** Refined extraction achieving target accuracy

#### Day 3: Tier 3 & 4 Validation
- [ ] Implement `validation/statistical_validator.py`
  - [ ] Distribution validation (KS test)
  - [ ] Outlier detection
  - [ ] Temporal plausibility checks
  - [ ] Digit preference analysis
- [ ] Implement `validation/pattern_validator.py`
  - [ ] Negation handling validation
  - [ ] Range handling validation
  - [ ] Unit conversion validation
- [ ] Run all Tier 3 & 4 checks
- [ ] Generate results JSON files

**Deliverable:** `statistical_validation.json`, `pattern_validation.json`

**Success Criteria:**
- <1% out-of-range values
- <2% statistical outliers
- <1% implausible transitions
- <5% negation false positives
- 100% unit conversion accuracy

#### Day 4: Comprehensive Report Generation
- [ ] Implement `validation/report_generator.py`
  - [ ] HTML report with all metrics
  - [ ] Bland-Altman plots
  - [ ] Distribution plots
  - [ ] Error taxonomy tables
  - [ ] Comparison to published benchmarks
- [ ] Generate `validation_report.html`
- [ ] Write limitations section

**Deliverable:** Publication-quality validation report

#### Day 5: Orchestrator & Final Testing
- [ ] Complete `module_03_vitals_processing.py`
  - [ ] Full pipeline execution
  - [ ] Checkpoint/resume functionality
  - [ ] Progress tracking
  - [ ] Error handling
- [ ] Run end-to-end integration tests
- [ ] Performance benchmarking
- [ ] Documentation finalization

**Deliverable:** Complete validated Module 3

**Success Criteria:**
- All validation targets met (≥90% accuracy)
- Full pipeline runs successfully
- Comprehensive documentation
- Publication-ready validation report

#### Week 10 Checkpoint
**Review Questions:**
- Are all validation targets achieved?
- Is the pipeline robust and well-tested?
- Is documentation complete and clear?
- Is the system ready for production use?

**Final Deliverables:**
- `vitals_features_final.h5` (primary output)
- `validation_report.html` (publication supplement)
- Complete test suite (all tests passing)
- User documentation (API, User Guide)

---

## Buffer Period (Weeks 11-12)

**Purpose:** Contingency time for addressing any issues

**Common Use Cases:**
- Pattern refinement if accuracy <90%
- Performance optimization
- Additional validation
- Bug fixes
- Documentation improvements
- Integration with downstream modules

---

## Success Metrics by Phase

### Phase 1 (Weeks 1-5): Extraction
| Metric | Target |
|--------|--------|
| Phy.txt extraction time | <30 min |
| Hnp.csv extraction time | <2 hours |
| Prg.csv extraction time | <8 hours |
| Coverage (patients with any vitals) | ≥95% |
| Initial cross-validation agreement | ≥80% |

### Phase 2 (Weeks 6-7): Integration
| Metric | Target |
|--------|--------|
| Patients with admission vitals | ≥70% |
| Overall data completeness | ≥80% |
| Conflict rate | <20% |
| Feature generation time | <1 hour |

### Phase 3 (Weeks 8-10): Validation
| Metric | Target |
|--------|--------|
| Cross-validation clinical agreement | ≥90% |
| Manual review accuracy | ≥90% |
| Inter-rater reliability (κ) | ≥0.80 |
| Out-of-range values | <1% |
| Implausible transitions | <1% |
| Negation false positives | <5% |

---

## Risk Mitigation Timeline

| Week | Primary Risk | Mitigation Action |
|------|-------------|-------------------|
| 3 | H&P accuracy <85% | Additional pattern development in Week 4 |
| 5 | Prg extraction too slow | Implement Dask parallelization |
| 7 | Feature bugs | Extensive unit testing, sample validation |
| 9 | IRR <0.80 | Adjudication, codebook refinement, re-review |
| 10 | Accuracy <90% | Use buffer weeks for pattern refinement or NER |

---

## Communication & Reporting

### Weekly Deliverables
- [ ] Progress report (accomplishments, blockers, next steps)
- [ ] Updated metrics dashboard
- [ ] Code pushed to repository
- [ ] Tests passing

### Bi-weekly Check-ins
- [ ] Demo current functionality
- [ ] Review metrics vs. targets
- [ ] Discuss any architecture changes
- [ ] Plan next sprint

### Final Presentation (Week 10)
- [ ] Architecture overview
- [ ] Validation results
- [ ] Clinical insights
- [ ] Next steps for modeling integration

---

## Development Best Practices

### Version Control
- [ ] Commit daily with descriptive messages
- [ ] Branch for each submodule (`feature/3.1-phy-extractor`)
- [ ] Pull request reviews before merging to main
- [ ] Tag releases (`v1.0-extraction-complete`)

### Testing
- [ ] Write tests BEFORE implementation (TDD)
- [ ] Unit tests for each function
- [ ] Integration tests for each submodule
- [ ] End-to-end tests for full pipeline
- [ ] Target: >80% code coverage

### Documentation
- [ ] Docstrings for all functions (Google style)
- [ ] Update API.md as functions are added
- [ ] Comment complex logic
- [ ] Update USER_GUIDE.md with examples

### Code Quality
- [ ] Use type hints
- [ ] Follow PEP 8 style
- [ ] Use Black for formatting
- [ ] Run flake8 linter
- [ ] Code reviews for all PRs

---

## Handoff Checklist (End of Week 10)

### Code Deliverables
- [ ] All submodules implemented and tested
- [ ] All tests passing (>80% coverage)
- [ ] Code formatted and linted
- [ ] Repository clean (no debug code)

### Data Deliverables
- [ ] `vitals_features_final.h5` generated
- [ ] `validation_report.html` complete
- [ ] Quality metrics documented
- [ ] Sample data for testing

### Documentation Deliverables
- [ ] ARCHITECTURE.md (complete)
- [ ] API.md (all functions documented)
- [ ] USER_GUIDE.md (with examples)
- [ ] VALIDATION_PROTOCOL.md (detailed methods)
- [ ] README.md (quick start guide)

### Knowledge Transfer
- [ ] Code walkthrough session
- [ ] Validation results presentation
- [ ] Known issues documented
- [ ] Future enhancements roadmap

---

## Appendix: Daily Standup Template

**What I accomplished yesterday:**
-

**What I plan to accomplish today:**
-

**Blockers/concerns:**
-

**Metrics update:**
- Tests passing: X/Y
- Code coverage: Z%
- Current task progress: N%

---

**Document Version:** 1.0
**Last Updated:** 2025-11-09
**Next Review:** Weekly during implementation
