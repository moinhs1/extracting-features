# Module 3 Brainstorming Session Summary

**Date:** 2025-11-09
**Session Goal:** Design comprehensive vitals acquisition and storage architecture
**Outcome:** ✅ Complete architecture design with actionable implementation plan

---

## Session Overview

We conducted a comprehensive brainstorming session to design Module 3 (Vitals Processing) for the TDA 11.1 PE prediction project. The session proceeded through:

1. **Discovery Phase**: Explored where vitals data is located in the TDA system
2. **Requirements Gathering**: Asked 10 clarifying questions to understand scope and priorities
3. **Design Phase**: Created comprehensive architecture based on your answers
4. **Documentation Phase**: Generated complete technical specifications

---

## Key Discoveries

### Vitals Data Sources Identified

**Primary Source: Phy.txt (Structured)**
- Path: `/home/moin/TDA_11_1/Data/FNR_20240409_091633_Phy.txt`
- Volume: 33M rows
- Content: Structured vital signs from outpatient and inpatient encounters
- Coverage: Temperature (2.5M), HR (2.4M), BP (1.7M), RR (1.3M), SpO2 (1.6M)
- Extraction: Direct structured field parsing (easy)

**Secondary Source: Hnp.csv (H&P Notes)**
- Path: `/home/moin/TDA_11_1/Data/Hnp.csv`
- Volume: 257K notes (3.4 GB)
- Content: Unstructured History & Physical notes
- Coverage: ~80% contain vitals
- **Critical Value**: Admission baseline vitals at PE presentation
- Extraction: Advanced NLP/regex (challenging)

**Tertiary Source: Prg.csv (Progress Notes)**
- Path: `/home/moin/TDA_11_1/Data/Prg.csv`
- Volume: 8.7M notes (42 GB)
- Content: Unstructured progress notes from hospitalizations
- Coverage: ~35% contain vitals
- **Critical Value**: Serial measurements for disease trajectory
- Extraction: Advanced NLP/regex with range/narrative handling (very challenging)

### Why All Three Sources Matter

**Complementary Coverage:**
- Phy.txt: Pre-admission baseline (outpatient)
- Hnp.csv: Admission presentation (disease severity)
- Prg.csv: Post-admission trajectory (treatment response)

**Clinical Significance:**
- Patients with only Phy.txt → Outpatient-managed PE (less severe)
- Patients with all three sources → Full disease progression (most severe)
- This pattern itself is a predictive feature

---

## Your Requirements (Q1-Q10 Answers)

### Q1: Data Source Strategy
**Answer:** All sources (comprehensive)
**Implication:** Extract from Phy.txt + Hnp.csv + Prg.csv for maximum coverage

### Q2: NLP Extraction Sophistication
**Answer:** Advanced NLP approach
**Implication:** Context-aware extraction with medical domain knowledge, not just simple regex

### Q3: Data Fusion Strategy
**Answer:** Extract all values with timestamps first, THEN do contextual selection
**Implication:**
- Raw Layer 1: Keep source-specific values
- Layer 2: Contextual selection (admission from Hnp, baseline from Phy, trajectory from Prg)

### Q4: Admission Vitals Handling
**Answer:** Make separate columns for admission vitals
**Implication:** ADMISSION_HR, ADMISSION_SBP, etc. as distinct features

### Q5: Temporal Resolution
**Answer:** Maximum resolution - 5-minute intervals when available
**Implication:** High-resolution acute phase grid to capture rapid physiological changes

### Q6: Data Fusion Granularity
**Answer:** Option A+: Maximum Information Preservation with Source Provenance Tracking
**Implication:** 6-layer architecture preserving:
1. Raw source-specific values
2. Merged values with source attribution
3. Conflict detection & quality metrics
4. Temporal precision tracking
5. Temporal consistency validation
6. Encounter pattern features

### Q7: Validation Strategy
**Answer:** Most rigorous - 4-tier hierarchical validation
**Implication:**
- Tier 1: Cross-validation with structured Phy.txt data
- Tier 2: Strategic manual review (200 notes, dual annotation, IRR)
- Tier 3: Statistical monitoring (distributions, outliers, temporal plausibility)
- Tier 4: Pattern-specific (negation, ranges, unit conversion)

### Q8: Implementation Approach
**Answer:** Complete solution, broken into actionable submodules
**Implication:** 10 independent submodules enabling parallel development and testing

### Q9: Computational Constraints
**Answer:** No constraints
**Implication:** Can process all 8.7M progress notes, use high temporal resolution, store all provenance layers

### Q10: Clinical Value Priority
**Answer:** All sources valuable
**Implication:** Each source provides unique clinical signal - don't compromise on any

---

## Design Decisions Made

### Architecture: Six-Layer Information Preservation

Rather than merging data and losing information, we designed a system that preserves **six layers of information**:

**Layer 1: Raw Source-Specific Values**
- HR_phy, HR_hnp, HR_prg with original timestamps
- Enables reprocessing without re-extraction

**Layer 2: Hierarchical Merged Values + Source Attribution**
- Single "best" value per vital/timepoint
- Priority: Prg > Hnp > Phy (inpatient > admission > outpatient)
- Track which source provided value and how many sources available

**Layer 3: Conflict Detection & Quality Metrics**
- Flag when sources disagree beyond clinical thresholds
- Calculate completeness per source
- Quantify data redundancy

**Layer 4: Temporal Precision Tracking**
- Time delta from each measurement to target grid point
- Average and maximum offsets per patient
- Exact timestamps preserved

**Layer 5: Temporal Consistency Validation**
- Rate of change between consecutive measurements
- Flag physiologically implausible transitions
- Track temporal coverage and density

**Layer 6: Encounter Pattern Features**
- Binary flags: has_outpatient_vitals, has_admission_vitals, has_inpatient_vitals
- Categorical: encounter_pattern (outpatient_only → full_trajectory)
- This single feature captures disease severity

### Modular Design: 10 Independent Submodules

**Phase 1: Extraction (Parallel)**
- 3.1: Phy.txt extractor (structured) - 2 days
- 3.2: Hnp.csv NLP extractor (admission vitals) - 5-7 days
- 3.3: Prg.csv NLP extractor (trajectory) - 7-10 days

**Phase 2: Harmonization (Sequential)**
- 3.4: Vitals harmonizer (map variants to canonical) - 2-3 days
- 3.5: Unit converter & QC filter - 2-3 days

**Phase 3: Integration (Sequential)**
- 3.6: Multi-source temporal aligner - 5-7 days
- 3.7: Provenance & quality metrics calculator - 4-5 days
- 3.8: Feature engineering pipeline - 6-8 days

**Phase 4: Validation**
- 3.9: 4-tier validation framework - 8-10 days
- 3.10: Main orchestrator - 3-4 days

**Total: 10 weeks** (includes parallelization + 2-week buffer)

### Temporal Phases with Maximum Resolution

**BASELINE**: [-365d, -30d] @ daily
**PRE_ACUTE**: [-30d, -7d] @ daily
**ACUTE**: [-7d, +1d] @ hourly
**HIGH_RES_ACUTE**: [-24h, +24h] @ **5 minutes** ← Your requirement
**SUBACUTE**: [+2d, +14d] @ hourly
**RECOVERY**: [+15d, +90d] @ daily

### Advanced NLP Features

**Context Awareness:**
- Negation detection: "blood pressure not obtained" → don't extract
- Historical filtering: "prior HR 88" → exclude
- Range parsing: "HR 70-85" → extract (min=70, mean=77.5, max=85)
- Narrative interpretation: "tachycardic" → HR~110 with confidence <1.0

**Multi-Pattern Matching:**
- Each vital has 5-10 regex pattern variants
- Patterns account for spacing, punctuation, unit variations
- Context validators reduce false positives

**Quality Tracking:**
- Confidence scores: 1.0 for structured, <1.0 for NLP
- Extraction flags: warnings for edge cases
- Pattern attribution: which regex matched

### Comprehensive Feature Engineering

**Basic Statistics (per vital, per phase):**
- mean, median, min, max, std, first, last, n_measurements, time_coverage

**Trajectory Features:**
- slope (linear regression), direction (increasing/decreasing/stable)
- volatility (coefficient of variation), range (max-min)
- time_to_normalization (hours until vital enters normal range)

**Clinical Composites:**
```python
shock_index = HR / SBP  # >1.0 = shock
pulse_pressure = SBP - DBP  # <25 = low cardiac output
MAP = DBP + (SBP - DBP) / 3  # <65 = inadequate perfusion
modified_shock_index = HR / MAP
delta_index = HR - RR  # Negative suggests severe PE
```

**Clinical Flags (per phase):**
- any_tachycardia (binary), prop_tachycardia (proportion), max_tachycardia_duration (hours)
- Same for: tachypnea, hypoxemia, hypotension, fever, bradycardia, hypothermia

**Special: Admission Vitals**
- ADMISSION_* features from first H&P note after index
- Separate from ACUTE phase aggregates
- Critical for PE severity assessment

---

## Validation Framework

### Tier 1: Cross-Validation (Objective)

**Method:** Match note extractions to structured Phy.txt values
**Coverage:** ~30-40% of extractions (where overlap exists)
**Metrics:**
- Clinical agreement (within tolerance): Target ≥90%
- Pearson correlation: Target r ≥0.90
- Mean Absolute Error: HR within 5 bpm, BP within 10 mmHg
- Bland-Altman: bias near zero with tight limits

**Use:** Primary quantitative evidence of extraction accuracy

### Tier 2: Manual Review (Gold Standard)

**Method:** Dual independent annotation of 200 stratified notes
**Strata:**
- 50 high-discrepancy (cross-val failures)
- 75 no structured match (validate uncovered cases)
- 40 critical values (extreme vitals)
- 35 edge cases (negation, ranges, unusual)

**Metrics:**
- Overall accuracy: Target ≥90%
- Inter-rater reliability: κ ≥0.80, ICC ≥0.90
- Error taxonomy: categorize and prioritize fixes

**Use:** Validate extraction on cases Tier 1 can't cover; guide pattern refinement

### Tier 3: Statistical Monitoring (Population-Level)

**Method:** Compare distributions and temporal patterns to reference populations
**Checks:**
- KS test vs. MIMIC-III / published PE cohorts
- Outlier rate: <2% with modified Z-score >3.5
- Out-of-range values: <1%
- Implausible temporal transitions: <1%
- Digit preference analysis

**Use:** Detect systematic biases or extraction artifacts

### Tier 4: Pattern-Specific (Edge Case)

**Method:** Targeted validation of known challenging patterns
**Checks:**
- Negation handling: <5% false positive rate
- Range handling: consistency check
- Unit conversion: 100% accuracy on explicit units

**Use:** Ensure specific extraction challenges are handled correctly

### Integration

All four tiers provide **independent lines of evidence** that complement each other:
- Tier 1 provides objective metrics but limited coverage
- Tier 2 provides gold standard on cases Tier 1 misses
- Tier 3 catches population-level biases invisible to sampling
- Tier 4 validates specific known failure modes

**Combined Target:** ≥90% accuracy across all vitals, supported by multiple validation methods

---

## Documentation Created

### 1. ARCHITECTURE.md (~12,000 words)
Complete technical specification including:
- Clinical context & goals
- Detailed data source comparison (Phy vs Hnp vs Prg)
- 6-layer information architecture
- 10 submodules with:
  - Purpose, complexity, time estimates
  - Key functions with signatures
  - Input/output schemas
  - Dependencies
  - Test file references
- 4-tier validation framework with benchmarks
- Design decisions & rationale
- Risk mitigation strategies
- Future enhancements
- Configuration formats (YAML, JSON)
- HDF5 output schema
- Success criteria tables
- Appendices (canonical names, clinical formulas, error taxonomy)

### 2. SUBMODULES_QUICK_REFERENCE.md
Fast lookup guide with:
- Dependency graph (visual)
- Summary table (complexity, time, parallelization)
- Quick reference card per submodule (1 page each)
- Critical paths (fast/MVP/complete)
- Data flow diagram
- Storage requirements
- Quick start commands
- Common issues & solutions

### 3. IMPLEMENTATION_ROADMAP.md
Week-by-week plan with:
- 10-week timeline with daily task breakdowns
- Deliverables per week
- Success criteria & checkpoint questions
- Risk mitigation timeline
- Testing & documentation best practices
- Handoff checklist
- Daily standup template

### 4. README.md
Project overview with:
- Quick start guide
- Architecture summary
- Usage examples (CLI + Python API)
- Configuration guide
- Performance benchmarks
- Success criteria
- Known limitations
- Future enhancements
- Dependencies
- Testing instructions

### 5. BRAINSTORMING_SESSION_SUMMARY.md (this document)
Session record capturing:
- Discovery findings
- Your requirements (Q1-Q10)
- Design decisions made
- Documentation created
- Next steps

---

## Key Metrics & Targets

### Data Quality
- Extraction accuracy: ≥90% (cross-validation + manual review)
- Inter-rater reliability: κ ≥0.80
- Out-of-range values: <1%
- Implausible transitions: <1%

### Coverage
- Patients with any vitals: ≥95%
- Patients with admission vitals: ≥70%
- Patients with high-res acute vitals: ≥50%
- Overall data completeness: ≥80%

### Performance
- Phy.txt extraction: <30 min
- Hnp.csv extraction: <2 hours
- Prg.csv extraction: <8 hours
- Full pipeline end-to-end: <12 hours
- Total storage: ~24 GB

---

## Implementation Paths

### Option 1: Fast Path (4 weeks)
```
3.1 (Phy only) → 3.4 → 3.5 → 3.6 → 3.7 → 3.8
```
**Deliverable:** Structured vitals only (lower coverage but fast validation)

### Option 2: Minimal Viable Product (6 weeks)
```
3.1 + 3.2 (Phy + Hnp) → 3.4 → 3.5 → 3.6 → 3.7 → 3.8
```
**Deliverable:** Structured + admission vitals (good coverage + PE severity)

### Option 3: Complete System (10 weeks) ← **RECOMMENDED**
```
3.1 + 3.2 + 3.3 → 3.4 → 3.5 → 3.6 → 3.7 → 3.8 → 3.9
```
**Deliverable:** All sources + full validation (publication-ready)

**Your Choice:** Option 3 (complete solution)

---

## Critical Success Factors

### Technical
1. **NLP Pattern Quality**: Iterative refinement based on validation
2. **Temporal Alignment**: Correct implementation of multi-resolution grids
3. **Provenance Tracking**: Maintaining data lineage through all transformations
4. **Validation Rigor**: Achieving ≥90% across all tiers

### Operational
1. **Modular Testing**: Each submodule independently validated
2. **Checkpointing**: Resume capability for long-running processes
3. **Error Handling**: Graceful failure and logging for debugging
4. **Documentation**: Keep API docs and examples current

### Clinical
1. **Feature Relevance**: PE-specific composites (shock index, delta index)
2. **Temporal Context**: Admission vitals vs. trajectory vs. baseline
3. **Interpretability**: Source provenance and quality metrics for model confidence
4. **Clinical Validation**: Ensure features align with known PE pathophysiology

---

## Risks & Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| NLP accuracy <90% | Medium | High | Iterative refinement; upgrade to medical NER if needed |
| Prg extraction too slow | Medium | Low | Dask parallelization; user has no constraints |
| Cross-val coverage low | Medium | Medium | Increase manual review sample; use statistical validation |
| High conflict rate | Low | Medium | Investigate patterns; implement confidence weighting |
| Manual review extends timeline | Medium | Low | Built-in 2-week buffer; recruit additional reviewers |

---

## Next Steps

### Immediate (This Week)
1. **Review design documents** with clinical team and stakeholders
2. **Set up development environment** (Python, dependencies, data access)
3. **Create sample test dataset** (1000 patients) for rapid iteration
4. **Begin Week 1 implementation**: Submodule 3.1 (Phy.txt extraction)

### Week 1-2 (Foundation)
- Implement structured data pipeline (3.1, 3.4, 3.5)
- Establish testing patterns and code structure
- Validate end-to-end structured pipeline

### Week 3-5 (NLP Extraction)
- Develop and refine NLP extractors (3.2, 3.3)
- Iterative pattern development based on samples
- Initial cross-validation

### Week 6-7 (Integration)
- Multi-source temporal alignment (3.6)
- Provenance and feature engineering (3.7, 3.8)
- Generate final feature set

### Week 8-10 (Validation)
- 4-tier comprehensive validation (3.9)
- Pattern refinement to achieve ≥90% accuracy
- Generate validation report and finalize orchestrator (3.10)

### Week 11-12 (Buffer)
- Address any validation gaps
- Performance optimization
- Documentation finalization
- Integration with downstream modeling modules

---

## Questions Answered

During this brainstorming session, we systematically addressed:

✅ **Where are vitals located?** → Phy.txt (structured), Hnp.csv (H&P notes), Prg.csv (progress notes)
✅ **Which sources to use?** → All three (comprehensive coverage)
✅ **How sophisticated should NLP be?** → Advanced (context-aware, negation handling)
✅ **How to merge sources?** → 6-layer provenance preservation
✅ **How to handle admission vitals?** → Separate ADMISSION_* features
✅ **What temporal resolution?** → Maximum (5-minute bins in acute phase)
✅ **How to validate?** → 4-tier rigorous framework (≥90% target)
✅ **How to structure implementation?** → 10 independent submodules
✅ **What about computational constraints?** → None (can process everything)
✅ **Which sources are most valuable?** → All provide unique clinical signal

---

## Session Outcome

✅ **Complete Architecture**: 6-layer information preservation system
✅ **Actionable Plan**: 10 submodules with week-by-week roadmap
✅ **Rigorous Validation**: 4-tier framework achieving ≥90% accuracy
✅ **Comprehensive Documentation**: ~20,000 words across 5 documents
✅ **Clear Next Steps**: Begin Week 1 (Submodule 3.1)

**Status:** ✅ Design Phase Complete → Ready for Implementation

---

## Folder Structure Created

```
module_3_vitals_processing/
├── docs/
│   ├── ARCHITECTURE.md                 # Complete technical design
│   ├── SUBMODULES_QUICK_REFERENCE.md   # Fast lookup guide
│   ├── IMPLEMENTATION_ROADMAP.md       # Week-by-week plan
│   └── BRAINSTORMING_SESSION_SUMMARY.md # This document
├── extractors/                         # Submodules 3.1-3.3
├── processing/                         # Submodules 3.4-3.8
├── validation/                         # Submodule 3.9
├── utils/                              # Helper functions
├── tests/                              # Comprehensive test suite
├── outputs/
│   ├── discovery/                      # Intermediate outputs
│   ├── features/                       # Final features
│   └── validation/                     # Validation reports
├── config/                             # Configuration files
├── module_03_vitals_processing.py      # Main orchestrator
├── requirements.txt                    # Dependencies
└── README.md                           # Project overview
```

---

## Final Recommendations

### Priority 1: Clinical Review
- Have clinical team review ARCHITECTURE.md Section 1 (Clinical Context)
- Validate that PE-specific features (shock index, delta index) are appropriate
- Confirm temporal phases align with PE pathophysiology
- Verify QC thresholds are clinically sensible

### Priority 2: Sample Data Validation
- Extract vitals for 10-20 patients manually (spot check)
- Verify structured extraction looks correct
- Test initial NLP patterns on 5-10 sample notes
- Ensure output formats are usable

### Priority 3: Test-Driven Development
- Write tests BEFORE implementing each function
- Use sample data for rapid iteration
- Achieve >80% code coverage
- Integration tests per submodule

### Priority 4: Continuous Validation
- Don't wait until Week 8 to validate
- Cross-validate each extractor immediately after development
- Manual review small samples continuously
- Statistical checks on every intermediate output

### Priority 5: Documentation as Code
- Keep API.md current as functions are written
- Update examples in USER_GUIDE.md
- Document design decisions in comments
- Maintain changelog

---

**Session Complete:** 2025-11-09
**Next Action:** Begin Week 1 - Submodule 3.1 (Phy.txt extraction)
**Estimated Completion:** Week 10 (with 2-week buffer through Week 12)
