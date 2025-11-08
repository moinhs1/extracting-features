# PE Trajectory Pipeline - Quick Reference Checklist

## 📋 Module Overview at a Glance

| Module | Purpose | Input Size | Output Size | Runtime | Status |
|--------|---------|------------|-------------|---------|--------|
| **1. Core Infrastructure** | Time Zero, temporal reference, outcomes | PE dataset | ~50 MB + PKL | 3-5 min | ✅ Complete |
| **2. Lab Processing** | Extract labs with QC, temporal features | 16 GB | ~2 GB | 15 min | ⬜ Not Started |
| **3. Vitals Extraction** | Parse vital values from text, hourly grid | 44 GB | ~5 GB | 45 min | ⬜ Not Started |
| **4. Medication Processing** | Temporal meds, dosing, response | 7.4 GB | ~500 MB | 8 min | ⬜ Not Started |
| **5. Clinical NLP** | Extract features from 92 GB notes | 92 GB | ~2 GB | 30-180 min | ⬜ Not Started |
| **6. Temporal Alignment** | Align all sources to hourly grid | All above | ~10 GB | 15 min | ⬜ Not Started |
| **7. Trajectory Features** | Rolling windows, change points, CSD | Module 6 | ~5 GB | 45 min | ⬜ Not Started |
| **8. Format Conversion** | GRU-D, GBTM, ML formats | Modules 6-7 | ~30 GB | 25 min | ⬜ Not Started |

**Total Pipeline: ~3-6 hours runtime, ~70 GB storage**

---

## 🎯 Critical Questions - ANSWER BEFORE STARTING

### Before Module 1 (REQUIRED):
- [ ] **Q1:** Admission time column name in PE dataset? _____________
- [ ] **Q2:** Discharge time column name in PE dataset? _____________  
- [ ] **Q3:** Do you have outcomes data? Where? _____________
- [ ] **Q4:** Derive outcomes from procedure codes? Yes/No _____________

### Before Module 3 (REQUIRED):
- [ ] **Q5:** Provide 5-10 sample rows from Vitals `Report_Text` field
  ```
  Example 1: ___________________________________
  Example 2: ___________________________________
  Example 3: ___________________________________
  ...
  ```

### Strategic Decisions:
- [ ] **Q6:** NLP level for Module 5?
  - [ ] Basic (fast, keyword-based)
  - [ ] Hybrid (moderate, NER + rules)
  - [ ] Advanced (slow, Clinical-Longformer)
  - GPU available? Yes/No _____________

- [ ] **Q7:** Which trajectory methods?
  - [ ] GRU-D
  - [ ] GBTM/lcmm
  - [ ] XGBoost
  - [ ] Neural CDEs
  - [ ] FDA/PACE
  - [ ] Dynamic Time Warping
  - [ ] Other: _____________

- [ ] **Q8:** Train/val/test split strategy?
  - [ ] Random 70/15/15
  - [ ] Temporal split
  - [ ] Other: _____________

---

## 📊 Data Flow Diagram

```
RAW DATA (120 GB)
    ↓
MODULE 1: Temporal Reference
    ↓
    ├─→ MODULE 2: Labs (16 GB) ──┐
    ├─→ MODULE 3: Vitals (44 GB) ─┤
    ├─→ MODULE 4: Meds (7.4 GB) ──┼─→ MODULE 6: Alignment
    └─→ MODULE 5: Notes (92 GB) ──┘           ↓
                                    MODULE 7: Trajectory Features
                                              ↓
                                    MODULE 8: Format Conversion
                                              ↓
                                    READY FOR ANALYSIS
```

---

## ⚠️ Red Flags Fixed by Each Module

| Red Flag | Fixed By | How |
|----------|----------|-----|
| Vitals only counting, not extracting | Module 3 | Parse Report_Text, extract HR/BP/RR/SpO2/Temp |
| No outcomes data | Module 1 | Extract or derive from codes |
| No temporal alignment | Module 6 | Common hourly grid, synchronized timestamps |
| No missing data encoding | Module 2,3,6 | Masks + time-since-last |
| Primitive keyword search | Module 5 | Advanced NLP with entities, temporal extraction |
| No QC thresholds | Module 2,3 | PE-specific physiological ranges |
| No trajectory features | Module 7 | Rolling windows, change points, CSD indicators |
| Wrong output formats | Module 8 | GRU-D tensors, GBTM long format, etc. |
| No medication dosing/timing | Module 4 | Temporal tracking, cumulative exposure |
| No radiology/echo | Future work | Separate extraction pipeline needed |

---

## 🚀 Implementation Checklist

### Week 1: Foundation ✅ COMPLETE
- [x] Answer Q1-Q4 above
- [x] Build Module 1
- [x] Validate temporal references
- [x] Confirm outcomes strategy
- [x] **Checkpoint:** Review patient_timelines.pkl, cohort stats
  - **Status:** Module 1 V2.0 complete with patient_timelines.pkl
  - **Test Results:** 10 patients, 100% encounter matching, all outcomes extracted
  - **Ready for:** Module 2 (Lab Processing)

### Week 2-3: Data Extraction
- [ ] Answer Q5 (vital samples)
- [ ] Build Module 2 (Labs)
- [ ] Build Module 3 (Vitals) - requires Q5 first!
- [ ] Build Module 4 (Meds)
- [ ] Decide NLP level (Q6)
- [ ] Build Module 5 (Notes)
- [ ] **Checkpoint:** Review QC reports from each module

### Week 4: Integration
- [ ] Build Module 6 (Alignment)
- [ ] Build Module 7 (Trajectory Features)
- [ ] **Checkpoint:** Validate aligned_trajectories.h5

### Week 5: Analysis Prep
- [ ] Decide which methods (Q7)
- [ ] Build Module 8 (Format Conversion)
- [ ] Create train/val/test splits (Q8)
- [ ] **Checkpoint:** Test formats with small model runs

### Week 5-6: Validation
- [ ] Run QC validation
- [ ] Generate visualizations
- [ ] Document pipeline
- [ ] **Final Deliverable:** Complete pipeline ready for trajectory modeling

---

## 💾 Storage Requirements

```
Working Space Needed:
├── Intermediate HDF5 files: ~15-30 GB
├── Method-specific formats: ~20-40 GB
├── QC reports/plots: ~1-2 GB
├── Logs: ~100 MB
└── TOTAL: ~40-70 GB free space required
```

---

## ⏱️ Estimated Timelines

**Optimistic (Everything works first try):** 3-4 weeks

**Realistic (Debugging, validation, iterations):** 5-6 weeks

**Conservative (Issues, missing data, multiple NLP approaches):** 7-8 weeks

---

## 📞 Decision Points Where You'll Need to Provide Input

1. **After Module 1:** Review temporal stats, confirm Time Zero makes sense
2. **Before Module 3:** MUST provide vital text samples
3. **After Module 2:** Review lab QC, adjust thresholds if needed
4. **After Module 5 Level 1:** Decide if upgrade to advanced NLP
5. **After Module 7:** Review trajectory features, add custom features?
6. **Before Module 8:** Confirm which analysis methods to support

---

## 🎓 Learning Opportunities

Each module teaches something about clinical data:
- **Module 2:** Lab kinetics, PE biomarker behavior
- **Module 3:** Vital sign patterns, hemodynamic monitoring
- **Module 4:** Anticoagulation strategies, treatment timing
- **Module 5:** Clinical documentation patterns, decision-making
- **Module 6:** Temporal data alignment challenges
- **Module 7:** What makes a trajectory "unstable" or "deteriorating"
- **Module 8:** How different ML methods consume temporal data

---

## ✅ Success Criteria

**Module 1:** 
- ✓ >95% patients have valid PE diagnosis time
- ✓ Temporal statistics look reasonable
- ✓ Outcomes data identified/extracted

**Module 2:**
- ✓ >90% critical labs extracted
- ✓ <1% values flagged as outliers
- ✓ Temporal features calculated successfully

**Module 3:**
- ✓ Vital values parsed with >90% success rate
- ✓ Hourly time series created
- ✓ QC thresholds applied

**Module 4:**
- ✓ Medication categories identified
- ✓ Temporal tracking working
- ✓ Treatment response features calculated

**Module 5:**
- ✓ Note features extracted
- ✓ Temporal mentions tracked
- ✓ (If advanced) Embeddings generated

**Module 6:**
- ✓ All sources aligned to common grid
- ✓ Missing data encoded systematically
- ✓ Integrated tensor created successfully

**Module 7:**
- ✓ Trajectory features calculated
- ✓ Features validated against clinical expectations
- ✓ Ready for downstream modeling

**Module 8:**
- ✓ All required formats generated
- ✓ Formats validated with test runs
- ✓ Splits consistent across formats

---

## 📝 Notes Section (Your answers)

**Outcomes data location:**
_____________________________________________

**Vital text format notes:**
_____________________________________________

**NLP decision reasoning:**
_____________________________________________

**Priority trajectory methods:**
_____________________________________________

**Other considerations:**
_____________________________________________

---

## 🔄 Next Immediate Steps

1. **Read full architecture document:** pe_trajectory_pipeline_architecture.md
2. **Answer Q1-Q4** (required for Module 1)
3. **Get vital text samples** (Q5 - needed before Module 3)
4. **Confirm you're ready to start Module 1**
5. **I'll write Module 1 code**

**Ready to proceed? Let me know your answers to Q1-Q4!**
