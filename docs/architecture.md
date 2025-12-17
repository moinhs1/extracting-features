# System Architecture
*Last Updated: 2025-12-17*

## High-Level Pipeline Overview

```mermaid
flowchart TB
    subgraph DataSources["Data Sources"]
        RPDR[(RPDR Data<br/>Enc, Prc, Dia, Med, Dem)]
        Gemma[(Gemma Predictions<br/>PE Classification)]
        Combined[(Combined Predictions<br/>Report Timestamps)]
    end

    subgraph Module1["Module 1: Core Infrastructure"]
        M1_Load[Load PE Cohort<br/>Filter Gemma_PE_Present=True]
        M1_Match[4-Tier Encounter Matching]
        M1_Extract[Outcome Extraction<br/>Mortality, ICU, Interventions]
        M1_Timeline[Create Patient Timelines]
    end

    subgraph Module2["Module 2: Laboratory Processing"]
        M2_Phase1[Phase 1: Discovery<br/>Scan Lab Data]
        M2_Tier1[Tier 1: LOINC Exact Match]
        M2_Tier3[Tier 3: Hierarchical Clustering]
        M2_Phase2[Phase 2: Feature Engineering<br/>72 Features per Test]
    end

    subgraph Module3["Module 3: Vitals Processing"]
        M3_Extract[Extract Vital Signs<br/>HR, BP, SpO2, RR, Temp]
        M3_Features[Temporal Features]
    end

    subgraph Module4["Module 4: Medication Processing ✅"]
        M4_Extract[Layer 1: Canonical Extract<br/>1.71M records]
        M4_RxNorm[RxNorm Mapping<br/>92.4% coverage]
        M4_Class[Layer 2: Therapeutic Classes<br/>53 indicators]
        M4_Indiv[Layer 3: Individual Meds<br/>581 indicators]
        M4_Embed[Layer 4: Embeddings<br/>769 + 1,582]
        M4_Dose[Layer 5: Dose Intensity<br/>86K records, 97.2% DDD]
    end

    subgraph Module6["Module 6: Procedure Encoding ✅ NEW"]
        M6_Extract[Layer 1: Canonical Extract<br/>22M records, 7 temporal flags]
        M6_Map[CCS + SNOMED Mapping<br/>85% target coverage]
        M6_CCS[Layer 2: CCS Indicators<br/>~230 categories]
        M6_PE[Layer 3: PE Features<br/>63+ clinical features]
        M6_Embed[Layer 4: Embeddings<br/>HDF5 output]
        M6_World[Layer 5: World Model<br/>Actions + States]
    end

    subgraph Outputs["Output Files"]
        OUT_CSV[(outcomes.csv<br/>8,713 patients)]
        OUT_PKL[(patient_timelines.pkl)]
        OUT_LAB[(lab_features.csv)]
        OUT_H5[(lab_sequences.h5)]
        OUT_MED[(medication outputs)]
        OUT_PRC[(procedure outputs)]
    end

    %% Data flow
    Gemma --> M1_Load
    Combined --> M1_Load
    RPDR --> M1_Match
    M1_Load --> M1_Match
    M1_Match --> M1_Extract
    RPDR --> M1_Extract
    M1_Extract --> M1_Timeline
    M1_Timeline --> OUT_CSV
    M1_Timeline --> OUT_PKL

    OUT_PKL --> M2_Phase1
    RPDR --> M2_Phase1
    M2_Phase1 --> M2_Tier1
    M2_Phase1 --> M2_Tier3
    M2_Tier1 --> M2_Phase2
    M2_Tier3 --> M2_Phase2
    M2_Phase2 --> OUT_LAB
    M2_Phase2 --> OUT_H5

    OUT_PKL --> M3_Extract
    RPDR --> M3_Extract
    M3_Extract --> M3_Features

    OUT_PKL --> M4_Extract
    RPDR --> M4_Extract
    M4_Extract --> M4_RxNorm
    M4_RxNorm --> M4_Class
    M4_Class --> M4_Indiv
    M4_Class --> M4_Embed
    M4_Class --> M4_Dose
    M4_Indiv --> OUT_MED
    M4_Embed --> OUT_MED
    M4_Dose --> OUT_MED

    OUT_PKL --> M6_Extract
    RPDR --> M6_Extract
    M6_Extract --> M6_Map
    M6_Map --> M6_CCS
    M6_CCS --> M6_PE
    M6_PE --> M6_Embed
    M6_PE --> M6_World
    M6_Embed --> OUT_PRC
    M6_World --> OUT_PRC
```

## Module 1: Core Infrastructure

```mermaid
flowchart LR
    subgraph Input["Input Data"]
        A1[ALL_PE_POSITIVE_With_Gemma_Predictions.csv]
        A2[Combined_PE_Predictions_All_Cohorts.txt]
        A3[Enc.txt - Encounters]
        A4[Prc.txt - Procedures]
        A5[Dia.txt - Diagnoses]
        A6[Med.txt - Medications]
        A7[Dem.txt - Demographics]
    end

    subgraph Cohort["Cohort Definition"]
        B1[Filter Gemma_PE_Present = True]
        B2[Merge for Report_Date_Time]
        B3[13,638 reports → 8,713 patients]
    end

    subgraph Matching["Encounter Matching"]
        C1[Tier 1: Date + Hospital Match]
        C2[Tier 2: Date Range Match]
        C3[Tier 3: Nearest Encounter]
        C4[Tier 4: Default Window]
    end

    subgraph Extraction["Outcome Extraction"]
        D1[Mortality: 30d/90d/1yr]
        D2[ICU Admission]
        D3[Ventilation/Intubation]
        D4[Dialysis]
        D5[Advanced Interventions]
        D6[Vasopressors/Inotropes]
        D7[Bleeding Events]
        D8[Readmissions/Shock]
    end

    subgraph Output["Outputs"]
        E1[outcomes.csv<br/>33 MB, 100+ columns]
        E2[patient_timelines.pkl<br/>36 MB]
    end

    A1 --> B1
    A2 --> B2
    B1 --> B2
    B2 --> B3
    B3 --> C1
    A3 --> C1
    C1 --> C2 --> C3 --> C4
    C4 --> D1
    A4 --> D2
    A4 --> D3
    A4 --> D4
    A4 --> D5
    A4 --> D6
    A5 --> D7
    A6 --> D6
    A3 --> D8
    A5 --> D8
    D1 & D2 & D3 & D4 & D5 & D6 & D7 & D8 --> E1
    E1 --> E2
```

## Module 2: Laboratory Processing

```mermaid
flowchart TB
    subgraph Phase1["Phase 1: Discovery & Harmonization"]
        P1_Scan[Scan Lab Data<br/>Identify Unique Tests]
        P1_T1[Tier 1: LOINC Exact Match<br/>96.7% coverage]
        P1_T2[Tier 2: LOINC Family Match<br/>0% in our data]
        P1_T3[Tier 3: Hierarchical Clustering<br/>3.3% coverage]
        P1_Map[Generate Harmonization Map<br/>48 test groups]
    end

    subgraph Phase2["Phase 2: Feature Engineering"]
        P2_Extract[Extract Measurements<br/>7.6M+ values]
        P2_QC[Quality Control<br/>Outlier Detection]
        P2_Feat[Calculate 72 Features<br/>Per Test Per Patient]
        P2_Triple[Triple Encoding<br/>Values + Masks + Deltas]
    end

    subgraph Features["72 Features per Test"]
        F1[Summary Stats<br/>mean, std, min, max, median]
        F2[Time Features<br/>first, last, count, rate]
        F3[Trend Features<br/>slope, delta, variability]
        F4[Phase Features<br/>BASELINE, ACUTE, SUBACUTE, RECOVERY]
    end

    P1_Scan --> P1_T1
    P1_T1 --> P1_T2
    P1_T2 --> P1_T3
    P1_T3 --> P1_Map
    P1_Map --> P2_Extract
    P2_Extract --> P2_QC
    P2_QC --> P2_Feat
    P2_Feat --> P2_Triple
    P2_Feat --> F1
    P2_Feat --> F2
    P2_Feat --> F3
    P2_Feat --> F4
```

## Module 4: Medication Processing ✅ COMPLETE

```mermaid
flowchart TB
    subgraph Layer1["Layer 1: Canonical Extraction"]
        L1_Parse[Parse Med.txt<br/>1.71M records]
        L1_Dose[Regex Dose Parsing<br/>89.9% success]
        L1_Bronze[Bronze: canonical_records.parquet<br/>23 MB]
    end

    subgraph RxNorm["RxNorm Mapping"]
        RX_DB[(RxNorm SQLite<br/>394K concepts)]
        RX_Map[Multi-strategy Mapping<br/>Exact → Fuzzy → has_form]
        RX_Silver[Silver: mapped_medications.parquet<br/>92.4% coverage]
    end

    subgraph Layer2["Layer 2: Therapeutic Classes"]
        L2_YAML[therapeutic_classes.yaml<br/>53 classes + union_of]
        L2_Build[Class Indicator Builder<br/>Vectorized + union classes]
        L2_Gold[Gold: class_indicators.parquet<br/>25K rows × 162 cols]
    end

    subgraph Layer3["Layer 3: Individual Medications"]
        L3_Filter[Prevalence Filter<br/>≥20 patients + exceptions]
        L3_Build[Individual Builder<br/>Vectorized pivot ops]
        L3_Gold[Gold: individual_indicators.parquet<br/>26K rows × 1,747 cols]
    end

    subgraph Layer4["Layer 4: Embeddings"]
        L4_CoOccur[Co-occurrence Word2Vec<br/>769 meds × 128d]
        L4_PK[Pharmacokinetic Features<br/>1,582 meds × 10d]
        L4_HDF5[medication_embeddings.h5<br/>585 KB]
    end

    subgraph Layer5["Layer 5: Dose Intensity"]
        L5_DDD[WHO DDD Normalization<br/>97.2% coverage]
        L5_Daily[Daily Aggregation<br/>86K records]
        L5_Gold[Gold: dose_intensity.parquet]
    end

    subgraph Exports["Method-Specific Exports"]
        EX_GBTM[GBTM: CSV<br/>54K rows × 14 features]
        EX_GRUD[GRU-D: HDF5<br/>8,394 × 168 × 12]
        EX_XGB[XGBoost: Parquet<br/>8,219 × 831 features]
    end

    L1_Parse --> L1_Dose
    L1_Dose --> L1_Bronze
    L1_Bronze --> RX_Map
    RX_DB --> RX_Map
    RX_Map --> RX_Silver
    RX_Silver --> L2_Build
    L2_YAML --> L2_Build
    L2_Build --> L2_Gold
    RX_Silver --> L3_Build
    L3_Filter --> L3_Build
    L3_Build --> L3_Gold
    RX_Silver --> L4_CoOccur
    L4_CoOccur --> L4_HDF5
    L4_PK --> L4_HDF5
    RX_Silver --> L5_DDD
    L5_DDD --> L5_Daily
    L5_Daily --> L5_Gold
    L2_Gold --> EX_GBTM
    L5_Gold --> EX_GBTM
    RX_Silver --> EX_GRUD
    L2_Gold --> EX_XGB
    L3_Gold --> EX_XGB
    L5_Gold --> EX_XGB
```

## Module 6: Procedure Encoding ✅ COMPLETE (NEW)

```mermaid
flowchart TB
    subgraph Layer1["Layer 1: Canonical Extraction"]
        L1_Parse[Parse Prc.txt<br/>22M records]
        L1_Temporal[7 Temporal Flags<br/>lifetime, provoking, diagnostic, etc.]
        L1_Bronze[Bronze: canonical_procedures.parquet]
    end

    subgraph Mapping["CCS + SNOMED Mapping"]
        MAP_CCS[(CCS Crosswalk<br/>~230 categories)]
        MAP_Direct[Direct CPT→CCS<br/>71% of records]
        MAP_Fuzzy[Fuzzy Matching<br/>EPIC codes]
        MAP_Silver[Silver: mapped_procedures.parquet<br/>85% target coverage]
    end

    subgraph Layer2["Layer 2: CCS Indicators"]
        L2_CCS[CCS Category Indicators<br/>per patient-timewindow]
        L2_Risk[Surgical Risk Classification<br/>very_high → minimal]
        L2_Gold[Gold: ccs_indicators.parquet]
    end

    subgraph Layer3["Layer 3: PE-Specific Features"]
        L3_History[Lifetime History<br/>prior IVC, surgeries]
        L3_Provoking[Provoking Procedures<br/>surgery within 30d]
        L3_Diagnostic[Diagnostic Workup<br/>CTA, echo with datetimes]
        L3_Treatment[Initial Treatment<br/>CDT, thrombolysis, ECMO]
        L3_Escalation[Escalation<br/>transfusions, arrest]
        L3_Gold[Gold: pe_features.parquet<br/>63+ features]
    end

    subgraph Layer4["Layer 4: Embeddings"]
        L4_Complexity[Procedural Complexity<br/>16 dims]
        L4_CoOccur[CCS Co-occurrence<br/>Word2Vec 128d]
        L4_HDF5[procedure_embeddings.h5]
    end

    subgraph Layer5["Layer 5: World Model"]
        L5_Static[Static State<br/>lifetime history, provoking]
        L5_Dynamic[Dynamic State<br/>support level, complications]
        L5_Actions[Action Vectors<br/>discretion-weighted]
        L5_Gold[Gold: world_model_states/]
    end

    subgraph Exports["Method-Specific Exports"]
        EX_GBTM[GBTM: CSV<br/>daily features]
        EX_GRUD[GRU-D: HDF5<br/>168h × features]
        EX_XGB[XGBoost: Parquet<br/>~500 features]
    end

    L1_Parse --> L1_Temporal
    L1_Temporal --> L1_Bronze
    L1_Bronze --> MAP_Direct
    MAP_CCS --> MAP_Direct
    L1_Bronze --> MAP_Fuzzy
    MAP_Direct --> MAP_Silver
    MAP_Fuzzy --> MAP_Silver
    MAP_Silver --> L2_CCS
    L2_CCS --> L2_Risk
    L2_Risk --> L2_Gold
    MAP_Silver --> L3_History
    MAP_Silver --> L3_Provoking
    MAP_Silver --> L3_Diagnostic
    MAP_Silver --> L3_Treatment
    MAP_Silver --> L3_Escalation
    L3_History --> L3_Gold
    L3_Provoking --> L3_Gold
    L3_Diagnostic --> L3_Gold
    L3_Treatment --> L3_Gold
    L3_Escalation --> L3_Gold
    MAP_Silver --> L4_Complexity
    MAP_Silver --> L4_CoOccur
    L4_Complexity --> L4_HDF5
    L4_CoOccur --> L4_HDF5
    L3_Gold --> L5_Static
    L3_Gold --> L5_Dynamic
    L3_Gold --> L5_Actions
    L5_Static --> L5_Gold
    L5_Dynamic --> L5_Gold
    L5_Actions --> L5_Gold
    L2_Gold --> EX_GBTM
    L3_Gold --> EX_GBTM
    MAP_Silver --> EX_GRUD
    L2_Gold --> EX_XGB
    L3_Gold --> EX_XGB
    L4_HDF5 --> EX_XGB
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph Raw["Raw RPDR Data"]
        R1[63.4M Lab Rows]
        R2[Encounters]
        R3[22M Procedures]
        R4[Diagnoses]
        R5[Medications]
    end

    subgraph M1["Module 1"]
        M1A[8,713 Patients<br/>Time Zero Anchored]
    end

    subgraph M2["Module 2"]
        M2A[7.6M Lab Measurements<br/>48 Harmonized Tests]
    end

    subgraph M3["Module 3"]
        M3A[Vital Signs<br/>HR, BP, SpO2]
    end

    subgraph M4["Module 4"]
        M4A[1.71M Med Records<br/>53 Classes, 581 Meds]
    end

    subgraph M6["Module 6"]
        M6A[22M Procedure Records<br/>63+ PE Features]
    end

    subgraph ML["ML Ready"]
        ML1[3,456 Lab Features<br/>per patient]
        ML2[Temporal Sequences<br/>HDF5 format]
        ML3[World Model States<br/>Actions + States]
    end

    R1 --> M2
    R2 & R3 & R4 & R5 --> M1
    M1 --> M2
    M1 --> M3
    M1 --> M4
    M1 --> M6
    R3 --> M6
    R5 --> M4
    M2 --> ML1
    M2 --> ML2
    M4 --> ML2
    M6 --> ML3
```

## File Structure

```
TDA_11_25/
├── Data/
│   ├── ALL_PE_POSITIVE_With_Gemma_Predictions.csv  # Cohort source
│   ├── Combined_PE_Predictions_All_Cohorts.txt     # Timestamps
│   ├── Enc.txt                                      # Encounters
│   ├── Prc.txt                                      # Procedures
│   ├── Dia.txt                                      # Diagnoses
│   ├── Med.txt                                      # Medications
│   └── Dem.txt                                      # Demographics
│
├── module_1_core_infrastructure/
│   ├── module_01_core_infrastructure.py            # Main script (1,400+ lines)
│   └── outputs/
│       ├── outcomes.csv                            # 33 MB, 8,713 patients
│       └── patient_timelines.pkl                   # 36 MB
│
├── module_2_laboratory_processing/
│   ├── module_02_laboratory_processing.py          # Main script (1,230 lines)
│   ├── loinc_matcher.py                            # LOINC database matching
│   ├── unit_converter.py                           # Lab unit conversions
│   ├── hierarchical_clustering.py                  # Tier 3 clustering
│   ├── visualization_generator.py                  # Interactive dashboards
│   ├── Loinc/                                      # LOINC database files
│   └── outputs/
│       ├── full_lab_features.csv                   # 35 MB
│       └── full_lab_sequences.h5                   # 646 MB
│
├── module_3_vitals_processing/
│   ├── extractors/
│   ├── tests/
│   └── outputs/
│
└── docs/
    ├── brief.md                                    # Session brief
    ├── architecture.md                             # This file
    ├── progress.md                                 # Progress tracker
    └── plans/                                      # Implementation plans
```

## Key Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Cohort Source | Gemma PE-Positive | Most accurate PE classification |
| Time Zero | Report_Date_Time | CT PE study timestamp |
| Encounter Matching | 4-Tier Strategy | 99.5% Tier 1 match rate |
| Lab Harmonization | 3-Tier LOINC | 100% coverage with clinical accuracy |
| Feature Engineering | 72 features/test | Comprehensive temporal coverage |
| Performance | Pre-group by EMPI | O(1) vs O(n) lookups |

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 5.0 | 2025-12-17 | Module 6 COMPLETE - 5-layer procedure encoding, 145 tests, world model support |
| 4.0 | 2025-12-12 | Module 4 COMPLETE - All 8 phases, bug fixes (heparin mapping, DDD expansion) |
| 3.0 | 2025-12-11 | Module 4 Layers 1-4 complete (581 med indicators, embeddings) |
| 2.5 | 2025-12-10 | Module 4 Phases 2-4 (RxNorm mapping, 53 therapeutic classes) |
| 2.0 | 2025-11-25 | Expanded cohort (8,713 patients), Module 1 optimization |
| 1.0 | 2025-11-09 | Initial Module 1 + Module 2 complete (3,565 patients) |
