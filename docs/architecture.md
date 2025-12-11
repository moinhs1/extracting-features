# System Architecture
*Last Updated: 2025-12-11*

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

    subgraph Module4["Module 4: Medication Processing"]
        M4_Extract[Layer 1: Canonical Extract<br/>1.71M records]
        M4_RxNorm[RxNorm Mapping<br/>92.4% coverage]
        M4_Class[Layer 2: Therapeutic Classes<br/>53 indicators]
        M4_Indiv[Layer 3: Individual Meds<br/>581 indicators]
    end

    subgraph Outputs["Output Files"]
        OUT_CSV[(outcomes.csv<br/>8,713 patients)]
        OUT_PKL[(patient_timelines.pkl)]
        OUT_LAB[(lab_features.csv)]
        OUT_H5[(lab_sequences.h5)]
        OUT_MED[(medication outputs)]
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
    M4_Indiv --> OUT_MED
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

## Module 4: Medication Processing

```mermaid
flowchart TB
    subgraph Layer1["Layer 1: Canonical Extraction"]
        L1_Parse[Parse Med.txt<br/>1.71M records]
        L1_Dose[Regex Dose Parsing<br/>89.9% success]
        L1_Bronze[Bronze: canonical_records.parquet<br/>23 MB]
    end

    subgraph RxNorm["RxNorm Mapping"]
        RX_DB[(RxNorm SQLite<br/>394K concepts)]
        RX_Map[Multi-strategy Mapping<br/>Exact → Fuzzy → Contains]
        RX_Silver[Silver: mapped_medications.parquet<br/>92.4% coverage]
    end

    subgraph Layer2["Layer 2: Therapeutic Classes"]
        L2_YAML[therapeutic_classes.yaml<br/>53 classes]
        L2_Build[Class Indicator Builder<br/>Vectorized]
        L2_Gold[Gold: class_indicators.parquet<br/>25K rows × 162 cols]
    end

    subgraph Layer3["Layer 3: Individual Medications"]
        L3_Filter[Prevalence Filter<br/>≥20 patients + exceptions]
        L3_Build[Individual Builder<br/>Vectorized pivot ops]
        L3_Gold[Gold: individual_indicators.parquet<br/>26K rows × 1,747 cols]
        L3_Sparse[Sparse HDF5<br/>98.4% sparsity]
    end

    L1_Parse --> L1_Dose
    L1_Dose --> L1_Bronze
    L1_Bronze --> RX_Map
    RX_DB --> RX_Map
    RX_Map --> RX_Silver
    RX_Silver --> L2_Build
    L2_YAML --> L2_Build
    L2_Build --> L2_Gold
    RX_Silver --> L3_Filter
    L3_Filter --> L3_Build
    L3_Build --> L3_Gold
    L3_Build --> L3_Sparse
```

## Data Flow Architecture

```mermaid
flowchart LR
    subgraph Raw["Raw RPDR Data"]
        R1[63.4M Lab Rows]
        R2[Encounters]
        R3[Procedures]
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

    subgraph ML["ML Ready"]
        ML1[3,456 Lab Features<br/>per patient]
        ML2[Temporal Sequences<br/>HDF5 format]
    end

    R1 --> M2
    R2 & R3 & R4 & R5 --> M1
    M1 --> M2
    M1 --> M3
    M2 --> ML1
    M2 --> ML2
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
| 3.0 | 2025-12-11 | Module 4 Layers 1-3 complete (581 med indicators, 98.4% sparse) |
| 2.5 | 2025-12-10 | Module 4 Phases 2-4 (RxNorm mapping, 53 therapeutic classes) |
| 2.0 | 2025-11-25 | Expanded cohort (8,713 patients), Module 1 optimization |
| 1.0 | 2025-11-09 | Initial Module 1 + Module 2 complete (3,565 patients) |
