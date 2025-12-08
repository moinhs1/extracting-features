# Submodule 3.3: Prg NLP Extractor Design

**Date:** 2024-12-08
**Status:** Approved

## Overview

Extract vital signs from Prg.txt (Progress Notes) - 4.6M rows, 29.7GB. Follows established Hnp extractor architecture with Prg-specific adaptations.

## Key Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Code reuse | Maximum - import from `hnp_patterns.py` | Minimize duplication, shared maintenance |
| False positive handling | Context-aware + section-based filtering | Progress notes have more noise (allergies, med lists) |
| Temperature method | New `temp_method` field | Clinically meaningful, easy to extract |
| Scale handling | Checkpointing every 5 chunks | 30GB file needs resume capability |
| Section patterns | Prg-specific config | Different structure than H&P notes |

## Architecture

```
module_3_vitals_processing/
├── extractors/
│   ├── hnp_patterns.py          # Existing - shared patterns
│   ├── hnp_extractor.py         # Existing
│   ├── prg_patterns.py          # NEW - Prg-specific patterns
│   └── prg_extractor.py         # NEW - main extractor
├── config/
│   └── vitals_config.py         # Add Prg paths
└── tests/
    └── test_prg_extractor.py    # NEW
```

## Section Patterns (Data-Driven)

### Extract Sections (where vitals appear)

Based on analysis of 300K rows from actual Prg.txt:

```python
PRG_SECTION_PATTERNS = {
    # High frequency (>5000 occurrences)
    'physical_exam': (r'Physical\s+Exam(?:ination)?[:\s]', 0),
    'objective': (r'Objective[:\s]', 0),
    'exam': (r'\bExam[:\s]', 0),
    'vitals': (r'Vitals?[:\s]', 0),

    # Specific vitals headers
    'vital_signs': (r'Vital\s+[Ss]igns?[:\s]', 0),
    'vital_signs_recent': (r'Vital\s+signs?:\s*\(most\s+recent\)', 0),
    'on_exam': (r'ON\s+EXAM[:\s]', 0),

    # Combined headers (common in Prg)
    'physical_exam_vitals': (r'Physical\s+Exam[:\s]+Vitals?[:\s]', 0),
    'physical_exam_gen': (r'Physical\s+Exam[:\s]+Gen(?:eral)?[:\s]', 0),
    'objective_temp': (r'Objective[:\s]+Temperature[:\s]', 0),

    # SOAP format
    'assessment_plan': (r'Assessment\s*(?:&|and|/)?\s*Plan[:\s]', 0),
}
```

### Skip Sections (false positive sources)

```python
PRG_SKIP_PATTERNS = [
    # Allergies/Reactions (HR values as side effects)
    r'Allerg(?:ies|ic|en)[:\s]',
    r'[Rr]eaction\(?s?\)?[:\s]',

    # Medication lists
    r'Medications?[:\s]',
    r'(?:Outpatient\s+)?Prescriptions?[:\s]',
    r'Scheduled\s+Meds[:\s]',

    # History sections (historical mentions)
    r'Past\s+(?:Medical\s+)?History[:\s]',
    r'Family\s+History[:\s]',
    r'(?:History\s+of\s+)?Present\s+Illness[:\s]',
    r'Social\s+History[:\s]',
    r'Surgical\s+History[:\s]',

    # Other non-vitals
    r'Review\s+of\s+Systems[:\s]',
    r'ROS[:\s]',
]
```

## Vitals Patterns (Prg-Specific Additions)

### Blood Pressure (extend Hnp)

```python
PRG_BP_PATTERNS = [
    # Spelled out format
    (r'Blood\s+pressure\s+(\d{2,3})/(\d{2,3})', 0.95),
    # With ranges
    (r'BP:\s*\(\d+-\d+\)/\(\d+-\d+\)\s*(\d{2,3})/(\d{2,3})', 0.9),
]
```

### Heart Rate/Pulse (extend Hnp)

```python
PRG_HR_PATTERNS = [
    # P format common in Prg
    (r'\bP\s+(\d{2,3})\b', 0.85),
    # With abnormal flag
    (r'Pulse\s*\(!\)\s*(\d{2,3})', 0.9),
    # With ranges
    (r'Heart\s+Rate:\s*\[\d+-\d+\]\s*(\d{2,3})', 0.95),
]
```

### Temperature with Method (NEW)

```python
TEMP_PATTERNS_PRG = [
    # Temp(Src) format with method: Temp(Src) 36.7 °C (98 °F) (Oral)
    (r'Temp\(?Src\)?\s*(\d{2,3}\.?\d?)\s*[?°]?\s*([CF])[^)]*\((\w+)\)', 1.0),
    # Temp with method: Temp 36.8 °C (98.2 °F) (Temporal)
    (r'Temp\s+(\d{2,3}\.?\d?)\s*[?°]?\s*([CF])[^)]*\((\w+)\)', 0.95),
]

TEMP_METHOD_MAP = {
    'oral': 'oral', 'orally': 'oral',
    'temporal': 'temporal',
    'rectal': 'rectal', 'rectally': 'rectal',
    'axillary': 'axillary', 'axilla': 'axillary',
    'tympanic': 'tympanic', 'ear': 'tympanic',
}
```

### SpO2 (extend Hnp)

```python
PRG_SPO2_PATTERNS = [
    # O2 sat alternate
    (r'O2\s*sat\s*(\d{2,3})', 0.85),
    # With space before %
    (r'SpO2\s*:?\s*(\d{2,3})\s*%', 0.95),
]
```

### Respiratory Rate (extend Hnp)

```python
PRG_RR_PATTERNS = [
    # Resp format
    (r'Resp[:\s]+(\d{1,2})\b', 0.9),
]
```

## Checkpointing System

```python
@dataclass
class ExtractionCheckpoint:
    input_path: str
    output_path: str
    rows_processed: int
    chunks_completed: int
    last_chunk_start: int
    records_extracted: int
    started_at: datetime
    updated_at: datetime

CHECKPOINT_FILE = "prg_extraction_checkpoint.json"
CHECKPOINT_INTERVAL = 5  # Save every 5 chunks (~50K rows)
```

### Incremental Output

```
module_3_vitals_processing/outputs/discovery/
├── hnp_vitals_raw.parquet      # Existing
├── prg_vitals_raw.parquet      # Final merged output
└── prg_chunks/                  # During processing
    ├── chunk_0000.parquet
    ├── chunk_0001.parquet
    └── ...
```

## Output Schema

Extends Hnp with one new field:

| Column | Type | Description |
|--------|------|-------------|
| EMPI | str | Patient identifier |
| timestamp | datetime | Vital sign timestamp |
| timestamp_source | str | 'explicit' or 'estimated' |
| timestamp_offset_hours | float | Hours offset if estimated |
| vital_type | str | HR, SBP, DBP, RR, SPO2, TEMP |
| value | float | Numeric value |
| units | str | bpm, mmHg, breaths/min, %, C/F |
| source | str | 'prg' |
| extraction_context | str | Section name |
| confidence | float | 0.0-1.0 |
| is_flagged_abnormal | bool | Had (!) marker |
| report_number | str | Report identifier |
| report_date_time | datetime | Report timestamp |
| **temp_method** | str | **NEW: oral, rectal, temporal, axillary, tympanic (nullable)** |

## Confidence Adjustment Logic

```python
def get_extraction_confidence(position: int, text: str, base_confidence: float) -> float:
    """Adjust confidence based on section context."""

    # Check if in skip section (look backwards up to 500 chars)
    context_before = text[max(0, position-500):position]

    for skip_pattern in PRG_SKIP_PATTERNS:
        if re.search(skip_pattern, context_before, re.IGNORECASE):
            # Check if we've entered a valid section since
            for section_name, (pattern, _) in PRG_SECTION_PATTERNS.items():
                if re.search(pattern, context_before, re.IGNORECASE):
                    return base_confidence  # Valid section after skip
            return 0.0  # Still in skip section - skip entirely

    return base_confidence
```

## Implementation Tasks

1. Create `prg_patterns.py` with section and vitals patterns
2. Create `prg_extractor.py` with:
   - Checkpoint save/load
   - Section-aware extraction
   - Temperature method extraction
   - Incremental parquet output
3. Update `vitals_config.py` with Prg paths
4. Create comprehensive tests
5. Add CLI entry point

## Expected Output

- ~4.6M input rows
- Estimated 2-5M vitals records (many notes lack vitals)
- Processing time: 2-4 hours with checkpointing
