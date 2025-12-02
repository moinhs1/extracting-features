# Submodule 3.2: Hnp NLP Extractor - Design Document

**Date:** 2025-12-02
**Status:** Approved
**Author:** Claude (Opus 4.5)

---

## Overview

Extract vital signs from 136,950 H&P (History & Physical) notes using a hybrid regex approach with context-aware negation detection, timestamp extraction/estimation, and pattern-based confidence scoring.

**Input:** `/home/moin/TDA_11_25/Data/Hnp.txt` (2.3 GB, pipe-delimited)
**Output:** `outputs/discovery/hnp_vitals_raw.parquet`

---

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Extraction approach | Hybrid (section-first, full-text fallback) | Maximizes coverage while minimizing false positives |
| Multiple values per note | Keep all, tag with extraction_context | Preserves ED vs Exam vitals; downstream harmonizer decides |
| Timestamps | Extract explicit, estimate by section offset | ED vitals ~6h before H&P filing; adds temporal granularity |
| Temperature units | Store original + unit flag | Defer conversion to Submodule 3.5 |
| Negation handling | Context window (±10 words) | More robust than keyword list; catches "BP was not measured" |
| Confidence scoring | Pattern-based (1.0 → 0.6) | Explicit labels score higher than implicit patterns |
| Processing | Parallel multiprocessing | 2.3GB file benefits from multi-core processing |

---

## Input File Format

**Columns (pipe-delimited):**
```
EMPI|EPIC_PMRN|MRN_Type|MRN|Report_Number|Report_Date_Time|Report_Description|Report_Status|Report_Type|Report_Text
```

**Key Column:** `Report_Text` - free-text clinical notes containing vital signs

---

## Extraction Pipeline

### Step 1: Section Identification

Identify clinical sections to prioritize extraction and assign timestamp offsets:

```python
SECTION_PATTERNS = {
    'exam': (r'(?:Physical\s+)?Exam(?:ination)?(?:\s+ON\s+ADMISSION)?[:\s]', -1),
    'vitals': (r'Vitals?(?:\s+Signs?)?[:\s]|Vital\s+signs', -1),
    'ed_course': (r'ED\s+Course[:\s]|Emergency\s+Department|Triage\s+Vitals', -6),
    'current': (r'Current[:\s]|Last\s+vitals', 0),
}
```

Extract ~500 character window after each section header.

### Step 2: Vital Sign Patterns

Patterns refined from actual data analysis (encoding: `?C`/`?F` for degree symbols, `(!)` for abnormal flags):

#### Temperature
```python
TEMP_PATTERNS = [
    # Full label: "Temperature: 37.1 °C (98.8 °F)" or "Temp 36.8 ?C"
    (r'(?:Temperature|Temp|T)\s*:?\s*(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])', 1.0),
    # Tcurrent notation
    (r'Tcurrent\s+(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])', 0.9),
    # Standalone with degree: "36.5 °C" in vitals context
    (r'(\d{2,3}\.\d)\s*[°?]\s*([CF])', 0.8),
]
```

#### Heart Rate
```python
HR_PATTERNS = [
    # Full label: "Heart Rate: 84" or "Heart Rate: (!) 107"
    (r'Heart\s*Rate\s*:?\s*\(?\!?\)?\s*(\d{2,3})', 1.0),
    # HR prefix: "HR 71", "HR (!) 112"
    (r'HR\s*:?\s*\(?\!?\)?\s*(\d{2,3})', 0.95),
    # Pulse variations: "Pulse 86", "P 91", "P (!) 108"
    (r'(?:Pulse|P)\s*:?\s*\(?\!?\)?\s*(\d{2,3})', 0.9),
    # Range then value: "[62-72] 72" - capture last
    (r'\[\d{2,3}-\d{2,3}\]\s*(\d{2,3})', 0.85),
]
```

#### Blood Pressure
```python
BP_PATTERNS = [
    # Full label: "Blood pressure 109/69", "BP: 132/80"
    (r'(?:Blood\s*[Pp]ressure|BP)\s*:?\s*\(?\!?\)?\s*(\d{2,3})/(\d{2,3})', 1.0),
    # Range then value: "(115-154)/(59-69) 145/67"
    (r'\(\d+-\d+\)/\(\d+-\d+\)\s*(\d{2,3})/(\d{2,3})', 0.9),
    # Standalone in vitals context
    (r'(\d{2,3})/(\d{2,3})\s*(?:mmHg)?', 0.7),
]
```

#### Respiratory Rate
```python
RR_PATTERNS = [
    # Full label: "Respiratory Rate: 16"
    (r'Respiratory\s*Rate\s*:?\s*(\d{1,2})', 1.0),
    # Abbreviated: "RR 18", "Resp 16", "TRR 20"
    (r'(?:RR|Resp|TRR)\s*:?\s*(\d{1,2})', 0.9),
    # Range then value
    (r'\[\d{1,2}-\d{1,2}\]\s*(\d{1,2})', 0.85),
]
```

#### SpO2
```python
SPO2_PATTERNS = [
    # Full: "SpO2: 96 %", "O2 Sat: 98%", "SaO2 >99%"
    (r'(?:SpO2|SaO2|O2\s*Sat(?:uration)?)\s*:?\s*>?(\d{2,3})\s*%?', 1.0),
    # Standalone percentage in vitals context
    (r'(\d{2,3})\s*%\s*(?:on|RA|room)', 0.8),
]
```

### Step 3: Negation Detection

Check ±10 words around each match for negation phrases:

```python
NEGATION_PATTERNS = [
    r'no vitals',
    r'not obtained',
    r'unable to (?:obtain|measure|assess)',
    r'refused',
    r'not measured',
    r'not documented',
    r'vitals unavailable',
    r'There were no vitals',
]
```

### Step 4: Timestamp Handling

1. **Try explicit extraction:**
```python
TIMESTAMP_PATTERNS = [
    r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)',
    r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{4})',  # Military time
]
```

2. **Fallback estimation (offset from Report_Date_Time):**

| Context | Offset |
|---------|--------|
| ED/Triage | -6 hours |
| Exam/Vitals section | -1 hour |
| Current/Last | 0 hours |
| Unknown | -2 hours |

### Step 5: Confidence Scoring

Pattern-based confidence tiers:

| Confidence | Criteria |
|------------|----------|
| 1.0 | Explicit label + value ("BP: 130/80", "Heart Rate: 88") |
| 0.9-0.95 | Standard abbreviation ("HR 71", "Temp 36.8 ?C") |
| 0.8-0.85 | Value with context (units, section) |
| 0.6-0.7 | Ambiguous/fallback patterns |

---

## Module Structure

```
module_3_vitals_processing/
├── extractors/
│   ├── hnp_extractor.py          # Main extractor
│   ├── hnp_patterns.py           # Regex patterns & constants
│   └── phy_extractor.py          # Existing (Submodule 3.1)
├── tests/
│   ├── test_hnp_extractor.py     # Unit tests
│   └── test_phy_extractor.py     # Existing
```

---

## Function Specifications

### `identify_sections(text: str) -> Dict[str, str]`
Find Exam/Vitals/ED sections in note text.
- **Input:** Full Report_Text
- **Output:** `{'exam': '...500 chars...', 'ed_course': '...', ...}`
- **Tests:** 5-6

### `extract_vital(text: str, vital_type: str) -> List[Dict]`
Apply regex patterns for one vital type.
- **Input:** Text window, vital type (HR, BP, etc.)
- **Output:** List of `{value, units, confidence, position, is_flagged_abnormal}`
- **Tests:** 8-10 per vital type

### `check_negation(text: str, match_position: int, window: int = 50) -> bool`
Check context window for negation phrases.
- **Input:** Full text, character position of match, window size
- **Output:** True if negation found
- **Tests:** 5-6

### `extract_timestamp(text: str, section: str, report_datetime: datetime) -> Tuple[datetime, str, float]`
Extract explicit timestamp or estimate from section context.
- **Input:** Text, section name, Report_Date_Time
- **Output:** `(timestamp, source='explicit'|'estimated', offset_hours)`
- **Tests:** 6-8

### `parse_temperature_unit(value: float, unit_str: Optional[str]) -> Tuple[float, str]`
Handle temperature unit detection.
- **Input:** Numeric value, unit string (may be None)
- **Output:** `(value, unit)` with auto-detection if needed
- **Tests:** 5-6

### `process_hnp_row(row: pd.Series) -> List[Dict]`
Process single H&P note row.
- **Input:** DataFrame row with Report_Text, EMPI, etc.
- **Output:** List of vital sign records
- **Tests:** 4-5

### `extract_hnp_vitals(input_path: str, output_path: str, n_workers: int = None, chunk_size: int = 10000) -> pd.DataFrame`
Main entry point with parallel processing.
- **Input:** File paths, worker count, chunk size
- **Output:** DataFrame written to parquet
- **Tests:** 2-3 (integration)

---

## Output Schema

```python
{
    'EMPI': str,                      # Patient ID
    'timestamp': datetime,            # Extracted or estimated
    'timestamp_source': str,          # 'explicit' or 'estimated'
    'timestamp_offset_hours': float,  # Offset applied (for audit)
    'vital_type': str,                # HR, SBP, DBP, RR, SPO2, TEMP
    'value': float,                   # Numeric value
    'units': str,                     # Original units (C, F, %, mmHg, bpm)
    'source': str,                    # Always 'hnp'
    'extraction_context': str,        # ED, Exam, Vitals, HPI, Other
    'confidence': float,              # 0.6-1.0 pattern-based
    'is_flagged_abnormal': bool,      # True if (!) present in source
    'report_number': str,             # For traceability
    'report_date_time': datetime,     # Original note timestamp
}
```

---

## Built-in Validation

| Check | Action |
|-------|--------|
| HR range | Reject if <20 or >300 |
| BP range | Reject SBP <40 or >350, DBP <20 or >250 |
| BP sanity | Swap if SBP < DBP |
| RR range | Reject if <4 or >80 |
| SpO2 range | Reject if <40 or >100 |
| Temp range | Reject if <25°C or >45°C (after unit normalization) |
| Temp unit detection | If no unit and value >50, assume Fahrenheit |

---

## Expected Output

- **Input:** 136,950 H&P notes
- **Notes with vitals:** ~80% (~109K)
- **Estimated records:** 400K-600K vital measurements
- **Processing time:** ~10-20 minutes (parallel)

---

## Test Coverage

| Category | Estimated Tests |
|----------|-----------------|
| Section identification | 5-6 |
| HR extraction | 8-10 |
| BP extraction | 8-10 |
| RR extraction | 6-8 |
| SpO2 extraction | 6-8 |
| Temperature extraction | 8-10 |
| Negation detection | 5-6 |
| Timestamp handling | 6-8 |
| Row processing | 4-5 |
| Integration | 2-3 |
| **Total** | **~55-65** |

---

## CLI Interface

```bash
# Default paths
python3 -m module_3_vitals_processing.extractors.hnp_extractor

# Custom paths
python3 -m module_3_vitals_processing.extractors.hnp_extractor \
  -i /path/to/Hnp.txt \
  -o /path/to/output.parquet \
  -w 4  # Number of workers
```

---

## Dependencies

- pandas
- pyarrow (parquet)
- multiprocessing (stdlib)
- re (stdlib)
- datetime (stdlib)

No new dependencies required.

---

**END OF DESIGN DOCUMENT**
