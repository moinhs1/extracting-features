# Unified Vital Extraction Design

> **Status:** APPROVED
> **Created:** 2025-12-12
> **Goal:** Merge comprehensive patterns from legacy script into robust extraction pipeline

---

## Background

Three extraction codebases exist:
- `ultimate_vital_extraction.py` (legacy): 150+ patterns in 3 stages, but no clinical safeguards
- `hnp_extractor.py`: Robust architecture (negation, deduplication, validation) but ~25 patterns
- `prg_extractor.py`: Extends HNP with skip section filtering, temp method capture

**Opportunity:** Combine ultimate's comprehensive patterns with hnp/prg's robust pipeline.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   unified_patterns.py                        │
│  • All patterns organized by vital type (HR, BP, RR, etc.)  │
│  • 3-tier confidence: standard (0.90-1.0), optimized        │
│    (0.80-0.90), specialized (0.65-0.80)                     │
│  • ~150 patterns from ultimate + existing                   │
│  • Includes O2_FLOW, O2_DEVICE, BMI patterns               │
└─────────────────────────┬───────────────────────────────────┘
                          │
┌─────────────────────────▼───────────────────────────────────┐
│                  unified_extractor.py                        │
│  • extract_vital(text, vital_type) → List[Dict]             │
│  • extract_all_vitals(text) → Dict[vital_type, List]        │
│  • Negation detection (current 8 patterns)                  │
│  • Skip section filtering (allergies, meds, history)        │
│  • Position deduplication (seen_positions)                  │
│  • Physiological validation (ranges, pulse pressure)        │
└─────────────────────────┬───────────────────────────────────┘
                          │
          ┌───────────────┴───────────────┐
          ▼                               ▼
┌─────────────────────┐         ┌─────────────────────┐
│  hnp_extractor.py   │         │  prg_extractor.py   │
│  (thin wrapper)     │         │  (thin wrapper)     │
│  • File I/O         │         │  • File I/O         │
│  • Column mapping   │         │  • Checkpointing    │
│  • source='hnp'     │         │  • source='prg'     │
└─────────────────────┘         └─────────────────────┘
```

---

## Key Decisions

### 1. Additional Vital Types
**Decision:** Include O2 Flow Rate, O2 Device, and BMI

**Rationale:** O2 flow/device are clinically relevant for PE hypoxemia assessment; BMI useful for dosing/risk stratification.

**Output:** Separate `supplemental_vitals.parquet` files to avoid disrupting existing pipeline.

### 2. Pattern Organization
**Decision:** Unified pattern library (`unified_patterns.py`)

**Rationale:**
- Patterns are source-agnostic (work on any clinical text)
- Avoids duplicating 150+ patterns across two files
- Easier to maintain confidence tiers in one place

### 3. Confidence Scoring
**Decision:** Hybrid approach (stage baseline + individual adjustments)

```python
CONFIDENCE_TIERS = {
    'standard': 0.95,    # Explicit label + unit
    'optimized': 0.85,   # Label or strong context
    'specialized': 0.70, # Contextual/bare patterns
}
```

Individual patterns adjusted up/down based on specificity.

### 4. Section Handling
**Decision:** Skip-only filtering (no section weighting)

**Skip sections:**
- Allergies/Reactions
- Medications/Prescriptions
- Past/Family/Social/Surgical History
- Review of Systems

**Rationale:** Section headers are inconsistent; skip filtering prevents the main false positives (medication doses as HR values) without over-engineering.

### 5. Negation Detection
**Decision:** Keep current 8 patterns (maximize extraction)

```python
NEGATION_PATTERNS = [
    r'no\s+vitals',
    r'not\s+obtained',
    r'unable\s+to\s+(?:obtain|measure|assess)',
    r'refused',
    r'not\s+measured',
    r'not\s+documented',
    r'vitals?\s+unavailable',
    r'There\s+were\s+no\s+vitals',
]
```

**Rationale:** More aggressive negation filtering risks losing valid extractions like "normal HR 72". Let confidence scores handle precision.

### 6. Implementation Approach
**Decision:** New unified extractor + thin wrappers

**Rationale:**
- Extraction logic is nearly identical between HNP and PRG
- Cleaner separation: file I/O vs extraction logic
- Easier to test core extraction independently
- Can add new sources by writing thin wrappers

---

## Unified Patterns Structure

```python
# Each pattern: (regex, confidence, tier_name)
HR_PATTERNS = [
    # Standard tier (0.90-1.0)
    (r'(?:Heart\s*Rate|HR)\s*:?\s*(\d{2,3})\s*(?:bpm|BPM)', 0.95, 'standard'),
    (r'(?:Pulse|P)\s*:?\s*(\d{2,3})\s*(?:bpm|BPM)', 0.92, 'standard'),

    # Optimized tier (0.80-0.90)
    (r'(?:EKG|ECG)[^0-9]*(?:rate|HR)[^0-9]*(\d{2,3})', 0.85, 'optimized'),
    (r'(?:sinus|normal sinus)\s*(?:rhythm)[^0-9]*(\d{2,3})', 0.85, 'optimized'),
    (r'(?:tachycardic|bradycardia)[^0-9]*(?:at|with|to)[^0-9]*(\d{1,3})', 0.82, 'optimized'),

    # Specialized tier (0.65-0.80)
    (r'(?:rate|pulse)[^\d]*(\d{2,3})(?:\s*bpm)?', 0.70, 'specialized'),
    # ... more patterns from ultimate_vital_extraction.py
]

BP_PATTERNS = [...]   # ~30 patterns
RR_PATTERNS = [...]   # ~27 patterns
SPO2_PATTERNS = [...]  # ~26 patterns
TEMP_PATTERNS = [...]  # ~20 patterns
O2_FLOW_PATTERNS = [...] # ~11 patterns
O2_DEVICE_PATTERNS = [...] # ~9 patterns (returns string)
BMI_PATTERNS = [...]  # ~7 patterns
```

---

## Validation Ranges

| Vital | Range | Rationale |
|-------|-------|-----------|
| HR | 30-220 bpm | <30 is asystole, >220 rare |
| SBP | 50-260 mmHg | Allows clinical extremes |
| DBP | 25-150 mmHg | >150 is rare |
| Pulse Pressure | 10-120 mmHg | Physiological constraint |
| RR | 6-50 /min | <6 is apnea, >50 very rare |
| SPO2 | 50-100% | <50 usually artifact |
| TEMP_F | 93-108°F | Clinical plausible range |
| TEMP_C | 33.5-42.5°C | Clinical plausible range |
| O2_FLOW | 0.5-60 L/min | 0.5L NC to 60L high-flow |
| BMI | 12-70 kg/m² | Clinical extremes |

---

## Core Extraction Logic

```python
def extract_vital(
    text: str,
    vital_type: str,
    patterns: List[Tuple],
    valid_range: Tuple[float, float]
) -> List[Dict]:
    """Extract single vital type from text."""
    results = []
    seen_positions = set()

    for pattern, confidence, tier in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            position = match.start()

            # 1. Position deduplication (within 15 chars)
            if any(abs(position - p) < 15 for p in seen_positions):
                continue

            # 2. Skip section check
            if is_in_skip_section(text, position):
                continue

            # 3. Negation check
            if check_negation(text, position):
                continue

            # 4. Extract and validate value
            try:
                value = float(match.group(1))
            except (ValueError, IndexError):
                continue

            # 5. Range validation
            if not (valid_range[0] <= value <= valid_range[1]):
                continue

            # 6. Add result
            results.append({
                'value': value,
                'confidence': confidence,
                'position': position,
                'tier': tier,
                'is_flagged_abnormal': is_abnormal(value, vital_type),
            })
            seen_positions.add(position)

    return results
```

**Special handlers:**
- `extract_blood_pressure()`: Captures SBP/DBP together, swaps if transposed, validates pulse pressure
- `extract_temperature()`: Captures unit, auto-detects F if >50, normalizes to Celsius

---

## Output Files

```
outputs/discovery/
├── hnp_vitals_raw.parquet      # Core 5 vitals (unchanged schema)
├── prg_vitals_raw.parquet      # Core 5 vitals (unchanged schema)
├── hnp_supplemental.parquet    # NEW: O2_FLOW, O2_DEVICE, BMI
└── prg_supplemental.parquet    # NEW: O2_FLOW, O2_DEVICE, BMI
```

**Core vitals schema (unchanged):**
```
EMPI, timestamp, vital_type, value, units, source,
confidence, position, is_flagged_abnormal, report_number
```

**Supplemental vitals schema:**
```
EMPI, timestamp, vital_type, value, units, source,
confidence, report_number
```

---

## Pipeline Integration

| Layer | Change Required |
|-------|-----------------|
| Layer 1 | None - reads same core parquet files |
| Layer 2 | None - same 7 vitals |
| Layer 3 | None - features built on Layer 2 |
| Layer 4 | None - embeddings built on Layer 2 |
| Layer 5 | None now; future update can add O2/BMI to reserved slots |

**Backward compatibility:** Existing pipeline works unchanged.

---

## Files to Create

| File | Lines (est.) | Description |
|------|--------------|-------------|
| `unified_patterns.py` | ~400 | All patterns with confidence tiers |
| `unified_extractor.py` | ~250 | Core extraction logic |
| `test_unified_patterns.py` | ~200 | Pattern coverage tests |
| `test_unified_extractor.py` | ~150 | Extraction logic tests |
| `test_supplemental_vitals.py` | ~100 | O2/BMI tests |

## Files to Refactor

| File | Change |
|------|--------|
| `hnp_extractor.py` | Thin wrapper (~150 lines, down from ~680) |
| `prg_extractor.py` | Thin wrapper (~200 lines, down from ~570) |
| `hnp_patterns.py` | Keep for backward compat, mark deprecated |
| `prg_patterns.py` | Keep for backward compat, mark deprecated |

---

## Testing Strategy

**Test categories:**

1. **Pattern matching tests** (~50 tests)
   - Positive matches (standard formats)
   - Negative matches (dates, page numbers, medication doses)
   - Confidence tier verification

2. **Validation tests** (~20 tests)
   - Range validation per vital
   - Pulse pressure validation
   - Temperature unit detection and normalization

3. **Skip section tests** (~10 tests)
   - Values in allergy/medication/history sections rejected
   - Values in vitals/exam sections accepted

4. **Integration tests** (~10 tests)
   - Full text extraction end-to-end
   - Output schema validation

**Existing tests:** `test_hnp_extractor.py` (74) and `test_prg_extractor.py` (61) should pass unchanged.

---

## Expected Outcomes

- **Pattern count:** ~25 → ~150 (6x increase)
- **Extraction coverage:** Significant improvement on edge cases
- **Maintainability:** Single place to update patterns
- **New data:** O2 flow/device/BMI captured for future use

---

**Document Version:** 1.0
**Approved:** 2025-12-12
