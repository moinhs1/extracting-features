# module_3_vitals_processing/extractors/hnp_patterns.py
"""Regex patterns and constants for Hnp.txt extraction."""

# Hnp.txt columns (pipe-delimited)
HNP_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number',
    'Report_Date_Time', 'Report_Description', 'Report_Status',
    'Report_Type', 'Report_Text'
]

# Section patterns: (regex, timestamp_offset_hours)
SECTION_PATTERNS = {
    'exam': (r'(?:Physical\s+)?Exam(?:ination)?(?:\s+ON\s+ADMISSION)?[:\s]', -1),
    'vitals': (r'Vitals?(?:\s+Signs?)?[:\s]|Vital\s+signs', -1),
    'ed_course': (r'ED\s+Course[:\s]|Emergency\s+Department|Triage\s+Vitals', -6),
    'current': (r'Current[:\s]|Last\s+vitals', 0),
}

# Heart Rate patterns: (regex, confidence)
HR_PATTERNS = [
    (r'Heart\s*Rate\s*:?\s*\(?\!?\)?\s*(\d{2,3})', 1.0),
    (r'HR\s*:?\s*\(?\!?\)?\s*(\d{2,3})', 0.95),
    (r'(?:Pulse|P)\s+:?\s*\(?\!?\)?\s*(\d{2,3})', 0.9),
    (r'\[\d{2,3}-\d{2,3}\]\s*(\d{2,3})', 0.85),
]

# Blood Pressure patterns: (regex, confidence) - captures (SBP, DBP)
BP_PATTERNS = [
    (r'(?:Blood\s*[Pp]ressure|BP)\s*:?\s*\(?\!?\)?\s*(\d{2,3})/(\d{2,3})', 1.0),
    (r'\(\d+-\d+\)/\(\d+-\d+\)\s*(\d{2,3})/(\d{2,3})', 0.9),
    (r'(\d{2,3})/(\d{2,3})\s*(?:mmHg)', 0.8),
    (r'(\d{2,3})/(\d{2,3})', 0.7),
]

# Respiratory Rate patterns: (regex, confidence)
RR_PATTERNS = [
    (r'Respiratory\s*Rate\s*:?\s*(\d{1,2})\b', 1.0),
    (r'(?:RR|Resp|TRR)\s*:?\s*(\d{1,2})\b', 0.9),
    (r'\[\d{1,2}-\d{1,2}\]\s*(\d{1,2})\b', 0.85),
]

# SpO2 patterns: (regex, confidence)
SPO2_PATTERNS = [
    (r'(?:SpO2|SaO2|O2\s*Sat(?:uration)?)\s*:?\s*>?(\d{2,3})\s*%?', 1.0),
    (r'(\d{2,3})\s*%\s*(?:on|RA|room)', 0.8),
]

# Temperature patterns: (regex, confidence) - captures (value, unit)
TEMP_PATTERNS = [
    (r'(?:Temperature|Temp)\s*:?\s*(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])', 1.0),
    (r'Tcurrent\s+(\d{2,3}\.?\d?)\s*[°?]?\s*([CF])', 0.9),
    (r'T\s+(\d{2,3}\.?\d?)([CF])', 0.9),  # T 98.6F (no space before unit)
    (r'T\s+(\d{2,3}\.?\d?)\s*[°?]\s*([CF])', 0.9),  # T 98.6 °F
    (r'(?:Temperature|Temp)\s*:?\s*(\d{2,3}\.?\d?)\b', 0.85),  # Temp: 98.6 (no unit)
    (r'(\d{2,3}\.\d)\s*[°?]\s*([CF])', 0.8),
]

# Negation patterns
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

# Timestamp patterns for explicit extraction
TIMESTAMP_PATTERNS = [
    r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{1,2}:\d{2}(?::\d{2})?\s*(?:AM|PM)?)',
    r'(\d{1,2}/\d{1,2}/\d{2,4})\s+(\d{4})',
]

# Value ranges for validation
VALID_RANGES = {
    'HR': (20, 300),
    'SBP': (40, 350),
    'DBP': (20, 250),
    'RR': (4, 80),
    'SPO2': (40, 100),
    'TEMP_C': (25, 45),
    'TEMP_F': (77, 113),
}

# Default timestamp offsets by section (hours)
DEFAULT_TIMESTAMP_OFFSET = -2
