"""Regex patterns and constants for Prg.txt extraction."""

# Prg.txt columns (same as Hnp.txt)
PRG_COLUMNS = [
    'EMPI', 'EPIC_PMRN', 'MRN_Type', 'MRN', 'Report_Number',
    'Report_Date_Time', 'Report_Description', 'Report_Status',
    'Report_Type', 'Report_Text'
]

# Section patterns: (regex, timestamp_offset_hours)
# Based on analysis of 300K rows from actual Prg.txt
PRG_SECTION_PATTERNS = {
    # High frequency (>5000 occurrences in sample)
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

# Skip sections (false positive sources)
# HR/BP values in these sections are NOT vital measurements
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
