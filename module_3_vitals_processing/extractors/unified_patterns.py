"""
Unified pattern library for vital sign extraction.

All patterns organized by vital type with 3-tier confidence scoring:
- Standard (0.90-1.0): Explicit label + unit
- Optimized (0.80-0.90): Label or strong context
- Specialized (0.65-0.80): Contextual/bare patterns

Each pattern tuple: (regex, confidence, tier_name)
"""
import re

# Validation ranges (tightened for clinical plausibility)
VALID_RANGES = {
    'HR': (30, 220),
    'SBP': (50, 260),
    'DBP': (25, 150),
    'PULSE_PRESSURE': (10, 120),
    'RR': (6, 50),
    'SPO2': (50, 100),
    'TEMP_C': (33.5, 42.5),
    'TEMP_F': (93, 108),
    'O2_FLOW': (0.5, 60),
    'BMI': (12, 70),
}

# Negation patterns (keep existing 8 for max extraction)
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

# Skip section patterns (false positive sources)
SKIP_SECTION_PATTERNS = [
    r'Allerg(?:ies|ic|en)[:\s]',
    r'[Rr]eaction\(?s?\)?[:\s]',
    r'Medications?[:\s]',
    r'(?:Outpatient\s+)?Prescriptions?[:\s]',
    r'Scheduled\s+Meds[:\s]',
    r'Past\s+(?:Medical\s+)?History[:\s]',
    r'Family\s+History[:\s]',
    r'(?:History\s+of\s+)?Present\s+Illness[:\s]',
    r'Social\s+History[:\s]',
    r'Surgical\s+History[:\s]',
    r'Review\s+of\s+Systems[:\s]',
    r'ROS[:\s]',
]

# Heart Rate patterns - sorted by confidence (highest first)
HR_PATTERNS = [
    # Standard tier (0.90-1.0) - explicit label with/without unit
    (r'Heart\s*Rate\s*:?\s*\(?\!?\)?\s*(\d{2,3})\s*(?:bpm|BPM|beats\s*per\s*minute)?', 0.95, 'standard'),
    (r'HR\s*:?\s*\(?\!?\)?\s*(\d{2,3})\s*(?:bpm|BPM)?', 0.95, 'standard'),
    (r'(?:Pulse|P)\s*:?\s*\(?\!?\)?\s*(\d{2,3})\s*(?:bpm|BPM)?', 0.90, 'standard'),
    (r'\[\d{2,3}-\d{2,3}\]\s*(\d{2,3})', 0.90, 'standard'),  # Reference range format

    # Optimized tier (0.80-0.90) - strong context
    (r'(?:EKG|ECG)[^0-9]*(?:rate|HR)[^0-9]*(\d{2,3})', 0.88, 'optimized'),
    (r'(?:sinus|normal\s+sinus)\s*(?:rhythm|tachycardia|bradycardia)[^0-9]*(\d{2,3})', 0.88, 'optimized'),
    (r'(?:tachycardic|bradycardia)[^0-9]*(?:at|with|to)[^0-9]*(\d{1,3})', 0.85, 'optimized'),
    (r'(?:monitor|cardiac\s+monitor)[^0-9\n]*(?:rate|HR|pulse)[^0-9]*(\d{2,3})', 0.85, 'optimized'),
    (r'rate\s*(?:is|of|at|=)\s*(\d{2,3})', 0.82, 'optimized'),
    (r'heart\s+rate[^0-9]*of[^0-9]*(\d{1,3})', 0.85, 'optimized'),
    (r'pulse[^0-9]*of[^0-9]*(\d{1,3})', 0.82, 'optimized'),
    (r'(?:HR|Heart\s*Rate|Pulse)[^0-9]*in\s+the\s+(\d{2,3})s', 0.80, 'optimized'),

    # Specialized tier (0.65-0.80) - contextual patterns
    (r'vitals[^:]*:[^:]*(?:[^\/\d]*?)(?:\d{2,3}[/\\]\d{2,3})[^,]*,\s*(?:HR|P)?\s*(\d{2,3})', 0.75, 'specialized'),
    (r'VS[^\d\n]*[\d/,:.]+[^,\d\n]*,\s*(?:HR|P)?\s*(\d{2,3})', 0.72, 'specialized'),
    (r'(?:cardiac|cardio|heart)\s*(?:monitor|monitoring|telemetry)[^0-9]*(\d{2,3})', 0.70, 'specialized'),
    (r'(?:atrial|junctional|ventricular)[^0-9]*rhythm[^0-9]*(\d{2,3})', 0.70, 'specialized'),
    (r'(?<=\W)(?:HR|P)[\s:=]*(\d{2,3})[\s,]', 0.68, 'specialized'),
]
