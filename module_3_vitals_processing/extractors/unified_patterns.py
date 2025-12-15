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
    'HR': (25, 220),        # Allow profound bradycardia (complete heart block, hypothermia)
    'SBP': (50, 260),
    'DBP': (25, 150),
    'PULSE_PRESSURE': (10, 120),
    'RR': (6, 50),
    'SPO2': (55, 100),      # SpO2 < 55% extremely rare in documented vitals
    'TEMP_C': (33.5, 42.5),
    'TEMP_F': (93, 108),
    'O2_FLOW': (0.25, 80),  # Pediatric/neonatal 0.25L; Vapotherm up to 60-80L
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

# Blood Pressure patterns - captures (SBP, DBP)
# IMPORTANT: No bare (\d{2,3})/(\d{2,3}) pattern - matches dates!
BP_PATTERNS = [
    # Standard tier (0.90-1.0) - explicit label
    (r'(?:Blood\s*[Pp]ressure|BP)\s*:?\s*\(?\!?\)?\s*(\d{2,3})[/\\](\d{2,3})', 0.95, 'standard'),
    (r'(?:BP|Blood\s*Pressure)\s*:?\s*(\d{2,3})\s*[/\\]\s*(\d{2,3})\s*(?:mmHg|mm\s*Hg)?', 0.95, 'standard'),
    (r'\(\d+-\d+\)/\(\d+-\d+\)\s*(\d{2,3})[/\\](\d{2,3})', 0.92, 'standard'),  # Reference range
    (r'(\d{2,3})[/\\](\d{2,3})\s*(?:mmHg|mm\s*Hg)', 0.90, 'standard'),  # With unit

    # Optimized tier (0.80-0.90) - strong context
    (r'(?:vitals?|v/?s)[:\s].{0,30}?(\d{2,3})[/\\](\d{2,3})', 0.88, 'optimized'),
    (r'blood\s+pressure[^0-9]*(\d{2,3})[^0-9]*(?:over|/)[^0-9]*(\d{2,3})', 0.88, 'optimized'),
    (r'(?:systolic|SBP)[^0-9]*(\d{2,3})[^0-9]*(?:diastolic|DBP)[^0-9]*(\d{2,3})', 0.85, 'optimized'),
    (r'(?:BP|Blood\s*Pressure)[^0-9]*of[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.85, 'optimized'),
    (r'initial\s+(?:vitals|VS)[^0-9]*(?:BP|blood)[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.85, 'optimized'),
    (r'cuff[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.82, 'optimized'),
    (r'(?:avg|average)\s+BP[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.82, 'optimized'),

    # Specialized tier (0.65-0.80) - contextual
    (r'VS[^\d\n]*[\d/,:.]+[^\d\n]*(\d{2,3})[/\\](\d{2,3})', 0.75, 'specialized'),
    (r'pressure\s+of\s+(\d{2,3})[/\\](\d{2,3})', 0.72, 'specialized'),
    (r'(?:measured|documented|recorded)[^0-9]*(?:bp|blood\s+pressure)[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.70, 'specialized'),
    (r'(?:admission|initial|presenting)[^0-9]*(?:bp|blood\s+pressure)[^0-9]*(\d{2,3})[/\\](\d{2,3})', 0.70, 'specialized'),
    (r'bp\s*=\s*(\d{2,3})[/\\](\d{2,3})', 0.68, 'specialized'),
]

# Respiratory Rate patterns
RR_PATTERNS = [
    # Standard tier
    (r'Respiratory\s*Rate\s*:?\s*(\d{1,2})\b', 0.95, 'standard'),
    (r'(?:RR|Resp|TRR)\s*:?\s*(\d{1,2})\b', 0.92, 'standard'),
    (r'\[\d{1,2}-\d{1,2}\]\s*(\d{1,2})\b', 0.90, 'standard'),

    # Optimized tier
    (r'(?:RR|Respiratory\s*Rate|Resp|respirations)[^0-9]*of[^0-9]*(\d{1,2})', 0.88, 'optimized'),
    (r'respiratory\s+rate[^0-9]*(\d{1,2})', 0.88, 'optimized'),
    (r'(?:breathing|respirations)[^0-9]*(?:at|with)[^0-9]*(\d{1,2})', 0.85, 'optimized'),
    (r'(\d{1,2})\s*(?:breaths?[/\s]*min|breaths?\s*per\s*minute)', 0.85, 'optimized'),
    (r'breath(?:ing|s)?\s*(?:at|of)\s*(\d{1,2})', 0.82, 'optimized'),

    # Specialized tier
    (r'vitals[^:]*:[^:]*(?:[^,]*,){2}[^,]*(?:RR)?\s*(\d{1,2})', 0.75, 'specialized'),
    (r'VS[^\d\n]*[\d/,:.]+[^,\d\n]*,[^,\d\n]*,[^,\d\n]*(?:RR)?\s*(\d{1,2})', 0.72, 'specialized'),
    (r'(?:ventilator|vent)[^0-9]*(?:rate|rr)[^0-9]*(\d{1,2})', 0.70, 'specialized'),
    (r'(?<=\W)(?:RR)[\s:=]*(\d{1,2})[\s,]', 0.68, 'specialized'),
]

# SpO2 patterns
# IMPORTANT: Patterns are carefully designed to avoid false positives from:
# - Cardiac pressures (RA=Right Atrial, e.g., "RA 15, PCWP 30")
# - Exercise test data (e.g., "RA Standing 60")
# - Ejection fraction (EF 65%), FEV1 (>70% predicted)
# - Ages (55 y/o), addresses, medication doses
SPO2_PATTERNS = [
    # Standard tier (0.90-1.0) - explicit SpO2 keywords with value
    (r'(?:SpO2|SaO2|O2\s*Sat(?:uration)?)\s*:?\s*>?(\d{2,3})\s*%?', 0.95, 'standard'),
    (r'(?:oxygen\s+saturation|pulse\s+ox|pulseox|pox)\s*:?\s*(\d{2,3})\s*%?', 0.92, 'standard'),

    # Optimized tier (0.80-0.90) - strong context required
    # Value% on device/oxygen context
    (r'(\d{2,3})\s*%\s*(?:on|at)\s+(?:\d+\s*L\s*)?(?:NC|nasal\s+cannula|O2|oxygen|RA|room\s+air)', 0.90, 'optimized'),
    # "satting/saturating at X%"
    (r'satt?(?:ing|urat(?:ing|ion))?\s*(?:at\s+)?(\d{2,3})\s*%', 0.90, 'optimized'),
    # "sat/sats of X%"
    (r'(?:sat|sats|saturation)\s+(?:of\s+)?(\d{2,3})\s*%', 0.88, 'optimized'),
    # "O2 sat/level X%"
    (r'(?:O2|oxygen)\s+(?:sat(?:uration)?|level)\s*(?:of|at|:)?\s*(\d{2,3})\s*%?', 0.88, 'optimized'),
    # SpO2/sat keyword BEFORE "on RA/room air" then value
    (r'(?:sat(?:uration)?|SpO2|O2\s*sat)[^0-9]{0,15}(?:on\s+)?(?:RA|room\s*air)[^0-9]{0,10}(\d{2,3})\s*%?', 0.88, 'optimized'),
    # "RA/room air" then SpO2/sat keyword then value
    (r'(?:RA|room\s*air)[^0-9]{0,10}(?:sat(?:uration)?|SpO2|O2\s*sat)[^0-9]{0,5}(\d{2,3})\s*%?', 0.88, 'optimized'),
    # "sats on RA X%"
    (r'sats?\s+(?:on\s+)?(?:RA|room\s*air)[^0-9]{0,5}(\d{2,3})\s*%?', 0.88, 'optimized'),

    # Specialized tier (0.65-0.80) - requires % sign to distinguish from cardiac pressures
    # RA/room air with % sign - accepts full SpO2 range (55-100%)
    (r'(?:RA|room\s*air)[^0-9%]{0,10}(\d{2,3})\s*%', 0.82, 'specialized'),
    # RA/room air followed by 95-100 range without % (exclude cardiac pressure range 10-30)
    (r'(?:on\s+)?(?:RA|room\s*air)[,\s]+(9[5-9]|100)(?!\s*(?:mg|ml|L|kg|mmHg|HR|RV|PA|cm|\d))', 0.78, 'specialized'),
    # Monitor context with % sign required
    (r'(?:monitor|monitoring)[^0-9\n]{0,20}(?:O2|sat|SpO2|saturation)[^0-9\n]{0,10}(\d{2,3})\s*%', 0.75, 'specialized'),
]

# Temperature patterns - captures (value, unit)
TEMP_PATTERNS = [
    # Standard tier - with explicit unit
    (r'(?:Temperature|Temp)\s*:?\s*(\d{2,3}\.?\d?)\s*[°]?\s*([CF])', 0.95, 'standard'),
    (r'(?:Tmax|T-max|Tcurrent)\s*:?\s*(\d{2,3}\.?\d?)\s*[°]?\s*([CF])', 0.95, 'standard'),
    (r'T\s+(\d{2,3}\.?\d?)\s*[°]?\s*([CF])', 0.92, 'standard'),

    # Optimized tier
    (r'(?:Temperature|Temp)\s*:?\s*(\d{2,3}\.?\d?)\s*(?:degrees)?\s*([CF])', 0.88, 'optimized'),
    (r'temperature[^0-9]*(\d{2,3}\.?\d?)[^0-9]*([CF])', 0.88, 'optimized'),
    (r'(?:afebrile|febrile)[^0-9]*(?:at|with)?[^0-9]*(\d{2,3}\.?\d?)\s*[°]?\s*([CF])?', 0.85, 'optimized'),
    (r'(\d{2,3}\.\d)\s*[°]\s*([CF])', 0.82, 'optimized'),

    # Specialized tier - may need unit inference
    (r'(?:Temperature|Temp)\s*:?\s*(\d{2,3}\.?\d?)\b', 0.75, 'specialized'),
    (r'(?:T|temp)[\s:=]+(\d{2,3}\.?\d?)(?!\d)', 0.70, 'specialized'),
    (r'(?:afebrile|febrile)[^0-9]*(9\d\.?\d{0,2}|10[0-4]\.?\d{0,2}|3[5-9]\.?\d{0,2})', 0.68, 'specialized'),
]

# O2 Flow Rate patterns - captures flow in L/min
O2_FLOW_PATTERNS = [
    # Standard tier
    (r'(?:on|via)\s+(\d+(?:\.\d)?)\s*L\s*(?:NC|nasal\s+cannula)', 0.95, 'standard'),
    (r'(\d+(?:\.\d)?)\s*L(?:PM|/min|iters?\s*/\s*min)', 0.95, 'standard'),
    (r'O2\s+Flow\s+Rate[^0-9]*(\d+(?:\.\d)?)', 0.92, 'standard'),

    # Optimized tier
    (r'(\d+(?:\.\d)?)\s*L\s*(?:NC|nasal\s+cannula|NRB|non-rebreather|mask|FM|face\s+mask|HFNC|high\s*-?\s*flow)', 0.88, 'optimized'),
    (r'(?:nasal\s+cannula|NC|high\s+flow|face\s+mask|HFNC|NRB)[^0-9]*(?:at|with|delivering)[^0-9]*(\d+(?:\.\d)?)\s*L?', 0.85, 'optimized'),
    (r'(?:on|receiving|with|at)\s+(\d+(?:\.\d)?)\s*L?\s*(?:O2|oxygen)', 0.82, 'optimized'),

    # Specialized tier
    (r'(\d+(?:\.\d)?)\s*L\s*(?:O2|oxygen)', 0.75, 'specialized'),
    (r'flow[^0-9]*(?:rate|of)?[^0-9]*(\d+(?:\.\d)?)\s*L?', 0.70, 'specialized'),
    (r'O2[^0-9]*(\d+(?:\.\d)?)\s*L', 0.68, 'specialized'),
]

# O2 Device patterns - returns device string (not numeric)
O2_DEVICE_PATTERNS = [
    # Standard tier - specific devices
    (r'(?:on|via)\s+(nasal\s+cannula|NC)', 0.95, 'standard'),
    (r'(?:on|via)\s+(room\s+air|RA|ambient\s+air)', 0.95, 'standard'),
    (r'(?:on|via)\s+(HFNC|high\s*-?\s*flow\s+nasal\s+cannula|high\s+flow)', 0.95, 'standard'),
    (r'(?:on|via)\s+(NRB|non-rebreather|non\s+rebreather)', 0.92, 'standard'),
    (r'(?:on|via)\s+(face\s+mask|FM|simple\s+mask|venturi\s+mask)', 0.92, 'standard'),

    # Optimized tier
    (r'(nasal\s+cannula|NC|nasal\s+prongs)', 0.85, 'optimized'),
    (r'(room\s+air|RA)', 0.85, 'optimized'),
    (r'(CPAP|BiPAP|ventilator|mechanical\s+ventilation|intubated)', 0.88, 'optimized'),

    # Specialized tier
    (r'supplemental[^0-9\n]*(oxygen|O2)', 0.75, 'specialized'),
    (r'(?:oxygen\s+therapy|O2\s+therapy)', 0.72, 'specialized'),
]

# BMI patterns
BMI_PATTERNS = [
    # Standard tier
    (r'BMI\s*:?\s*(\d{1,2}(?:\.\d{1,2})?)', 0.95, 'standard'),
    (r'(?:Body\s+Mass\s+Index|body\s+mass\s+index)\s*:?\s*(\d{1,2}(?:\.\d{1,2})?)', 0.95, 'standard'),

    # Optimized tier
    (r'BMI[^0-9]*of[^0-9]*(\d{1,2}(?:\.\d{1,2})?)', 0.88, 'optimized'),
    (r'BMI[^0-9]*(\d{1,2}(?:\.\d{1,2})?)\s*(?:kg/m2|kg/m\^?2)', 0.88, 'optimized'),
    (r'calculated\s+BMI[^0-9]*(\d{1,2}(?:\.\d{1,2})?)', 0.85, 'optimized'),

    # Specialized tier
    (r'(?:overweight|obese|obesity)[^0-9\n]*(?:BMI)[^0-9\n]*(\d{1,2}(?:\.\d{1,2})?)', 0.75, 'specialized'),
    (r'BMI\s*(?:of|is|was|=)[^0-9]*(\d{1,2}\.?\d{0,2})', 0.72, 'specialized'),
]
