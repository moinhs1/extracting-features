"""ICD code definitions for PE-specific features."""

VTE_CODES = {
    "pe": {
        "icd10": ["I26.0", "I26.9", "I26.90", "I26.92", "I26.93", "I26.94", "I26.99"],
        "icd9": ["415.11", "415.12", "415.13", "415.19"],
    },
    "dvt_lower": {
        "icd10": ["I82.4", "I82.5"],  # Lower extremity DVT
        "icd9": ["453.4", "453.5", "453.8"],
    },
    "dvt_upper": {
        "icd10": ["I82.6"],  # Upper extremity DVT
        "icd9": ["453.82", "453.83"],
    },
}
