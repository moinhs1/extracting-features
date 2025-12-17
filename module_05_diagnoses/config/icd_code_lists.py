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

CANCER_CODES = {
    "lung": {"icd10": ["C34"], "icd9": ["162"]},
    "gi": {"icd10": ["C15", "C16", "C17", "C18", "C19", "C20", "C21", "C22", "C23", "C24", "C25"], "icd9": ["150", "151", "152", "153", "154", "155", "156", "157"]},
    "gu": {"icd10": ["C64", "C65", "C66", "C67", "C68"], "icd9": ["188", "189"]},
    "hematologic": {"icd10": ["C81", "C82", "C83", "C84", "C85", "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96"], "icd9": ["200", "201", "202", "203", "204", "205", "206", "207", "208"]},
    "breast": {"icd10": ["C50"], "icd9": ["174", "175"]},
    "metastatic": {"icd10": ["C77", "C78", "C79", "C80"], "icd9": ["196", "197", "198", "199"]},
    "chemotherapy": {"icd10": ["Z51.1", "Z51.11", "Z51.12"], "icd9": ["V58.11", "V58.12"]},
}

CARDIOVASCULAR_CODES = {
    "heart_failure": {"icd10": ["I50"], "icd9": ["428"]},
    "heart_failure_reduced": {"icd10": ["I50.2", "I50.4"], "icd9": []},
    "heart_failure_preserved": {"icd10": ["I50.3"], "icd9": []},
    "coronary_artery_disease": {"icd10": ["I25", "I21", "I22"], "icd9": ["414", "410", "412"]},
    "atrial_fibrillation": {"icd10": ["I48"], "icd9": ["427.31"]},
    "pulmonary_hypertension": {"icd10": ["I27.0", "I27.2"], "icd9": ["416.0", "416.8"]},
    "valvular_heart_disease": {"icd10": ["I34", "I35", "I36", "I37", "I38", "I39"], "icd9": ["394", "395", "396", "397", "424"]},
}

PULMONARY_CODES = {
    "copd": {"icd10": ["J44"], "icd9": ["491", "492", "496"]},
    "asthma": {"icd10": ["J45", "J46"], "icd9": ["493"]},
    "interstitial_lung_disease": {"icd10": ["J84"], "icd9": ["516"]},
    "home_oxygen": {"icd10": ["Z99.81"], "icd9": ["V46.2"]},
    "respiratory_failure": {"icd10": ["J96"], "icd9": ["518.81", "518.82", "518.83", "518.84"]},
}

BLEEDING_CODES = {
    "gi_bleeding": {"icd10": ["K92.0", "K92.1", "K92.2", "K25.0", "K25.2", "K25.4", "K25.6", "K26.0", "K26.2", "K26.4", "K26.6"], "icd9": ["578", "531.0", "531.2", "531.4", "531.6", "532.0", "532.2", "532.4", "532.6"]},
    "intracranial_hemorrhage": {"icd10": ["I60", "I61", "I62"], "icd9": ["430", "431", "432"]},
    "other_major_bleeding": {"icd10": ["D62", "R58"], "icd9": ["285.1", "459.0"]},
    "peptic_ulcer": {"icd10": ["K25", "K26", "K27", "K28"], "icd9": ["531", "532", "533", "534"]},
    "thrombocytopenia": {"icd10": ["D69.6", "D69.59"], "icd9": ["287.4", "287.5"]},
    "coagulopathy": {"icd10": ["D68", "D65", "D66", "D67"], "icd9": ["286"]},
}

RENAL_CODES = {
    "ckd_stage1": {"icd10": ["N18.1"], "icd9": []},
    "ckd_stage2": {"icd10": ["N18.2"], "icd9": []},
    "ckd_stage3": {"icd10": ["N18.3", "N18.30", "N18.31", "N18.32"], "icd9": ["585.3"]},
    "ckd_stage4": {"icd10": ["N18.4"], "icd9": ["585.4"]},
    "ckd_stage5": {"icd10": ["N18.5", "N18.6"], "icd9": ["585.5", "585.6"]},
    "dialysis": {"icd10": ["Z99.2"], "icd9": ["V45.1", "V56"]},
    "aki": {"icd10": ["N17"], "icd9": ["584"]},
}

PROVOKING_FACTOR_CODES = {
    "recent_surgery": {"icd10": ["Z96", "Z98"], "icd9": ["V43", "V44", "V45"]},
    "trauma": {"icd10": ["S", "T0", "T1"], "icd9": ["8", "9"]},
    "immobilization": {"icd10": ["Z74.0", "R26.3"], "icd9": ["V49.84"]},
    "pregnancy": {"icd10": ["O", "Z33", "Z34", "Z39"], "icd9": ["V22", "V23", "V24", "6"]},
    "hormonal_therapy": {"icd10": ["Z79.3", "Z79.890"], "icd9": ["V25.0"]},
    "central_venous_catheter": {"icd10": ["Z45.2", "T82.7"], "icd9": ["V58.81", "996.62"]},
}

# Complication codes - references other code lists where appropriate
COMPLICATION_CODES = {
    "aki": RENAL_CODES["aki"],
    "bleeding_any": {
        "icd10": BLEEDING_CODES["gi_bleeding"]["icd10"] + BLEEDING_CODES["other_major_bleeding"]["icd10"],
        "icd9": BLEEDING_CODES["gi_bleeding"]["icd9"] + BLEEDING_CODES["other_major_bleeding"]["icd9"],
    },
    "bleeding_major": BLEEDING_CODES["gi_bleeding"],
    "intracranial_hemorrhage": BLEEDING_CODES["intracranial_hemorrhage"],
    "respiratory_failure": PULMONARY_CODES["respiratory_failure"],
    "cardiogenic_shock": {"icd10": ["R57.0"], "icd9": ["785.51"]},
    "cardiac_arrest": {"icd10": ["I46"], "icd9": ["427.5"]},
    "recurrent_vte": {
        "icd10": VTE_CODES["pe"]["icd10"] + VTE_CODES["dvt_lower"]["icd10"] + VTE_CODES["dvt_upper"]["icd10"],
        "icd9": VTE_CODES["pe"]["icd9"] + VTE_CODES["dvt_lower"]["icd9"] + VTE_CODES["dvt_upper"]["icd9"],
    },
    "cteph": {"icd10": ["I27.24", "I27.29"], "icd9": ["416.8"]},
}
