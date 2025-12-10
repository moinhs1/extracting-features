# /home/moin/TDA_11_25/module_04_medications/extractors/__init__.py
"""
Module 4 Extractors
===================

Layer 1 (Bronze): Canonical medication extraction
Layer 1→Silver: RxNorm mapping
"""

from .dose_parser import (
    extract_dose,
    extract_route,
    extract_frequency,
    extract_drug_name,
    parse_medication_string,
)

from .canonical_extractor import (
    load_med_chunk,
    iter_med_chunks,
    load_patient_timelines,
    filter_to_cohort,
    compute_hours_from_t0,
    filter_study_window,
    parse_medications,
    transform_to_canonical,
    extract_canonical_records,
    extract_vocabulary,
)

from .rxnorm_mapper import (
    exact_match,
    fuzzy_match,
    ingredient_match,
    map_medication,
    map_vocabulary,
    get_mapping_stats,
    get_ingredient_for_rxcui,
)

__all__ = [
    # Dose parsing
    'extract_dose',
    'extract_route',
    'extract_frequency',
    'extract_drug_name',
    'parse_medication_string',
    # Canonical extraction
    'load_med_chunk',
    'iter_med_chunks',
    'load_patient_timelines',
    'filter_to_cohort',
    'compute_hours_from_t0',
    'filter_study_window',
    'parse_medications',
    'transform_to_canonical',
    'extract_canonical_records',
    'extract_vocabulary',
    # RxNorm mapping
    'exact_match',
    'fuzzy_match',
    'ingredient_match',
    'map_medication',
    'map_vocabulary',
    'get_mapping_stats',
    'get_ingredient_for_rxcui',
]
