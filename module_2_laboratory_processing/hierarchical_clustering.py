"""
Hierarchical clustering for unmapped lab tests.
"""

import numpy as np
import re
from typing import Dict, List, Tuple, Set
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform


def calculate_token_similarity(name1: str, name2: str) -> float:
    """
    Calculate Jaccard similarity between tokenized test names.

    Args:
        name1: First test name (e.g., "LOW DENSITY LIPOPROTEIN")
        name2: Second test name (e.g., "HIGH DENSITY LIPOPROTEIN")

    Returns:
        float: Similarity in [0, 1], where 1=identical tokens
    """
    # Tokenize: split on whitespace, convert to uppercase, remove empty
    tokens1 = set(name1.upper().split())
    tokens2 = set(name2.upper().split())

    # Remove common stop words that don't add meaning
    stop_words = {'TEST', 'BLOOD', 'SERUM', 'PLASMA', 'LEVEL', 'TOTAL'}
    tokens1 = tokens1 - stop_words
    tokens2 = tokens2 - stop_words

    # Jaccard similarity
    intersection = tokens1 & tokens2
    union = tokens1 | tokens2

    if len(union) == 0:
        return 0.0

    return len(intersection) / len(union)


def calculate_unit_incompatibility(unit1: str, unit2: str) -> float:
    """
    Calculate unit incompatibility score.

    Args:
        unit1: First unit (e.g., "mg/dL")
        unit2: Second unit (e.g., "mmol/L")

    Returns:
        float: Incompatibility in [0, 1], where 0=same unit, 1=incompatible
    """
    # Normalize units (lowercase, strip whitespace)
    u1 = unit1.lower().strip() if unit1 else ""
    u2 = unit2.lower().strip() if unit2 else ""

    # Exact match
    if u1 == u2:
        return 0.0

    # Try pint for dimension analysis
    try:
        import pint
        ureg = pint.UnitRegistry()

        # Parse units
        unit_obj1 = ureg(u1)
        unit_obj2 = ureg(u2)

        # Same dimensionality = convertible
        if unit_obj1.dimensionality == unit_obj2.dimensionality:
            return 0.3  # Compatible but need conversion
        else:
            return 1.0  # Incompatible dimensions

    except:
        # Fallback: simple string matching for common patterns
        normalize_map = {
            'mgdl': 'mg/dL',
            'mg/dl': 'mg/dL',
            'mmol': 'mmol/L',
            'umol/l': 'µmol/L',
            'g/dl': 'g/dL',
            'u/l': 'U/L'
        }

        u1_norm = normalize_map.get(u1.replace(' ', ''), u1)
        u2_norm = normalize_map.get(u2.replace(' ', ''), u2)

        if u1_norm == u2_norm:
            return 0.0

        # Check if both are concentration units (likely convertible)
        conc_units = {'mg/dL', 'mmol/L', 'µmol/L', 'g/dL', 'ng/mL', 'pg/mL'}
        if u1_norm in conc_units and u2_norm in conc_units:
            return 0.5  # Possibly convertible, needs review

        # Otherwise, incompatible
        return 1.0


def calculate_combined_distance(
    test1: Dict,
    test2: Dict,
    unit_weight: float = 0.4
) -> float:
    """
    Combined distance metric: 60% name similarity + 40% unit compatibility.

    Args:
        test1: Test dictionary with 'name' and 'unit' keys
        test2: Test dictionary with 'name' and 'unit' keys
        unit_weight: Weight for unit incompatibility (default 0.4)

    Returns:
        float: Distance in [0, 1], where 0=identical, 1=completely different
    """
    # Calculate name similarity and convert to distance
    name_similarity = calculate_token_similarity(test1['name'], test2['name'])
    name_distance = 1 - name_similarity

    # Calculate unit incompatibility (already a distance)
    unit_distance = calculate_unit_incompatibility(test1['unit'], test2['unit'])

    # Weighted combination
    combined = (1 - unit_weight) * name_distance + unit_weight * unit_distance

    return combined


def perform_hierarchical_clustering(
    unmapped_tests: List[Dict],
    threshold: float = 0.9,
    unit_weight: float = 0.4
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """
    Cluster unmapped tests using Ward's method with combined distance metric.

    Args:
        unmapped_tests: List of test dictionaries with 'name', 'unit', etc.
        threshold: Similarity threshold for cutting dendrogram (0-1)
        unit_weight: Weight for unit compatibility in distance (default 0.4)

    Returns:
        clusters: Dict mapping cluster_id to list of test indices
        linkage_matrix: Linkage matrix for dendrogram plotting
        distances: Full distance matrix for heatmap
    """
    n = len(unmapped_tests)

    if n == 0:
        return {}, None, None

    if n == 1:
        return {0: [0]}, None, None

    # Calculate pairwise distance matrix
    distances = np.zeros((n, n))
    for i in range(n):
        for j in range(i+1, n):
            dist = calculate_combined_distance(
                unmapped_tests[i],
                unmapped_tests[j],
                unit_weight=unit_weight
            )
            distances[i, j] = dist
            distances[j, i] = dist

    # Convert to condensed distance matrix for scipy
    condensed_dist = squareform(distances)

    # Perform hierarchical clustering (Ward's method)
    linkage_matrix = linkage(condensed_dist, method='ward')

    # Cut dendrogram at threshold
    # Threshold is a similarity threshold (0-1), convert to distance threshold
    # At threshold=0.9, we want to accept distances up to ~0.5 (generous)
    # At threshold=0.5, we want to accept distances up to the max
    distance_threshold = (1 - threshold) * 5.0  # Scale factor for Ward distance

    # Get cluster labels
    cluster_labels = fcluster(linkage_matrix, distance_threshold, criterion='distance')

    # Group tests by cluster
    clusters = {}
    for idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(idx)

    return clusters, linkage_matrix, distances


def detect_isoenzyme_pattern(test_names: List[str]) -> bool:
    """
    Detect if cluster contains isoenzymes (should be separated).

    Args:
        test_names: List of test names in cluster

    Returns:
        bool: True if isoenzyme pattern detected
    """
    # Patterns for isoenzymes
    patterns = [
        r'LDH\s*[1-5]',           # LDH1, LDH2, ..., LDH5
        r'CK[-\s]?(MB|MM|BB)',    # CK-MB, CK-MM, CK-BB
        r'TROPONIN\s*[IT]',       # Troponin I, Troponin T (different biomarkers)
    ]

    for pattern in patterns:
        matches = [re.search(pattern, name, re.IGNORECASE) for name in test_names]
        if sum(1 for m in matches if m) >= 2:
            return True  # At least 2 isoenzymes in cluster

    return False


def flag_suspicious_clusters(
    clusters: Dict[int, List[int]],
    unmapped_tests: List[Dict]
) -> Dict[int, List[str]]:
    """
    Identify clusters that need manual review.

    Args:
        clusters: Dict mapping cluster_id to list of test indices
        unmapped_tests: List of test dictionaries

    Returns:
        flags: Dict mapping cluster_id to list of flag reasons
    """
    flags = {}

    for cluster_id, test_indices in clusters.items():
        cluster_flags = []

        # Get test info
        test_names = [unmapped_tests[i]['name'] for i in test_indices]
        test_units = [unmapped_tests[i]['unit'] for i in test_indices]

        # Flag 1: Very large cluster (>10 tests)
        if len(test_indices) > 10:
            cluster_flags.append(f"large_cluster ({len(test_indices)} tests)")

        # Flag 2: Isoenzyme pattern
        if detect_isoenzyme_pattern(test_names):
            cluster_flags.append("isoenzyme_pattern")

        # Flag 3: Unit mismatch (incompatible units)
        unique_units = set(test_units)
        if len(unique_units) > 1:
            # Check if all are convertible
            incompatible = False
            units_list = list(unique_units)
            for i in range(len(units_list)):
                for j in range(i+1, len(units_list)):
                    incomp = calculate_unit_incompatibility(units_list[i], units_list[j])
                    if incomp > 0.5:  # Not convertible
                        incompatible = True
                        break
                if incompatible:
                    break

            if incompatible:
                cluster_flags.append(f"unit_mismatch ({', '.join(unique_units)})")

        # Flag 4: Generic terms (likely too broad)
        generic_terms = ['PANEL', 'PROFILE', 'COMPREHENSIVE', 'BASIC', 'COMPLETE']
        if any(term in ' '.join(test_names).upper() for term in generic_terms):
            cluster_flags.append("generic_terms")

        if cluster_flags:
            flags[cluster_id] = cluster_flags

    return flags
