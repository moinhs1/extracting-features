"""
Unit conversion system for lab tests.
"""

from typing import Dict, Optional, Tuple


# Common lab test unit conversions
# Format: {test_component: {'target': unit, 'factors': {source_unit: factor}}}
DEFAULT_CONVERSIONS = {
    'glucose': {
        'target': 'mg/dL',
        'factors': {
            'mmol/L': 18.018,
            'mg/dL': 1.0
        }
    },
    'creatinine': {
        'target': 'mg/dL',
        'factors': {
            'µmol/L': 0.0113,
            'umol/L': 0.0113,
            'mg/dL': 1.0
        }
    },
    'cholesterol': {
        'target': 'mg/dL',
        'factors': {
            'mmol/L': 38.67,
            'mg/dL': 1.0
        }
    },
    'triglycerides': {
        'target': 'mg/dL',
        'factors': {
            'mmol/L': 88.57,
            'mg/dL': 1.0
        }
    },
    'bilirubin': {
        'target': 'mg/dL',
        'factors': {
            'µmol/L': 0.0585,
            'umol/L': 0.0585,
            'mg/dL': 1.0
        }
    },
    'calcium': {
        'target': 'mg/dL',
        'factors': {
            'mmol/L': 4.008,
            'mg/dL': 1.0
        }
    },
}


class UnitConverter:
    """Handles unit conversion for lab tests."""

    def __init__(self, custom_conversions: Optional[Dict] = None):
        """
        Initialize unit converter.

        Args:
            custom_conversions: Optional dict of custom conversions to add
        """
        self.conversions = DEFAULT_CONVERSIONS.copy()
        if custom_conversions:
            self.conversions.update(custom_conversions)

    def normalize_unit(self, unit: str) -> str:
        """
        Normalize unit string for matching.

        Args:
            unit: Unit string to normalize

        Returns:
            str: Normalized unit
        """
        # Convert to lowercase and strip
        unit = unit.lower().strip()

        # Remove spaces
        unit = unit.replace(' ', '')

        # Common mappings
        normalize_map = {
            'mgdl': 'mg/dL',
            'mg/dl': 'mg/dL',
            'mmol': 'mmol/L',
            'umol/l': 'µmol/L',
            'umoll': 'µmol/L',
            'g/dl': 'g/dL',
            'gdl': 'g/dL',
            'u/l': 'U/L',
            'ul': 'U/L'
        }

        return normalize_map.get(unit, unit)

    def get_conversion_factor(
        self,
        test_component: str,
        source_unit: str
    ) -> Optional[float]:
        """
        Get conversion factor from source unit to target unit.

        Args:
            test_component: Test component name (e.g., 'glucose')
            source_unit: Source unit (e.g., 'mmol/L')

        Returns:
            float or None: Conversion factor, or None if not found
        """
        # Normalize inputs
        component = test_component.lower().strip()
        unit = self.normalize_unit(source_unit)

        # Look up conversion
        if component in self.conversions:
            factors = self.conversions[component]['factors']
            # Try normalized unit, then original
            return factors.get(unit) or factors.get(source_unit)

        return None

    def convert_value(
        self,
        value: float,
        test_component: str,
        source_unit: str
    ) -> Tuple[float, str, bool]:
        """
        Convert a value to standard units.

        Args:
            value: Original value
            test_component: Test component name
            source_unit: Source unit

        Returns:
            (converted_value, target_unit, success):
                - converted_value: Converted value, or original if no conversion
                - target_unit: Target unit, or source if no conversion
                - success: True if conversion applied, False if not found
        """
        factor = self.get_conversion_factor(test_component, source_unit)

        if factor is None:
            # No conversion found, return original
            return value, source_unit, False

        # Apply conversion
        converted = value * factor

        # Get target unit
        component = test_component.lower().strip()
        target_unit = self.conversions[component]['target']

        return converted, target_unit, True
