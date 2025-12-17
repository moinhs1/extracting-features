# pipeline.py
"""
Module 07: Social & Physical History Pipeline
=============================================

Main pipeline integrating all extractors and builders.
"""

import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import logging
import sys

sys.path.insert(0, str(Path(__file__).parent))

from config.social_physical_config import (
    PHY_FILE,
    PATIENT_TIMELINES_PKL,
    BRONZE_DIR,
    SILVER_DIR,
    GOLD_DIR,
    ALL_RELEVANT_CONCEPTS,
    ensure_directories,
)
from extractors.phy_extractor import PhyExtractor
from transformers.body_measurements_builder import BodyMeasurementsBuilder
from transformers.social_history_builder import SocialHistoryBuilder
from transformers.pain_builder import PainBuilder
from transformers.functional_status_builder import FunctionalStatusBuilder

logger = logging.getLogger(__name__)


class SocialPhysicalPipeline:
    """Main pipeline for social and physical history extraction."""

    def __init__(
        self,
        phy_path: str,
        index_dates: Dict[str, datetime],
        hnp_path: Optional[str] = None,
        prg_path: Optional[str] = None,
    ):
        """
        Initialize pipeline.

        Args:
            phy_path: Path to Phy.txt file
            index_dates: Dict mapping EMPI -> index date
            hnp_path: Optional path to Hnp.txt for clinical notes
            prg_path: Optional path to Prg.txt for progress notes
        """
        self.phy_path = Path(phy_path)
        self.index_dates = index_dates
        self.hnp_path = Path(hnp_path) if hnp_path else None
        self.prg_path = Path(prg_path) if prg_path else None

    def process_data(self, phy_data: pd.DataFrame, empis: List[str]) -> pd.DataFrame:
        """
        Process pre-loaded Phy data and build features.

        Args:
            phy_data: DataFrame with Phy.txt data
            empis: List of patient EMPIs to process

        Returns:
            DataFrame with one row per patient
        """
        # Initialize builders
        body_builder = BodyMeasurementsBuilder(phy_data, self.index_dates)
        social_builder = SocialHistoryBuilder(phy_data, self.index_dates)
        pain_builder = PainBuilder(phy_data, self.index_dates)
        functional_builder = FunctionalStatusBuilder(phy_data, self.index_dates)

        all_features = []
        for empi in empis:
            features = {'empi': empi}

            # Body measurements
            body = body_builder.build_all_features(empi)
            body.pop('empi', None)
            features.update(body)

            # Social history
            social = social_builder.build_all_features(empi)
            social.pop('empi', None)
            features.update(social)

            # Pain
            pain = pain_builder.build_all_features(empi)
            pain.pop('empi', None)
            features.update(pain)

            # Functional status
            functional = functional_builder.build_all_features(empi)
            functional.pop('empi', None)
            features.update(functional)

            all_features.append(features)

        return pd.DataFrame(all_features)

    def run(
        self,
        output_dir: Optional[str] = None,
        test_mode: bool = False,
        test_n_rows: int = 100000,
    ) -> pd.DataFrame:
        """
        Run full extraction pipeline.

        Args:
            output_dir: Output directory (default: module gold dir)
            test_mode: If True, only process subset
            test_n_rows: Rows for test mode

        Returns:
            DataFrame with all patient features
        """
        print("=" * 60)
        print("Module 07: Social & Physical History Extraction")
        print("=" * 60)

        ensure_directories()
        output_dir = Path(output_dir) if output_dir else GOLD_DIR

        # Extract from Phy.txt
        print("\n1. Extracting from Phy.txt...")
        extractor = PhyExtractor(str(self.phy_path))

        if test_mode:
            # Load limited rows for testing
            phy_data = extractor.parse_chunk(self.phy_path)
            if len(phy_data) > test_n_rows:
                phy_data = phy_data.head(test_n_rows)
        else:
            phy_data = extractor.extract_all_relevant()

        print(f"   Loaded {len(phy_data):,} relevant records")

        # Get EMPIs to process
        empis = list(self.index_dates.keys())
        print(f"\n2. Processing {len(empis)} patients...")

        # Build features
        features_df = self.process_data(phy_data, empis)

        # Save outputs
        print(f"\n3. Saving to {output_dir}...")
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "social_physical_features.parquet"
        features_df.to_parquet(output_path, index=False)

        # Summary
        print("\n" + "=" * 60)
        print("Extraction Summary")
        print("=" * 60)
        print(f"   Patients processed: {len(features_df)}")
        print(f"   Features per patient: {len(features_df.columns)}")

        # Key feature stats
        if 'bmi_at_index' in features_df.columns:
            bmi_coverage = features_df['bmi_at_index'].notna().mean()
            print(f"   BMI coverage: {bmi_coverage:.1%}")

        if 'smoking_status_at_index' in features_df.columns:
            smoking_known = (features_df['smoking_status_at_index'] != 'unknown').mean()
            print(f"   Smoking status known: {smoking_known:.1%}")

        if 'ivdu_ever' in features_df.columns:
            ivdu_rate = features_df['ivdu_ever'].mean()
            print(f"   IVDU ever rate: {ivdu_rate:.1%}")

        print(f"\n   Output: {output_path}")
        print("=" * 60)

        return features_df


# =============================================================================
# PATIENT TIMELINE LOADING
# =============================================================================

def load_patient_timelines(path: Optional[Path] = None) -> Dict[str, datetime]:
    """
    Load patient timelines from Module 1.

    Args:
        path: Optional path to patient_timelines.pkl

    Returns:
        Dictionary mapping EMPI -> index date
    """
    path = path or PATIENT_TIMELINES_PKL

    # Custom unpickler for PatientTimeline class
    class _Unpickler(pickle.Unpickler):
        def find_class(self, module, name):
            if name == 'PatientTimeline':
                module1_path = str(Path(__file__).parent.parent / "module_1_core_infrastructure")
                if module1_path not in sys.path:
                    sys.path.insert(0, module1_path)
                import module_01_core_infrastructure
                return module_01_core_infrastructure.PatientTimeline
            return super().find_class(module, name)

    with open(path, 'rb') as f:
        timelines = _Unpickler(f).load()

    return {
        empi: pd.Timestamp(timeline.time_zero).to_pydatetime()
        for empi, timeline in timelines.items()
    }


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extract social/physical history features")
    parser.add_argument('--test', action='store_true', help='Run in test mode')
    parser.add_argument('--n', type=int, default=100000, help='Rows for test mode')
    args = parser.parse_args()

    # Load patient timelines
    print("Loading patient timelines...")
    index_dates = load_patient_timelines()
    print(f"Loaded {len(index_dates)} patients")

    # Run pipeline
    pipeline = SocialPhysicalPipeline(
        phy_path=str(PHY_FILE),
        index_dates=index_dates,
    )
    pipeline.run(test_mode=args.test, test_n_rows=args.n)
