"""
Phy.txt Extractor
=================

Extracts structured social/physical history from Phy.txt.
"""

import pandas as pd
from pathlib import Path
from typing import Generator, Optional, Set, Union
from io import StringIO
import logging

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.social_physical_config import ALL_RELEVANT_CONCEPTS

logger = logging.getLogger(__name__)


class PhyExtractor:
    """Extract structured data from Phy.txt."""

    COLUMN_DTYPES = {
        'EMPI': str,
        'EPIC_PMRN': str,
        'MRN_Type': str,
        'MRN': str,
        'Concept_Name': str,
        'Code_Type': str,
        'Code': str,
        'Result': str,
        'Units': str,
        'Provider': str,
        'Clinic': str,
        'Hospital': str,
        'Inpatient_Outpatient': str,
        'Encounter_number': str,
    }

    def __init__(self, phy_path: str, chunk_size: int = 100_000):
        """
        Initialize extractor.

        Args:
            phy_path: Path to Phy.txt
            chunk_size: Rows per chunk for streaming
        """
        self.phy_path = Path(phy_path)
        self.chunk_size = chunk_size
        self.all_concepts: Set[str] = set(ALL_RELEVANT_CONCEPTS)

    def parse_chunk(self, data: Union[str, StringIO, Path]) -> pd.DataFrame:
        """
        Parse a chunk of Phy.txt data.

        Args:
            data: File path, StringIO, or string data

        Returns:
            Parsed DataFrame
        """
        df = pd.read_csv(
            data,
            sep='|',
            dtype=self.COLUMN_DTYPES,
            low_memory=False,
        )
        return df

    def stream_relevant_chunks(self) -> Generator[pd.DataFrame, None, None]:
        """
        Stream Phy.txt, filtering to relevant concepts.

        Yields:
            DataFrames containing only relevant concept records
        """
        logger.info(f"Streaming {self.phy_path} in chunks of {self.chunk_size}")

        for chunk in pd.read_csv(
            self.phy_path,
            sep='|',
            chunksize=self.chunk_size,
            dtype=self.COLUMN_DTYPES,
            low_memory=False,
        ):
            # Filter to relevant concepts
            filtered = chunk[chunk['Concept_Name'].isin(self.all_concepts)]
            if not filtered.empty:
                yield filtered

    def extract_all_relevant(self, output_path: Optional[Path] = None) -> pd.DataFrame:
        """
        Extract all relevant records and optionally save.

        Args:
            output_path: Optional path to save parquet

        Returns:
            DataFrame with all relevant records
        """
        chunks = []
        for i, chunk in enumerate(self.stream_relevant_chunks()):
            chunks.append(chunk)
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {(i + 1) * self.chunk_size:,} rows")

        if not chunks:
            return pd.DataFrame()

        df = pd.concat(chunks, ignore_index=True)

        if output_path:
            df.to_parquet(output_path, index=False)
            logger.info(f"Saved {len(df):,} rows to {output_path}")

        return df
