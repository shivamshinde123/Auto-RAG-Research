"""Local CSV data source connector.

Reads CSV files from a configured directory, extracting text from
a specified column.
"""

import csv
import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.data_sources import register
from src.data_sources.base import BaseDataSource

logger = logging.getLogger(__name__)


@register("local_csv")
class LocalCsvDataSource(BaseDataSource):

    def validate_config(self) -> bool:
        path = self.config.get("path")
        if not path:
            raise ValueError("local_csv config missing required 'path' field")
        text_column = self.config.get("text_column")
        if not text_column:
            raise ValueError("local_csv config missing required 'text_column' field")
        return True

    def health_check(self) -> bool:
        self.validate_config()
        path = Path(self.config["path"])
        if not path.exists():
            raise RuntimeError(f"CSV directory does not exist: {path}")
        if not path.is_dir():
            raise RuntimeError(f"CSV path is not a directory: {path}")
        csv_files = list(path.glob("*.csv"))
        if not csv_files:
            raise RuntimeError(f"No .csv files found in: {path}")

        text_column = self.config["text_column"]
        for csv_file in csv_files:
            with open(csv_file, newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames is None:
                    raise RuntimeError(f"CSV file is empty: {csv_file.name}")
                if text_column not in reader.fieldnames:
                    raise RuntimeError(
                        f"text_column '{text_column}' not found in {csv_file.name}. "
                        f"Available columns: {', '.join(reader.fieldnames)}"
                    )

        logger.info("health_check passed: %d CSV files in %s", len(csv_files), path)
        return True

    def load(self) -> List[Document]:
        self.validate_config()
        path = Path(self.config["path"])
        text_column = self.config["text_column"]
        documents: List[Document] = []

        csv_files = sorted(path.glob("*.csv"))
        if not csv_files:
            logger.warning("No .csv files found in %s", path)
            return documents

        for csv_file in csv_files:
            try:
                with open(csv_file, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames is None:
                        logger.warning("Skipping empty CSV: %s", csv_file.name)
                        continue
                    if text_column not in reader.fieldnames:
                        logger.error(
                            "text_column '%s' not found in %s, skipping",
                            text_column,
                            csv_file.name,
                        )
                        continue

                    for row_index, row in enumerate(reader):
                        text = row.get(text_column, "")
                        if text and text.strip():
                            documents.append(
                                Document(
                                    page_content=text,
                                    metadata={
                                        "source": str(csv_file),
                                        "source_type": "local_csv",
                                        "file_name": csv_file.name,
                                        "row_index": row_index,
                                    },
                                )
                            )
            except Exception as e:
                logger.error("Failed to read %s: %s", csv_file.name, e)

        logger.info(
            "Loaded %d documents from %d CSV files", len(documents), len(csv_files)
        )
        return documents
