"""Local TXT data source connector.

Reads plain text files from a configured directory with automatic
encoding detection via chardet.
"""

import logging
from pathlib import Path
from typing import List

import chardet
from langchain_core.documents import Document

from src.data_sources import register
from src.data_sources.base import BaseDataSource

logger = logging.getLogger(__name__)


@register("local_txt")
class LocalTxtDataSource(BaseDataSource):

    def validate_config(self) -> bool:
        path = self.config.get("path")
        if not path:
            raise ValueError("local_txt config missing required 'path' field")
        return True

    def health_check(self) -> bool:
        self.validate_config()
        path = Path(self.config["path"])
        if not path.exists():
            raise RuntimeError(f"TXT directory does not exist: {path}")
        if not path.is_dir():
            raise RuntimeError(f"TXT path is not a directory: {path}")
        txt_files = list(path.glob("*.txt"))
        if not txt_files:
            raise RuntimeError(f"No .txt files found in: {path}")
        logger.info("health_check passed: %d TXT files in %s", len(txt_files), path)
        return True

    def load(self) -> List[Document]:
        self.validate_config()
        path = Path(self.config["path"])
        documents: List[Document] = []

        txt_files = sorted(path.glob("*.txt"))
        if not txt_files:
            logger.warning("No .txt files found in %s", path)
            return documents

        for txt_path in txt_files:
            try:
                text = self._read_with_detected_encoding(txt_path)
                if text.strip():
                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(txt_path),
                                "source_type": "local_txt",
                                "file_name": txt_path.name,
                                "char_count": len(text),
                            },
                        )
                    )
            except Exception as e:
                logger.error("Failed to read %s: %s", txt_path.name, e)

        logger.info("Loaded %d documents from %d TXT files", len(documents), len(txt_files))
        return documents

    def _read_with_detected_encoding(self, file_path: Path) -> str:
        raw = file_path.read_bytes()
        if not raw:
            return ""
        detected = chardet.detect(raw)
        encoding = detected.get("encoding") or "utf-8"
        return raw.decode(encoding)
