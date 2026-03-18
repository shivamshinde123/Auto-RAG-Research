"""Local PDF data source connector.

Reads PDF files from a configured directory using PyMuPDF (primary)
with pdfplumber as a fallback parser.
"""

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.data_sources import register
from src.data_sources.base import BaseDataSource

logger = logging.getLogger(__name__)


@register("local_pdf")
class LocalPdfDataSource(BaseDataSource):

    def validate_config(self) -> bool:
        path = self.config.get("path")
        if not path:
            raise ValueError("local_pdf config missing required 'path' field")
        return True

    def health_check(self) -> bool:
        self.validate_config()
        path = Path(self.config["path"])
        if not path.exists():
            raise RuntimeError(f"PDF directory does not exist: {path}")
        if not path.is_dir():
            raise RuntimeError(f"PDF path is not a directory: {path}")
        pdf_files = list(path.glob("*.pdf"))
        if not pdf_files:
            raise RuntimeError(f"No PDF files found in: {path}")
        logger.info("health_check passed: %d PDF files in %s", len(pdf_files), path)
        return True

    def load(self) -> List[Document]:
        self.validate_config()
        path = Path(self.config["path"])
        documents: List[Document] = []

        pdf_files = sorted(path.glob("*.pdf"))
        if not pdf_files:
            logger.warning("No PDF files found in %s", path)
            return documents

        for pdf_path in pdf_files:
            try:
                docs = self._load_with_pymupdf(pdf_path)
                documents.extend(docs)
            except Exception as e:
                logger.warning(
                    "PyMuPDF failed on %s (%s), trying pdfplumber", pdf_path.name, e
                )
                try:
                    docs = self._load_with_pdfplumber(pdf_path)
                    documents.extend(docs)
                except Exception as e2:
                    logger.error(
                        "Both parsers failed on %s: %s", pdf_path.name, e2
                    )

        logger.info("Loaded %d document pages from %d PDF files", len(documents), len(pdf_files))
        return documents

    def _load_with_pymupdf(self, pdf_path: Path) -> List[Document]:
        import fitz

        docs: List[Document] = []
        doc = fitz.open(str(pdf_path))
        try:
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                if text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(pdf_path),
                                "source_type": "local_pdf",
                                "file_name": pdf_path.name,
                                "page_number": page_num + 1,
                                "total_pages": len(doc),
                            },
                        )
                    )
        finally:
            doc.close()
        return docs

    def _load_with_pdfplumber(self, pdf_path: Path) -> List[Document]:
        import pdfplumber

        docs: List[Document] = []
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text()
                if text and text.strip():
                    docs.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": str(pdf_path),
                                "source_type": "local_pdf",
                                "file_name": pdf_path.name,
                                "page_number": page_num + 1,
                                "total_pages": len(pdf.pages),
                            },
                        )
                    )
        return docs
