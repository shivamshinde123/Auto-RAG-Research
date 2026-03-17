"""Abstract base class for all data source connectors."""

from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document


class BaseDataSource(ABC):
    """Base class that all data source connectors must implement.

    Each connector loads documents from a specific source (local files,
    cloud storage, APIs, etc.) and returns them as LangChain Document objects.
    """

    def __init__(self, config: dict):
        self.config = config

    @abstractmethod
    def load(self) -> List[Document]:
        """Load documents from this data source.

        Returns a list of LangChain Document objects. Every document must have:
        - page_content: the text content
        - metadata with at least: source, source_type, file_name (if applicable)
        """

    @abstractmethod
    def validate_config(self) -> bool:
        """Validate that all required config fields are present.

        Returns True if valid.
        Raises ValueError with a clear message if something is missing.
        """

    @abstractmethod
    def health_check(self) -> bool:
        """Check connectivity and access before the main run starts.

        For local sources: checks if the folder exists and has files.
        For remote sources: checks credentials and connectivity.

        Returns True if healthy.
        Raises ConnectionError or RuntimeError with a clear message on failure.
        """
