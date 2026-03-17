"""local_pdf data source connector — stub to be implemented."""

from typing import List
from langchain_core.documents import Document
from src.data_sources import register
from src.data_sources.base import BaseDataSource


@register("local_pdf")
class LocalPdfDataSource(BaseDataSource):

    def load(self) -> List[Document]:
        raise NotImplementedError("local_pdf connector not yet implemented")

    def validate_config(self) -> bool:
        raise NotImplementedError("local_pdf connector not yet implemented")

    def health_check(self) -> bool:
        raise NotImplementedError("local_pdf connector not yet implemented")
