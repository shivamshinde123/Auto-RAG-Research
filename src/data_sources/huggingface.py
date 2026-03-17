"""huggingface data source connector — stub to be implemented."""

from typing import List
from langchain_core.documents import Document
from src.data_sources import register
from src.data_sources.base import BaseDataSource


@register("huggingface")
class HuggingFaceDataSource(BaseDataSource):

    def load(self) -> List[Document]:
        raise NotImplementedError("huggingface connector not yet implemented")

    def validate_config(self) -> bool:
        raise NotImplementedError("huggingface connector not yet implemented")

    def health_check(self) -> bool:
        raise NotImplementedError("huggingface connector not yet implemented")
