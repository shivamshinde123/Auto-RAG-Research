"""web data source connector — stub to be implemented."""

from typing import List
from langchain_core.documents import Document
from src.data_sources import register
from src.data_sources.base import BaseDataSource


@register("web")
class WebDataSource(BaseDataSource):

    def load(self) -> List[Document]:
        raise NotImplementedError("web connector not yet implemented")

    def validate_config(self) -> bool:
        raise NotImplementedError("web connector not yet implemented")

    def health_check(self) -> bool:
        raise NotImplementedError("web connector not yet implemented")
