"""Tests for data source base class and registry."""

import pytest
from langchain_core.documents import Document

from src.data_sources.base import BaseDataSource
from src.data_sources import get_data_source, _REGISTRY, _ensure_registered


class TestBaseDataSource:
    def test_cannot_instantiate_abc(self):
        """BaseDataSource is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseDataSource(config={})

    def test_subclass_must_implement_all_methods(self):
        """A subclass that doesn't implement all methods can't be instantiated."""

        class IncompleteSource(BaseDataSource):
            def load(self):
                return []

        with pytest.raises(TypeError):
            IncompleteSource(config={})


class TestRegistry:
    def test_all_connector_types_registered(self):
        """All expected connector types are in the registry."""
        _ensure_registered()

        expected_types = {
            "local_pdf",
            "local_txt",
            "local_csv",
            "gdrive",
            "s3",
            "notion",
            "web",
            "huggingface",
        }
        assert expected_types == set(_REGISTRY.keys())

    def test_get_data_source_returns_correct_type(self):
        """get_data_source returns an instance of the correct connector."""
        source = get_data_source({"type": "local_pdf", "path": "data/pdfs/"})
        assert isinstance(source, BaseDataSource)
        assert source.config["type"] == "local_pdf"

    def test_get_data_source_unknown_type(self):
        """get_data_source raises ValueError for unknown types."""
        with pytest.raises(ValueError, match="Unknown data source type 'nonexistent'"):
            get_data_source({"type": "nonexistent"})

    def test_get_data_source_missing_type(self):
        """get_data_source raises ValueError when type is missing."""
        with pytest.raises(ValueError, match="missing 'type' field"):
            get_data_source({})

    def test_implemented_connectors_have_load(self):
        """Implemented connectors have working load/validate_config methods."""
        source = get_data_source({"type": "local_pdf", "path": "data/pdfs/"})
        # validate_config should work (path is set)
        assert source.validate_config() is True
