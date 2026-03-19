"""Tests for Notion data source connector."""

import os
from unittest.mock import MagicMock, patch

import pytest

from src.data_sources.notion import NotionDataSource


class TestNotionValidateConfig:
    def test_missing_database_id(self):
        source = NotionDataSource({"type": "notion"})
        with pytest.raises(ValueError, match="database_id"):
            source.validate_config()

    def test_missing_api_key(self, monkeypatch):
        monkeypatch.delenv("NOTION_API_KEY", raising=False)
        source = NotionDataSource({"type": "notion", "database_id": "abc"})
        with pytest.raises(ValueError, match="NOTION_API_KEY"):
            source.validate_config()

    def test_valid_config(self, monkeypatch):
        monkeypatch.setenv("NOTION_API_KEY", "ntn_test")
        source = NotionDataSource({"type": "notion", "database_id": "abc"})
        assert source.validate_config() is True


class TestNotionLoad:
    @patch("src.data_sources.notion.requests")
    def test_load_pages(self, mock_requests, monkeypatch):
        monkeypatch.setenv("NOTION_API_KEY", "ntn_test")

        # Mock database query
        query_resp = MagicMock()
        query_resp.status_code = 200
        query_resp.json.return_value = {
            "results": [{
                "id": "page1",
                "last_edited_time": "2026-01-01T00:00:00Z",
                "properties": {
                    "Name": {"type": "title", "title": [{"plain_text": "Test Page"}]},
                },
            }],
            "has_more": False,
        }

        # Mock block children
        blocks_resp = MagicMock()
        blocks_resp.status_code = 200
        blocks_resp.json.return_value = {
            "results": [{
                "type": "paragraph",
                "paragraph": {"rich_text": [{"plain_text": "Hello from Notion"}]},
                "has_children": False,
            }],
            "has_more": False,
        }
        blocks_resp.raise_for_status = MagicMock()
        query_resp.raise_for_status = MagicMock()

        mock_requests.post.return_value = query_resp
        mock_requests.get.side_effect = [MagicMock(status_code=200), blocks_resp]

        source = NotionDataSource({"type": "notion", "database_id": "db1"})

        # Mock health check's GET
        with patch.object(source, "validate_config"):
            docs = source.load()

        assert len(docs) == 1
        assert "Hello from Notion" in docs[0].page_content
        assert docs[0].metadata["title"] == "Test Page"


class TestNotionExtractText:
    def test_extract_various_block_types(self, monkeypatch):
        monkeypatch.setenv("NOTION_API_KEY", "ntn_test")
        source = NotionDataSource({"type": "notion", "database_id": "db1"})

        blocks = [
            {"type": "heading_1", "heading_1": {"rich_text": [{"plain_text": "Title"}]}, "has_children": False},
            {"type": "paragraph", "paragraph": {"rich_text": [{"plain_text": "Body text"}]}, "has_children": False},
            {"type": "bulleted_list_item", "bulleted_list_item": {"rich_text": [{"plain_text": "Item 1"}]}, "has_children": False},
            {"type": "code", "code": {"rich_text": [{"plain_text": "print('hi')"}]}, "has_children": False},
        ]

        text = source._extract_text({}, blocks)
        assert "# Title" in text
        assert "Body text" in text
        assert "- Item 1" in text
        assert "print('hi')" in text


class TestNotionHealthCheck:
    @patch("src.data_sources.notion.requests")
    def test_health_check_passes(self, mock_requests, monkeypatch):
        monkeypatch.setenv("NOTION_API_KEY", "ntn_test")
        resp = MagicMock()
        resp.status_code = 200
        mock_requests.get.return_value = resp

        source = NotionDataSource({"type": "notion", "database_id": "db1"})
        assert source.health_check() is True

    @patch("src.data_sources.notion.requests")
    def test_health_check_fails(self, mock_requests, monkeypatch):
        monkeypatch.setenv("NOTION_API_KEY", "ntn_test")
        resp = MagicMock()
        resp.status_code = 401
        resp.text = "Unauthorized"
        mock_requests.get.return_value = resp

        source = NotionDataSource({"type": "notion", "database_id": "db1"})
        with pytest.raises(ConnectionError, match="Cannot access"):
            source.health_check()
