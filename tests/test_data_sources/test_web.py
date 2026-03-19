"""Tests for web data source connector."""

from unittest.mock import MagicMock, patch

import pytest

from src.data_sources.web import WebDataSource


class TestWebValidateConfig:
    def test_missing_urls(self):
        source = WebDataSource({"type": "web"})
        with pytest.raises(ValueError, match="urls"):
            source.validate_config()

    def test_empty_urls(self):
        source = WebDataSource({"type": "web", "urls": []})
        with pytest.raises(ValueError, match="urls"):
            source.validate_config()

    def test_valid_config(self):
        source = WebDataSource({"type": "web", "urls": ["https://example.com"]})
        assert source.validate_config() is True


class TestWebLoad:
    @patch("src.data_sources.web.requests")
    @patch.object(WebDataSource, "_check_robots", return_value=True)
    def test_load_extracts_content(self, mock_robots, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        resp.text = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <nav>Navigation</nav>
            <main><p>Main content here</p></main>
            <footer>Footer</footer>
        </body>
        </html>
        """
        resp.raise_for_status = MagicMock()
        mock_requests.get.return_value = resp

        source = WebDataSource({"type": "web", "urls": ["https://example.com"]})
        docs = source.load()

        assert len(docs) == 1
        assert "Main content" in docs[0].page_content
        assert "Navigation" not in docs[0].page_content
        assert "Footer" not in docs[0].page_content
        assert docs[0].metadata["page_title"] == "Test Page"

    @patch("src.data_sources.web.requests")
    @patch.object(WebDataSource, "_check_robots", return_value=True)
    def test_load_handles_timeout(self, mock_robots, mock_requests):
        import requests
        mock_requests.get.side_effect = requests.Timeout("timeout")
        mock_requests.Timeout = requests.Timeout

        source = WebDataSource({"type": "web", "urls": ["https://slow.example.com"]})
        docs = source.load()
        assert len(docs) == 0

    @patch("src.data_sources.web.requests")
    @patch.object(WebDataSource, "_check_robots", return_value=False)
    def test_load_respects_robots(self, mock_robots, mock_requests):
        source = WebDataSource({"type": "web", "urls": ["https://blocked.example.com"]})
        docs = source.load()
        assert len(docs) == 0
        mock_requests.get.assert_not_called()

    @patch("src.data_sources.web.requests")
    @patch.object(WebDataSource, "_check_robots", return_value=True)
    def test_load_handles_failed_url(self, mock_robots, mock_requests):
        mock_requests.get.side_effect = Exception("connection error")

        source = WebDataSource({"type": "web", "urls": ["https://bad.example.com"]})
        docs = source.load()
        assert len(docs) == 0


class TestWebHealthCheck:
    @patch("src.data_sources.web.requests")
    def test_health_check_passes(self, mock_requests):
        resp = MagicMock()
        resp.status_code = 200
        mock_requests.head.return_value = resp

        source = WebDataSource({"type": "web", "urls": ["https://example.com"]})
        assert source.health_check() is True

    @patch("src.data_sources.web.requests")
    def test_health_check_fails(self, mock_requests):
        mock_requests.head.side_effect = Exception("unreachable")

        source = WebDataSource({"type": "web", "urls": ["https://bad.example.com"]})
        with pytest.raises(ConnectionError, match="None of the configured URLs"):
            source.health_check()
