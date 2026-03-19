"""Tests for Google Drive data source connector."""

from unittest.mock import MagicMock, patch

import pytest

from src.data_sources.gdrive import GdriveDataSource


class TestGdriveValidateConfig:
    def test_missing_folder_id(self):
        source = GdriveDataSource({"type": "gdrive"})
        with pytest.raises(ValueError, match="folder_id"):
            source.validate_config()

    def test_missing_credentials_file(self, tmp_path):
        source = GdriveDataSource({
            "type": "gdrive",
            "folder_id": "abc123",
            "credentials_path": str(tmp_path / "nonexistent.json"),
        })
        with pytest.raises(ValueError, match="credentials not found"):
            source.validate_config()

    def test_valid_config(self, tmp_path):
        creds = tmp_path / "credentials.json"
        creds.write_text("{}")
        source = GdriveDataSource({
            "type": "gdrive",
            "folder_id": "abc123",
            "credentials_path": str(creds),
        })
        assert source.validate_config() is True


class TestGdriveLoad:
    @patch.object(GdriveDataSource, "_build_service")
    @patch.object(GdriveDataSource, "validate_config")
    def test_load_pdf_file(self, mock_validate, mock_service):
        mock_svc = MagicMock()
        mock_service.return_value = mock_svc

        # Mock list files
        mock_svc.files().list().execute.return_value = {
            "files": [{
                "id": "file1",
                "name": "test.pdf",
                "mimeType": "application/pdf",
                "modifiedTime": "2026-01-01T00:00:00Z",
            }]
        }

        # Mock file download
        mock_svc.files().get_media().execute.return_value = b"fake pdf content"

        source = GdriveDataSource({"type": "gdrive", "folder_id": "folder1"})

        with patch.object(source, "_extract_pdf") as mock_extract:
            from langchain_core.documents import Document
            mock_extract.return_value = [
                Document(page_content="PDF text", metadata={"source": "gdrive://file1"})
            ]
            docs = source.load()

        assert len(docs) == 1
        assert docs[0].page_content == "PDF text"

    @patch.object(GdriveDataSource, "_build_service")
    @patch.object(GdriveDataSource, "validate_config")
    def test_load_skips_unsupported_files(self, mock_validate, mock_service):
        mock_svc = MagicMock()
        mock_service.return_value = mock_svc

        mock_svc.files().list().execute.return_value = {
            "files": [{
                "id": "file1",
                "name": "image.png",
                "mimeType": "image/png",
                "modifiedTime": "2026-01-01T00:00:00Z",
            }]
        }

        source = GdriveDataSource({"type": "gdrive", "folder_id": "folder1"})
        docs = source.load()
        assert len(docs) == 0


class TestGdriveHealthCheck:
    @patch.object(GdriveDataSource, "_build_service")
    @patch.object(GdriveDataSource, "validate_config")
    def test_health_check_passes(self, mock_validate, mock_service):
        mock_svc = MagicMock()
        mock_service.return_value = mock_svc
        mock_svc.files().list().execute.return_value = {"files": []}

        source = GdriveDataSource({"type": "gdrive", "folder_id": "folder1"})
        assert source.health_check() is True

    @patch.object(GdriveDataSource, "_build_service")
    @patch.object(GdriveDataSource, "validate_config")
    def test_health_check_fails(self, mock_validate, mock_service):
        mock_svc = MagicMock()
        mock_service.return_value = mock_svc
        mock_svc.files().list().execute.side_effect = Exception("API error")

        source = GdriveDataSource({"type": "gdrive", "folder_id": "folder1"})
        with pytest.raises(ConnectionError, match="Cannot access"):
            source.health_check()
