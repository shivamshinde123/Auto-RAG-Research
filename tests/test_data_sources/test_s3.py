"""Tests for S3 data source connector."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.data_sources.s3 import S3DataSource


class TestS3ValidateConfig:
    def test_missing_bucket(self):
        source = S3DataSource({"type": "s3"})
        with pytest.raises(ValueError, match="bucket"):
            source.validate_config()

    def test_missing_aws_key(self, monkeypatch):
        monkeypatch.delenv("AWS_ACCESS_KEY_ID", raising=False)
        source = S3DataSource({"type": "s3", "bucket": "my-bucket"})
        with pytest.raises(ValueError, match="AWS_ACCESS_KEY_ID"):
            source.validate_config()

    def test_missing_aws_secret(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.delenv("AWS_SECRET_ACCESS_KEY", raising=False)
        source = S3DataSource({"type": "s3", "bucket": "my-bucket"})
        with pytest.raises(ValueError, match="AWS_SECRET_ACCESS_KEY"):
            source.validate_config()

    def test_valid_config(self, monkeypatch):
        monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test")
        monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test")
        source = S3DataSource({"type": "s3", "bucket": "my-bucket"})
        assert source.validate_config() is True


class TestS3Load:
    @patch.object(S3DataSource, "_get_client")
    @patch.object(S3DataSource, "validate_config")
    def test_load_txt_files(self, mock_validate, mock_client):
        client = MagicMock()
        mock_client.return_value = client

        paginator = MagicMock()
        paginator.paginate.return_value = [{
            "Contents": [
                {"Key": "docs/file1.txt", "LastModified": "2026-01-01"},
            ]
        }]
        client.get_paginator.return_value = paginator

        source = S3DataSource({"type": "s3", "bucket": "test-bucket", "prefix": "docs/"})

        with patch.object(source, "_process_file") as mock_process:
            from langchain_core.documents import Document
            mock_process.return_value = [
                Document(page_content="text content", metadata={"source_type": "s3"})
            ]
            docs = source.load()

        assert len(docs) == 1

    @patch.object(S3DataSource, "_get_client")
    @patch.object(S3DataSource, "validate_config")
    def test_load_skips_unsupported(self, mock_validate, mock_client):
        client = MagicMock()
        mock_client.return_value = client

        paginator = MagicMock()
        paginator.paginate.return_value = [{
            "Contents": [
                {"Key": "docs/image.png", "LastModified": "2026-01-01"},
            ]
        }]
        client.get_paginator.return_value = paginator

        source = S3DataSource({"type": "s3", "bucket": "test-bucket"})
        docs = source.load()
        assert len(docs) == 0

    def test_extract_txt(self, tmp_path):
        txt_file = tmp_path / "test.txt"
        txt_file.write_text("Hello world", encoding="utf-8")

        source = S3DataSource({"type": "s3", "bucket": "b"})
        docs = source._extract_txt(str(txt_file), "b", {"Key": "test.txt", "LastModified": ""})
        assert len(docs) == 1
        assert docs[0].page_content == "Hello world"
        assert docs[0].metadata["source_type"] == "s3"

    def test_extract_txt_empty(self, tmp_path):
        txt_file = tmp_path / "empty.txt"
        txt_file.write_text("", encoding="utf-8")

        source = S3DataSource({"type": "s3", "bucket": "b"})
        docs = source._extract_txt(str(txt_file), "b", {"Key": "empty.txt"})
        assert len(docs) == 0


class TestS3HealthCheck:
    @patch.object(S3DataSource, "_get_client")
    @patch.object(S3DataSource, "validate_config")
    def test_health_check_passes(self, mock_validate, mock_client):
        client = MagicMock()
        mock_client.return_value = client

        source = S3DataSource({"type": "s3", "bucket": "test-bucket"})
        assert source.health_check() is True

    @patch.object(S3DataSource, "_get_client")
    @patch.object(S3DataSource, "validate_config")
    def test_health_check_fails(self, mock_validate, mock_client):
        client = MagicMock()
        client.head_bucket.side_effect = Exception("access denied")
        mock_client.return_value = client

        source = S3DataSource({"type": "s3", "bucket": "bad-bucket"})
        with pytest.raises(ConnectionError, match="Cannot access"):
            source.health_check()
