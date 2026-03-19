"""AWS S3 data source connector.

Reads PDF and TXT files from a configured S3 bucket/prefix.
Uses IAM credentials from environment variables.
"""

import logging
import os
import tempfile
from pathlib import Path
from typing import List

from langchain_core.documents import Document

from src.data_sources import register
from src.data_sources.base import BaseDataSource

logger = logging.getLogger(__name__)


@register("s3")
class S3DataSource(BaseDataSource):

    def validate_config(self) -> bool:
        bucket = self.config.get("bucket")
        if not bucket:
            raise ValueError(
                "s3 config missing required 'bucket' field. "
                "Add 'bucket: <bucket_name>' to the s3 [[data_sources]] block."
            )
        if not os.environ.get("AWS_ACCESS_KEY_ID"):
            raise ValueError(
                "AWS_ACCESS_KEY_ID environment variable not set. "
                "Export it or add it to your .env file."
            )
        if not os.environ.get("AWS_SECRET_ACCESS_KEY"):
            raise ValueError(
                "AWS_SECRET_ACCESS_KEY environment variable not set. "
                "Export it or add it to your .env file."
            )
        return True

    def health_check(self) -> bool:
        self.validate_config()
        client = self._get_client()
        bucket = self.config["bucket"]
        try:
            client.head_bucket(Bucket=bucket)
        except Exception as e:
            raise ConnectionError(f"Cannot access S3 bucket '{bucket}': {e}")
        logger.info("health_check passed: S3 bucket %s accessible", bucket)
        return True

    def load(self) -> List[Document]:
        self.validate_config()
        client = self._get_client()
        bucket = self.config["bucket"]
        prefix = self.config.get("prefix", "")
        documents: List[Document] = []

        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.endswith("/"):
                    continue

                ext = Path(key).suffix.lower()
                if ext not in (".pdf", ".txt"):
                    logger.debug("Skipping unsupported file: %s", key)
                    continue

                try:
                    docs = self._process_file(client, bucket, obj)
                    documents.extend(docs)
                except Exception as e:
                    logger.error("Failed to process s3://%s/%s: %s", bucket, key, e)

        logger.info("Loaded %d documents from s3://%s/%s", len(documents), bucket, prefix)
        return documents

    def _get_client(self):
        import boto3
        return boto3.client("s3")

    def _process_file(self, client, bucket: str, obj: dict) -> List[Document]:
        key = obj["Key"]
        ext = Path(key).suffix.lower()

        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
            tmp_path = tmp.name
            client.download_file(bucket, key, tmp_path)

        try:
            if ext == ".pdf":
                return self._extract_pdf(tmp_path, bucket, obj)
            elif ext == ".txt":
                return self._extract_txt(tmp_path, bucket, obj)
            return []
        finally:
            Path(tmp_path).unlink(missing_ok=True)

    def _extract_pdf(self, file_path: str, bucket: str, obj: dict) -> List[Document]:
        import fitz

        docs: List[Document] = []
        doc = fitz.open(file_path)
        try:
            for page_num in range(len(doc)):
                text = doc[page_num].get_text()
                if text.strip():
                    docs.append(Document(
                        page_content=text,
                        metadata={
                            "source": f"s3://{bucket}/{obj['Key']}",
                            "source_type": "s3",
                            "file_name": Path(obj["Key"]).name,
                            "s3_key": obj["Key"],
                            "bucket_name": bucket,
                            "last_modified": str(obj.get("LastModified", "")),
                            "page_number": page_num + 1,
                        },
                    ))
        finally:
            doc.close()
        return docs

    def _extract_txt(self, file_path: str, bucket: str, obj: dict) -> List[Document]:
        text = Path(file_path).read_text(encoding="utf-8")
        if not text.strip():
            return []
        return [Document(
            page_content=text,
            metadata={
                "source": f"s3://{bucket}/{obj['Key']}",
                "source_type": "s3",
                "file_name": Path(obj["Key"]).name,
                "s3_key": obj["Key"],
                "bucket_name": bucket,
                "last_modified": str(obj.get("LastModified", "")),
            },
        )]
