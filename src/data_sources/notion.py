"""Notion data source connector.

Fetches pages from a Notion database and extracts text content
from page blocks recursively.
"""

import logging
import os
from typing import List

import requests
from langchain_core.documents import Document

from src.data_sources import register
from src.data_sources.base import BaseDataSource

logger = logging.getLogger(__name__)

NOTION_API_URL = "https://api.notion.com/v1"
NOTION_VERSION = "2022-06-28"


@register("notion")
class NotionDataSource(BaseDataSource):

    def validate_config(self) -> bool:
        database_id = self.config.get("database_id")
        if not database_id:
            raise ValueError("notion config missing required 'database_id' field")
        if not os.environ.get("NOTION_API_KEY"):
            raise ValueError(
                "NOTION_API_KEY environment variable not set. "
                "Create an integration at https://www.notion.so/my-integrations "
                "and export NOTION_API_KEY."
            )
        return True

    def health_check(self) -> bool:
        self.validate_config()
        headers = self._headers()
        database_id = self.config["database_id"]
        resp = requests.get(
            f"{NOTION_API_URL}/databases/{database_id}",
            headers=headers,
            timeout=10,
        )
        if resp.status_code != 200:
            raise ConnectionError(
                f"Cannot access Notion database '{database_id}': "
                f"HTTP {resp.status_code}"
            )
        logger.info("health_check passed: Notion database %s accessible", database_id)
        return True

    def load(self) -> List[Document]:
        self.validate_config()
        headers = self._headers()
        database_id = self.config["database_id"]
        documents: List[Document] = []

        # Query all pages in the database
        pages = self._query_database(headers, database_id)

        for page in pages:
            try:
                page_id = page["id"]
                title = self._get_page_title(page)
                last_edited = page.get("last_edited_time", "")

                # Fetch and extract block content
                blocks = self._get_blocks(headers, page_id)
                text = self._extract_text(headers, blocks)

                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": f"notion://{page_id}",
                            "source_type": "notion",
                            "notion_page_id": page_id,
                            "title": title,
                            "last_edited_time": last_edited,
                        },
                    ))
            except Exception as e:
                logger.error("Failed to process Notion page %s: %s", page.get("id", "?"), e)

        logger.info("Loaded %d documents from Notion database %s", len(documents), database_id)
        return documents

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {os.environ['NOTION_API_KEY']}",
            "Notion-Version": NOTION_VERSION,
            "Content-Type": "application/json",
        }

    def _query_database(self, headers: dict, database_id: str) -> list:
        pages = []
        start_cursor = None
        while True:
            body = {}
            if start_cursor:
                body["start_cursor"] = start_cursor
            resp = requests.post(
                f"{NOTION_API_URL}/databases/{database_id}/query",
                headers=headers,
                json=body,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            pages.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")
        return pages

    def _get_page_title(self, page: dict) -> str:
        props = page.get("properties", {})
        for prop in props.values():
            if prop.get("type") == "title":
                title_parts = prop.get("title", [])
                return "".join(t.get("plain_text", "") for t in title_parts)
        return "Untitled"

    def _get_blocks(self, headers: dict, block_id: str) -> list:
        blocks = []
        start_cursor = None
        while True:
            params = {}
            if start_cursor:
                params["start_cursor"] = start_cursor
            resp = requests.get(
                f"{NOTION_API_URL}/blocks/{block_id}/children",
                headers=headers,
                params=params,
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            blocks.extend(data.get("results", []))
            if not data.get("has_more"):
                break
            start_cursor = data.get("next_cursor")
        return blocks

    def _extract_text(self, headers: dict, blocks: list) -> str:
        parts = []
        for block in blocks:
            block_type = block.get("type", "")
            block_data = block.get(block_type, {})

            # Extract rich text from common block types
            rich_text = block_data.get("rich_text", [])
            text = "".join(rt.get("plain_text", "") for rt in rich_text)

            if block_type in ("heading_1", "heading_2", "heading_3"):
                parts.append(f"\n{'#' * int(block_type[-1])} {text}\n")
            elif block_type in ("paragraph", "quote", "callout"):
                if text:
                    parts.append(text)
            elif block_type in ("bulleted_list_item", "numbered_list_item"):
                if text:
                    parts.append(f"- {text}")
            elif block_type == "to_do":
                checked = block_data.get("checked", False)
                marker = "[x]" if checked else "[ ]"
                parts.append(f"{marker} {text}")
            elif block_type == "toggle":
                if text:
                    parts.append(text)
            elif block_type == "code":
                if text:
                    parts.append(f"```\n{text}\n```")

            # Recurse into children
            if block.get("has_children"):
                try:
                    children = self._get_blocks(headers, block["id"])
                    child_text = self._extract_text(headers, children)
                    if child_text.strip():
                        parts.append(child_text)
                except Exception as e:
                    logger.debug("Failed to get children of block %s: %s", block["id"], e)

        return "\n".join(parts)
