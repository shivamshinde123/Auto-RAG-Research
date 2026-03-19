"""Web URL scraper data source connector.

Scrapes configured URLs using BeautifulSoup, extracts main content,
and respects robots.txt.
"""

import logging
from datetime import datetime, timezone
from typing import List
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document

from src.data_sources import register
from src.data_sources.base import BaseDataSource

logger = logging.getLogger(__name__)

STRIP_TAGS = {"nav", "footer", "sidebar", "script", "style", "header", "aside"}
DEFAULT_TIMEOUT = 10


@register("web")
class WebDataSource(BaseDataSource):

    ALLOWED_SCHEMES = {"http", "https"}

    def validate_config(self) -> bool:
        urls = self.config.get("urls")
        if not urls or not isinstance(urls, list) or len(urls) == 0:
            raise ValueError("web config missing required 'urls' field (must be a non-empty list)")
        for url in urls:
            parsed = urlparse(url)
            if parsed.scheme not in self.ALLOWED_SCHEMES:
                raise ValueError(
                    f"URL scheme '{parsed.scheme}' is not allowed. "
                    f"Only {self.ALLOWED_SCHEMES} are supported."
                )
            if not parsed.netloc:
                raise ValueError(f"URL '{url}' is missing a valid host")
        return True

    def health_check(self) -> bool:
        self.validate_config()
        urls = self.config["urls"]
        reachable = False
        for url in urls:
            try:
                resp = requests.head(url, timeout=DEFAULT_TIMEOUT, allow_redirects=True)
                if resp.status_code < 400:
                    reachable = True
                    break
            except Exception:
                continue

        if not reachable:
            raise ConnectionError("None of the configured URLs are reachable")
        logger.info("health_check passed: at least one URL reachable")
        return True

    def load(self) -> List[Document]:
        self.validate_config()
        urls = self.config["urls"]
        documents: List[Document] = []

        for url in urls:
            try:
                if not self._check_robots(url):
                    logger.warning("robots.txt disallows access to %s, skipping", url)
                    continue

                resp = requests.get(url, timeout=DEFAULT_TIMEOUT)
                resp.raise_for_status()

                soup = BeautifulSoup(resp.text, "html.parser")
                title = soup.title.string.strip() if soup.title and soup.title.string else url

                # Strip unwanted elements
                for tag_name in STRIP_TAGS:
                    for tag in soup.find_all(tag_name):
                        tag.decompose()

                text = soup.get_text(separator="\n", strip=True)
                if text.strip():
                    documents.append(Document(
                        page_content=text,
                        metadata={
                            "source": url,
                            "source_type": "web",
                            "url": url,
                            "page_title": title,
                            "scrape_timestamp": datetime.now(timezone.utc).isoformat(),
                        },
                    ))
            except Exception as e:
                if "Timeout" in type(e).__name__:
                    logger.warning("Timeout scraping %s, skipping", url)
                else:
                    logger.error("Failed to scrape %s: %s", url, e)

        logger.info("Loaded %d documents from %d URLs", len(documents), len(urls))
        return documents

    def _check_robots(self, url: str) -> bool:
        """Check if robots.txt allows scraping the URL."""
        try:
            parsed = urlparse(url)
            robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
            rp = RobotFileParser()
            rp.set_url(robots_url)
            rp.read()
            return rp.can_fetch("*", url)
        except Exception:
            # If we can't read robots.txt, allow access
            return True
