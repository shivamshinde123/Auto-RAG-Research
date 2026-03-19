"""Dataset loader that orchestrates all enabled data sources."""

import hashlib
import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.data_sources import get_data_source

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Compute MD5 hash of document content for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def load_documents(
    data_source_configs: list[dict],
) -> tuple[List[Document], Optional[List[dict]]]:
    """Load and merge documents from all enabled data sources.

    Args:
        data_source_configs: List of data source config dicts, each with at
            least a 'type' key and an 'enabled' key.

    Returns:
        Tuple of (documents, qa_pairs). qa_pairs is None unless a huggingface
        connector is enabled and returns QA pairs.
    """
    all_documents: List[Document] = []
    all_qa_pairs: Optional[List[dict]] = None

    enabled_configs = [c for c in data_source_configs if c.get("enabled", False)]
    if not enabled_configs:
        logger.warning("No enabled data sources found")
        return [], None

    # Health check all sources first
    sources = []
    for config in enabled_configs:
        try:
            source = get_data_source(config)
            source.health_check()
            sources.append((config, source))
        except Exception as e:
            logger.error(
                "Health check failed for %s: %s — skipping",
                config.get("type", "unknown"), e,
            )

    if not sources:
        logger.error("All data source health checks failed")
        return [], None

    # Load from each source
    for config, source in sources:
        source_type = config.get("type", "unknown")
        try:
            # Special handling for huggingface: collect QA pairs
            if source_type == "huggingface" and hasattr(source, "load_with_qa"):
                docs, qa_pairs = source.load_with_qa()
                if qa_pairs:
                    all_qa_pairs = (all_qa_pairs or []) + qa_pairs
            else:
                docs = source.load()

            logger.info("Loaded %d documents from %s", len(docs), source_type)
            all_documents.extend(docs)
        except Exception as e:
            logger.error("Failed to load from %s: %s — continuing", source_type, e)

    # Deduplicate by content hash
    seen_hashes: set[str] = set()
    unique_documents: List[Document] = []
    for doc in all_documents:
        h = _content_hash(doc.page_content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_documents.append(doc)

    deduped = len(all_documents) - len(unique_documents)
    if deduped > 0:
        logger.info("Deduplicated %d documents (removed %d duplicates)", len(unique_documents), deduped)

    logger.info(
        "Total: %d documents from %d sources%s",
        len(unique_documents),
        len(sources),
        f", {len(all_qa_pairs)} QA pairs" if all_qa_pairs else "",
    )
    return unique_documents, all_qa_pairs
