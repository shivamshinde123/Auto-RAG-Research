"""Dataset loader that loads PDFs and generates QA pairs from them.

Orchestrates: PDF loading via data source connectors -> deduplication
by content hash -> LLM-based QA pair generation from document text.
"""

import hashlib
import json
import logging
from typing import List, Set, Tuple

from langchain_core.documents import Document

from src.data_sources import get_data_source

logger = logging.getLogger(__name__)


def _content_hash(text: str) -> str:
    """Compute MD5 hash of document content for deduplication."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _generate_qa_pairs(documents: List[Document], num_pairs: int, llm_model: str) -> List[dict]:
    """Generate QA pairs from document content using an LLM.

    Picks chunks of text from the documents and asks the LLM to create
    question-answer pairs based on the content.
    """
    import openai

    # Gather text chunks for QA generation (use first N docs, limited to ~2000 chars each)
    text_chunks = []
    for doc in documents:
        text = doc.page_content.strip()
        if len(text) > 100:  # Skip very short chunks
            text_chunks.append(text[:2000])
        if len(text_chunks) >= num_pairs * 2:
            break

    if not text_chunks:
        logger.warning("No suitable text chunks for QA generation")
        return []

    # Build prompt with sampled chunks
    sampled = text_chunks[:min(len(text_chunks), num_pairs + 5)]
    context_block = "\n\n---\n\n".join(sampled)

    prompt = f"""Based on the following document excerpts, generate exactly {num_pairs} question-answer pairs.
Each question should be answerable from the provided text. Keep answers concise and factual.

Return ONLY a JSON array with objects having "question" and "answer" keys. No other text.

Document excerpts:
{context_block}"""

    try:
        client = openai.OpenAI()
        # Low temperature for factual, consistent QA generation
        response = client.chat.completions.create(
            model=llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )
        content = response.choices[0].message.content.strip()

        # LLM sometimes wraps JSON in markdown code fences — strip them
        if content.startswith("```"):
            content = content.split("\n", 1)[1]
            if content.endswith("```"):
                content = content[:-3]

        try:
            qa_raw = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error("LLM returned invalid JSON for QA generation: %s\nResponse: %s", e, content[:500])
            return []

        if not isinstance(qa_raw, list):
            logger.error("LLM returned %s instead of a JSON array", type(qa_raw).__name__)
            return []

        # Normalize keys: LLM returns "answer", RAGAS expects "ground_truth"
        qa_pairs = []
        for item in qa_raw[:num_pairs]:
            if "question" not in item or "answer" not in item:
                logger.warning("Skipping malformed QA pair (missing keys): %s", item)
                continue
            qa_pairs.append({
                "question": item["question"],
                "ground_truth": item["answer"],
            })

        logger.info("Generated %d QA pairs from document content", len(qa_pairs))
        return qa_pairs

    except openai.APIError as e:
        logger.error("OpenAI API error during QA generation: %s", e, exc_info=True)
        return []
    except Exception as e:
        logger.error("Failed to generate QA pairs: %s", e, exc_info=True)
        return []


def load_documents(
    data_source_configs: List[dict],
    num_qa_pairs: int = 20,
    llm_model: str = "gpt-4o-mini",
) -> Tuple[List[Document], List[dict]]:
    """Load documents from PDF sources and generate QA pairs.

    Args:
        data_source_configs: List of data source config dicts.
        num_qa_pairs: Number of QA pairs to generate from the documents.
        llm_model: LLM model to use for QA pair generation.

    Returns:
        Tuple of (documents, qa_pairs).
    """
    all_documents: List[Document] = []

    enabled_configs = [c for c in data_source_configs if c.get("enabled", False)]
    if not enabled_configs:
        logger.warning("No enabled data sources found")
        return [], []

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
        return [], []

    # Load from each source
    for config, source in sources:
        source_type = config.get("type", "unknown")
        try:
            docs = source.load()
            logger.info("Loaded %d documents from %s", len(docs), source_type)
            all_documents.extend(docs)
        except Exception as e:
            logger.error("Failed to load from %s: %s — continuing", source_type, e)

    # Deduplicate by content hash
    seen_hashes: Set[str] = set()
    unique_documents: List[Document] = []
    for doc in all_documents:
        h = _content_hash(doc.page_content)
        if h not in seen_hashes:
            seen_hashes.add(h)
            unique_documents.append(doc)

    deduped = len(all_documents) - len(unique_documents)
    if deduped > 0:
        logger.info("Deduplicated %d documents (removed %d duplicates)", len(unique_documents), deduped)

    logger.info("Total: %d documents from %d sources", len(unique_documents), len(sources))

    # Generate QA pairs from the loaded documents
    qa_pairs = _generate_qa_pairs(unique_documents, num_qa_pairs, llm_model)

    logger.info("Total: %d documents, %d QA pairs", len(unique_documents), len(qa_pairs))
    return unique_documents, qa_pairs
