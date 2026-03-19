"""HuggingFace datasets data source connector.

Loads SQuAD and HotpotQA datasets, extracting both documents and QA pairs
for evaluation.
"""

import logging
from typing import List, Optional

from langchain_core.documents import Document

from src.data_sources import register
from src.data_sources.base import BaseDataSource

logger = logging.getLogger(__name__)


@register("huggingface")
class HuggingFaceDataSource(BaseDataSource):

    def validate_config(self) -> bool:
        dataset_name = self.config.get("dataset_name")
        if not dataset_name:
            raise ValueError(
                "huggingface config missing required 'dataset_name' field. "
                "Add 'dataset_name: <name>' (e.g., dataset_name: squad) to the huggingface [[data_sources]] block."
            )
        split = self.config.get("split")
        if not split:
            raise ValueError(
                "huggingface config missing required 'split' field. "
                "Add 'split: <split>' (e.g., split: train, split: validation) to the huggingface [[data_sources]] block."
            )
        return True

    def health_check(self) -> bool:
        self.validate_config()
        from datasets import load_dataset_builder

        dataset_name = self.config["dataset_name"]
        try:
            load_dataset_builder(dataset_name)
        except Exception as e:
            raise ConnectionError(f"Cannot access HuggingFace dataset '{dataset_name}': {e}")
        logger.info("health_check passed: HuggingFace dataset %s accessible", dataset_name)
        return True

    def load(self) -> List[Document]:
        """Load documents only (without QA pairs)."""
        docs, _ = self.load_with_qa()
        return docs

    def load_with_qa(self) -> tuple[List[Document], List[dict]]:
        """Load documents and QA pairs from the dataset.

        Returns:
            Tuple of (documents, qa_pairs).
        """
        self.validate_config()
        from datasets import load_dataset

        dataset_name = self.config["dataset_name"]
        split = self.config["split"]
        sample_size = self.config.get("sample_size")

        ds = load_dataset(dataset_name, split=split)

        if sample_size and sample_size < len(ds):
            ds = ds.select(range(sample_size))

        if "squad" in dataset_name.lower():
            return self._process_squad(ds, dataset_name, split)
        elif "hotpot" in dataset_name.lower():
            return self._process_hotpotqa(ds, dataset_name, split)
        else:
            logger.warning("Unknown dataset format '%s', attempting SQuAD-style extraction", dataset_name)
            return self._process_squad(ds, dataset_name, split)

    def _process_squad(self, ds, dataset_name: str, split: str) -> tuple[List[Document], List[dict]]:
        documents: List[Document] = []
        qa_pairs: List[dict] = []
        seen_contexts: set[str] = set()

        for i, example in enumerate(ds):
            context = example.get("context", "")
            question = example.get("question", "")
            answers = example.get("answers", {})

            # Extract ground truth answer
            if isinstance(answers, dict):
                answer_texts = answers.get("text", [])
                ground_truth = answer_texts[0] if answer_texts else ""
            elif isinstance(answers, list):
                ground_truth = answers[0] if answers else ""
            else:
                ground_truth = str(answers)

            # Add context as document (deduplicate)
            if context and context not in seen_contexts:
                seen_contexts.add(context)
                documents.append(Document(
                    page_content=context,
                    metadata={
                        "source": f"huggingface://{dataset_name}",
                        "source_type": "huggingface",
                        "dataset_name": dataset_name,
                        "split": split,
                        "index": i,
                    },
                ))

            # Add QA pair
            if question:
                qa_pairs.append({
                    "question": question,
                    "ground_truth": ground_truth,
                })

        logger.info(
            "Loaded %d documents and %d QA pairs from %s/%s",
            len(documents), len(qa_pairs), dataset_name, split,
        )
        return documents, qa_pairs

    def _process_hotpotqa(self, ds, dataset_name: str, split: str) -> tuple[List[Document], List[dict]]:
        documents: List[Document] = []
        qa_pairs: List[dict] = []
        seen_contexts: set[str] = set()

        for i, example in enumerate(ds):
            question = example.get("question", "")
            answer = example.get("answer", "")

            # Supporting facts as documents
            sentences = example.get("context", {})
            if isinstance(sentences, dict):
                titles = sentences.get("title", [])
                sent_lists = sentences.get("sentences", [])
                for title, sents in zip(titles, sent_lists):
                    context = " ".join(sents) if isinstance(sents, list) else str(sents)
                    if context and context not in seen_contexts:
                        seen_contexts.add(context)
                        documents.append(Document(
                            page_content=context,
                            metadata={
                                "source": f"huggingface://{dataset_name}",
                                "source_type": "huggingface",
                                "dataset_name": dataset_name,
                                "split": split,
                                "index": i,
                                "title": title,
                            },
                        ))

            if question:
                qa_pairs.append({
                    "question": question,
                    "ground_truth": answer,
                })

        logger.info(
            "Loaded %d documents and %d QA pairs from %s/%s",
            len(documents), len(qa_pairs), dataset_name, split,
        )
        return documents, qa_pairs
