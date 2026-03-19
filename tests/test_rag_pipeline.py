"""Tests for rag_pipeline module."""

from unittest.mock import MagicMock, patch

import pytest
from langchain_core.documents import Document

from src.rag_pipeline import chunk_documents, get_embedding_model, run_pipeline


class TestChunkDocuments:
    def test_basic_chunking(self):
        docs = [Document(page_content="a " * 500, metadata={"source": "test"})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk.page_content) <= 110  # allow slight overshoot

    def test_small_document_no_split(self):
        docs = [Document(page_content="short text", metadata={"source": "test"})]
        chunks = chunk_documents(docs, chunk_size=512, chunk_overlap=50)
        assert len(chunks) == 1
        assert chunks[0].page_content == "short text"

    def test_chunk_overlap(self):
        # Create a document large enough to split
        text = " ".join([f"word{i}" for i in range(200)])
        docs = [Document(page_content=text, metadata={"source": "test"})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_empty_documents(self):
        chunks = chunk_documents([], chunk_size=512, chunk_overlap=50)
        assert chunks == []

    def test_metadata_preserved(self):
        docs = [Document(page_content="a " * 500, metadata={"source": "test", "key": "val"})]
        chunks = chunk_documents(docs, chunk_size=100, chunk_overlap=10)
        for chunk in chunks:
            assert chunk.metadata["source"] == "test"
            assert chunk.metadata["key"] == "val"


class TestGetEmbeddingModel:
    @patch("src.rag_pipeline.OpenAIEmbeddings", create=True)
    def test_openai_embedding(self, mock_cls):
        with patch.dict("sys.modules", {"langchain_openai": MagicMock()}):
            from langchain_openai import OpenAIEmbeddings
            get_embedding_model("text-embedding-ada-002")

    def test_unknown_model_raises(self):
        with pytest.raises(ValueError, match="Unknown embedding model"):
            get_embedding_model("nonexistent-model")


class TestRunPipeline:
    @patch("src.rag_pipeline.get_llm")
    @patch("src.rag_pipeline.build_vector_store")
    def test_run_pipeline_returns_results(self, mock_build_vs, mock_get_llm):
        # Mock vector store and retriever
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [
            Document(page_content="context chunk 1"),
            Document(page_content="context chunk 2"),
        ]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_build_vs.return_value = mock_vs

        # Mock LLM
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="generated answer")
        mock_get_llm.return_value = mock_llm

        documents = [
            Document(page_content="Some document text " * 50, metadata={"source": "test"}),
        ]
        qa_pairs = [
            {"question": "What is this?", "ground_truth": "test answer"},
        ]
        config = {
            "chunk_size": 512,
            "chunk_overlap": 50,
            "top_k": 5,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_model": "gpt-4o-mini",
        }

        results, chunk_count = run_pipeline(documents, qa_pairs, config)

        assert len(results) == 1
        assert results[0]["question"] == "What is this?"
        assert results[0]["answer"] == "generated answer"
        assert results[0]["ground_truth"] == "test answer"
        assert len(results[0]["contexts"]) == 2
        assert chunk_count > 0

    @patch("src.rag_pipeline.get_llm")
    @patch("src.rag_pipeline.build_vector_store")
    def test_run_pipeline_empty_qa_pairs(self, mock_build_vs, mock_get_llm):
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = MagicMock()
        mock_build_vs.return_value = mock_vs

        documents = [Document(page_content="text " * 100, metadata={"source": "test"})]
        results, chunk_count = run_pipeline(documents, [], {"chunk_size": 512})
        assert results == []
        assert chunk_count == 0

    @patch("src.rag_pipeline.get_llm")
    @patch("src.rag_pipeline.build_vector_store")
    def test_run_pipeline_multiple_qa_pairs(self, mock_build_vs, mock_get_llm):
        mock_retriever = MagicMock()
        mock_retriever.invoke.return_value = [Document(page_content="ctx")]
        mock_vs = MagicMock()
        mock_vs.as_retriever.return_value = mock_retriever
        mock_build_vs.return_value = mock_vs

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content="answer")
        mock_get_llm.return_value = mock_llm

        documents = [Document(page_content="text " * 100, metadata={"source": "test"})]
        qa_pairs = [
            {"question": "Q1", "ground_truth": "A1"},
            {"question": "Q2", "ground_truth": "A2"},
            {"question": "Q3", "ground_truth": "A3"},
        ]

        results, chunk_count = run_pipeline(documents, qa_pairs, {"chunk_size": 512})
        assert len(results) == 3
        assert [r["question"] for r in results] == ["Q1", "Q2", "Q3"]

    def test_run_pipeline_empty_documents(self):
        results, chunk_count = run_pipeline([], [{"question": "Q"}], {"chunk_size": 512})
        assert results == []
        assert chunk_count == 0
