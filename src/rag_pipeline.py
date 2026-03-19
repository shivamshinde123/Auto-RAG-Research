"""RAG pipeline with configurable chunking, embeddings, and retrieval."""

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[Document]:
    """Split documents into chunks using RecursiveCharacterTextSplitter."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(documents)
    logger.info(
        "Chunked %d documents into %d chunks (size=%d, overlap=%d)",
        len(documents), len(chunks), chunk_size, chunk_overlap,
    )
    return chunks


def get_embedding_model(model_name: str):
    """Return a LangChain embedding model instance by name."""
    if model_name == "text-embedding-ada-002":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-ada-002")
    elif model_name in ("BGE-large", "bge-large"):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="BAAI/bge-large-en-v1.5")
    elif model_name in ("all-MiniLM-L6-v2", "all-minilm-l6-v2"):
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    else:
        raise ValueError(f"Unknown embedding model: {model_name}")


def build_vector_store(chunks: List[Document], embedding_model_name: str):
    """Build a FAISS vector store from document chunks."""
    from langchain_community.vectorstores import FAISS

    embeddings = get_embedding_model(embedding_model_name)
    vector_store = FAISS.from_documents(chunks, embeddings)
    logger.info(
        "Built FAISS vector store with %d chunks using %s",
        len(chunks), embedding_model_name,
    )
    return vector_store


def get_llm(model_name: str):
    """Return a LangChain LLM instance by name."""
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model=model_name, temperature=0)


def run_pipeline(
    documents: List[Document],
    qa_pairs: List[dict],
    config: dict,
) -> List[dict]:
    """Run the full RAG pipeline: chunk, embed, retrieve, generate.

    Args:
        documents: List of LangChain Documents to index.
        qa_pairs: List of dicts with 'question' and 'ground_truth' keys.
        config: Dict with chunk_size, chunk_overlap, top_k, embedding_model, llm_model.

    Returns:
        List of dicts ready for RAGAS evaluation, each containing:
        question, answer, contexts, ground_truth.
    """
    chunk_size = config.get("chunk_size", 512)
    chunk_overlap = config.get("chunk_overlap", 50)
    top_k = config.get("top_k", 5)
    embedding_model = config.get("embedding_model", "all-MiniLM-L6-v2")
    llm_model = config.get("llm_model", "gpt-4o-mini")

    # Step 1: Chunk documents
    chunks = chunk_documents(documents, chunk_size, chunk_overlap)
    if not chunks:
        logger.warning("No chunks produced from %d documents", len(documents))
        return []

    # Step 2: Build vector store
    vector_store = build_vector_store(chunks, embedding_model)
    retriever = vector_store.as_retriever(search_kwargs={"k": top_k})

    # Step 3: Set up LLM
    llm = get_llm(llm_model)

    # Step 4: For each QA pair, retrieve and generate
    results = []
    for i, qa in enumerate(qa_pairs):
        question = qa["question"]
        ground_truth = qa.get("ground_truth", "")

        # Retrieve relevant chunks
        retrieved_docs = retriever.invoke(question)
        contexts = [doc.page_content for doc in retrieved_docs]

        # Generate answer
        context_text = "\n\n".join(contexts)
        prompt = (
            f"Answer the question based on the following context.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {question}\n\n"
            f"Answer:"
        )
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)

        results.append({
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
        })

        if (i + 1) % 10 == 0:
            logger.info("Processed %d/%d QA pairs", i + 1, len(qa_pairs))

    logger.info(
        "Pipeline complete: %d results (model=%s, embedding=%s, top_k=%d)",
        len(results), llm_model, embedding_model, top_k,
    )
    return results
