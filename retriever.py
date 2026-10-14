"""
Hybrid retrieval module.

Combines semantic search (vector DB) + BM25 (lexical) + cross-encoder reranking.
"""
import pickle
from pathlib import Path

from langchain_community.retrievers import BM25Retriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_classic.retrievers import EnsembleRetriever
from sentence_transformers import CrossEncoder

from config import Settings

_retriever = None
_cross_encoder = None


def get_retriever():
    # 1. Load vector store from disk (config.index_dir)
    # 2. Create semantic retriever from vector store
    # 3. Load chunks and create BM25 retriever
    # 4. Combine into ensemble retriever (semantic + BM25)
    # 5. Add cross-encoder reranker on top
    # 6. Return the final retriever    retriever = _load_retriever()
    global _retriever
    if _retriever is not None:
        return _retriever

    index_dir = Path(Settings.index_dir)
    if not index_dir.exists():
        raise FileNotFoundError(
            f"Index not found at '{index_dir}/'. Run 'python ingest.py' first."
        )

    embeddings = OpenAIEmbeddings(
        model=Settings.embedding_model,
        openai_api_key=Settings.api_key.get_secret_value(),
    )

    vector_store = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    semantic_retriever = vector_store.as_retriever(
        search_kwargs={"k": Settings.retrieval_top_k}
    )

    chunks_path = index_dir / "chunks.pkl"
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)

    bm25_retriever = BM25Retriever.from_documents(chunks)
    bm25_retriever.k = Settings.retrieval_top_k

    _retriever = EnsembleRetriever(
        retrievers=[semantic_retriever, bm25_retriever],
        weights=[0.5, 0.5],
    )
    return _retriever


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder("BAAI/bge-reranker-base")
    return _cross_encoder


def retrieve_and_rerank(query: str) -> list:
    docs = get_retriever().invoke(query)

    if not docs:
        return []

    cross_encoder = get_cross_encoder()
    pairs = [(query, doc.page_content) for doc in docs]
    scores = cross_encoder.predict(pairs)

    ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)
    return [doc for _, doc in ranked[: Settings.rerank_top_n]]
