"""
Knowledge ingestion pipeline.

Loads documents from data/ directory, splits into chunks,
generates embeddings, and saves the index to disk.

Usage: python ingest.py
"""
import pickle
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import Settings


def ingest():
    # 1. Load documents from config.data_dir (PDF, TXT, MD)
    # 2. Split into chunks using TextSplitter
    # 3. Generate embeddings
    # 4. Build vector store (FAISS, Qdrant, Chroma, etc.)
    # 5. Save index to config.index_dir
    # 6. Save chunks for BM25 retriever (pickle or JSON)
    data_dir = Path(Settings.data_dir)
    index_dir = Path(Settings.index_dir)
    index_dir.mkdir(exist_ok=True)

    docs = []
    for pdf_file in sorted(data_dir.glob("*.pdf")):
        print(f"Loading {pdf_file.name}...")
        loader = PyPDFLoader(str(pdf_file))
        docs.extend(loader.load())

    for txt_file in sorted(data_dir.glob("*.txt")):
        print(f"Loading {txt_file.name}...")
        loader = TextLoader(str(txt_file), encoding="utf-8")
        docs.extend(loader.load())

    for md_file in sorted(data_dir.glob("*.md")):
        print(f"Loading {md_file.name}...")
        loader = TextLoader(str(md_file), encoding="utf-8")
        docs.extend(loader.load())

    if not docs:
        print(f"No documents found in {data_dir}/")
        return

    print(f"Loaded {len(docs)} pages total")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Settings.chunk_size,
        chunk_overlap=Settings.chunk_overlap,
    )
    chunks = splitter.split_documents(docs)
    print(f"Created {len(chunks)} chunks")

    embeddings = OpenAIEmbeddings(
        model=Settings.embedding_model,
        openai_api_key=Settings.api_key.get_secret_value(),
    )

    print("Generating embeddings and building FAISS index...")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(index_dir))
    print(f"FAISS index saved to {index_dir}/")

    chunks_path = index_dir / "chunks.pkl"
    with open(chunks_path, "wb") as f:
        pickle.dump(chunks, f)
    print(f"Saved {len(chunks)} chunks for BM25 to {chunks_path}")


if __name__ == "__main__":
    ingest()
