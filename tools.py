import os
import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool

from config import Settings, SEARCH_ENGINE
from retriever import retrieve_and_rerank


@tool
def web_search(query: str) -> list[dict] | str:
    """Performs web search and returns list of results with title, href, and body (snippet).
    Use for finding current information not covered by the knowledge base."""
    try:
        text = DDGS().text(query, max_results=Settings.max_search_results, backend=SEARCH_ENGINE)
        if text is None:
            return "web_search error: failed to fetch search result"
        for item in text:
            if "body" in item:
                item["body"] = item["body"][:Settings.max_url_content_length]
        return text
    except Exception as e:
        return f"web_search error: {type(e).__name__}: {e}"


@tool
def read_url(url: str) -> str:
    """Loads and returns the text content of a web page at the given URL."""
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return "read_url error: failed to fetch URL"
        text = trafilatura.extract(downloaded)
        if text is None:
            return "read_url error: failed to extract text from page"
        return text[:Settings.max_url_content_length]
    except Exception as e:
        return f"read_url error: {type(e).__name__}: {e}"


@tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base (ingested PDF/TXT documents) using hybrid semantic + BM25
    retrieval with cross-encoder reranking. Use this FIRST for questions that may be covered
    by the loaded documents. Returns top relevant text passages with source and page number."""
    try:
        docs = retrieve_and_rerank(query)
        if not docs:
            return "knowledge_search: no relevant documents found"
        results = []
        for i, doc in enumerate(docs, 1):
            source = os.path.basename(doc.metadata.get("source", "unknown"))
            page = doc.metadata.get("page", "?")
            results.append(f"[{i}] Source: {source}, Page: {page}\n{doc.page_content}")
        return "\n\n".join(results)
    except FileNotFoundError as e:
        return f"knowledge_search error: {e}"
    except Exception as e:
        return f"knowledge_search error: {type(e).__name__}: {e}"


@tool
def save_report(filename: str, content: str) -> str:
    """Saves a markdown report to a file in the output directory.
    Returns the full path to the created file."""
    os.makedirs(Settings.output_dir, exist_ok=True)
    path = os.path.join(Settings.output_dir, filename)
    with open(path, "wt", encoding="utf-8") as f:
        f.write(content)
    return path
