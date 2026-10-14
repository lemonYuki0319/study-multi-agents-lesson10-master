"""
Golden Dataset Generator using Ragas TestsetGenerator.

Генерує тестові пари (запит → еталонна відповідь) УКРАЇНСЬКОЮ МОВОЮ
на основі документів з RAG-індексу (index/chunks.pkl).

Використання:
    python goldengen.py

Результат: golden_dataset.json
"""

import json
import pickle
import sys
from collections import Counter
from pathlib import Path
from typing import Any, List

# goldengen.py знаходиться у tests/ — додаємо корінь проєкту до sys.path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from config import Settings

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, SystemMessage
from langchain_core.outputs import ChatResult
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]
from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
from ragas.testset import TestsetGenerator  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Налаштування
# ---------------------------------------------------------------------------

TESTSET_SIZE = 20
# Завжди зберігати поруч із goldengen.py, незалежно від CWD
OUTPUT_PATH = Path(__file__).parent / "golden_dataset.json"

UKRAINIAN_SYSTEM_PROMPT = (
    "ОБОВ'ЯЗКОВО: Генеруй ВИКЛЮЧНО українською мовою. "
    "Всі запитання, відповіді та будь-який текст — ТІЛЬКИ українською мовою. "
    "Жодного англійського тексту у запитаннях та відповідях."
)

EVOLUTION_TO_CATEGORY = {
    "simple": "happy_path",
    "multi_context": "happy_path",
    "reasoning": "edge_case",
    "multi_hop": "edge_case",
    "conditional": "edge_case",
    "negative": "failure_case",
}


# ---------------------------------------------------------------------------
# LLM-обгортка для генерації українською
# ---------------------------------------------------------------------------

class UkrainianChatWrapper(BaseChatModel):
    """
    Обгортка над ChatOpenAI, що автоматично додає системний промпт
    із вимогою генерувати текст українською мовою.
    """

    inner: ChatOpenAI

    @property
    def _llm_type(self) -> str:
        return "ukrainian_chat_wrapper"

    @property
    def _identifying_params(self) -> dict:
        return {"inner_model": self.inner.model_name}

    def _generate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        augmented = [SystemMessage(content=UKRAINIAN_SYSTEM_PROMPT)] + list(messages)
        return self.inner._generate(augmented, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs: Any) -> ChatResult:
        augmented = [SystemMessage(content=UKRAINIAN_SYSTEM_PROMPT)] + list(messages)
        return await self.inner._agenerate(augmented, stop=stop, run_manager=run_manager, **kwargs)


# ---------------------------------------------------------------------------
# Кроки генерації
# ---------------------------------------------------------------------------

def load_chunks() -> list:
    chunks_path = Path(Settings.index_dir) / "chunks.pkl"
    if not chunks_path.exists():
        raise FileNotFoundError(
            f"Індекс не знайдено: {chunks_path}\n"
            "Спочатку запустіть: python ingest.py"
        )
    with open(chunks_path, "rb") as f:
        chunks = pickle.load(f)
    print(f"Завантажено {len(chunks)} chunk'ів з {chunks_path}")
    return chunks


def build_generator() -> TestsetGenerator:
    api_key = Settings.api_key.get_secret_value()
    base_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, api_key=api_key)
    ukrainian_llm = UkrainianChatWrapper(inner=base_llm)
    embeddings = OpenAIEmbeddings(model=Settings.embedding_model, api_key=api_key)
    return TestsetGenerator(
        llm=LangchainLLMWrapper(ukrainian_llm),
        embedding_model=LangchainEmbeddingsWrapper(embeddings),
    )


def generate_testset(chunks: list):
    generator = build_generator()
    print(f"Генерую {TESTSET_SIZE} тестових прикладів (Ukrainian)...")
    testset = generator.generate_with_langchain_docs(documents=chunks, testset_size=TESTSET_SIZE)
    return testset.to_pandas()


def build_golden_dataset(df) -> list[dict]:
    golden = []
    for _, row in df.iterrows():
        question = str(row.get("user_input", "")).strip()
        reference = str(row.get("reference", "")).strip()
        if not question or not reference:
            continue
        et = str(row.get("evolution_type", "simple")).lower()
        category = next((v for k, v in EVOLUTION_TO_CATEGORY.items() if k in et), "happy_path")
        golden.append({"input": question, "expected_output": reference, "category": category})
    return golden


def balance_categories(golden: list[dict]) -> list[dict]:
    counts = Counter(ex["category"] for ex in golden)
    missing = {cat for cat in ("happy_path", "edge_case", "failure_case") if counts[cat] == 0}
    if not missing:
        return golden
    print(f"\nУвага: категорії {missing} порожні. Перепризначаємо частину прикладів.")
    happy = [ex for ex in golden if ex["category"] == "happy_path"]
    for cat in sorted(missing):
        if len(happy) > 3:
            ex = happy.pop()
            ex["category"] = cat
            print(f"  Перепризначено: '{ex['input'][:60]}…' → {cat}")
    return golden


def save(golden: list[dict]) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(golden, f, ensure_ascii=False, indent=2)
    print(f"\nЗбережено {len(golden)} прикладів → {OUTPUT_PATH}")


def print_summary(golden: list[dict]) -> None:
    counts = Counter(ex["category"] for ex in golden)
    print("\n=== Golden Dataset — підсумок ===")
    print(f"Всього прикладів: {len(golden)}")
    for cat in ("happy_path", "edge_case", "failure_case"):
        print(f"  {cat}: {counts[cat]}")
    print("\nПерші 3 приклади (preview):")
    for ex in golden[:3]:
        print(f"\n  [{ex['category']}]")
        print(f"  Запит:     {ex['input'][:90]}")
        print(f"  Відповідь: {ex['expected_output'][:90]}")
    print(
        "\n⚠️  ОБОВ'ЯЗКОВО зробіть manual review файлу tests/golden_dataset.json:\n"
        "   - Перевірте що всі запитання й відповіді написані УКРАЇНСЬКОЮ\n"
        "   - Виправте або видаліть неякісні приклади\n"
        "   - Уточніть категорії (happy_path / edge_case / failure_case)\n"
        "   - Мета: 10–15 якісних прикладів після очищення"
    )


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=== Ragas Golden Dataset Generator (Ukrainian) ===\n")

    chunks = load_chunks()
    df = generate_testset(chunks)
    print(f"Ragas згенерував {len(df)} рядків")

    golden = build_golden_dataset(df)
    golden = balance_categories(golden)
    save(golden)
    print_summary(golden)
