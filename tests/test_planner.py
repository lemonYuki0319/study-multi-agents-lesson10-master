"""
Component tests — Planner Agent.

Перевіряємо якість ResearchPlan:
 - наявність конкретних (не розмитих) пошукових запитів
 - вказані джерела (knowledge_base / web)
 - зрозумілий output_format

Запуск:
    deepeval test run tests/test_planner.py
"""

import json
import pytest

from deepeval import evaluate
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.planner import get_planner_agent


# ---------------------------------------------------------------------------
# Метрика
# ---------------------------------------------------------------------------

plan_quality = GEval(
    name="Plan Quality",
    evaluation_steps=[
        "Перевір, що plan містить конкретні пошукові запити (не розмиті фрази на кшталт 'дослідити тему')",
        "Перевір, що sources_to_check містить хоча б одне з: 'knowledge_base', 'web'",
        "Перевір, що search_queries покривають різні аспекти запиту (мінімум 2 різних запити)",
        "Перевір, що output_format описує чітку структуру фінального звіту",
        "Запитання та запити можуть бути як українською, так і англійською — це прийнятно",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


# ---------------------------------------------------------------------------
# Хелпер
# ---------------------------------------------------------------------------

def run_planner(query: str) -> str:
    """Запускає Planner і повертає JSON-рядок із ResearchPlan."""
    agent = get_planner_agent()
    config = {"configurable": {"thread_id": f"test_planner_{abs(hash(query))}"}}
    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)

    structured = result.get("structured_response")
    if structured is not None:
        return structured.model_dump_json()
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Тести
# ---------------------------------------------------------------------------

@pytest.mark.deepeval
def test_plan_quality():
    """
    Happy path: для типового RAG-запиту Planner має повернути якісний план
    із конкретними пошуковими запитами та вказаними джерелами.
    """
    query = (
        "Порівняй Naive RAG та sentence-window retrieval. "
        "Які переваги та недоліки кожного підходу?"
    )
    output = run_planner(query)

    test_case = LLMTestCase(input=query, actual_output=output)
    plan_quality.measure(test_case)
    assert plan_quality.score >= plan_quality.threshold, (
        f"Plan Quality score {plan_quality.score:.2f} < threshold {plan_quality.threshold}. "
        f"Reason: {plan_quality.reason}"
    )


@pytest.mark.deepeval
def test_plan_has_specific_queries():
    """
    Planner для вузького технічного запиту має сформувати
    мінімум 2 конкретні пошукові запити (не порожній список).
    """
    query = "Що таке cross-encoder reranking у контексті RAG-систем?"
    output = run_planner(query)

    # Структурна перевірка: мінімум 2 search_queries
    try:
        data = json.loads(output)
        search_queries = data.get("search_queries", [])
        assert len(search_queries) >= 2, (
            f"Очікувано ≥2 search_queries, отримано {len(search_queries)}: {search_queries}"
        )
    except json.JSONDecodeError:
        # Якщо plan повернуто не як JSON — перевіряємо через GEval
        pass

    test_case = LLMTestCase(input=query, actual_output=output)
    plan_quality.measure(test_case)
    assert plan_quality.score >= plan_quality.threshold, (
        f"Plan Quality score {plan_quality.score:.2f} < threshold {plan_quality.threshold}. "
        f"Reason: {plan_quality.reason}"
    )


@pytest.mark.deepeval
def test_plan_for_multilingual_query():
    """
    Edge case: Planner отримує запит мішаною мовою (укр + англ терміни) —
    має коректно декомпозувати у план.
    """
    query = "Як працює hybrid retrieval (BM25 + semantic search) і навіщо cross-encoder reranking?"
    output = run_planner(query)

    test_case = LLMTestCase(input=query, actual_output=output)
    plan_quality.measure(test_case)
    assert plan_quality.score >= plan_quality.threshold, (
        f"Plan Quality score {plan_quality.score:.2f} < threshold {plan_quality.threshold}. "
        f"Reason: {plan_quality.reason}"
    )
