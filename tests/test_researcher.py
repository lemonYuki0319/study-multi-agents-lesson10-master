"""
Component tests — Research Agent.

Перевіряємо Groundedness відповіді:
кожне фактичне твердження у відповіді має бути підкріплено
контекстом із knowledge base (retrieval context).

Запуск:
    deepeval test run tests/test_researcher.py
"""

import pytest

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.research import get_research_agent
from retriever import retrieve_and_rerank


# ---------------------------------------------------------------------------
# Метрика
# ---------------------------------------------------------------------------

groundedness = GEval(
    name="Groundedness",
    evaluation_steps=[
        "Витягни кожне фактичне твердження з 'actual output'",
        "Для кожного твердження перевір, чи воно безпосередньо підтримується 'retrieval context'",
        "Твердження, яких немає в retrieval context, вважаються непідкріпленими — навіть якщо вони загально правильні",
        "Score = кількість підкріплених тверджень / загальна кількість тверджень",
        "Якщо фактичних тверджень немає взагалі — score 0.5",
    ],
    evaluation_params=[
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    model="gpt-4o-mini",
    threshold=0.65,  # Baseline: 0.65 (score 0.69 зафіксовано при першому запуску)
)


# ---------------------------------------------------------------------------
# Хелпери
# ---------------------------------------------------------------------------

def run_researcher(query: str) -> str:
    """Запускає Research Agent і повертає текст відповіді."""
    agent = get_research_agent()
    config = {"configurable": {"thread_id": f"test_researcher_{abs(hash(query))}"}}
    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)
    return result["messages"][-1].content


def get_context(query: str) -> list[str]:
    """Отримує retrieval context із knowledge base для даного запиту."""
    docs = retrieve_and_rerank(query)
    return [doc.page_content for doc in docs]


# ---------------------------------------------------------------------------
# Тести
# ---------------------------------------------------------------------------

@pytest.mark.deepeval
def test_research_grounded_happy_path():
    """
    Happy path: Researcher має дати відповідь про RAG,
    заземлену на документах з knowledge base.
    """
    query = "Поясни механізм роботи RAG (Retrieval-Augmented Generation): компоненти та переваги"
    output = run_researcher(query)
    context = get_context(query)

    assert context, "Knowledge base повернула порожній контекст — перевір індекс"

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        retrieval_context=context,
    )
    groundedness.measure(test_case)
    assert groundedness.score >= groundedness.threshold, (
        f"Groundedness {groundedness.score:.2f} < threshold {groundedness.threshold}. "
        f"Reason: {groundedness.reason}"
    )


@pytest.mark.deepeval
def test_research_grounded_llm_topic():
    """
    Happy path: Researcher відповідає на запит про LLM-архітектуру —
    відповідь має спиратися на завантажені документи.
    """
    query = "Що таке transformer-архітектура та яку роль вона відіграє у великих мовних моделях (LLM)?"
    output = run_researcher(query)
    context = get_context(query)

    assert context, "Knowledge base повернула порожній контекст"

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        retrieval_context=context,
    )
    groundedness.measure(test_case)
    assert groundedness.score >= groundedness.threshold, (
        f"Groundedness {groundedness.score:.2f} < threshold {groundedness.threshold}. "
        f"Reason: {groundedness.reason}"
    )


@pytest.mark.deepeval
def test_research_grounded_edge_case():
    """
    Edge case: запит охоплює кілька документів одночасно (LangChain + RAG),
    відповідь має бути заземлена в retrieval context попри складність.
    Threshold 0.5 — базовий рівень для складних мультидокументних запитів.
    """
    edge_groundedness = GEval(
        name="Groundedness (edge case)",
        evaluation_steps=[
            "Витягни кожне фактичне твердження з 'actual output'",
            "Для кожного твердження перевір наявність підтримки в 'retrieval context'",
            "Score = підкріплені / загальні (або 0.5 якщо тверджень немає)",
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model="gpt-4o-mini",
        threshold=0.5,  # Знижений поріг для складного мультидокументного запиту
    )

    # Запит охоплює декілька PDF одночасно — складний, але документи це покривають
    query = "Як LangChain використовує концепцію chain-of-thought та яку роль це відіграє у RAG-пайплайнах?"
    output = run_researcher(query)
    context = get_context(query)

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        retrieval_context=context if context else ["Інформації з даної теми не знайдено"],
    )
    edge_groundedness.measure(test_case)
    assert edge_groundedness.score >= edge_groundedness.threshold, (
        f"Groundedness (edge) {edge_groundedness.score:.2f} < threshold {edge_groundedness.threshold}. "
        f"Reason: {edge_groundedness.reason}"
    )
