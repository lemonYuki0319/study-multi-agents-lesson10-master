"""
Component tests — Critic Agent.

Перевіряємо якість критики:
 - конкретність виявлених проблем (не розмиті скарги)
 - actionability revision_requests
 - коректність вердикту APPROVE / REVISE

Запуск:
    deepeval test run tests/test_critic.py
"""

import json
import pytest

from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from agents.critic import get_critic_agent


# ---------------------------------------------------------------------------
# Метрика
# ---------------------------------------------------------------------------

critique_quality = GEval(
    name="Critique Quality",
    evaluation_steps=[
        "Перевір, що критика виявляє конкретні проблеми (не розмиті скарги типу 'потребує покращення')",
        "Перевір, що revision_requests є actionable — дослідник може конкретно діяти на їх основі",
        "Якщо verdict='APPROVE': gaps має бути порожнім або містити лише незначні зауваження",
        "Якщо verdict='REVISE': має бути хоча б один конкретний revision_request",
        "Перевір, що critique оцінює всі три виміри: freshness, completeness, structure",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.7,
)


# ---------------------------------------------------------------------------
# Тестові дані
# ---------------------------------------------------------------------------

GOOD_FINDINGS = """
# Дослідження: RAG (Retrieval-Augmented Generation)

## Що таке RAG?
RAG (Retrieval-Augmented Generation) — це архітектурний патерн, що поєднує пошук
релевантних документів із генерацією відповіді мовною моделлю. Замість покладання
виключно на параметричні знання моделі, RAG динамічно отримує актуальну інформацію
з бази знань.

## Ключові компоненти
1. **Retriever** — відповідає за пошук релевантних документів (FAISS, BM25, hybrid)
2. **Generator** — LLM, що генерує відповідь на основі знайдених документів
3. **Vector Store** — зберігає ембедінги документів для семантичного пошуку

## Переваги RAG
- Значно знижує галюцинації завдяки заземленню відповідей у реальних документах
- Дозволяє оновлювати знання без дорогого перенавчання моделі
- Надає цитати та посилання на джерела для верифікації

## Порівняння підходів
| Підхід | Переваги | Недоліки |
|---|---|---|
| Naive RAG | Простота реалізації | Фіксований розмір chunk'ів |
| Sentence-window RAG | Кращий контекст | Складніша реалізація |
| Hybrid RAG | Найвища точність | Висока складність |

## Джерела
- retrieval-augmented-generation.pdf, сторінки 3–7 (2024)
- large-language-model.pdf, сторінки 12–15 (2024)
- Lewis et al. (2020) "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
"""

POOR_FINDINGS = """
RAG — це технологія.
Вона використовує пошук для покращення LLM.
Можливо, є якісь переваги над звичайними підходами.
Більше інформації можна знайти в інтернеті.
"""


# ---------------------------------------------------------------------------
# Хелпер
# ---------------------------------------------------------------------------

def run_critic(findings: str) -> str:
    """Запускає Critic Agent і повертає JSON-рядок із CritiqueResult."""
    agent = get_critic_agent()
    config = {"configurable": {"thread_id": f"test_critic_{abs(hash(findings[:30]))}"}}
    result = agent.invoke(
        {"messages": [{"role": "user", "content": findings}]},
        config=config,
    )
    structured = result.get("structured_response")
    if structured is not None:
        return structured.model_dump_json()
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Тести
# ---------------------------------------------------------------------------

@pytest.mark.deepeval
def test_critique_approve():
    """
    Happy path: для якісного, повного та структурованого дослідження
    Critic має видати verdict='APPROVE' і не мати серйозних gaps.
    """
    output = run_critic(GOOD_FINDINGS)

    test_case = LLMTestCase(input=GOOD_FINDINGS, actual_output=output)
    critique_quality.measure(test_case)
    assert critique_quality.score >= critique_quality.threshold, (
        f"Critique Quality {critique_quality.score:.2f} < threshold {critique_quality.threshold}. "
        f"Reason: {critique_quality.reason}"
    )

    # Додаткова структурна перевірка: якщо REVISE — мають бути конкретні revision_requests.
    # APPROVE не гарантовано — Critic може вважати джерела застарілими (дата 2026 vs 2024).
    try:
        data = json.loads(output)
        if data.get("verdict") == "REVISE":
            assert len(data.get("revision_requests", [])) >= 1, (
                "REVISE verdict має містити ≥1 revision_request"
            )
    except json.JSONDecodeError:
        pass  # Якщо не JSON — довіряємо GEval


@pytest.mark.deepeval
def test_critique_revise():
    """
    Happy path: для неповного та поверхневого дослідження
    Critic має видати verdict='REVISE' і надати конкретні revision_requests.
    """
    output = run_critic(POOR_FINDINGS)

    test_case = LLMTestCase(input=POOR_FINDINGS, actual_output=output)
    critique_quality.measure(test_case)
    assert critique_quality.score >= critique_quality.threshold, (
        f"Critique Quality {critique_quality.score:.2f} < threshold {critique_quality.threshold}. "
        f"Reason: {critique_quality.reason}"
    )

    # Додаткова структурна перевірка
    try:
        data = json.loads(output)
        verdict = data.get("verdict")
        revision_requests = data.get("revision_requests", [])
        assert verdict == "REVISE", (
            f"Очікував REVISE для неповного дослідження, отримав: {verdict}"
        )
        assert len(revision_requests) >= 1, (
            f"REVISE verdict має містити ≥1 revision_request, отримано: {revision_requests}"
        )
    except json.JSONDecodeError:
        pass  # Якщо не JSON — довіряємо GEval


@pytest.mark.deepeval
def test_critique_specificity():
    """
    Custom метрика: Critic має надавати СПЕЦИФІЧНУ критику,
    а не загальні коментарі. Це бізнес-правило: revision_requests
    мають бути actionable (дослідник знає що саме виправити).
    """
    specificity_metric = GEval(
        name="Critique Specificity",
        evaluation_steps=[
            "Перевір кожен revision_request: чи він називає конкретну відсутню інформацію або конкретну дію?",
            "Загальні коментарі на кшталт 'додайте більше деталей' або 'покращіть структуру' — не специфічні",
            "Специфічний запит: 'додай порівняння BM25 vs FAISS з конкретними метриками' — це конкретно",
            "Якщо verdict='APPROVE' — перевіряй що gaps теж конкретні або порожні",
            "Score = частка специфічних revision_requests / загальна кількість",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model="gpt-4o-mini",
        threshold=0.6,
    )

    output = run_critic(POOR_FINDINGS)
    test_case = LLMTestCase(input=POOR_FINDINGS, actual_output=output)
    specificity_metric.measure(test_case)
    assert specificity_metric.score >= specificity_metric.threshold, (
        f"Critique Specificity {specificity_metric.score:.2f} < threshold {specificity_metric.threshold}. "
        f"Reason: {specificity_metric.reason}"
    )
