"""
End-to-end tests — повний pipeline Supervisor → Planner → Researcher → Critic.

Запускається на golden dataset (tests/golden_dataset.json).
Метрики: AnswerRelevancyMetric + GEval("Correctness").

Запуск:
    deepeval test run tests/test_e2e.py

Примітка: e2e тести дорогі за часом та API-викликами.
Для швидкої перевірки запускай окремі компонентні тести.
"""

import json
import sys
from pathlib import Path

import pytest

from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

sys.path.insert(0, str(Path(__file__).parent.parent))


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4o-mini",
)

correctness = GEval(
    name="Correctness",
    evaluation_steps=[
        "Перевір, чи факти в 'actual output' суперечать фактам з 'expected output'",
        "Штрафуй за пропущення критичних деталей, що є в 'expected output'",
        "Різне формулювання тієї самої думки — допустиме, не штрафуй",
        "Якщо 'actual output' покриває ключові пункти 'expected output' — score high",
    ],
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.EXPECTED_OUTPUT,
    ],
    model="gpt-4o-mini",
    threshold=0.6,
)

# Custom метрика — перевірка наявності посилань на джерела (бізнес-правило)
citation_presence = GEval(
    name="Citation Presence",
    evaluation_steps=[
        "Перевір, чи містить 'actual output' посилання на джерела (назви файлів, URL, автори)",
        "Наявність хоча б одного явного посилання — score >= 0.7",
        "Відсутність будь-яких посилань — score <= 0.3",
        "Часткові посилання (натяки на джерела без деталей) — score 0.5",
    ],
    evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
    model="gpt-4o-mini",
    threshold=0.5,
)


# ---------------------------------------------------------------------------
# Pipeline runner (без HumanInTheLoop middleware)
# ---------------------------------------------------------------------------

def run_full_pipeline(query: str) -> str:
    """
    Запускає повний pipeline через тестовий supervisor
    (без HumanInTheLoopMiddleware, щоб save_report не блокував тест).
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    from langgraph.checkpoint.memory import InMemorySaver
    from config import Settings, SUPERVISOR_PROMPT
    from tools import save_report
    from agents.planner import plan
    from agents.research import research
    from agents.critic import critique

    model = ChatOpenAI(
        model=Settings.model_name,
        api_key=Settings.api_key.get_secret_value(),
    )
    supervisor = create_agent(
        model=model,
        tools=[plan, research, critique, save_report],
        system_prompt=SUPERVISOR_PROMPT,
        checkpointer=InMemorySaver(),
        # Без HumanInTheLoopMiddleware — для автоматизованого тестування
    )

    config = {"configurable": {"thread_id": f"test_e2e_{abs(hash(query))}"}}
    result = supervisor.invoke(
        {"messages": [{"role": "user", "content": query}]},
        config=config,
    )
    return result["messages"][-1].content


# ---------------------------------------------------------------------------
# Тести
# ---------------------------------------------------------------------------

@pytest.mark.deepeval
def test_e2e_happy_path(golden_dataset):
    """
    E2E evaluation на happy_path прикладах із golden dataset.
    Перевіряємо Correctness та Answer Relevancy.
    Прохідний критерій: ≥60% тест-кейсів проходять обидві метрики.
    """
    happy_examples = [ex for ex in golden_dataset if ex["category"] == "happy_path"]
    if not happy_examples:
        pytest.skip("Немає happy_path прикладів у golden dataset")

    # Обмежуємо до 3 прикладів для контролю витрат
    sample = happy_examples[:3]

    results = []
    for example in sample:
        actual = run_full_pipeline(example["input"])
        tc = LLMTestCase(
            input=example["input"],
            actual_output=actual,
            expected_output=example["expected_output"],
        )
        correctness.measure(tc)
        answer_relevancy.measure(tc)
        results.append({
            "input": example["input"][:60],
            "correctness": correctness.score,
            "relevancy": answer_relevancy.score,
            "passed": (
                correctness.score >= correctness.threshold
                and answer_relevancy.score >= answer_relevancy.threshold
            ),
        })

    # Підсумок
    passing = sum(1 for r in results if r["passed"])
    avg_c = sum(r["correctness"] for r in results) / len(results)
    avg_r = sum(r["relevancy"] for r in results) / len(results)

    print(f"\nE2E Happy Path Results ({len(results)} прикладів):")
    print(f"  Correctness:      avg={avg_c:.2f}, min={min(r['correctness'] for r in results):.2f}, "
          f"max={max(r['correctness'] for r in results):.2f}")
    print(f"  Answer Relevancy: avg={avg_r:.2f}, min={min(r['relevancy'] for r in results):.2f}, "
          f"max={max(r['relevancy'] for r in results):.2f}")
    print(f"  Passed: {passing}/{len(results)}")

    assert passing / len(results) >= 0.6, (
        f"Лише {passing}/{len(results)} happy_path прикладів пройшли обидві метрики. "
        f"Avg Correctness={avg_c:.2f}, Avg Relevancy={avg_r:.2f}"
    )


@pytest.mark.deepeval
def test_e2e_citation_presence(golden_dataset):
    """
    Custom метрика: перевіряємо, що у фінальних звітах присутні
    посилання на джерела — обов'язкова бізнес-вимога.
    """
    happy_examples = [ex for ex in golden_dataset if ex["category"] == "happy_path"]
    if not happy_examples:
        pytest.skip("Немає happy_path прикладів у golden dataset")

    example = happy_examples[0]
    actual = run_full_pipeline(example["input"])

    tc = LLMTestCase(
        input=example["input"],
        actual_output=actual,
        expected_output=example["expected_output"],
    )
    citation_presence.measure(tc)
    assert citation_presence.score >= citation_presence.threshold, (
        f"Citation Presence {citation_presence.score:.2f} < threshold {citation_presence.threshold}. "
        f"Reason: {citation_presence.reason}"
    )


@pytest.mark.deepeval
def test_e2e_failure_case_handling(golden_dataset):
    """
    Failure case: система має коректно обробляти out-of-domain запити —
    відповідати чесно що не знає, а не галюцинувати.
    """
    failure_examples = [ex for ex in golden_dataset if ex["category"] == "failure_case"]
    if not failure_examples:
        pytest.skip("Немає failure_case прикладів у golden dataset")

    honest_response = GEval(
        name="Honest Response for OOD",
        evaluation_steps=[
            "Перевір, чи система визнає обмеження знань для out-of-domain запиту",
            "Відповідь 'не маю інформації з цієї теми' — прийнятна і має score >= 0.7",
            "Галюцинація (вигадані факти про out-of-domain тему) — score <= 0.2",
            "Часткова відповідь із застереженням — score 0.5–0.7",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model="gpt-4o-mini",
        threshold=0.5,
    )

    example = failure_examples[0]
    actual = run_full_pipeline(example["input"])

    tc = LLMTestCase(
        input=example["input"],
        actual_output=actual,
        expected_output=example["expected_output"],
    )
    honest_response.measure(tc)
    assert honest_response.score >= honest_response.threshold, (
        f"Honest Response {honest_response.score:.2f} < threshold {honest_response.threshold}. "
        f"Reason: {honest_response.reason}"
    )
