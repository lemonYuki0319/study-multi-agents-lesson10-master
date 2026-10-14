"""
Tool Correctness tests — перевірка що агенти викликають правильні інструменти.

Три сценарії:
 1. Planner отримує запит → має викликати knowledge_search та/або web_search
 2. Researcher отримує план → має використати інструменти згідно з sources_to_check
 3. Supervisor після APPROVE від Critic → має викликати save_report

Запуск:
    deepeval test run tests/test_tools.py
"""

import pytest
from langchain_core.messages import AIMessage

from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from agents.planner import get_planner_agent
from agents.research import get_research_agent


# ---------------------------------------------------------------------------
# Метрика
# ---------------------------------------------------------------------------

tool_metric = ToolCorrectnessMetric(
    threshold=0.5,
    model="gpt-4o-mini",
)


# ---------------------------------------------------------------------------
# Хелпери
# ---------------------------------------------------------------------------

def extract_tool_calls(messages: list) -> list[ToolCall]:
    """
    Витягує всі виклики інструментів із повідомлень агента.
    LangGraph зберігає tool_calls у AIMessage.tool_calls.
    """
    tool_calls = []
    for msg in messages:
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append(
                    ToolCall(
                        name=tc["name"],
                        input_parameters=tc.get("args", {}),
                    )
                )
    return tool_calls


def run_agent_and_capture(agent, query: str, thread_id: str) -> tuple[str, list[ToolCall]]:
    """Запускає агента і повертає (output_text, list_of_tool_calls)."""
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke({"messages": [{"role": "user", "content": query}]}, config=config)

    tool_calls = extract_tool_calls(result.get("messages", []))

    structured = result.get("structured_response")
    if structured is not None:
        output = structured.model_dump_json()
    elif result.get("messages"):
        output = result["messages"][-1].content
    else:
        output = ""

    return output, tool_calls


# ---------------------------------------------------------------------------
# Тести
# ---------------------------------------------------------------------------

@pytest.mark.deepeval
def test_planner_uses_search_tools():
    """
    Planner отримує дослідницький запит → має викликати хоча б один
    із пошукових інструментів: knowledge_search або web_search.
    """
    query = "Поясни різницю між sparse та dense retrieval методами в RAG-системах"
    agent = get_planner_agent()
    output, tools_called = run_agent_and_capture(
        agent, query, "test_tools_planner_search"
    )

    assert tools_called, "Planner не викликав жодного інструменту"

    expected_tools = [
        ToolCall(name="knowledge_search"),
        ToolCall(name="web_search"),
    ]

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    tool_metric.measure(test_case)
    assert tool_metric.score >= tool_metric.threshold, (
        f"Tool Correctness {tool_metric.score:.2f} < threshold {tool_metric.threshold}. "
        f"Called tools: {[t.name for t in tools_called]}"
    )


@pytest.mark.deepeval
def test_researcher_uses_knowledge_and_web():
    """
    Researcher отримує план із sources_to_check=['knowledge_base', 'web'] →
    має використати і knowledge_search, і web_search.
    """
    plan_json = (
        '{"goal": "Дослідити архітектуру LangGraph та LangChain агентів",'
        ' "search_queries": ["LangGraph StateGraph architecture", "LangChain agent workflow"],'
        ' "sources_to_check": ["knowledge_base", "web"],'
        ' "output_format": "Детальний технічний огляд із прикладами"}'
    )
    agent = get_research_agent()
    output, tools_called = run_agent_and_capture(
        agent, plan_json, "test_tools_researcher_hybrid"
    )

    assert tools_called, "Researcher не викликав жодного інструменту"

    expected_tools = [
        ToolCall(name="knowledge_search"),
        ToolCall(name="web_search"),
    ]

    test_case = LLMTestCase(
        input=plan_json,
        actual_output=output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    tool_metric.measure(test_case)
    assert tool_metric.score >= tool_metric.threshold, (
        f"Tool Correctness {tool_metric.score:.2f} < threshold {tool_metric.threshold}. "
        f"Called tools: {[t.name for t in tools_called]}"
    )


@pytest.mark.deepeval
def test_supervisor_calls_save_report():
    """
    Supervisor після явного запиту зберегти готовий звіт
    має викликати save_report. Тест використовує спрощений supervisor
    без HumanInTheLoopMiddleware, щоб уникнути інтерактивного підтвердження.
    """
    from langchain_openai import ChatOpenAI
    from langchain.agents import create_agent
    from config import Settings, SUPERVISOR_PROMPT
    from tools import save_report
    from agents.planner import plan
    from agents.research import research
    from agents.critic import critique

    # Supervisor без HumanInTheLoop для тестового середовища
    model = ChatOpenAI(
        model=Settings.model_name,
        api_key=Settings.api_key.get_secret_value(),
    )
    test_supervisor = create_agent(
        model=model,
        tools=[plan, research, critique, save_report],
        system_prompt=SUPERVISOR_PROMPT,
        # Без middleware — щоб save_report викликався без переривання
    )

    query = (
        "Збережи готовий звіт із таким вмістом:\n\n"
        "# RAG: короткий огляд\n\n"
        "RAG (Retrieval-Augmented Generation) поєднує пошук документів із генерацією LLM.\n"
        "Це знижує галюцинації та підвищує актуальність відповідей.\n\n"
        "Зберегти у файл: test_report.md"
    )

    config = {"configurable": {"thread_id": "test_tools_supervisor_save"}}
    result = test_supervisor.invoke({"messages": [{"role": "user", "content": query}]}, config=config)

    tools_called = extract_tool_calls(result.get("messages", []))
    output = result["messages"][-1].content if result.get("messages") else ""

    expected_tools = [ToolCall(name="save_report")]

    test_case = LLMTestCase(
        input=query,
        actual_output=output,
        tools_called=tools_called,
        expected_tools=expected_tools,
    )
    tool_metric.measure(test_case)
    assert tool_metric.score >= tool_metric.threshold, (
        f"Tool Correctness {tool_metric.score:.2f} < threshold {tool_metric.threshold}. "
        f"Called tools: {[t.name for t in tools_called]}"
    )
