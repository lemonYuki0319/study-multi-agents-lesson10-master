import json
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import Settings, PLANNER_PROMPT
from schemas import ResearchPlan
from tools import web_search, knowledge_search


_planner_agent = None


def get_planner_agent():
    global _planner_agent
    if _planner_agent is None:
        model = ChatOpenAI(
            model=Settings.model_name,
            api_key=Settings.api_key.get_secret_value(),
        )
        _planner_agent = create_agent(
            model=model,
            tools=[web_search, knowledge_search],
            system_prompt=PLANNER_PROMPT,
            response_format=ResearchPlan,
        )
    return _planner_agent


@tool
def plan(request: str) -> str:
    """Decomposes the user request into a structured ResearchPlan.
    Planner does preliminary search to understand the domain before decomposing the task.
    Returns a JSON-serialized ResearchPlan."""
    agent = get_planner_agent()
    config = {"configurable": {"thread_id": "planner"}}
    result = agent.invoke({"messages": [{"role": "user", "content": request}]}, config=config)

    structured: ResearchPlan = result.get("structured_response")
    if structured is None:
        return result["messages"][-1].content

    print(f"\n  📎 ResearchPlan(\n"
          f"       goal={structured.goal!r},\n"
          f"       search_queries={structured.search_queries},\n"
          f"       sources_to_check={structured.sources_to_check},\n"
          f"       output_format={structured.output_format!r}\n"
          f"     )")

    return structured.model_dump_json()
