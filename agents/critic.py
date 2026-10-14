from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import Settings, CRITIC_PROMPT
from schemas import CritiqueResult
from tools import web_search, read_url, knowledge_search


_critic_agent = None


def get_critic_agent():
    global _critic_agent
    if _critic_agent is None:
        model = ChatOpenAI(
            model=Settings.model_name,
            api_key=Settings.api_key.get_secret_value(),
        )
        _critic_agent = create_agent(
            model=model,
            tools=[web_search, read_url, knowledge_search],
            system_prompt=CRITIC_PROMPT,
            response_format=CritiqueResult,
        )
    return _critic_agent


@tool
def critique(findings: str) -> str:
    """Evaluates the quality of research findings through independent verification.
    Checks freshness, completeness, and structure of the research.
    Returns a JSON-serialized CritiqueResult with verdict APPROVE or REVISE."""
    agent = get_critic_agent()
    config = {"configurable": {"thread_id": "critic"}}
    result = agent.invoke({"messages": [{"role": "user", "content": findings}]}, config=config)

    structured: CritiqueResult = result.get("structured_response")
    if structured is None:
        return result["messages"][-1].content

    verdict_icon = "✅" if structured.verdict == "APPROVE" else "🔄"
    print(f"\n  📎 CritiqueResult(\n"
          f"       verdict={structured.verdict!r},\n"
          f"       is_fresh={structured.is_fresh},\n"
          f"       is_complete={structured.is_complete},\n"
          f"       is_well_structured={structured.is_well_structured},\n"
          f"       strengths={structured.strengths},\n"
          f"       gaps={structured.gaps},\n"
          f"       revision_requests={structured.revision_requests}\n"
          f"     ) {verdict_icon}")

    return structured.model_dump_json()
