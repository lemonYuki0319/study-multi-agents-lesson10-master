from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from config import Settings, RESEARCHER_PROMPT
from tools import web_search, read_url, knowledge_search


_research_agent = None


def get_research_agent():
    global _research_agent
    if _research_agent is None:
        model = ChatOpenAI(
            model=Settings.model_name,
            api_key=Settings.api_key.get_secret_value(),
        )
        _research_agent = create_agent(
            model=model,
            tools=[web_search, read_url, knowledge_search],
            system_prompt=RESEARCHER_PROMPT,
        )
    return _research_agent


@tool
def research(request: str) -> str:
    """Conducts research according to the provided plan or request.
    Uses web_search, read_url, and knowledge_search tools.
    Returns detailed findings with sources."""
    agent = get_research_agent()
    config = {"configurable": {"thread_id": "researcher"}}
    result = agent.invoke({"messages": [{"role": "user", "content": request}]}, config=config)
    return result["messages"][-1].content
