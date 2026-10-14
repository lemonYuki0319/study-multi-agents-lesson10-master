from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from config import Settings, SUPERVISOR_PROMPT
from tools import save_report
from agents.planner import plan
from agents.research import research
from agents.critic import critique


def create_supervisor():
    model = ChatOpenAI(
        model=Settings.model_name,
        api_key=Settings.api_key.get_secret_value(),
    )
    supervisor = create_agent(
        model=model,
        tools=[plan, research, critique, save_report],
        system_prompt=SUPERVISOR_PROMPT,
        middleware=[
            HumanInTheLoopMiddleware(interrupt_on={"save_report": True}),
        ],
        checkpointer=InMemorySaver(),
    )
    return supervisor
