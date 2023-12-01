from langchain.llms import Bedrock
from langchain.agents import tool
from datetime import date

@tool
def time(text: str) -> str:
    return str(date.today())   

llm = Bedrock( model_id="anthropic.claude-v2")
from langchain.agents import load_tools, initialize_agent
from langchain.agents import AgentType
tools = load_tools(["llm-math"], llm=llm)
agent= initialize_agent(
    tools + [time],
    llm,
    agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    handle_parsing_errors=True,
    verbose = True)

q = "what time is it?"
print(agent(q))