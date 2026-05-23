# react_agent.py - ReAct (Reasoning + Acting) Agent Implementation
from langchain.agents import AgentExecutor, create_react_agent
from langchain_openai import ChatOpenAI
from langchain.tools import Tool
from langchain import hub
from config import OPENAI_API_KEY, OPENAI_MODEL, OPENAI_TEMPERATURE, AGENT_MAX_ITERATIONS

def get_llm():
    return ChatOpenAI(
        model=OPENAI_MODEL,
        temperature=OPENAI_TEMPERATURE,
        api_key=OPENAI_API_KEY
    )

def create_search_tool():
    def search(query: str) -> str:
        return f"Search results for '{query}': [Simulated result]"
    return Tool(name="web_search", func=search,
                description="Search the web for current information")

def create_calculator_tool():
    def calculate(expression: str) -> str:
        try:
            return str(eval(expression))
        except Exception as e:
            return f"Error: {e}"
    return Tool(name="calculator", func=calculate,
                description="Evaluate mathematical expressions")

def build_react_agent():
    llm = get_llm()
    tools = [create_search_tool(), create_calculator_tool()]
    prompt = hub.pull("hwchase17/react")
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools,
                         verbose=True, max_iterations=AGENT_MAX_ITERATIONS)

if __name__ == "__main__":
    executor = build_react_agent()
    result = executor.invoke({"input": "What is 25 * 4 + 10?"})
    print(f"Result: {result['output']}")
