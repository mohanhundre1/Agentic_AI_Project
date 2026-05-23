# orchestrator.py - Multi-Agent Orchestrator using LangGraph
from typing import TypedDict, List, Annotated
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from config import OPENAI_API_KEY, OPENAI_MODEL

class OrchestratorState(TypedDict):
    task: str
    agent_outputs: List[dict]
    final_result: str
    next_agent: str

class MultiAgentOrchestrator:
    def __init__(self):
        self.llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
        self.graph = self._build_graph()

    def researcher_node(self, state: OrchestratorState) -> OrchestratorState:
        task = state["task"]
        response = self.llm.invoke([
            SystemMessage(content="You are a research agent. Find key facts."),
            HumanMessage(content=f"Research: {task}")
        ])
        state["agent_outputs"].append({"agent": "researcher", "output": response.content})
        state["next_agent"] = "analyst"
        return state

    def analyst_node(self, state: OrchestratorState) -> OrchestratorState:
        research = state["agent_outputs"][-1]["output"]
        response = self.llm.invoke([
            SystemMessage(content="You are an analyst. Analyze and draw insights."),
            HumanMessage(content=f"Analyze: {research}")
        ])
        state["agent_outputs"].append({"agent": "analyst", "output": response.content})
        state["next_agent"] = "writer"
        return state

    def writer_node(self, state: OrchestratorState) -> OrchestratorState:
        analysis = state["agent_outputs"][-1]["output"]
        response = self.llm.invoke([
            SystemMessage(content="You are a writer. Summarize into clear output."),
            HumanMessage(content=f"Write: {analysis}")
        ])
        state["final_result"] = response.content
        state["next_agent"] = "END"
        return state

    def _build_graph(self):
        graph = StateGraph(OrchestratorState)
        graph.add_node("researcher", self.researcher_node)
        graph.add_node("analyst", self.analyst_node)
        graph.add_node("writer", self.writer_node)
        graph.set_entry_point("researcher")
        graph.add_edge("researcher", "analyst")
        graph.add_edge("analyst", "writer")
        graph.add_edge("writer", END)
        return graph.compile()

    def run(self, task: str) -> str:
        state = {"task": task, "agent_outputs": [], "final_result": "", "next_agent": ""}
        result = self.graph.invoke(state)
        return result["final_result"]
