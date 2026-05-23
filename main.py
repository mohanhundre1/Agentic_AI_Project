# main.py - Entry Point for Agentic AI Project
import argparse
from config import OPENAI_API_KEY

def run_react_agent(task: str):
    from react_agent import build_react_agent
    print(f"\n--- Running ReAct Agent ---")
    agent = build_react_agent()
    result = agent.invoke({"input": task})
    print(f"Result: {result['output']}")

def run_memory_agent(task: str):
    from memory_agent import MemoryAgent
    print(f"\n--- Running Memory Agent ---")
    agent = MemoryAgent()
    response = agent.chat(task)
    print(f"Response: {response}")

def run_rag_agent(task: str):
    from rag_agent import RAGAgent
    print(f"\n--- Running RAG Agent ---")
    agent = RAGAgent()
    agent.ingest_documents([
        "Agentic AI is a paradigm where AI systems can plan and execute multi-step tasks.",
        "LangChain provides tools for building LLM-powered applications.",
        "LangGraph enables building stateful, multi-actor workflows with LLMs."
    ])
    result = agent.ask(task)
    print(f"Answer: {result['answer']}")

def run_orchestrator(task: str):
    from orchestrator import MultiAgentOrchestrator
    print(f"\n--- Running Multi-Agent Orchestrator ---")
    orch = MultiAgentOrchestrator()
    result = orch.run(task)
    print(f"Final Result: {result}")

def main():
    parser = argparse.ArgumentParser(description="Agentic AI Project")
    parser.add_argument("--agent", choices=["react","memory","rag","orchestrator"], default="react")
    parser.add_argument("--task", type=str, default="What is Agentic AI?")
    args = parser.parse_args()

    if not OPENAI_API_KEY:
        print("ERROR: Set OPENAI_API_KEY in your .env file")
        return

    if args.agent == "react":
        run_react_agent(args.task)
    elif args.agent == "memory":
        run_memory_agent(args.task)
    elif args.agent == "rag":
        run_rag_agent(args.task)
    elif args.agent == "orchestrator":
        run_orchestrator(args.task)

if __name__ == "__main__":
    main()
