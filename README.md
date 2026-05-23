# Agentic AI Project

A comprehensive project for building **Agentic AI systems** with autonomous agents, multi-step reasoning, and tool-calling capabilities.

## Project Structure

```
Agentic_AI_Project/
├── main.py              # CLI entry point for all agents
├── config.py            # Configuration and environment variables
├── agent_base.py        # Abstract base class for all agents
├── react_agent.py       # ReAct (Reasoning + Acting) agent
├── memory_agent.py      # Conversational agent with memory
├── rag_agent.py         # Retrieval Augmented Generation agent
├── orchestrator.py      # Multi-agent orchestrator (LangGraph)
├── vector_store.py      # ChromaDB vector database for memory
├── tools.py             # Custom LangChain tools
├── prompts.py           # Reusable prompt templates
├── utils.py             # Utility functions and logging
├── requirements.txt     # Python dependencies
└── .gitignore           # Git ignore rules
```

## Agent Types

| Agent | Description |
|-------|-------------|
| `react_agent` | Uses ReAct framework with tools (search, calculator) |
| `memory_agent` | Maintains conversation history with buffer memory |
| `rag_agent` | Answers questions using document retrieval (RAG) |
| `orchestrator` | Multi-agent workflow with researcher, analyst, writer |

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/mohanhundre1/Agentic_AI_Project.git
cd Agentic_AI_Project

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set environment variables
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 5. Run an agent
python main.py --agent react --task "What is the capital of India?"
python main.py --agent memory --task "My name is Mohan"
python main.py --agent rag --task "What is LangGraph?"
python main.py --agent orchestrator --task "Explain Agentic AI"
```

## Tech Stack

- **LangChain** - Agent framework
- **LangGraph** - Multi-agent orchestration
- **ChromaDB** - Vector database for memory
- **OpenAI GPT-4** - Language model
- **FastAPI** - API server

## License

MIT License
