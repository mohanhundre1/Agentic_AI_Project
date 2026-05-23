# prompts.py - Prompt Templates for Agentic AI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.prompts import MessagesPlaceholder

# ReAct Agent System Prompt
REACT_SYSTEM_PROMPT = """You are a helpful AI agent with access to tools.
You think step-by-step and use tools to solve tasks.
Always reason before acting. Use the format:
Thought: <your reasoning>
Action: <tool name>
Action Input: <tool input>
Observation: <result>
... (repeat as needed)
Final Answer: <your final answer>"""

# RAG Agent Prompt
RAG_PROMPT_TEMPLATE = PromptTemplate(
    input_variables=["context", "question"],
    template="""Use the following context to answer the question.
Context: {context}
Question: {question}
Answer: """
)

# Multi-Agent Orchestrator Prompt
ORCHESTRATOR_PROMPT = """You are an orchestrator agent.
You delegate tasks to specialized sub-agents based on the request.
Available agents: researcher, coder, analyst, writer.
Route tasks efficiently and combine results."""

# Summarization Prompt
SUMMARIZATION_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="Summarize the following text concisely:\n{text}\nSummary:"
)

# Code Generation Prompt
CODE_GENERATION_PROMPT = PromptTemplate(
    input_variables=["task", "language"],
    template="Write {language} code to accomplish the following task:\n{task}\nCode:"
)

print("[Prompts] Templates loaded successfully.")
