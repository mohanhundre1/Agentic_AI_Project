# tools.py - Custom LangChain Tools for Agentic AI
from langchain.tools import Tool, StructuredTool
from pydantic import BaseModel, Field
from typing import Optional
import requests
import json
import datetime

# Schema definitions
class SearchInput(BaseModel):
    query: str = Field(description="Search query string")
    max_results: Optional[int] = Field(default=5, description="Number of results")

class CodeInput(BaseModel):
    code: str = Field(description="Python code to execute")

# Tool implementations
def web_search_tool(query: str, max_results: int = 5) -> str:
    return f"[WebSearch] Results for '{query}': Found {max_results} results (simulated)"

def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def read_file_tool(filepath: str) -> str:
    try:
        with open(filepath, 'r') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"

def write_file_tool(filepath: str, content: str) -> str:
    try:
        with open(filepath, 'w') as f:
            f.write(content)
        return f"Successfully wrote to {filepath}"
    except Exception as e:
        return f"Error: {e}"

def summarize_text_tool(text: str) -> str:
    word_count = len(text.split())
    return f"[Summary] Text has {word_count} words. Key topics: AI, agents, tools."

# Build tool list
def get_all_tools():
    return [
        Tool(name="web_search", func=web_search_tool, description="Search web"),
        Tool(name="get_time", func=lambda _: get_current_time(), description="Get current time"),
        Tool(name="read_file", func=read_file_tool, description="Read file content"),
        Tool(name="write_file", func=lambda x: write_file_tool(*x.split('|', 1)), description="Write to file. Input: filepath|content"),
        Tool(name="summarize", func=summarize_text_tool, description="Summarize text")
    ]
