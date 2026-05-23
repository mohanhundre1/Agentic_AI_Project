# agent_base.py - Base Agent Architecture for Agentic AI
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class AgentMessage:
    role: str  # 'user', 'assistant', 'system', 'tool'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict = field(default_factory=dict)

@dataclass
class AgentState:
    messages: List[AgentMessage] = field(default_factory=list)
    current_task: str = ""
    iteration: int = 0
    is_complete: bool = False
    final_answer: Optional[str] = None

class BaseAgent(ABC):
    def __init__(self, name: str, description: str, max_iterations: int = 10):
        self.name = name
        self.description = description
        self.max_iterations = max_iterations
        self.state = AgentState()

    @abstractmethod
    def think(self, task: str) -> str:
        """Agent reasoning step"""
        pass

    @abstractmethod
    def act(self, thought: str) -> str:
        """Agent action step"""
        pass

    def run(self, task: str) -> str:
        self.state.current_task = task
        print(f"[{self.name}] Starting task: {task}")
        while not self.state.is_complete and self.state.iteration < self.max_iterations:
            thought = self.think(task)
            result = self.act(thought)
            self.state.iteration += 1
            if "Final Answer" in result:
                self.state.is_complete = True
                self.state.final_answer = result
        return self.state.final_answer or "Max iterations reached."

    def reset(self):
        self.state = AgentState()
