# memory_agent.py - Agent with Persistent Memory using LangChain
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory, VectorStoreRetrieverMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from config import OPENAI_API_KEY, OPENAI_MODEL, AGENT_MEMORY_K

class MemoryAgent:
    def __init__(self, memory_type: str = "buffer"):
        self.llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
        self.memory_type = memory_type
        self.memory = self._create_memory()
        self.chain = self._build_chain()
        self.conversation_history = []

    def _create_memory(self):
        if self.memory_type == "buffer":
            return ConversationBufferWindowMemory(k=AGENT_MEMORY_K,
                                                  return_messages=True)
        return ConversationBufferWindowMemory(k=AGENT_MEMORY_K)

    def _build_chain(self):
        prompt = PromptTemplate(
            input_variables=["history", "input"],
            template="""You are a helpful AI assistant with memory.
Conversation history:\n{history}\nHuman: {input}\nAssistant:"""
        )
        return ConversationChain(llm=self.llm, memory=self.memory,
                                  prompt=prompt, verbose=True)

    def chat(self, user_input: str) -> str:
        response = self.chain.predict(input=user_input)
        self.conversation_history.append({"user": user_input, "agent": response})
        return response

    def get_history(self):
        return self.conversation_history

    def clear_memory(self):
        self.memory.clear()
        self.conversation_history = []
        print("[MemoryAgent] Memory cleared.")

if __name__ == "__main__":
    agent = MemoryAgent()
    print(agent.chat("My name is Mohan."))
    print(agent.chat("What is my name?"))
