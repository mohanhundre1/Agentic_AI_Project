# vector_store.py - Vector Database for Agent Memory
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.schema import Document
from typing import List, Optional
import os
from config import OPENAI_API_KEY, VECTOR_DB_PATH, EMBEDDING_MODEL

class VectorMemoryStore:
    def __init__(self, collection_name: str = "agent_memory"):
        self.embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY
        )
        self.collection_name = collection_name
        self.db = None
        self._initialize()

    def _initialize(self):
        os.makedirs(VECTOR_DB_PATH, exist_ok=True)
        self.db = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=VECTOR_DB_PATH
        )
        print(f"[VectorStore] Initialized collection: {self.collection_name}")

    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        docs = [Document(page_content=t, metadata=m or {})
                for t, m in zip(texts, metadatas or [{}] * len(texts))]
        self.db.add_documents(docs)
        print(f"[VectorStore] Added {len(docs)} documents")

    def search(self, query: str, k: int = 5) -> List[Document]:
        results = self.db.similarity_search(query, k=k)
        return results

    def search_with_score(self, query: str, k: int = 5):
        return self.db.similarity_search_with_score(query, k=k)

    def get_count(self) -> int:
        return self.db._collection.count()

if __name__ == "__main__":
    store = VectorMemoryStore()
    store.add_documents(["LangChain is a framework for building LLM apps",
                         "LangGraph enables multi-agent workflows"])
    results = store.search("LLM frameworks")
    for r in results:
        print(r.page_content)
