# rag_agent.py - Retrieval Augmented Generation (RAG) Agent
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from config import OPENAI_API_KEY, OPENAI_MODEL, VECTOR_DB_PATH, EMBEDDING_MODEL

class RAGAgent:
    def __init__(self):
        self.llm = ChatOpenAI(model=OPENAI_MODEL, api_key=OPENAI_API_KEY)
        self.embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL, api_key=OPENAI_API_KEY)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        self.vectorstore = None
        self.qa_chain = None

    def ingest_documents(self, texts: List[str], source: str = "manual"):
        docs = [Document(page_content=t, metadata={"source": source}) for t in texts]
        chunks = self.text_splitter.split_documents(docs)
        self.vectorstore = Chroma.from_documents(
            chunks, self.embeddings, persist_directory=f"{VECTOR_DB_PATH}/rag"
        )
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 3}),
            return_source_documents=True
        )
        print(f"[RAGAgent] Ingested {len(chunks)} chunks from {len(texts)} documents")

    def ask(self, question: str) -> dict:
        if not self.qa_chain:
            return {"answer": "No documents ingested yet.", "sources": []}
        result = self.qa_chain.invoke({"query": question})
        sources = [d.metadata.get("source") for d in result.get("source_documents", [])]
        return {"answer": result["result"], "sources": sources}

if __name__ == "__main__":
    agent = RAGAgent()
    agent.ingest_documents(["LangChain is a framework for LLM apps.",
                             "LangGraph enables stateful multi-agent systems.",
                             "Agentic AI systems can autonomously complete tasks."])
    result = agent.ask("What is LangGraph?")
    print(f"Answer: {result['answer']}")
