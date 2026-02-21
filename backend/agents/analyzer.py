# backend/agents/analyzer.py

from typing import List, Dict
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage


load_dotenv()


class AnalyzerAgent:
    """
    Agent responsible for deep reasoning over retrieved documents.
    """

    def __init__(self):
        self.model = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=0.2
        )

    def run(self, query: str, retrieved_chunks: List[Dict]) -> str:
        """
        Perform analytical reasoning on retrieved content.
        """
        context = "\n\n".join(
            f"- {chunk['content']}" for chunk in retrieved_chunks
        )

        system_prompt = (
            "You are an analytical AI assistant. "
            "Your task is to carefully analyze the provided document excerpts "
            "and extract relevant insights related to the user query. "
            "Do not answer the query directly. "
            "Focus on identifying key facts, sections, and relationships."
        )

        human_prompt = (
            f"User Query:\n{query}\n\n"
            f"Retrieved Document Excerpts:\n{context}\n\n"
            "Provide a detailed analytical breakdown."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        # response = self.model(messages)
        response = self.model.invoke(messages)
        return response.content
