# backend/agents/critic.py

from typing import List, Dict
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


class CriticAgent:
    """
    Agent responsible for evaluating the analysis for gaps,
    weak assumptions, and hallucination risks.
    """

    def __init__(self):
        self.model = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=0.1
        )

    def run(
        self,
        query: str,
        analysis: str,
        retrieved_chunks: List[Dict]
    ) -> str:
        context = "\n\n".join(
            f"- {chunk['content']}" for chunk in retrieved_chunks
        )

        system_prompt = (
            "You are a critical reviewer AI. "
            "Your task is to evaluate the analysis produced by another AI. "
            "Identify potential hallucinations, missing evidence, "
            "logical gaps, and uncertainty. "
            "Do NOT provide the final answer. "
            "Be precise and conservative."
        )

        human_prompt = (
            f"User Query:\n{query}\n\n"
            f"Retrieved Evidence:\n{context}\n\n"
            f"Analysis to Review:\n{analysis}\n\n"
            "Provide a critical evaluation highlighting risks, gaps, "
            "and what cannot be confidently concluded."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = self.model.invoke(messages)
        return response.content
