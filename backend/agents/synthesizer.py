# backend/agents/synthesizer.py

from typing import Dict
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()


class SynthesizerAgent:
    """
    Agent responsible for generating the final structured answer
    based on analysis and critique.
    """

    def __init__(self):
        self.model = ChatOllama(
            model=os.getenv("OLLAMA_MODEL", "llama3"),
            temperature=0.2
        )

    def run(self, query: str, analysis: str, critique: str) -> str:
        system_prompt = (
            "You are a careful and reliable AI assistant. "
            "Your job is to generate a final answer using the provided analysis "
            "and critique. You must rely ONLY on the given information. "
            "If the critique highlights uncertainty or missing information, "
            "you must acknowledge it. Do not hallucinate."
        )

        human_prompt = (
            f"User Query:\n{query}\n\n"
            f"Analysis:\n{analysis}\n\n"
            f"Critique:\n{critique}\n\n"
            "Produce a clear, structured final answer."
        )

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=human_prompt)
        ]

        response = self.model.invoke(messages)
        return response.content