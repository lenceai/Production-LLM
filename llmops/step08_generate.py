"""
step08_generate.py
Version: 1.0
Usage: Generates an answer using the generator model and context.
"""
from typing import List, Dict, Any
from .models import Generator, Generation

def generate_answer(generator: Generator, question: str, contexts: List[Dict[str, Any]]) -> Generation:
    return generator.generate(question, contexts)

