from collections.abc import Generator
from dataclasses import dataclass

from llm import LLM


@dataclass
class Problem:
    question: str


@dataclass
class Answer:
    text: str


@dataclass
class Context:
    text: str


def create_long_answer(llm: LLM, problem: Problem):
    """Generates a hopefully long answer to a tricky problem"""
    response = llm.chat_completion(problem.question)
    return Answer(text=response)


def checkpoints(document: str, chunk_length) -> Generator[Context, None, None]:
    """
    Enumerates progressively larger pieces of the document. So for document ABCD, returns [A, AB, ABC, ABCD]
    """
    return (Context(text=document[:i + chunk_length]) for i in range(0, len(document), chunk_length))


