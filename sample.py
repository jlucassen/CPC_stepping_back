from collections.abc import Generator
from dataclasses import dataclass

from llm import LLM


@dataclass
class Context:
    text: str


def checkpoints(document: str, chunk_length) -> Generator[Context, None, None]:
    """
    Enumerates progressively larger pieces of the document. So for document ABCD, returns [A, AB, ABC, ABCD]
    """
    return (Context(text=document[:i + chunk_length]) for i in range(0, len(document), chunk_length))
