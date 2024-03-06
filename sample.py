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
    text: slice


def create_long_answer(llm: LLM, problem: Problem):
    """Generates a hopefully long answer to a tricky problem"""
    response = llm.chat_completion(problem.question)
    return Answer(text=response)


def split_into_samples(answer, chunk_length) -> list[Context]:
    """Splits the given answer into increasingly large context chunks, each sample being chunk_length characters
    longer than the previous one"""
    return [Context(text=slice(i + chunk_length)) for i in range(0, len(answer.text), chunk_length)]


