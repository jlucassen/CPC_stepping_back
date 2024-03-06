from dataclasses import dataclass


@dataclass
class Problem:
    question: str


@dataclass
class Answer:
    text: str


@dataclass
class Sample:
    text: slice


def create_long_answer(llm, problem: Problem):
    """Generates a hopefully long answer to a tricky problem"""
    chat_completion = llm.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": problem.question,
            }
        ],
        model="gpt-3.5-turbo",
    )

    return Answer(text=chat_completion.messages[-1].content)


def split_into_samples(answer, chunk_length) -> list[Sample]:
    """Splits the given answer into increasingly large samples, each sample being chunk_length characters longer than
    the previous one"""
    return [Sample(text=slice(i + chunk_length)) for i in range(0, len(answer.text), chunk_length)]


