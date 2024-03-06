from openai import OpenAI

import judge
import sample
import solver
from llm import LLM


llm = LLM(OpenAI(), "gpt-3.5-turbo")

problem = sample.Problem("What is the meaning of life?")

answer = sample.create_long_answer(llm, problem)
print(answer.text)
contexts = sample.split_into_samples(answer, 1000)
results = solver.solve(llm, contexts)
print(results)
evaluation = judge.judge_results(results)
print(evaluation)