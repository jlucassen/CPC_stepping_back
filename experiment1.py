from openai import OpenAI
from dotenv import load_dotenv

import judge
import sample
import solver
from llm import LLM

load_dotenv()
llm = LLM(OpenAI(), "gpt-3.5-turbo")

problem = sample.Problem("What is the meaning of life?")

answer = sample.create_long_answer(llm, problem)
print(answer.text)
contexts = sample.split_into_samples(answer, 1000)
results = list(solver.solve(llm, contexts))
print(results)
evaluation = list(judge.judge_results(results))
print(evaluation)
