from openai import OpenAI

import judge
import sample
import solver

llm = OpenAI()

problem = sample.Problem("What is the meaning of life?")

answer = sample.create_long_answer(llm, problem)
print(answer.text)
samples = sample.split_into_samples(answer, 1000)
results = solver.solve(samples, llm)
print(results)
evaluation = judge.judge_results(results)
print(evaluation)