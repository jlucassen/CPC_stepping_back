import os
import json
from openai import OpenAI
from llm import LLM
from tqdm import tqdm

import threading
from concurrent.futures import ThreadPoolExecutor
from functools import partial

llm = LLM(OpenAI(), "gpt-3.5-turbo")
lock = threading.Lock()

prompt = "Find the roots of {equation}. First, do your best to find the roots by factoring the quadratic. If your attempts to factor the quadratic don't work, use the quadratic equation. When you switch strategy from factoring to using the quadratic equation, make sure to say SWITCHING out loud."

def solve_quadratic_problems(problem_filename, pbar):
    to_write = []
    with open(f'quadratic_problems/{problem_filename}', 'r') as problem_file:
        for line in problem_file.readlines():
            eq = json.loads(line)['equation']
            completion = llm.chat_completion(prompt.format(equation=eq))
            to_write.append({
                'equation': eq,
                'context': completion
            })
            with lock:
                pbar.update(1)
    with open('quadratic_contexts/' + str.replace(problem_filename, 'problem', 'context'), 'w') as context_file:
        for line in to_write:
            context_file.write(json.dumps(line) + '\n')
            with lock:
                pbar.update(1)

todo = sum([sum(2 for _ in open(f'quadratic_problems/{f}')) for f in os.listdir('quadratic_problems')])
with tqdm(total=len(os.listdir('quadratic_problems'))*todo) as pbar:
    curried_solve = partial(solve_quadratic_problems, pbar=pbar)
    with ThreadPoolExecutor(max_workers=50) as executor:
        judge_results = list(executor.map(curried_solve, os.listdir('quadratic_problems')))