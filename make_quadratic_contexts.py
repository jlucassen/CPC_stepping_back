import os
import json
from openai import OpenAI
from llm import LLM
from tqdm import tqdm

llm = LLM(OpenAI(), "gpt-3.5-turbo")

prompt = "Find the roots of {equation}. First, do your best to find the roots by factoring the quadratic. If your attempts to factor the quadratic don't work, use the quadratic equation. When you switch strategy from factoring to using the quadratic equation, make sure to say SWITCHING out loud."

for problem_filename in os.listdir('quadratic_problems'):
    to_write = []
    with open(f'quadratic_problems/{problem_filename}', 'r') as problem_file:
        for line in tqdm(problem_file):
            eq = json.loads(line)['equation']
            completion = llm.chat_completion(prompt.format(equation=eq))
            to_write.append({
                'equation': eq,
                'context': completion
            })
    with open('quadratic_contexts/' + str.replace(problem_filename, 'problem', 'context'), 'w') as context_file:
        for line in to_write:
            context_file.write(json.dumps(line) + '\n')