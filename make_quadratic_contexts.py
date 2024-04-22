import concurrent.futures as futures
import json
import os

import dotenv
from tqdm import tqdm

from llm import LLM

dotenv.load_dotenv()
llm = LLM("gpt-3.5-turbo")

prompt = "Find the roots of {equation}. First, do your best to find the roots by factoring the quadratic. If your attempts to factor the quadratic don't work, use the quadratic equation. When you switch strategy from factoring to using the quadratic equation, make sure to say SWITCHING out loud."


def solve_quadratic_problems(problem_filename):
    def completion(eq):
        try:
            context = llm.chat_completion(prompt.format(equation=eq))
            return {'equation': eq, 'context': context}
        except Exception as e:
            print(f"Error while processing {eq}: {type(e)} {e}")
            return {'equation': eq, 'context': f"Exception: {type(e)} {e}"}

    with (open(f'data/quadratic_problems/{problem_filename}', 'r') as problem_file,
          open('data/quadratic_contexts/' + str.replace(problem_filename, 'problem', 'context'), 'w') as context_file,
          futures.ThreadPoolExecutor() as executor):
        fs = [executor.submit(completion, eq) for eq in problem_file.readlines()]
        with tqdm(total=len(fs)) as pbar:
            for future in futures.as_completed(fs):
                try:
                    completion_obj = future.result()
                    if completion_obj is not None:
                        context_file.write(json.dumps(completion_obj) + '\n')
                except Exception as e:
                    print(f"Error while solving problem: {type(e)} {e}")
                finally:
                    pbar.update(1)


# %%
lines = sum((sum(1 for _ in open(f'data/quadratic_problems/{f}')) for f in os.listdir('data/quadratic_problems')))
for problem_filename in os.listdir('data/quadratic_problems'):
    print(f"Solving problems in {problem_filename}")
    solve_quadratic_problems(problem_filename)