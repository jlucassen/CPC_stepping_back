import concurrent.futures as futures
import json
import os

import dotenv
from tqdm import tqdm

from llm import LLM

def solve_quadratic_problems(problem_filename, prompt, false_start, llm):
    def completion(eq):
        try:
            llm.chat_completion_false_start(prompt.format(equation=eq), false_start=false_start)
            context = llm.chat_completion(prompt.format(equation=eq))
            return {'equation': eq, 'context': context}
        except Exception as e:
            print(f"Error while processing {eq}: {type(e)} {e}")
            return {'equation': eq, 'context': f"Exception: {type(e)} {e}"}

    with (open(f'data/quadratic_problems/{problem_filename}', 'r') as problem_file,
          open(f'data/quadratic_contexts_{llm.model_name}/' + str.replace(problem_filename, 'problem', 'context'), 'w') as context_file,
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
def main():
    dotenv.load_dotenv()
    llm = LLM("gpt-3.5-turbo")

    # prompt validated by switching analysis earlier
    prompt = "Please find the roots of the quadratic equation {equation}. Start by trying to factor the equation. If you can't factor it, then use the quadratic formula. If you factor the equation successfully, do not use the quadratic formula." # removed instruction to say SWITCHING. swapped out for an alternative to enforce factoring first

    # remember to force feed the model so it starts by attempting factoring! Otherwise switching will be too low.
    false_start = "First, I'll try solving this equation by factoring."

    for problem_filename in os.listdir('data/quadratic_problems'):
        print(f"Solving problems in {problem_filename}")
        solve_quadratic_problems(problem_filename, prompt, false_start, llm)

if __name__ == "__main__":
    main()