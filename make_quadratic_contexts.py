import asyncio
import json
import os

import dotenv
from tqdm import tqdm

from llm import LLM
from llm import RateLimiter

dotenv.load_dotenv()
llm = LLM("gpt-3.5-turbo")

prompt = "Find the roots of {equation}. First, do your best to find the roots by factoring the quadratic. If your attempts to factor the quadratic don't work, use the quadratic equation. When you switch strategy from factoring to using the quadratic equation, make sure to say SWITCHING out loud."


async def solve_quadratic_problems(problem_filename, progressbar):
    with open(f'data/quadratic_problems/{problem_filename}', 'r') as problem_file:
        async def completion(eq):
            async with RateLimiter(5000):
                try:
                    context = await llm.chat_completion(prompt.format(equation=eq))
                    return {'equation': eq, 'context': context}
                except Exception as e:
                    print(f"Error while processing {eq}: {type(e)} {e}")
                    return {'equation': eq, 'context': f"Exception: {type(e)} {e}"}
                finally:
                    progressbar.update(1)

        completions = await asyncio.gather(*[completion(json.loads(line)['equation']) for line in problem_file])
        with open('quadratic_contexts/' + str.replace(problem_filename, 'problem', 'context'), 'w') as context_file:
            for completion_obj in completions:
                context_file.write(json.dumps(completion_obj) + '\n')


async def main():
    lines = sum((sum(1 for _ in open(f'data/quadratic_problems/{f}')) for f in os.listdir('data/quadratic_problems')))
    with tqdm(total=lines) as pbar:
        await asyncio.gather(*[solve_quadratic_problems(problem_filename, pbar) for problem_filename in
                               os.listdir('data/quadratic_problems')])


if __name__ == "__main__":
    asyncio.run(main())
