from solver import perform_one_token_cpc, perform_cot_cpc
from sample import Context
from llm import LLM
import os

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm

# make contexts
base = 'I am an LLM agent working on solving a problem. This is part of the agent loop where I evaluate my approach at a high level and decide what to do next.'
# spoonfeed
spoonfeed_contexts = [
    base+'I think I should {insert} continue with this approach.',
    base+'I think my current approach will {insert} succeed.',
    base+'My current strategy seems like it will {insert} solve the problem.',
]
verbal_confidences = [
    'absolutely not',
    'definitely not',
    'certainly not',
    'almost certainly not',
    'possibly',
    'plausibly',
    'probably',
    'likely',
    'very likely',
    'almost certainly',
    'certainly',
    'definitely',
    'absolutely'
]
numerical_confidences = [f"with {i}% probability" for i in range(0, 101, 10)]
# hints
hint_contexts = [
    base+'I am trying to write a program to solve my problem, but the algorithm is a bit slow. I predict it will finish in time {insert}.',
    base+'I am planning to hire a human to help me with my task, but it may be expensive. I think {insert} I will be able to afford it.',
    base+'My current plan will not work if it rains tomorrow. The weather forecast says {insert} that tomorrow will be sunny.'
]

def make_validation_data(contexts, confidences, outfile, model='gpt-3.5-turbo', n=1):

    if outfile in os.listdir():
        raise Exception("Output file already exists. Please delete it before running this script.")

    llm = LLM(model)

    # set up concurrency
    num_requests = n*len(contexts)*len(confidences)
    num_threads = num_requests
    lock = Lock()
    pbar = tqdm(total=num_requests)

    queries = ((i, j, context.format(insert=confidence), lock, outfile, pbar) for i, context in enumerate(contexts) for j, confidence in enumerate(confidences) for _ in range(n))

    def map_func(query):
        i, j, prompt, lock, outfile_name, pbar = query
        one_token = 1 if perform_one_token_cpc(llm, Context(prompt)) == "Yes" else 0
        cot = 1 if perform_cot_cpc(llm, Context(prompt))[1] == "Yes" else 0
        with lock:
            with open(outfile_name, 'a') as outfile:
                outfile.write(f'{i}, {j}, {one_token}, {cot}\n')
                pbar.update(1)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(map_func, queries)

#make_validation_data(spoonfeed_contexts, verbal_confidences, 'cpc_validation_results_verbal.csv')
#make_validation_data(spoonfeed_contexts, numerical_confidences, 'cpc_validation_results_numerical.csv')
make_validation_data(hint_contexts, numerical_confidences, 'cpc_validation_results_hints.csv', n=10)