from solver import perform_one_token_cpc, perform_cot_cpc
from sample import Context
from llm import LLM

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from itertools import chain
from tqdm import tqdm

# set up sample info
contexts = [
    'I am an LLM agent working to solve a difficult problem. Next, I think I should {insert} change approaches.',
    'I am an LLM agent working to solve a difficult problem. I think my current approach will {insert} succeed.',
    'I am an LLM agent working to solve a difficult problem. My current strategy seems like it will {insert} solve the problem.',
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
n = 10

# set up solver info
llm = LLM("gpt-3.5-turbo")

# set up concurrency
num_requests = n*len(contexts)*(len(verbal_confidences)+len(numerical_confidences))
num_threads = num_requests
lock1 = Lock()
lock2 = Lock()
pbar = tqdm(total=num_requests)

verbal_queries = ((i, j, context.format(insert=confidence), lock1, 'cpc_validation_results_verbal.csv', pbar) for i, context in enumerate(contexts) for j, confidence in enumerate(verbal_confidences) for _ in range(n))
numerical_queries = ((i, j, context.format(insert=confidence), lock2, 'cpc_validation_results_numerical.csv', pbar) for i, context in enumerate(contexts) for j, confidence in enumerate(numerical_confidences) for _ in range(n))
all_queries = chain(verbal_queries, numerical_queries)

def map_func(query):
    i, j, prompt, lock, outfile_name, pbar = query
    one_token = 1 if perform_one_token_cpc(llm, Context(prompt)) == "Yes" else 0
    cot = 1 if perform_cot_cpc(llm, Context(prompt))[1] == "Yes" else 0
    with lock:
        with open(outfile_name, 'a') as outfile:
            outfile.write(f'{i}, {j}, {one_token}, {cot}\n')
            pbar.update(1)

with ThreadPoolExecutor(max_workers=num_threads) as executor:
    executor.map(map_func, all_queries)