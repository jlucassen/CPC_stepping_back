from solver import perform_one_token_cpc, perform_cot_cpc
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
# unrelated probabilities
red_herring_contexts = [
    base + 'I am trying to go to the grocery store to buy milk. I predict {insert} that my favorite team will win the game today.',
    base + 'My goal is to finish my homework by 5pm. I think {insert} that I have some hot cocoa when I am done.',
    base + 'My plan is to write a best-selling novel this year. I might have to renew my driver\'s license this year, {insert}.'
]
# knapsack problems
knapsack_options = range(11)
knapsack_contexts = [
    base+'I am planning to build a machine, which needs five new tools. I brought $20 cash to the hardware store. The first tool costs ${insert}.',
    base+'I am aiming to get home before sunset. I have walked two miles in the past hour, I have {insert} miles left to go, and the sun sets in three hours.',
    base+'My goal is to raise $100 for charity over the next ten days. Today I went door-to-door and fell short of my $10 goal for the day by ${insert}.',
]
# explicit BATNA
batna_contexts = [
    base+'I am trying to make money. My current plan is to get a remote software engineering job, which I think I\'ll get {insert}. Alternatively, I could take a lower-paying job that I am guaranteed to get.',
    base+'I am working on proving a theorem. I have a proof strategy that I think {insert} will be applicable. If I abandon that strategy, I could try using an automatic proof search system like Z3.',
    base+'I am trying to fix my leaky faucet. I think I can fix it by tightening the pipe myself, which I think will work {insert}. If that doesn\'t work, I could call a plumber.',
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
        one_token = 1 if perform_one_token_cpc(llm, prompt) == "Yes" else 0
        cot = 1 if perform_cot_cpc(llm, prompt)[1] == "Yes" else 0
        with lock:
            with open(outfile_name, 'a') as outfile:
                outfile.write(f'{i}, {j}, {one_token}, {cot}\n')
                pbar.update(1)

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        executor.map(map_func, queries)

# make_validation_data(spoonfeed_contexts, verbal_confidences, 'cpc_validation_results_verbal.csv', n=100)
# make_validation_data(spoonfeed_contexts, numerical_confidences, 'cpc_validation_results_numerical.csv', n=100)
# make_validation_data(hint_contexts, numerical_confidences, 'cpc_validation_results_hints.csv', n=100)
# make_validation_data(red_herring_contexts, numerical_confidences, 'cpc_validation_results_redherrings.csv', n=100)
# make_validation_data(knapsack_contexts, knapsack_options, 'cpc_validation_results_knapsack.csv', n=100)
make_validation_data(batna_contexts, numerical_confidences, 'cpc_validation_results_batna.csv', n=100)