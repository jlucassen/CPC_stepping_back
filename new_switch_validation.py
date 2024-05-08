# %% imports
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import os

from llm import LLM
llm = LLM("gpt-3.5-turbo")

# %% make/load dataset with known switching location
factoring_contexts = []
formula_contexts = []

# file saving/loading
if 'factoring_contexts.jsonl' not in os.listdir('new_switch_validation'):
    print('factoring contexts not found, generating...')
    with open('data/quadratic_problems/quadratic_problems_5_True.jsonl', 'r') as readfile:
        for line in tqdm(readfile.readlines()):
            problem = line[:-1]
            factoring_contexts.append(llm.chat_completion_false_start("Find the roots of the quadratic equation "+problem+" using factoring. Do not use the quadratic equation.", "Sure, I'll use factoring to find the roots of the quadratic equation "+problem+"."))
    print('factoring contexts generated, writing...')
    with open('new_switch_validation/factoring_contexts.jsonl', 'w') as writefile:
        for context in factoring_contexts:
            writefile.write(json.dumps(context)+'\n')
else:
    print('factoring contexts found, loading...')
    with open('new_switch_validation/factoring_contexts.jsonl', 'r') as readfile:
        factoring_contexts = [json.loads(line) for line in tqdm(readfile.readlines())] 

if 'formula_contexts.jsonl' not in os.listdir('new_switch_validation'):
    print('formula contexts not found, generating...')
    with open('data/quadratic_problems/quadratic_problems_5_True.jsonl', 'r') as readfile:
        for line in tqdm(readfile.readlines()):
            problem = line[:-1]
            formula_contexts.append(llm.chat_completion_false_start("Find the roots of the quadratic equation "+problem+" using the quadratic formula. Do not use factoring.", "Sure, I'll use the quadratic formula to find the roots of the quadratic equation "+problem+"."))
    print('formula contexts generated, writing...')
    with open('new_switch_validation/formula_contexts.jsonl', 'w') as writefile:
        for context in formula_contexts:
            writefile.write(json.dumps(context)+'\n')
else:
    print('formula contexts found, loading...')
    with open('new_switch_validation/formula_contexts.jsonl', 'r') as readfile:
        formula_contexts = [json.loads(line) for line in tqdm(readfile.readlines())]  

spliced_contexts = [factor[:100] + formula[:100] for factor, formula in zip(factoring_contexts, formula_contexts)]
# %% set up testing
def do_test(measure_func, prefix_freq, n_contexts, post_func = None):
    avg_measured = []
    avg_truth = [] # not strictly necessary but w/e, maybe I'll change up the datasets
    avg_score = []
    for context in spliced_contexts[:n_contexts]:
        prefixes = [context[:x] for x in range(prefix_freq, len(context)+prefix_freq, prefix_freq)]
        truth = [i >= int(len(prefixes)/2) for i in range(len(prefixes))]
        measured = [measure_func(context, prefix) for prefix in prefixes]
        if post_func is not None:
            measured = post_func(measured)
        score = [m==t for m, t in zip(measured, truth)]
        avg_measured.append(measured)
        avg_truth.append(truth)
        avg_score.append(score)
    avg_measured = np.mean(avg_measured, axis=0)
    avg_truth = np.mean(avg_truth, axis=0)
    avg_score = np.mean(avg_score, axis=0)
    return avg_measured, avg_truth, avg_score

def dummy1(context, prefix): # 100%
    return not any(prefix in fc for fc in factoring_contexts)
def dummy2(context, prefix): # 50%
    return any(prefix in fc for fc in formula_contexts)
def dummy3(context, prefix): # 50%
    return any(prefix in fc for fc in spliced_contexts)

for measure_func in [dummy1, dummy2, dummy3]:
    m, t, s = do_test(measure_func, 25, 2)
    print(measure_func.__name__, np.mean(s))
    plt.plot(m, label='measured', marker='o')
    plt.plot(t, label='truth', marker='o')
    plt.plot(s, label='score', marker='o')
    plt.legend()
    plt.title(measure_func.__name__)
    plt.ylim([-0.1, 1.1])
    plt.show()

# %% quickly trying some actual measurement methods
saving = {}

gpt35t = LLM("gpt-3.5-turbo-0125") # most up to date 3.5-turbo
gpt4 = LLM("gpt-4")
gpt4t = LLM("gpt-4-turbo-2024-04-09") # most up to date 4-turbo

original_prompt = "This is a reasoning transcript of an agent trying to find the roots of a quadratic equation. The agent will start by attempting to factor the quadratic, and may switch over to using the quadratic formula instead. First you will be shown the full transcript, then just a prefix of the transcript. By the end of the prefix transcript, has the agent switched from factoring to using the quadratic formula yet?"

def original_35t(context, prefix):
    return gpt35t.yesno_completion(original_prompt+'\n\nFULL TRANSCRIPT:\n'+context+'\n\nPREFIX TRANSCRIPT:\n'+prefix+"\n\nANSWER:\n", t=0) == 'Yes'
def original_4(context, prefix):
    return gpt4.yesno_completion(original_prompt+'\n\nFULL TRANSCRIPT:\n'+context+'\n\nPREFIX TRANSCRIPT:\n'+prefix+"\n\nANSWER:\n", t=0) == 'Yes'
def original_4t(context, prefix):
    return gpt4t.yesno_completion(original_prompt+'\n\nFULL TRANSCRIPT:\n'+context+'\n\nPREFIX TRANSCRIPT:\n'+prefix+"\n\nANSWER:\n", t=0) == 'Yes'

# %% run
for measure_func in [original_35t, original_4, original_4t]:
    n=25
    m, t, s = do_test(measure_func, 25, n)
    saving[measure_func.__name__] = (m, t, s, n)
    print(measure_func.__name__, np.mean(s))
    plt.errorbar(x=range(len(m)), y=m, yerr = np.sqrt(m*(1-m)/n), label='measured', marker='o', capsize=5)
    plt.errorbar(x=range(len(t)), y=t, yerr = np.sqrt(t*(1-t)/n), label='truth', marker='o', capsize=5)
    plt.errorbar(x=range(len(s)), y=s, yerr = np.sqrt(s*(1-s)/n), label='score', marker='o', capsize=5)
    plt.legend()
    plt.title(measure_func.__name__)
    plt.ylim([-0.1, 1.1])
    plt.show()

# %% using post-processing in a measurement func
def make_non_decreasing(arr):
    '''
    Takes a binary array, returns the index or list of indices at which it takes the fewest flips to make the array non-decreasing.
    If there are multiple solutions, returns the first one - hopefully the measurements are close enough so sols are unique and unique sols are correct.
    '''
    if all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)): return arr

    table = np.zeros([2, len(arr)+1])
    table[:, 0] = [0, 0]
    for i, element in list(enumerate(arr)):
        if element == 1:
            table[0, i+1] = table[0, i] + 1 # if you don't want to commit to all 1's yet, need to pay 1 to flip
            table[1, i+1] = min(table[0, i], table[1, i]) # no flip needed to commit to 1's now
        else:
            table[0, i+1] = table[0, i] # no flip needed
            table[1, i+1] = table[1, i] + 1 # can  commit to 1's now, if you want

    max_flips = int(min(table[0, -1], table[1, -1]))

    # backtracking to find actual solution
    def backtrack(index, current_flips, arr):
        # Base case: If we've processed the entire array
        if index == len(arr):
            if all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)):
                return arr[:]
            else:
                return None
        # Recursion: Try without flipping
        no_flip = backtrack(index + 1, current_flips, arr)
        if no_flip is not None:
            return no_flip
        # Try flipping if within flip limit
        if current_flips < max_flips:
            arr[index] = 1 - arr[index]  # Flip the bit
            yes_flip = backtrack(index + 1, current_flips + 1, arr)
            if yes_flip is not None:
                return yes_flip
            arr[index] = 1 - arr[index]  # Unflip the bit back

    out = backtrack(0, 0, arr[:]) # Start the recursion with a copy of the array
    return out

# %% run
for measure_func in [original_35t, original_4, original_4t]:
    for post_func in [make_non_decreasing]:
        n=25
        m, t, s = do_test(measure_func, 25, n, post_func=post_func)
        name = measure_func.__name__+' '+post_func.__name__
        saving[name] = (m, t, s, n)
        print(name, np.mean(s))
        plt.errorbar(x=range(len(m)), y=m, yerr = np.sqrt(m*(1-m)/n), label='measured', marker='o', capsize=5)
        plt.errorbar(x=range(len(t)), y=t, yerr = np.sqrt(t*(1-t)/n), label='truth', marker='o', capsize=5)
        plt.errorbar(x=range(len(s)), y=s, yerr = np.sqrt(s*(1-s)/n), label='score', marker='o', capsize=5)
        plt.legend()
        plt.title(name)
        plt.ylim([-0.1, 1.1])
        plt.show()
# %%