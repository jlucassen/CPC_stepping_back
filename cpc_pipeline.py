import pandas as pd
pd.set_option('display.max_columns', None)
from itertools import product
import nltk
import random
import os
from termcolor import colored
from llm import LLM
import matplotlib.pyplot as plt

from concurrent.futures import ThreadPoolExecutor

gpt35t = LLM("gpt-3.5-turbo")
gpt4 = LLM("gpt-4")

executor = ThreadPoolExecutor()

def myHash(text:str):
  '''
  Hash with settable seed, to allow deterministic hashing between Python instances
  '''
  hash=0
  for ch in text:
    hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
  return hash

def cpc_problems(problem_maker, args):
    filename = 'cpc_pipeline/'+problem_maker.__name__ + str(myHash(str(args)))+'.csv'
    if not os.path.exists(filename):
        print(f"Creating {filename}...")
        df = pd.DataFrame([[problem_maker(*setting)]+list(setting) for setting in product(*args.values())], columns=['problem']+list(args.keys()))
        df.to_csv(filename, index=False)
    else:
        print(colored(f"Reading {filename}...", 'blue'))
        df = pd.read_csv(filename)
    return df

def cpc_contexts(problem_df, row_solver):
    filename = 'cpc_pipeline/'+row_solver.__name__ + str(myHash(problem_df.to_string()))+'.csv'
    if not os.path.exists(filename):
        print(f"Creating {filename}...")
        problem_df = problem_df.copy()
        problem_df['context'] = problem_df['problem'].apply(lambda x: executor.submit(row_solver, x))
        problem_df['context'] = problem_df['context'].apply(lambda x: x.result())
        problem_df.to_csv(filename, index=False)
    else:
        print(colored(f"Reading {filename}...", 'blue'))
        problem_df = pd.read_csv(filename)
    return problem_df

def split_and_judge_switching(context_df, row_split_and_judge, chunk_size):
    row_results = context_df['context'].apply(lambda row: executor.submit(row_split_and_judge, row, chunk_size))
    row_results = row_results.apply(lambda x: x.result()) # list of (prefixes, processed_switching) for each context
    new_df_list = []
    switch_indices = [switching.index(1) if 1 in switching else len(switching) for _, switching in row_results]
    for i, (prefixes, processed_switching) in enumerate(row_results):
        row_info = context_df.iloc[i]
        for j, (prefix, switch_val) in enumerate(list(zip(prefixes, processed_switching))):
            new_df_list.append(list(row_info) + [j, prefix, switch_val, switch_indices[i]])
    return pd.DataFrame(new_df_list, columns=list(context_df.columns)+['index', 'prefix', 'switch', 'switch_index'])


def judge_cpc():
    pass

def do_analysis():
    pass

def cpc_pipeline():
    pass





filtered_corpus = [s for s in nltk.corpus.abc.words() if s.lower() == s]
def make_caesar_cipher(word_length):
    word = random.choice([s for s in filtered_corpus if len(s) == word_length])
    shift = random.randint(1, 25)
    return ''.join([chr((ord(c) + shift - 97) % 26 + 97) for c in word])

def solve_caesar_gpt35t(problem, prompt=None):
    if prompt is None:
        prompt = "This ciphertext is either an anagram or a Caesar cipher. First, assume it's an anagram, and try to unscramble it. If that doesn't work, assume it's a Caesar cipher and try to decrypt it. Ciphertext: {ciphertext}"
    return gpt35t.chat_completion(prompt.format(ciphertext=problem))

from new_switch_utils import make_non_decreasing

anagram_switching_prompt = "This is a reasoning transcript of an agent trying to decipher some ciphertext. The agent will start treating the ciphertext as an anagram and attempting to unscramble it. The agent may then give up and treat the ciphertext as a Caesar cipher instead. First you will be shown the full transcript, then just a prefix of the transcript. By the end of the prefix transcript, has the agent switched from treating the ciphertext as an anagram to treating it as a Caesar cipher yet?"
def judge_anagram_switching(context, prefix):
        return 1 if gpt4.yesno_completion(anagram_switching_prompt+'\n\nFULL TRANSCRIPT:\n'+context+'\n\nPREFIX TRANSCRIPT:\n'+prefix+"\n\nANSWER:\n") == 'Yes' else 0

def split_and_judge_gpt4(context, chunk_size):
    prefixes = [context[:pref_end] for pref_end in range(chunk_size, len(context)+chunk_size, chunk_size)]
    measured_switching = [judge_anagram_switching(context, prefix) for prefix in prefixes]
    processed_switching = random.choice(make_non_decreasing(measured_switching))
    return prefixes, processed_switching

def main():
    problem_df = cpc_problems(make_caesar_cipher, {'word_length': [5, 6, 7]})
    context_df = cpc_contexts(problem_df, solve_caesar_gpt35t)
    switching_df = split_and_judge_switching(context_df, split_and_judge_gpt4, 100)
    print(switching_df)

if __name__ == '__main__':
    main()