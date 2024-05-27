import pandas as pd
from itertools import product
import nltk
import random
import os
from termcolor import colored
from llm import LLM

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

def split_and_judge_switching(context_df):
    pass

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

def main():
    problem_df = cpc_problems(make_caesar_cipher, {'word_length': [5, 6, 7]})
    context_df = cpc_contexts(problem_df, solve_caesar_gpt35t)
    print(context_df)

if __name__ == '__main__':
    main()