import pandas as pd
from itertools import product
import nltk
import random
import os
from termcolor import colored
import dotenv

dotenv.load_dotenv()

from make_quadratic_problems import make_quadratic_problem

filtered_corpus = [s for s in nltk.corpus.abc.words() if s.lower() == s]
def make_caesar_cipher(word_length):
    word = random.choice([s for s in filtered_corpus if len(s) == word_length])
    shift = random.randint(1, 25)
    return ''.join([chr((ord(c) + shift - 97) % 26 + 97) for c in word]), word

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

def cpc_contexts():
    pass

def split_and_judge_switching():
    pass

def judge_cpc():
    pass

def do_analysis():
    pass

def cpc_pipeline():
    pass

def main():
    print(cpc_problems(make_caesar_cipher, {'word_length': [5, 6, 7]}))

if __name__ == '__main__':
    main()