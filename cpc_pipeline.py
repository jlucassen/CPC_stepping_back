import pandas as pd
from itertools import product
import nltk
import random

from make_quadratic_problems import make_quadratic_problem

filtered_corpus = [s for s in nltk.corpus.abc.words() if s.lower() == s]
def make_caesar_cipher(word_length):
    word = random.choice([s for s in filtered_corpus if len(s) == word_length])
    shift = random.randint(1, 25)
    return ''.join([chr((ord(c) + shift - 97) % 26 + 97) for c in word]), word

def cpc_problems(problem_maker, args):
    return pd.DataFrame([[problem_maker(*setting)]+list(setting) for setting in product(*args.values())], columns=['problem']+list(args.keys()))

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