import pandas as pd
from itertools import product

from make_quadratic_problems import make_quadratic_problem


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
    print(cpc_problems(make_quadratic_problem, {'difficulty': range(5, 50, 5), 'factorable': [True, False]}))

if __name__ == '__main__':
    main()