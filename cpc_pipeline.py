import pandas as pd
pd.set_option('display.max_columns', None)
import numpy as np
from itertools import product
import os
from termcolor import colored
import matplotlib.pyplot as plt

import concurrent.futures
executor = concurrent.futures.ThreadPoolExecutor()

from llm import LLM
gpt35t = LLM("gpt-3.5-turbo")
gpt4 = LLM("gpt-4")



def myHash(text:str):
  '''
  Hash with settable seed, to allow deterministic hashing between Python instances
  '''
  hash=0
  for ch in text:
    hash = ( hash*281  ^ ord(ch)*997) & 0xFFFFFFFF
  return hash

def cpc_problems(problem_maker, args, n, unwrap_colnames=['problem']):
    filename = 'cpc_pipeline/'+problem_maker.__name__ + str(myHash(problem_maker.__name__+str(args)+str(n)+str(unwrap_colnames)))+'.csv'
    if not os.path.exists(filename):
        print(f"Creating {filename}...")
        df_list = []
        for setting in product(*args.values()):
            for _ in range(n):
                df_list += [[problem_maker(*setting)]+list(setting) ]
        if unwrap_colnames == ['problem']:
            df = pd.DataFrame(df_list, columns=['problem']+list(args.keys()))
        else:
            df_list_unwrapped = [list(sublist[0]) + sublist[1:] for sublist in df_list] # assume problem_maker output is a tuple
            df = pd.DataFrame(df_list_unwrapped, columns=unwrap_colnames+list(args.keys()))
        df.to_csv(filename, index=False)
    else:
        print(colored(f"Reading {filename}...", 'blue'))
        df = pd.read_csv(filename)
    return df

def cpc_contexts(problem_df, row_solver):
    filename = 'cpc_pipeline/'+row_solver.__name__ + str(myHash(str(problem_df)+row_solver.__name__))+'.csv'
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
    filename = 'cpc_pipeline/'+row_split_and_judge.__name__ + str(myHash(str(context_df)+row_split_and_judge.__name__+str(chunk_size)))+'.csv'
    if not os.path.exists(filename):
        print(f"Creating {filename}...")
        row_results = context_df['context'].apply(lambda row: executor.submit(row_split_and_judge, row, chunk_size))
        row_results = row_results.apply(lambda x: x.result()) # list of (prefixes, processed_switching) for each context
        new_df_list = []
        switch_indices = [switching.index(1) if 1 in switching else None for _, switching in row_results]
        for i, (prefixes, processed_switching) in enumerate(row_results):
            row_info = context_df.iloc[i]
            for j, (prefix, switch_val) in enumerate(list(zip(prefixes, processed_switching))):
                new_df_list.append(list(row_info) + [j, prefix, switch_val, switch_indices[i]])
        switching_df = pd.DataFrame(new_df_list, columns=list(context_df.columns)+['index', 'prefix', 'switch', 'switch_index'])
        switching_df.to_csv(filename, index=False)
    else:
        print(colored(f"Reading {filename}...", 'blue'))
        switching_df = pd.read_csv(filename)
    return switching_df

def judge_cpc(switching_df, list_of_cpc_functions, radius_left = -1, radius_right = -1):
    filename = 'cpc_pipeline/cpc_'+str(myHash(str(switching_df)+str([x.__name__ for x in list_of_cpc_functions])+str(radius_left)+str(radius_right)))+'.csv'
    if not os.path.exists(filename):
        print(f"Creating {filename}...")
        switching_df = switching_df.copy()
        switching_df['dist_to_switch'] = switching_df['index'] - switching_df['switch_index']
        if radius_left > -1:
            switching_df = switching_df.loc(switching_df['dist_to_switch'] >= -radius_left) # throw out all entries further left than -radius_left
        if radius_right > -1:
            switching_df = switching_df.loc[switching_df['dist_to_switch'] <= radius_right] # throw out all entries further right than radius_right
        for cpc_function in list_of_cpc_functions:
            switching_df[cpc_function.__name__] = switching_df['prefix'].apply(lambda x: executor.submit(cpc_function, x))
            tuple_holder = switching_df[cpc_function.__name__].apply(lambda x: x.result())
            if isinstance(tuple_holder[0], tuple):
                switching_df[cpc_function.__name__] = tuple_holder.apply(lambda x: x[-1])
                for i in range(len(tuple_holder[0])-1):
                    switching_df[cpc_function.__name__ + '_'+str(i)] = tuple_holder.apply(lambda x: x[i])
            else:
                switching_df[cpc_function.__name__] = tuple_holder
        switching_df.to_csv(filename, index=False)
    else:
        print(colored(f"Reading {filename}...", 'blue'))
        switching_df = pd.read_csv(filename)
    return switching_df

def do_analysis(cpc_df, list_of_cpc_functions):
    df = cpc_df.copy()
    cpc_colnames = [cpc_function.__name__ for cpc_function in list_of_cpc_functions]
    
    fig, axs = plt.subplots(1, len(cpc_colnames), figsize=(10*len(cpc_colnames), 6), sharey=True)

    for i, cpc_colname in enumerate(cpc_colnames):
        ax = axs[i] if len(cpc_colnames) > 1 else axs
        ax.vlines(0, 0, 1, linestyles='dashed', colors='gray')
        ax.set_xlabel('Distance to Switch')
        ax.set_ylabel('Stepback Probability')
        ax.set_ylim(0, 1)
        ax.set_title(cpc_colname)

        df[cpc_colname+'_numeric'] = df[cpc_colname].map({'Yes': 1, 'No': 0})
        grouped = df.groupby('dist_to_switch')[cpc_colname+'_numeric'].agg(['mean', 'count']).reset_index()
        grouped['stdev'] = np.sqrt(grouped['mean'] * (1 - grouped['mean']) / grouped['count'])
        ax.errorbar(grouped['dist_to_switch'], grouped['mean'], yerr=grouped['stdev']*1.96, fmt='o', capsize=5)
    plt.show()