import concurrent.futures as futures
import json

import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from llm import LLM
from solver import perform_cot_cpc

"""
Investigate whether the llm converges to 'Yes' the longer the chain of thought is.
"""
load_dotenv()
gpt35t = LLM("gpt-3.5-turbo")

passages_data = json.load(open("data/passages1.json"))


def fill_percent_yes(llm, row, num_samples):
    """
    This function fills in the rows which have not yet been processed. Row is not yet processed if both
    percent_yes and 'error' columns are absent.
    """
    if 'percent_yes' in row or 'error' in row:
        return row
    try:
        with futures.ThreadPoolExecutor() as executor:
            completions = executor.map(lambda _: perform_cot_cpc(llm, row['context']), range(num_samples))
        thoughts, results = zip(*completions)
        row['percent_yes'] = results.count('Yes') / num_samples
        row['thoughts'] = list(thoughts)
    except Exception as e:
        row['error'] = f"{type(e).__name__}: {str(e)}"
    return row


def experiment3(llm, df, num_samples=20):
    """
    For the given context, how often does the llm say yes to cpc?
    Takes a dataframe with columns 'context', 'percent_yes', 'thoughts', and 'error'.
    """
    result = []
    for i, row in tqdm(df.iterrows(), total=df.shape[0]):
        result.append(fill_percent_yes(llm, row.copy(), num_samples))
    return pd.DataFrame(result)


# %%
passages = pd.DataFrame(
    [{'category': category, 'context': passages[:1000]}
     for category, passage_list in passages_data.items() for passages in passage_list])
# %%
experiment3_gpt35 = experiment3(gpt35t, passages.sample(1))
