# %%
import concurrent.futures as futures
import glob
import json

import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv

from llm import LLM
from solver import perform_one_token_cpc, perform_cot_cpc

"""
We interrupt an LLM while it is reasoning through a problem. If we fork the LLM at this point,
explicitly prompting one branch to consider using a different approach, do we see a difference between the two 
branches?
"""

load_dotenv()
gpt35 = LLM('gpt-3.5-turbo')
gpt4 = LLM('gpt-4')

quadratic_contexts_glob = glob.glob('data/quadratic_contexts_gpt35turbo/*.jsonl')
passages = []
for filename in quadratic_contexts_glob:
    # filename has format quadratic_contexts_[difficulty]_[is_factorizable].jsonl
    # Each jsonl file is a list of objects, one json object per line. each object has 'equation' and 'context' field
    split_name = filename.split('_')
    difficulty = split_name[-2]
    is_factorizable = split_name[-1].split('.')[0]

    with open(filename, 'r') as f:
        for line in f:
            item = json.loads(line)
            passages.append({
                'difficulty': difficulty,
                'is_factorizable': 'Yes' if is_factorizable == 'True' else 'No',
                'context': item['context'],
                'equation': item['equation']
            })
passages = pd.DataFrame(passages)


# %%
def process_row(llm, row: dict) -> dict:
    if 'one_token_cpc_result' in row or 'cot_cpc_result' in row or 'error' in row:
        print(f'Skipped processing passage {row["equation"]} because it was already processed')
        return row

    try:
        # Take as context the first part of the passage['context'], or the portion of the context
        # before the SWITCH token, whichever is shorter
        context_truncated = row['context']
        if 'SWITCH' in context_truncated:
            context_truncated = context_truncated.split('SWITCH')[0]
        context_truncated = context_truncated[:300]

        one_token_cpc_result = perform_one_token_cpc(llm, context_truncated)
        cot_cpc_thoughts, cot_cpc_result = perform_cot_cpc(llm, context_truncated)

        row['one_token_cpc_result'] = one_token_cpc_result
        row['cot_cpc_result'] = cot_cpc_result
        row['cot_cpc_thoughts'] = cot_cpc_thoughts
        row['did_switch'] = 'Yes' if 'SWITCH' in row['context'] else 'No'
        row['context_truncated'] = context_truncated
        return row
    except Exception as e:
        print(f'Error processing row: {type(e)} {str(e)}')
        row['error'] = f'{type(e)} {str(e)}'
        return row


def experiment2(llm, passages: pd.DataFrame) -> pd.DataFrame:
    """
    Processes the rows in the given dataframe which have not already been processed.

    :dataframe: a dataframe with at least columns 'equation', 'difficulty', 'is_factorizable', and 'context'.
    Some rows may already have been processed, in which case they will have 'one_token_cpc_result' and 'cot_cpc_result'
    columns, or else a value in the 'error' column.
    :return: dataframe with the results of processing all rows in the given `passages` dataframe
    """
    # turn passages into a rows dict including index (id, numerical)
    rows = []
    try:
        with futures.ThreadPoolExecutor() as executor:
            row_futures = [executor.submit(process_row, llm, row) for row in passages.to_dict(orient='records')]
            for future in tqdm(futures.as_completed(row_futures), total=len(row_futures)):
                if future.exception():
                    print(f'Error processing row: {type(future.exception())} {str(future.exception())}')
                    continue
                rows.append(future.result())
    except Exception as e:
        print(f'Error processing rows: {type(e)} {str(e)}')
    finally:
        return pd.DataFrame(rows)


# %%
# Test
test_passages = passages.sample(2)
gpt3_experiment2 = experiment2(gpt35, test_passages)
# gpt4_experiment2 = experiment2(gpt4, test_passages)

# %%
# Save the results
gpt3_experiment2.to_csv('gpt3_experiment2.csv', index=False)
# gpt4_experiment2.to_csv('gpt4_experiment2.csv', index=False)

# %%
# Run for all passages
gpt3_experiment2 = experiment2(gpt35, passages)
