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

cpc_prompts = {
    1: "At this point, we're going to stop and consider whether this approach is working or not and leads to a correct solution to the problem being worked on. Otherwise, we should step back and try a different approach.",
    2: "At this stage, we need to pause and evaluate if our current method is effective and if it's leading to a proper solution to the issue at hand. If not, we might need to reconsider and explore an alternative strategy.",
    3: "Right now, it's important to halt and assess whether our strategy is successful and solving the problem we're addressing. If it isn't, we should consider taking a step back and attempting another approach.",
    4: "Currently, we should stop and determine if this strategy is effective and results in a viable solution to the problem. If it proves inadequate, we need to retreat and try a different method.",
}

# %%
gpt = LLM('gpt-3.5-turbo')
quadratic_contexts_glob = glob.glob('data/quadratic_contexts_3_ex1/*.jsonl')

# gpt = LLM('gpt-4')
# quadratic_contexts_glob = glob.glob('data/quadratic_contexts_4/*.jsonl')

# %%
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
            for one_token_prompt in cpc_prompts:
                for cot_prompt in cpc_prompts:
                    passages.append({
                        'difficulty': difficulty,
                        'is_factorizable': 'Yes' if is_factorizable == 'True' else 'No',
                        'context': item['context'],
                        'equation': item['equation'],
                        'one_token_prompt': one_token_prompt,
                        'cot_prompt': cot_prompt
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
        # edited to avoid saying we should switch if the switch occurs at 900 and the interruption is at 300
        context_truncated = row['context']
        row['did_switch'] = 'No'
        if 'switch' in context_truncated.lower():
            switch_index = context_truncated.lower().index('switch')
            context_truncated = context_truncated[:switch_index]
            row['did_switch'] = 'Yes'
        else:
            context_truncated = context_truncated[:int(len(context_truncated) / 2)]

        one_token_cpc_result = perform_one_token_cpc(llm, context_truncated, cpc_prompts[row['one_token_prompt']])
        cot_cpc_thoughts, cot_cpc_result = perform_cot_cpc(llm, context_truncated, cpc_prompts[row['cot_prompt']])

        row['one_token_cpc_result'] = one_token_cpc_result
        row['cot_cpc_result'] = cot_cpc_result
        row['cot_cpc_thoughts'] = cot_cpc_thoughts
        
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
test_passages = passages.iloc[:10]
gpt3_experiment2 = experiment2(gpt, test_passages)
# gpt4_experiment2 = experiment2(gpt4, test_passages)

# %%
# Run for all passages
gpt3_experiment2 = experiment2(gpt, passages.sample(100))
# Save the results
gpt3_experiment2.to_csv('gpt3_experiment2_a.csv', index=False)
# gpt4_experiment2.to_csv('gpt4_experiment2.csv', index=False)
# %%
