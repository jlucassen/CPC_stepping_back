# %%
import glob
import json
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio

from llm import LLM, RateLimiter
from solver import perform_one_token_cpc, perform_cot_cpc

"""
We interrupt an LLM while it is reasoning through a problem. If we fork the LLM at this point,
explicitly prompting one branch to consider using a different approach, do we see a difference between the two 
branches?
"""

load_dotenv()
gpt35 = LLM('gpt-3.5-turbo', AsyncOpenAI())
gpt4 = LLM('gpt-4', AsyncOpenAI())

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
async def process_row(llm, passage):
    try:
        # Take as context the first 100 characters of passage['context'], not to include the SWITCH token if present
        context_truncated = passage['context']
        if 'SWITCH' in context_truncated:
            context_truncated = context_truncated.split('SWITCH')[0]
        context_truncated = context_truncated[:300]

        one_token_cpc_result = await perform_one_token_cpc(llm, context_truncated)
        cot_cpc_thoughts, cot_cpc_result = await perform_cot_cpc(llm, context_truncated)

        return {
            'difficulty': passage['difficulty'],
            'is_factorizable': passage['is_factorizable'],
            'did_switch': 'Yes' if 'SWITCH' in passage['context'] else 'No',
            'context': context_truncated,
            'equation': passage['equation'],
            'one_token_cpc_result': one_token_cpc_result,
            'cot_cpc_result': cot_cpc_result,
            'cot_cpc_thoughts': cot_cpc_thoughts
        }
    except Exception as e:
        e.add_note(str(passage))
        raise e


async def process_all_rows(llm, passages, output_df: pd.DataFrame = pd.DataFrame()) -> pd.DataFrame:
    """
    :output_df: a dataframe, possibly with previously processed rows, to which the results of processing the given
    `passages` dataframe will be appended
    :return: output_df with the results of processing all rows in the given `passages` dataframe
    """
    # Asynchronously process all rows in the given `passages` dataframe
    completed = 0
    rate_limiter = RateLimiter(4800)

    async def process_row_with_concurrency_limit(llm, passage):
        nonlocal completed
        # Check if the row is already processed and present in the output dataframe
        if not output_df[(output_df['equation'] == passage['equation']) &
                         (output_df['difficulty'] == passage['difficulty']) &
                         (output_df['error'].isnull()) &
                         (output_df['one_token_cpc_result'].notnull()) &
                         (output_df['cot_cpc_result'].notnull())].empty:
            completed += 1
            print(f'Skipped processing passage {completed} of {len(passages)}\n{passage}')
            return None  # Return None if the row is already processed

        async with rate_limiter:
            try:
                result = await process_row(llm, passage)
                completed += 1
                print(f'Completed processing passage {completed} of {len(passages)}\n{passage}')
                return result
            except Exception as e:
                print(f'Error processing row: {type(e)} {str(e)}')
                completed += 1
                return {'equation': passage['equation'], 'difficulty': passage['difficulty'], 'error': str(e)}

    tasks = [process_row_with_concurrency_limit(llm, row) for _, row in passages.iterrows()]
    results = []
    try:
        for future in asyncio.as_completed(tasks):
            result = await future
            if result is not None:  # Only append the result if it's not None
                results.append(result)
                # Save the result to the output dataframe
                output_df = output_df.append(result, ignore_index=True)
    finally:
        return output_df


async def experiment2(llm, passages: pd.DataFrame, results=None) -> pd.DataFrame:
    results = await process_all_rows(llm, passages, results)
    return pd.DataFrame(results)


# %%
test_passages = passages.sample(1)
gpt3_experiment2 = asyncio.run(experiment2(gpt35, test_passages))
gpt4_experiment2 = asyncio.run(experiment2(gpt4, test_passages))

# %%
# Save the results
gpt3_experiment2.to_csv('gpt3_experiment2.csv', index=False)
gpt4_experiment2.to_csv('gpt4_experiment2.csv', index=False)

# %%
# Run for all passages
gpt3_experiment2 = asyncio.run(experiment2(gpt35, passages))
