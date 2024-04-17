from concurrent import futures

import pandas as pd
from pandas import DataFrame

from llm import LLM

"""See if the answer differs one-word vs CoT"""

wouldyourather_data = pd.read_csv('data/wouldyourather.csv')
gpt35turbo = LLM('gpt-3.5-turbo')


def onetoken_wouldyourather(llm, row):
    prompt = f'Would you rather:\nOption A) {row["option_a"]}\nOption B) {row["option_b"]}\nOption'
    return ab_completion(llm, prompt)


def ab_completion(llm, prompt):
    # Force 'A' (32) or 'B' (33)
    return llm.single_token_completion(prompt, {'32': 100, '33': 100})


def cot_wouldyourather(llm, row):
    prompt = {
        'role': 'user',
        'content': f'Would you rather:\nOption A) {row["option_a"]}\nOption B) {row["option_b"]}\n' +
                   'Take a moment to show your work before you respond.'
    }
    cot = llm.chat_completion([prompt])
    answer = ab_completion(llm, [
        prompt,
        {
            'role': 'assistant',
            'content': cot
        },
        {
            'role': 'user',
            'content': 'In your previous message did you choose Option A or Option B? Option'
        }
    ])
    return cot, answer


def process_row(llm, row: dict) -> dict:
    if 'one_token_cpc_result' in row or 'cot_cpc_result' in row or 'error' in row:
        print(f'Skipped processing row {row["id"]} because it was already processed')
        return row

    try:
        one_token_result = onetoken_wouldyourather(llm, row)
        cot_thoughts, cot_result = cot_wouldyourather(llm, row)

        row['one_token_result'] = one_token_result
        row['cot_result'] = cot_result
        row['cot_thoughts'] = cot_thoughts
        return row
    except Exception as e:
        print(f'Error processing row: {type(e)} {str(e)}')
        row['error'] = f'{type(e)} {str(e)}'
        return row


def experiment4(llm, data: DataFrame):
    with futures.ThreadPoolExecutor() as executor:
        data.reset_index(inplace=True)
        return DataFrame(executor.map(lambda row: process_row(llm, row), data.to_dict('records')))


# %%
test_wouldyourather = wouldyourather_data.sample(5)
experiment4_wouldyourather = experiment4(gpt35turbo, test_wouldyourather)

# %% save as csv file
experiment4_wouldyourather.to_csv('results/experiment4_wouldyourather.csv', index=False)
