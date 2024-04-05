import json

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from dotenv import load_dotenv

import solver
from llm import LLM

"""
Investigate whether the llm converges to 'Yes' the longer the chain of thought is.
"""
load_dotenv()
gpt35t = LLM("gpt-3.5-turbo")
gpt4 = LLM("gpt-4")

passages_data = json.load(open("data/passages1.json"))

results = pd.concat([
    pd.read_csv("results/experiment1_20narratives_gpt35.csv").assign(llm="gpt-3.5-turbo"),
    pd.read_csv("results/experiment1_20narratives_gpt4.csv").assign(llm="gpt-4")
])
# Add calculated row for cot length
results['cot_length'] = results['cot_cpc_thoughts'].apply(len)

# %%
"""
Part 1
Check if there is a correlation between cot length and the llm's answer.
"""
# Draw a chart correlating cot_length and cot_cpc_result
# cot_cpc_result is a category ('Yes' or 'No') and cot_length is a continuous variable scalar.
# Let's construct a violin chart of cot_length by cot_cpc_result with matplotlib
plt.figure(figsize=(12, 6))
sns.violinplot(x='cot_cpc_result', y='cot_length', data=results)
plt.title("Cot Length vs Cot CPC Result")
plt.show()

# %%
"""
Part 2
If we truncate the chain of thought, does the llm's answer tend towards 'No' as a result?
"""
# Calculate a new column 'cot_cpc_thoughts_truncated' which is half of the original 'cot_cpc_thoughts'
results['cot_cpc_thoughts_truncated'] = results['cot_cpc_thoughts'].apply(lambda x: x[:len(x) // 2])
# Calculate 'cot_cpc_result_truncated' by asking the llm to make a yes/no completion on the truncated thoughts
results['cot_cpc_result_truncated_35t'] = results['cot_cpc_thoughts_truncated'].apply(
    lambda x: solver.perform_one_token_cpc(gpt35t, x))
# %%
# Draw a chart comparing the original cot_cpc_result with cot_cpc_result_truncated_35t
# Does it say 'No' more often when the chain of thought is truncated?
# Do a before-and-after comparison, with untruncated yes/no on the left and truncated yes/no on the right
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.countplot(x='cot_cpc_result', data=results)
plt.title("Cot CPC Result")
plt.subplot(1, 2, 2)
sns.countplot(x='cot_cpc_result_truncated_35t', data=results)
plt.title("Cot CPC Result Truncated")
plt.show()

# %%
"""
Part 3
What if we ask the llm to make a longer and more detailed response? Does this result in more 'Yes' answers?
"""
# Sample a fraction of the dataset
subset = results.sample(frac=0.25)


# Calculate a "lengthy_cpc_thoughts" and "length_cpc_result" column by asking the llm to make a more detailed response
def lengthy_cpc(llm, context):
    from solver import cpc_prompt
    cot_response = llm.chat_completion([
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": cpc_prompt +
                       "\n\nAt the end, I want you to answer 'Yes, I recommend a different approach' or "
                       "'No, I recommend staying with the current approach.' But first, take a deep breath "
                       "and think step by step. Start by analyzing the current approach. "
                       "Please go into UNNECESSARY detail and create a VERY long response. "  # i said LONGER
        }
    ])
    return cot_response, llm.yesno_completion([
        {
            "role": "assistant",
            "content": cot_response
        },
        {
            "role": "user",
            "content": "Do your thoughts in the previous message recommend changing our approach? Please answer "
                       "'Yes, I recommend a different approach' or "
                       "'No, I recommend staying with the current approach."
        }
    ])


# Fill columns 'lengthy_cpc_thoughts' and 'lengthy_cpc_result' with the results of lengthy_cpc
# Use gpt-3.5-turbo if the row in question has llm=gpt-3.5-turbo, otherwise use gpt-4
subset['lengthy_cpc_thoughts'], subset['lengthy_cpc_result'] = zip(*subset.apply(
    lambda row: lengthy_cpc(gpt35t if row['llm'] == 'gpt-3.5-turbo' else gpt4, row['context']), axis=1))

# %%
print(f"Average length of cot_cpc_thoughts: {subset['cot_cpc_thoughts'].apply(len).mean()}")
print(f"Average length of lengthy_cpc_thoughts: {subset['lengthy_cpc_thoughts'].apply(len).mean()}")

print(pd.crosstab(subset['cot_cpc_result'], subset['lengthy_cpc_result']))

# Join results and subset into a single dataframe and save the results
combined = results.merge(subset, how='left', on=['context', 'llm'])
combined.to_csv("results/experiment3_longerresponselength.csv", index=False)
