# %%
import pandas as pd

# %%
readfile = 'results/gpt3_experiment2_a.csv'
df = pd.read_csv(readfile)
# %%
print(sum(df['one_token_cpc_result'] == df['cot_cpc_result'])/len(df))
print(sum(df['one_token_cpc_result'] == df['did_switch'])/len(df))
print(sum(df['cot_cpc_result'] == df['did_switch'])/len(df))
print(sum(df['one_token_cpc_result'] == 'Yes')/len(df))
print(sum(df['cot_cpc_result'] == 'Yes')/len(df))
print(sum(df['did_switch'] == 'Yes')/len(df))
# %%
