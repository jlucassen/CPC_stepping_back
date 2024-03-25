import json
from dataclasses import dataclass
from openai import OpenAI
import os
import openai
from dotenv import load_dotenv
import sample
from llm import LLM
from solver import perform_one_token_cpc, perform_cot_cpc
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns

@dataclass
class Result:
    context: str
    one_token_cpc_result: str
    cot_cpc_result: str

def load_jsonl_contexts(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            data = json.loads(line)
            yield data['context']

def analyse_contexts(llm, file_path):
    results = []
    for text in load_jsonl_contexts(file_path):
        context_parts = sample.checkpoints(text, 1000)
        for context in context_parts:
            one_token_response = perform_one_token_cpc(llm, context)
            cot_response, _ = perform_cot_cpc(llm, context)
            one_token_switches = one_token_response.lower() == 'yes'
            cot_switches = 'yes' in cot_response.lower()
            print(f"Context: {context.text}")
            print(f"One-token response: {one_token_response}")
            print(f"CoT response: {cot_response}")
            print(f"One-token switches: {one_token_switches}")
            print(f"CoT switches: {cot_switches}")
            print("---")
            results.append({
                'context': context.text,
                'one_token_switches': one_token_switches,
                'cot_switches': cot_switches
            })
    return results

def confusion_matrix(evaluations):
    true_positives = sum(1 for result in evaluations if result['one_token_switches'] and result['cot_switches'])
    true_negatives = sum(1 for result in evaluations if not result['one_token_switches'] and not result['cot_switches'])
    false_positives = sum(1 for result in evaluations if result['one_token_switches'] and not result['cot_switches'])
    false_negatives = sum(1 for result in evaluations if not result['one_token_switches'] and result['cot_switches'])
    return true_positives, true_negatives, false_positives, false_negatives


load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")
llm = LLM("gpt-3.5-turbo")

# files to analyse
file_paths = [
    'quadratic_contexts/quadratic_contexts_5_False.jsonl',
    'quadratic_contexts/quadratic_contexts_15_False.jsonl',
    'quadratic_contexts/quadratic_contexts_25_False.jsonl',
    'quadratic_contexts/quadratic_contexts_35_False.jsonl',
    'quadratic_contexts/quadratic_contexts_45_False.jsonl'
]

results = []
for file_path in file_paths:
    evaluations = analyse_contexts(llm, file_path)
    true_positives, true_negatives, false_positives, false_negatives = confusion_matrix(evaluations)
    total_evaluations = len(evaluations)
    results.append({
        'file_path': file_path,
        'true_positives': true_positives,
        'true_negatives': true_negatives,
        'false_positives': false_positives,
        'false_negatives': false_negatives,
        'total_evaluations': total_evaluations
    })

df = pd.DataFrame(results)
df['TPR'] = df['true_positives'] / (df['true_positives'] + df['false_negatives'])
df['TNR'] = df['true_negatives'] / (df['true_negatives'] + df['false_positives'])
df['FPR'] = df['false_positives'] / (df['false_positives'] + df['true_negatives'])
df['FNR'] = df['false_negatives'] / (df['false_negatives'] + df['true_positives'])

# save CSV
df.to_csv('analysis_results.csv', index=False)


plt.figure(figsize=(14, 8))

#plot each rate
for rate in ['TPR', 'TNR', 'FPR', 'FNR']:
    sns.lineplot(data=df, x='file_path', y=rate, label=rate)

plt.title('Rates by File')
plt.xlabel('File Path')
plt.ylabel('Rate')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()

# Save the plot as an image file before showing it
plt.savefig('rates_analysis.png')

# Now show the plot if in ipynb
#plt.show()

#  print to console
print(df)