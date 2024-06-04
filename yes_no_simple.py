import os
import csv
import logging
from dotenv import load_dotenv
from llm import LLM
from solver_yes_no_simple import perform_one_token_cpc, perform_cot_cpc
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats

# debug
logging.basicConfig(filename='output.log', level=logging.INFO)

# API
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logging.error("No OPENAI_API_KEY found in env")
    raise ValueError("No OPENAI_API_KEY found in env")

def test_extraction(llm, dataset, function, prompt_variations, n=5):
    results = []
    total_iterations = len(dataset) * len(prompt_variations) * n

    def task(question, correct_answer, prompt_variation):
        accuracy_sum = 0
        for _ in range(n):
            response = function(llm, question, prompt_variation)
            correct = (response.strip().lower() == correct_answer.lower())
            accuracy_sum += int(correct)
        accuracy_rate = (accuracy_sum / n) * 100
        return {
            'question index': dataset.index((question, correct_answer)),
            'question': question,
            'prompt phrasing': prompt_variation,
            'response': response,
            'correct response': correct_answer,
            'accuracy rate': accuracy_rate
        }
    
    with tqdm(total=total_iterations, desc="Processing") as pbar:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_task = {
                executor.submit(task, question, correct_answer, prompt_variation): (question, correct_answer, prompt_variation)
                for question, correct_answer in dataset
                for prompt_variation in prompt_variations
            }
            for future in concurrent.futures.as_completed(future_to_task):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logging.error(f"Task for {future_to_task[future]} raised an exception: {e}")
                pbar.update(1)

    return results

def create_box_plot(results, filename):
    data = {}
    for result in results:
        prompt_variation = result['prompt phrasing']
        accuracy_rate = result['accuracy rate']
        if prompt_variation not in data:
            data[prompt_variation] = []
        data[prompt_variation].append(accuracy_rate)
        
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.boxplot([[float(x) for x in l] for l in list(data.values())], showfliers=False)
    ax.set_xticks(range(1, len(data.keys())+1))
    #ax.set_xticklabels(data.keys(), rotation=45, ha='right')
    ax.set_ylabel('Accuracy Rate')
    ax.set_xlabel('Prompt Variation')
    ax.set_ylim(0, 105)
    ax.set_title(f'Accuracy Rate by Prompt Variation, {filename[:filename.index("_")]}, {"one_token" if "yes" in filename else "cot"}')
    plt.tight_layout()
    plt.savefig(filename)

def write_to_csv(results, filename):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

def create_confidence_interval_plot(results, filename):
    data = {}
    for result in results:
        prompt_variation = result['prompt phrasing']
        accuracy_rate = result['accuracy rate']
        if prompt_variation not in data:
            data[prompt_variation] = []
        data[prompt_variation].append(accuracy_rate)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x_labels = []
    means = []
    lower_errors = []
    upper_errors = []
    
    for prompt_variation, accuracy_rates in data.items():
        x_labels.append(prompt_variation)
        mean = np.mean(accuracy_rates)
        ci = stats.norm.interval(0.95, loc=mean, scale=stats.sem(accuracy_rates))
        means.append(mean)
        lower_errors.append(mean - ci[0])
        upper_errors.append(ci[1] - mean)
    
    yerr = [lower_errors, upper_errors]
    
    ax.errorbar(range(len(x_labels)), means, yerr=yerr, fmt='o')
    ax.set_xticks(range(len(x_labels)))
    ax.set_xticklabels(x_labels, rotation=45, ha='right')
    ax.set_ylabel('Accuracy Rate')
    ax.set_ylim(0, 100)
    ax.set_title('95% Confidence Interval for Accuracy Rate by Prompt Variation')
    plt.tight_layout()
    plt.savefig(filename)

if __name__ == "__main__":
    # prompt variations
    yes_no_dataset = [
        ("Is there a prime number larger than any given finite set of primes?", "Yes"),
        ("Is it possible for a person to survive without a brain?", "No"),
        ("Can a person's blood type change over the course of their lifetime?", "No"),
        ("Can a person's fingerprints change naturally over the course of their lifetime?", "No"),
        ("Is it possible for a person to have two different colored eyes?", "Yes"),
        ("Is it possible to find a perfect cuboid (a rectangular box with integer side lengths, face diagonals, and body diagonal)?", "No"),
        ("Is it possible for a person to be allergic to water?", "Yes"),
        ("Is it possible to trisect an angle using only a compass and straightedge?", "No"),
        ("Can a polynomial equation of degree n always be solved using radicals for n greater than 4?", "No")
    ]

    prompt_variations = [
        "Please respond with 'Yes' if you agree or 'No' if you disagree.",
        "Respond with 'Yes' if you agree or 'No' if you disagree.",
        "Please respond with Yes if you agree or No if you disagree.",
        "Respond Yes or No.",
        "Do you agree? If yes, please say 'Yes'. If no, please say 'No'.",
        "Simply state 'Yes' if you find this correct, or 'No' if incorrect.",
        "Indicate agreement with 'Yes' or disagreement with 'No'.",
        "Can you confirm this statement? Answer 'Yes' or 'No'.",
        "Is the above statement true? Respond 'Yes' or 'No'.",
        "Just let me know, is this true? Yes or No?",
    ]

    cot_dataset = [
        ("Is the square of the sum of 5 and 3 equal to 64?", "No"),
        ("If you divide 48 by 6, is the quotient 8?", "Yes"),
        ("Does multiplying 7 by 9 give a product of 56?", "No"),
        ("Is the difference between 20 and 12 equal to 8?", "Yes"),
        ("When you subtract 15 from 30, do you get 25?", "No"),
        ("If you double 9 and then add 3, is the result 21?", "Yes"),
        ("Does dividing 100 by 4 give a quotient of 25?", "Yes"),
        ("If you add 17 and 8, is the sum less than 30?", "Yes"),
        ("Is the product of 6 and 5 equal to 30?", "Yes"),
        ("When you subtract 7 from 15, is the result greater than 5?", "Yes")
    ]

    # all models
    model_names = ['gpt-3.5-turbo', 'gpt-4', 'gpt-4-turbo']

    for model_name in model_names:
        llm = LLM(model_name=model_name)
        
        
        # yes_no_results = test_extraction(llm, yes_no_dataset, perform_one_token_cpc, prompt_variations)
        # write_to_csv(yes_no_results, f'{model_name}_yes_no_results.csv')
        # #yes_no_results = csv.DictReader(open(f'{model_name}_yes_no_results.csv'))
        # create_box_plot(yes_no_results, f'{model_name}_yes_no_accuracy_box_plot.png')
        # create_confidence_interval_plot(yes_no_results, f'{model_name}_yes_no_confidence_intervals.png')
        
        cot_results = test_extraction(llm, cot_dataset, perform_cot_cpc, prompt_variations)
        write_to_csv(cot_results, f'{model_name}_cot_results.csv')
        # cot_results = csv.DictReader(open(f'{model_name}_cot_results.csv'))
        create_box_plot(cot_results, f'{model_name}_cot_accuracy_box_plot.png')
        create_confidence_interval_plot(cot_results, f'{model_name}_cot_confidence_intervals.png')