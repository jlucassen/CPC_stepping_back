import os
import csv
from dotenv import load_dotenv
from llm import LLM
from solver_yes_no_simple import perform_one_token_cpc
import concurrent.futures
from tqdm import tqdm
import matplotlib.pyplot as plt 
# Load API key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("No OPENAI_API_KEY found in env")

def test_extraction(llm, dataset, function, prompt_variations, n=50):
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
            futures = []
            for question, correct_answer in dataset:
                for prompt_variation in prompt_variations:
                    future = executor.submit(task, question, correct_answer, prompt_variation)
                    futures.append(future)
                    future.add_done_callback(lambda p: pbar.update())
            
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

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
    ax.boxplot(data.values())
    ax.set_xticklabels(data.keys(), rotation=45, ha='right')
    ax.set_ylabel('Accuracy Rate')
    ax.set_title('Accuracy Rate by Prompt Variation')
    plt.tight_layout()
    plt.savefig(filename)


def write_to_csv(results, filename):
    keys = results[0].keys()
    with open(filename, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)

if __name__ == "__main__":
    # Initialize LLM with model details
    llm = LLM(model_name='gpt-3.5-turbo')
    
    # questions for Yes/No
    # yes_no_dataset = [
    #     ("The sky is blue", "Yes"), ("Two plus two equals five", "No"),
    #     ("Does water boiling at 100 degrees Celsius at sea level?", "Yes"),
    #     ("Can humans breathe normally in space without any special gear?","No"),
    #     ("Do all birds have the ability to fly?", "No"), ("Is the sun a star?", "Yes"),
    #     ("Is the Earth flat?","No")
    # ]

    yes_no_dataset =[
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

    # 1t prompt variations
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
    ""
    ]

    # cot_dataset = [
    #     ("Is it possible for a person to have situs inversus, where their organs are mirrored from their normal positions?", "Yes"),
    #     ("Can a Möbius strip be created with only one surface and one edge?", "Yes"),
    #     ("Can a square have the same area as a circle with rational radius?", "No"),
    #     ("Is it possible for a polynomial equation of degree 5 or higher to have no algebraic solution in terms of radicals?", "Yes"),
    #     ("Can a perpetual motion machine of the first kind (which creates energy) be constructed?", "No")
    # ]
    
    # running
    #yes_no_results = test_extraction(llm, yes_no_dataset, perform_one_token_cpc, prompt_variations)
    #cot_results = test_extraction(llm, cot_dataset, perform_cot_cpc, prompt_variations)
    
    # write out
    #write_to_csv(yes_no_results, 'yes_no_results.csv')
    #write_to_csv(cot_results, 'cot_results.csv')

    # create box plot
    create_box_plot(yes_no_results, 'yes_no_accuracy_box_plot.png')