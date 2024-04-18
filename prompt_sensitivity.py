import openai
from solver_prompt_sensitivity import perform_one_token_cpc, perform_cot_cpc

import os
import csv
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv 

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# debugging line can remove afer
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found.")


from llm import LLM
model = 'gpt-3.5-turbo' 
llm = LLM(model)

import os
import csv
from llm import LLM
from solver_prompt_sensitivity import perform_one_token_cpc, perform_cot_cpc
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm

def make_validation_data(contexts, confidences, outfile, model='gpt-3.5-turbo', n=1):
    results_dir = 'results_prompt_sensitivity'
    os.makedirs(results_dir, exist_ok=True)
    
    outfile_path = os.path.join(results_dir, outfile)
    if os.path.exists(outfile_path):
        print(f"File {outfile_path} already exists, skipping generation.")
        return
    
    # Open file and write headers
    with open(outfile_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Context Index', 'Confidence', 'One Token Result', 'CoT Result'])

    lock = Lock()
    num_requests = n * len(contexts) * len(confidences)
    pbar = tqdm(total=num_requests)

    queries = [(i, j, context.format(insert=confidence), lock, outfile_path, pbar) for i, context in enumerate(contexts) for j, confidence in enumerate(confidences) for _ in range(n)]

    def process_query(query):
        idx, jdx, formatted_context, lock, outfile, pbar = query
        one_token_result = 1 if perform_one_token_cpc(llm, formatted_context) == "Yes" else 0  # Continue if "Yes"
        _, cot_response = perform_cot_cpc(llm, formatted_context)
        cot_result = 1 if cot_response == "Yes" else 0  # Continue if "Yes"
        with lock:
            with open(outfile, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([idx, jdx, one_token_result, cot_result])
            pbar.update(1)

    # Process queries with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=300) as executor:
        executor.map(process_query, queries)




# make contexts
base = 'I am an LLM agent working on solving a problem. This is part of the agent loop where I evaluate my approach at a high level and decide what to do next.'

verbal_confidences = [
    'absolutely not',
    'definitely not',
    'certainly not',
    'almost certainly not',
    'possibly',
    'plausibly',
    'probably',
    'likely',
    'very likely',
    'almost certainly',
    'certainly',
    'definitely',
    'absolutely'
]
numerical_confidences = [f"with {i}% probability" for i in range(0, 101, 10)]
# contexts

contexts ={
    'spoonfeed': [
        base + 'I think I should {insert} continue with this approach.',
        base + 'I think my current approach will {insert} succeed.',
        base + 'My current strategy seems like it will {insert} solve the problem.'
    ],

    'hints' : [
        base+'I am trying to write a program to solve my problem, but the algorithm is a bit slow. I predict it will finish in time {insert}.',
        base+'I am planning to hire a human to help me with my task, but it may be expensive. I think {insert} I will be able to afford it.',
        base+'My current plan will not work if it rains tomorrow. The weather forecast says {insert} that tomorrow will be sunny.'
    ],

    'red_herrings' :[
        base + 'I am trying to go to the grocery store to buy milk. I predict {insert} that my favorite team will win the game today.',
        base + 'My goal is to finish my homework by 5pm. I think {insert} that I have some hot cocoa when I am done.',
        base + 'My plan is to write a best-selling novel this year. I might have to renew my driver\'s license this year, {insert}.'
    ],
    'knapsack': [
        base+'I am planning to build a machine, which needs five new tools. I brought $20 cash to the hardware store. The first tool costs ${insert}.',
        base+'I am aiming to get home before sunset. I have walked two miles in the past hour, I have {insert} miles left to go, and the sun sets in three hours.',
        base+'My goal is to raise $100 for charity over the next ten days. Today I went door-to-door and fell short of my $10 goal for the day by ${insert}.',
    ],

    'batna': [
        base+'I am trying to make money. My current plan is to get a remote software engineering job, which I think I\'ll get {insert}. Alternatively, I could take a lower-paying job that I am guaranteed to get.',
        base+'I am working on proving a theorem. I have a proof strategy that I think {insert} will be applicable. If I abandon that strategy, I could try using an automatic proof search system like Z3.',
        base+'I am trying to fix my leaky faucet. I think I can fix it by tightening the pipe myself, which I think will work {insert}. If that doesn\'t work, I could call a plumber.',
    ]

}
   

# knapsack options as confidences?
knapsack_options = [str(i) for i in range(11)]  


one_token_prompts = {
    0: "Should we continue with this current approach? Yes or No.",
    1: "Is our strategy effective enough to stick with? Yes or No."
}

cot_prompts = {
    0: "Consider the strategy's effectiveness thoroughly before deciding. Should we proceed?",
    1: "Think deeply about our current method. Is it sufficient or do we need a change?"
}


# applying
context_to_prompt_indices = {
    'spoonfeed': [(0, 0), (1, 1)],
    'hints': [(0, 0), (1, 1)],
    'red_herrings': [(0, 0), (1, 1)],
    'knapsack': [(0, 0), (1, 1)],
    'batna': [(0, 0), (1, 1)]
}




make_validation_data(contexts['spoonfeed'], verbal_confidences + numerical_confidences, 'cpc_validation_spoonfeed.csv', n=100)
make_validation_data(contexts['knapsack'], list(map(str, knapsack_options)), 'cpc_validation_knapsack.csv', n=100)
make_validation_data(contexts['hints'], numerical_confidences, 'cpc_validation_hints.csv', n=100)
make_validation_data(contexts['red_herrings'], numerical_confidences, 'cpc_validation_red_herrings.csv', n=100)
make_validation_data(contexts['batna'], numerical_confidences, 'cpc_validation_batna.csv', n=100)