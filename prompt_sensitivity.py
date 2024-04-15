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

def process_query(llm, context, confidence, idx, jdx, one_token_prompt_idx, cot_prompt_idx, outfile, lock):
    formatted_context = context.format(insert=confidence)
    one_token_result = perform_one_token_cpc(llm, formatted_context)  
    _, cot_result = perform_cot_cpc(llm, formatted_context)  
    with lock:
        with open(outfile, 'a', newline='') as f:
            writer = csv.writer(f)
            # log the indices of the prompts used
            writer.writerow([idx, jdx, one_token_prompt_idx, cot_prompt_idx, one_token_result, cot_result])

def make_validation_data(context_type, contexts, confidences, outfile, model='gpt-3.5-turbo', n=1):
    if os.path.exists(outfile):
        print(f"File {outfile} already exists, skipping generation.")
        return

    llm = LLM(model)
    lock = Lock()

    # for multiple prompt configurations per context type
    prompt_configurations = context_to_prompt_indices[context_type]

    for one_token_idx, cot_idx in prompt_configurations:
        # unique folder outputs
        config_outfile = f"{outfile.replace('.csv', '')}_{one_token_idx}_{cot_idx}.csv"

        with open(config_outfile, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Context Index', 'Confidence', 'One Token Prompt Index', 'CoT Prompt Index', 'One Token Result', 'CoT Result'])

        with ThreadPoolExecutor(max_workers=10) as executor:
            futures = []
            for i, context in enumerate(contexts):
                for j, confidence in enumerate(confidences):
                    for _ in range(n):
                        futures.append(executor.submit(process_query, llm, context, confidence, i, j, one_token_idx, cot_idx, config_outfile, lock))

            for future in tqdm(futures):
                future.result()


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



make_validation_data('spoonfeed', contexts['spoonfeed'], verbal_confidences, 'cpc_validation_results_verbal.csv', n=100)
# checking on spoodfeed then will add others