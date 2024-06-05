import openai
from solver import perform_one_token_cpc, perform_cot_cpc

import os
import csv
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm
from dotenv import load_dotenv 
import itertools

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# debugging line can remove after
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found.")


from llm import LLM
model = 'gpt-3.5-turbo' 
llm = LLM(model)


def make_validation_data(context_list, prompt_indices, confidences, outfile, n=1):
    results_dir = 'results_prompt_sensitivity'
    os.makedirs(results_dir, exist_ok=True)
    
    outfile_path = os.path.join(results_dir, outfile)
    if os.path.exists(outfile_path):
        print(f"File {outfile_path} already exists, skipping generation.")
        return
    
    with open(outfile_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Context Index', 'Confidence', 'Prompt Index for One Token', 'One Token Result', 'Prompt Index for CoT', 'CoT Result'])

    num_requests = n * len(context_list) * len(confidences)
    
    def process_query(idx, confidence, one_token_idx, cot_idx, formatted_context):
        try:
            one_token_output = perform_one_token_cpc(llm, formatted_context, prompt=cpc_prompts[one_token_idx])
            #print(f"One Token Output: {one_token_output}")
            one_token_result = 1 if one_token_output == "No" else 0
            
            cot_output, cot_yesno = perform_cot_cpc(llm, formatted_context, prompt=cpc_prompts[cot_idx])
            #print(f"CoT Output: {cot_output}")
            #print(f"CoT Yes/No: {cot_yesno}")
            cot_result = 1 if cot_yesno == "No" else 0
            
            return idx, confidence, one_token_idx, one_token_result, cot_idx, cot_result
        except Exception as e:
            print(f"Error processing query: {e}")
            return idx, confidence, one_token_idx, None, cot_idx, None

    with ThreadPoolExecutor(max_workers=100) as executor:
        futures = []
        for context_idx, context in enumerate(context_list):
            for confidence in confidences:
                for one_token_idx, cot_idx in prompt_indices:
                    for _ in range(n):
                        formatted_context = context.format(insert=confidence)
                        future = executor.submit(process_query, context_idx, confidence, one_token_idx, cot_idx, formatted_context)
                        futures.append(future)
        
        with tqdm(total=num_requests) as pbar:
            with open(outfile_path, 'a', newline='') as f:
                writer = csv.writer(f)
                for future in futures:
                    idx, confidence, one_token_idx, one_token_result, cot_idx, cot_result = future.result()
                    writer.writerow([idx, confidence, one_token_idx, one_token_result, cot_idx, cot_result])
                    pbar.update(1)



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
    'batna': [
        base+'I am trying to make money. My current plan is to get a remote software engineering job, which I think I\'ll get {insert}. Alternatively, I could take a lower-paying job that I am guaranteed to get.',
        base+'I am working on proving a theorem. I have a proof strategy that I think {insert} will be applicable. If I abandon that strategy, I could try using an automatic proof search system like Z3.',
        base+'I am trying to fix my leaky faucet. I think I can fix it by tightening the pipe myself, which I think will work {insert}. If that doesn\'t work, I could call a plumber.',
    ]

}
   

cpc_prompts = {
    0: "Would it be better to switch to a different strategy? Yes or No.",
    1: "Do you think a different approach would solve this problem more efficiently? Yes or No.",
    2: "Would it be worthwhile to change plans and solve this problem another way? Yes or No.",
    3: "Do you think some other method would be more effective? Yes or No.",
}





# applying
context_to_prompt_indices = {
    'spoonfeed': [(0, 0), (1, 1),(2,2),(3,3)],
    'hints': [(0, 0), (1, 1),(2,2),(3,3)],
    'red_herrings': [(0, 0), (1, 1),(2,2),(3,3)],
    'knapsack': [(0, 0), (1, 1),(2,2),(3,3)],
    'batna': [(0, 0), (1, 1),(2,2),(3,3)]
}



model = 'gpt-3.5-turbo' 
llm = LLM(model)
make_validation_data(contexts['spoonfeed'], context_to_prompt_indices['spoonfeed'], numerical_confidences, 'cpc_validation_spoonfeed.csv', n=10)
make_validation_data(contexts['hints'], context_to_prompt_indices['hints'],  numerical_confidences, 'cpc_validation_hints.csv', n=10)
make_validation_data(contexts['red_herrings'], context_to_prompt_indices['red_herrings'], numerical_confidences, 'cpc_validation_red_herrings.csv', n=10)
make_validation_data(contexts['batna'], context_to_prompt_indices['batna'], numerical_confidences, 'cpc_validation_batna.csv', n=10)

# model = 'gpt-4'
# llm = LLM(model)
# make_validation_data(contexts['spoonfeed'], context_to_prompt_indices['spoonfeed'], numerical_confidences, '4_cpc_validation_spoonfeed.csv', n=10)
# make_validation_data(contexts['hints'], context_to_prompt_indices['hints'],  numerical_confidences, '4_cpc_validation_hints.csv', n=10)
# make_validation_data(contexts['red_herrings'], context_to_prompt_indices['red_herrings'], numerical_confidences, '4_cpc_validation_red_herrings.csv', n=10)
# make_validation_data(contexts['batna'], context_to_prompt_indices['batna'], numerical_confidences, '4_cpc_validation_batna.csv', n=10)