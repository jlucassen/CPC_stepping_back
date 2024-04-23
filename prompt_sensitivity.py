from solver import perform_one_token_cpc, perform_cot_cpc

import os
import csv
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
from dotenv import load_dotenv 

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# debugging line can remove after
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable not found.")

from llm import LLM
model = 'gpt-3.5-turbo' 
llm = LLM(model)

def make_validation_data(context_list, confidences, outfile, n=1):
    prompt_indices = [(0, 0), (1, 1), (2, 2), (3, 3)]

    # set up outfile
    results_dir = 'results_prompt_sensitivity'
    os.makedirs(results_dir, exist_ok=True)
    outfile_path = os.path.join(results_dir, outfile)
    if os.path.exists(outfile_path):
        print(f"File {outfile_path} already exists, skipping generation.")
        return
    with open(outfile_path, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Context Index', 'Confidence', 'Prompt Index for One Token', 'One Token Result', 'Prompt Index for CoT', 'CoT Result'])

    # set up concurrency
    def process_query(idx, confidence, one_token_idx, cot_idx, formatted_context):
        try:
            one_token_output = perform_one_token_cpc(llm, formatted_context)#, prompt_override=one_token_prompts[one_token_idx])
            one_token_result = 1 if one_token_output == "Yes" else 0 # denote stepping back as 1, not stepping back as 0
            
            cot_output, cot_yesno = perform_cot_cpc(llm, formatted_context)#, prompt_override=cot_prompts[cot_idx])
            cot_result = 1 if cot_yesno == "Yes" else 0
            
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
        
        with open(outfile_path, 'a', newline='') as f:
            writer = csv.writer(f)
            for future in tqdm(futures):
                idx, confidence, one_token_idx, one_token_result, cot_idx, cot_result = future.result()
                writer.writerow([idx, confidence, one_token_idx, one_token_result, cot_idx, cot_result])

# make contexts
base = 'I am an LLM agent working on solving a problem. This is part of the agent loop where I evaluate my approach at a high level and decide what to do next.'
confidences = [f"{i}" for i in range(0, 101, 10)]
knapsack_options = [str(i) for i in range(11)]

# the higher the value of {insert}, the better the current strategy, the LOWER the stepback rate.
contexts = {
    'spoonfeed':
    [
        base+'I think I should {insert} continue with this approach.',
        base+'I think my current approach will {insert} succeed.',
        base+'My current strategy seems like it will {insert} solve the problem.',
    ],

    'hints':
    [
        base+'I am trying to write a program to solve my problem, but the algorithm is a bit slow. I predict it will finish in time {insert}.',
        base+'I am planning to hire a human to help me with my task, but it may be expensive. I think {insert} I will be able to afford it.',
        base+'My current plan will not work if it rains tomorrow. The weather forecast says {insert} that tomorrow will be sunny.',
    ],

    'red_herrings':
    [
        base+'I am trying to go to the grocery store to buy milk. I predict {insert} that my favorite team will win the game today.',
        base+'My goal is to finish my homework by 5pm. I think {insert} that I have some hot cocoa when I am done.',
        base+'My plan is to write a best-selling novel this year. I might have to renew my driver\'s license this year, {insert}.',
    ],
    'batna':
    [
        base+'I am trying to make money. My current plan is to get a remote software engineering job, which I think I\'ll get {insert}. Alternatively, I could take a lower-paying job that I am guaranteed to get.',
        base+'I am working on proving a theorem. I have a proof strategy that I think {insert} will be applicable. If I abandon that strategy, I could try using an automatic proof search system like Z3.',
        base+'I am trying to fix my leaky faucet. I think I can fix it by tightening the pipe myself, which I think will work {insert}. If that doesn\'t work, I could call a plumber.',
    ],
    'knapsack':
    [
        base+'I am planning to build a machine, which needs five new tools. I brought $20 cash to the hardware store. The first tool costs ${insert}.',
        base+'I am aiming to get home before sunset. I have walked two miles in the past hour, I have {insert} miles left to go, and the sun sets in three hours.',
        base+'My goal is to raise $100 for charity over the next ten days. Today I went door-to-door and fell short of my $10 goal for the day by ${insert}.',
    ],
}

# candidate cpc elicitation prompts
one_token_prompts = {
    0: "Is this strategy failing to meet my expectations? Yes or No.",
    1: "Does this plan inadequately address my goals? Yes or No.",
    2: "Should I consider a new approach due to the shortcomings of the present one? Yes or No.",
    3: "Is my current method failing us in achieving my objectives? Yes or No."
}
cot_prompts = {
    0: "Let me take some time to reflect critically on my current approach. Does it fall short of what is required? Provide a detailed analysis before concluding with Yes to switch, or No to proceed.",
    1: "Let me take some time to analyze the effectiveness of my strategy. Is it adequate for my needs or must I explore other options? Yes for new options, No to continue as is.",
    2: "Let me take some time to assess the sufficiency of my current plan. Are there critical areas where it fails? Conclude with Yes if a new strategy is needed, No if it remains viable.",
    3: "Let me take some time to deliberate on my current methodology. Is it proving to be suboptimal for my goals? End with Yes to abandon it, or No to keep it."
}

n = 10
make_validation_data(contexts['spoonfeed'], confidences, '3_cpc_validation_spoonfeed.csv', n=n)
make_validation_data(contexts['hints'], confidences, '3_cpc_validation_hints.csv', n=n)
make_validation_data(contexts['red_herrings'], confidences, '3_cpc_validation_red_herrings.csv', n=n)
make_validation_data(contexts['batna'], confidences, '3_cpc_validation_batna.csv', n=n)
make_validation_data(contexts['knapsack'], knapsack_options, '3_cpc_validation_knapsack.csv', n=n)

# from llm import LLM
# model = 'gpt-3.5-turbo' 
# llm = LLM(model)

# n = 1
# make_validation_data(contexts['spoonfeed'], confidences, '4_cpc_validation_spoonfeed.csv', n=n)
# make_validation_data(contexts['hints'], confidences, '4_cpc_validation_hints.csv', n=n)
# make_validation_data(contexts['red_herrings'], confidences, '4_cpc_validation_red_herrings.csv', n=n)
# make_validation_data(contexts['batna'], confidences, '4_cpc_validation_batna.csv', n=n)
# make_validation_data(contexts['knapsack'], knapsack_options, '4_cpc_validation_knapsack.csv', n=n)