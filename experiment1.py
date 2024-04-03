import json
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

import sample
from judge import JudgeResult
from llm import LLM
from solver import perform_one_token_cpc, perform_cot_cpc


"""
When the ai considers whether to step back or not, does its one-word answer differ from its CoT answer?
"""

@dataclass
class Result:
    context: str
    one_token_cpc_result: str
    cot_cpc_result: str


load_dotenv()
llm = LLM(OpenAI(), "gpt-3.5-turbo")

# Each 'passage' is a lengthy text where we are reasoning through a problem.
# Consider progressively larger context parts of each passage (iterating through 'checkpoints' in the passage):
passages = json.load(open("data/passages1.json"))
checkpoints = (text for document in passages for text in sample.checkpoints(document, 1000))

# and for each context part, ask the llm if the current approach is working or not
results = (
    Result(
        context=context,
        one_token_cpc_result=perform_one_token_cpc(llm, context),
        cot_cpc_result=perform_cot_cpc(llm, context)
    )
    for context in checkpoints
)

# For each result, determine whether the result is good (the two cpc methods agreed) or bad (they disagreed)
evaluations = (
    JudgeResult(
        result=result,
        score=1.0 if result.one_token_cpc_result.lower() == result.cot_cpc_result.lower() else 0.0
    )
    for result in results
)

for evaluation in evaluations:
    print(f"For context ending in '{evaluation.result.context.text[-70:]}'...")
    print(f"one_token_cpc_result={evaluation.result.one_token_cpc_result}, "
          f"cot_cpc_result={evaluation.result.cot_cpc_result}, score={evaluation.score}")
