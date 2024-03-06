import json
from dataclasses import dataclass

from openai import OpenAI
from dotenv import load_dotenv

import sample
from judge import JudgeResult
from llm import LLM
from solver import perform_one_token_cpc, perform_cot_cpc


@dataclass
class Result:
    context: sample.Context
    gut_check: str
    cot: str


load_dotenv()
llm = LLM(OpenAI(), "gpt-3.5-turbo")

# Each 'passage' is a lengthy text where we are reasoning through a problem.
# Consider progressively larger context parts of each passage (iterating through 'checkpoints' in the passage):
passages = json.load(open("data/passages1.json"))
checkpoints = (text for document in passages for text in sample.checkpoints(document, 1000))

# and for each context part, ask the llm if the current approach is working or not
results = list((
    Result(
        context=context,
        gut_check=perform_one_token_cpc(llm, context),
        cot=perform_cot_cpc(llm, context)
    )
    for context in checkpoints
))
print(results)

# For each result, determine whether the result is good (the two cpc methods agreed) or bad (they disagreed)
evaluation = list((
    JudgeResult(
        result=result,
        score=1.0 if result.gut_check.lower() == result.cot.lower() else 0.0
    )
    for result in results
))
print(evaluation)
