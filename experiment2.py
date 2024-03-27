# %%
import glob
import json
import pandas as pd
from dotenv import load_dotenv
from openai import AsyncOpenAI
import asyncio

from llm import LLM
from solver import perform_one_token_cpc, perform_cot_cpc

"""
We interrupt an LLM while it is reasoning through a problem. If we fork the LLM at this point,
explicitly prompting one branch to consider using a different approach, do we see a difference between the two 
branches?
"""

load_dotenv()
gpt35 = LLM("gpt-3.5-turbo", AsyncOpenAI())
gpt4 = LLM("gpt-4", AsyncOpenAI())

quadratic_contexts_glob = glob.glob("data/quadratic_contexts/*.jsonl")
passages = []
for filename in quadratic_contexts_glob:
    # filename has format quadratic_contexts_[difficulty]_[is_factorizable].jsonl
    # Each jsonl file is a list of objects, one json object per line. each object has 'equation' and 'context' field
    split_name = filename.split("_")
    difficulty = split_name[-2]
    is_factorizable = split_name[-1].split(".")[0]

    with open(filename, "r") as f:
        for line in f:
            item = json.loads(line)
            passages.append({
                "difficulty": difficulty,
                "is_factorizable": "Yes" if is_factorizable == "True" else "No",
                "context": item["context"],
                "equation": item["equation"]
            })
passages = pd.DataFrame(passages)


# %%
async def process_row(llm, passage):
    try:
        # Take as context the first 100 characters of passage["context"], not to include the SWITCH token if present
        context_truncated = passage["context"]
        if "SWITCH" in context_truncated:
            context_truncated = context_truncated.split("SWITCH")[0]
        context_truncated = context_truncated[:300]

        one_token_cpc_result = await perform_one_token_cpc(llm, context_truncated)
        cot_cpc_thoughts, cot_cpc_result = await perform_cot_cpc(llm, context_truncated)

        return {
            "difficulty": passage["difficulty"],
            "is_factorizable": passage["is_factorizable"],
            "did_switch": "Yes" if "SWITCH" in passage["context"] else "No",
            "context": context_truncated,
            "equation": passage["equation"],
            "one_token_cpc_result": one_token_cpc_result,
            "cot_cpc_result": cot_cpc_result,
            "cot_cpc_thoughts": cot_cpc_thoughts
        }
    except Exception as e:
        e.add_note(str(passage))
        raise e


async def process_all_rows(llm, passages):
    # Asynchronously process all rows in the given `passages` dataframe
    sem = asyncio.Semaphore(3)  # Limit to 3 concurrent tasks

    async def process_row_with_concurrency_limit(llm, passage, index):
        async with sem:
            print(f"Processing passage {index + 1} of {len(passages)}\n{passage}")
            return await process_row(llm, passage)

    tasks = [process_row_with_concurrency_limit(llm, row, index) for index, (_, row) in enumerate(passages.iterrows())]
    results = []
    for future in asyncio.as_completed(tasks, timeout=120):
        try:
            result = await future
            results.append(result)
        except Exception as e:
            print(f"Error processing row: {type(e)} {str(e)}")
            results.append({"error": str(e)})
    return results


async def experiment2(llm, passages: pd.DataFrame) -> pd.DataFrame:
    results = await process_all_rows(llm, passages)
    return pd.DataFrame(results)


# %%
test_passages = passages.sample(5)
gpt3_experiment2 = asyncio.run(experiment2(gpt35, test_passages))
gpt4_experiment2 = asyncio.run(experiment2(gpt4, test_passages))

# %%
# Save the results
gpt3_experiment2.to_csv("gpt3_experiment2.csv", index=False)
gpt4_experiment2.to_csv("gpt4_experiment2.csv", index=False)

# %%
# Run for all passages
gpt3_experiment2 = asyncio.run(experiment2(gpt35, passages))
gpt4_experiment2 = asyncio.run(experiment2(gpt4, passages))
