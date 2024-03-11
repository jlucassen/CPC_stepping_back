from dataclasses import dataclass
from collections.abc import Generator, Iterable
import re

from llm import LLM
from sample import Context


def perform_one_token_cpc(llm: LLM, context: Context):
    """Asks the llm to do a one-word completion on whether its priorities should change or not"""
    one_token_cpc_prompt = ("At this point, stop and consider whether this approach is working or not. If it seems "
                            "like a different approach might be better, we should step back and try something else.\n"
                            "At this point, should we change to a different approach? Please answer Yes or No.")
    return llm.yesno_completion(str(context.text) + "\n" + one_token_cpc_prompt)


def perform_cot_cpc(llm: LLM, sample: Context):
    """Asks the llm to make a more lengthy consideration of whether its priorities should change or not"""
    cot_check_prompt = ("Should my priorities change? Respond either [Yes] or [No]! "
                        "State your answer as [Yes] or [No] in brackets. Please think step by step.")
    response = llm.chat_completion(str(sample.text) + "\n" + cot_check_prompt)
    print(response)
    search = re.search(r"\[(Yes|No)]", response)
    if search:
        return search.group(1)
    else:
        return ""
