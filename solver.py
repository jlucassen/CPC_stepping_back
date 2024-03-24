from llm import LLM
from sample import Context


cpc_prompt = ("At this point, we're going to stop and consider whether this approach is working or not. "
              "Do you think our current approach is promising, and may lead to a correct solution to "
              "the problem being worked on? Otherwise, we should step back and try a different approach.")


def perform_one_token_cpc(llm: LLM, context: Context) -> str:
    """Asks the llm to do a one-word completion on whether its priorities should change or not"""
    one_token_cpc_prompt = (cpc_prompt +
                            "\nAt this point, should we change to a different approach? Yes or No.")
    return llm.yesno_completion(str(context.text) + "\n" + one_token_cpc_prompt)


def perform_cot_cpc(llm: LLM, sample: Context) -> (str, str):
    """Asks the llm to make a more lengthy consideration of whether its priorities should change or not"""
    cot_cpc_prompt = (cpc_prompt +
                      "\nAt this point, should we change to a different approach? Please think step by step.")
    cot_context = str(sample.text) + "\n" + cot_cpc_prompt
    cot_response = llm.chat_completion(cot_context)
    return cot_response, perform_one_token_cpc(llm, Context(cot_context + "\n" + cot_response))
