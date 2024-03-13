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
    cot_cpc_prompt = ("At this point, stop and consider whether this approach is working or not. "
                      "If it seems like a different approach might be better, we should step back "
                      "and try something else.\nAt this point, should we change to a different approach? "
                      "Please think step by step.")
    cot_context = str(sample.text) + "\n" + cot_cpc_prompt
    cot_response = llm.chat_completion(cot_context)
    print(cot_response)
    return perform_one_token_cpc(llm, Context(cot_context + "\n" + cot_response))
