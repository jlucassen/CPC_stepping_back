from llm import LLM

cpc_prompt = ("At this point, we're going to stop and consider whether this approach is working or not "
              "and leads to a correct solution to the problem being worked on. "
              "Otherwise, we should step back and try a different approach.")

yesno_extract = "Respond Yes or No."

cot_extract = 'Think through the question but do not answer yet.'

def perform_one_token_cpc(llm: LLM, context: str, prompt=cpc_prompt) -> str:
    """Asks the llm to do a one-word completion on whether its priorities should change or not."""
    return llm.yesno_completion([
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": prompt + ' ' + yesno_extract
        }
    ])

def perform_cot_cpc(llm: LLM, context: str, prompt=cpc_prompt) -> (str, str):
    """
    Asks the llm to make a more lengthy consideration of whether its priorities should change or not.
    :returns: a tuple of two strings; the first string is the llm's thoughts from the CoT prompt; the second is the
    Yes/No response summarizing the CoT result.
    """
    cot_response = llm.chat_completion([
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": prompt + ' ' + cot_extract
        }
    ])
    return cot_response, llm.yesno_completion([
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": prompt + ' ' + cot_extract
        },
        {
            "role": "assistant",
            "content": cot_response
        },
        {
            "role": "user",
            "content": prompt + ' ' + yesno_extract
        },
    ])
