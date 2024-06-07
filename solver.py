from llm import LLM
cpc_prompt = ("At this point, we're going to stop and consider whether this approach is working or not "
              "and leads to a correct solution to the problem being worked on. "
              "Otherwise, we should step back and try a different approach."
              "At this point, should we change to a different approach?")

yesno_extract = "Respond Yes or No." # same as Solver, shown to be best on trivia
cot_extract = 'Think through whether we should change to a different approach, but do not answer yet.' # shown to be best on CPC

def perform_one_token_cpc(llm: LLM, context: str, task_descr, cpc_prompt=cpc_prompt) -> str:
    """Asks the llm to do a one-word completion on whether its priorities should change or not."""
    messages_1t = [
        {
            "role": "user",
            "content": task_descr
        },
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": cpc_prompt + '\n' + yesno_extract
        }
    ]
    return llm.yesno_completion(messages_1t)

def perform_cot_cpc(llm: LLM, context: str, task_descr, cpc_prompt=cpc_prompt) -> (str, str):
    """
    Asks the llm to make a more lengthy consideration of whether its priorities should change or not.
    :returns: a tuple of two strings; the first string is the llm's thoughts from the CoT prompt; the second is the
    Yes/No response summarizing the CoT result.
    """
    messages_cot = [
        {
            "role": "user",
            "content": task_descr
        },
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": cpc_prompt + '\n' + cot_extract
        }
    ]
    cot_response = llm.chat_completion(messages_cot)
    messages_cot_answer = [{
            "role": "assistant",
            "content": cot_response
        },
        {
            "role": "user",
            "content": cpc_prompt + '\n' + yesno_extract
        }
    ]
    return cot_response, llm.yesno_completion(messages_cot + messages_cot_answer)