from llm import LLM

cpc_prompt = ("At this point, we're going to stop and consider whether this approach is working or not "
              "and leads to a correct solution to the problem being worked on. "
              "Otherwise, we should step back and try a different approach.")


async def perform_one_token_cpc(llm: LLM, context: str) -> str:
    """Asks the llm to do a one-word completion on whether its priorities should change or not"""
    one_token_cpc_prompt = [
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": cpc_prompt +
                       "\n\nAt this point, should we change to a different approach? Please answer "
                       "'Yes, I recommend a different approach' or "
                       "'No, I recommend staying with the current approach.'"
        }
    ]
    return await llm.yesno_completion(one_token_cpc_prompt)


async def perform_cot_cpc(llm: LLM, context: str) -> (str, str):
    """
    Asks the llm to make a more lengthy consideration of whether its priorities should change or not
    :returns: a tuple of two strings; the first string is the llm's thoughts from the CoT prompt; the second is the
    * Yes/No response summarizing the CoT result.
    """
    cot_prompt = [
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": cpc_prompt +
                       "\n\nAt the end, I want you to answer 'Yes, I recommend a different approach' or "
                       "'No, I recommend staying with the current approach.' But first, take a deep breath "
                       "and think step by step. Start by analyzing the current approach:"
        }
    ]
    cot_response = await llm.chat_completion(cot_prompt)
    return cot_response, await llm.yesno_completion(
        [
            {
                "role": "assistant",
                "content": cot_response
            },
            {
                "role": "user",
                "content": "Do your thoughts in the previous message recommend changing our approach? Please answer "
                           "'Yes, I recommend a different approach' or "
                           "'No, I recommend staying with the current approach."
            }
        ])
