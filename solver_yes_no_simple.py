from llm import LLM

cpc_prompt = ("Answer the question as best you can")

def perform_one_token_cpc(llm, context, prompt_variation):
    """Asks the llm to provide a simple yes_no answer to the question"""
    return llm.yesno_completion([
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": prompt_variation
        }
    ])

# def perform_one_token_cpc(llm: LLM, context: str) -> str:
#     """Asks the llm to provide a simple yes_no answer to the question"""
#     return llm.yesno_completion([
#         {
#             "role": "assistant",
#             "content": context
#         },
#         {
#             "role": "user",
#             "content": cpc_prompt +
#                        "\n\nPlease respond with 'Yes' if you agree or 'No' if you disagree."
                       
#         }
#     ])

def perform_cot_cpc(llm: LLM, context: str) -> str:
    """
    Asks the LLM to provide a chain of thought for the given question and clearly indicate the answer by typing 'YES' or 'NO'.
    """
    cot_response = llm.chat_completion([
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": cpc_prompt +
                       "\n\nWhichever point you have decided on an answer, indicate it by typing 'YES' or 'NO' in capitals."
        }
    ])
    return cot_response



def perform_cot_cpc(llm: LLM, context: str) -> (str, str):
    """
    Asks the LLM to provide a chain of thought for the given question and clearly indicate the answer by typing 'Yes' or 'No'.
    """
    cot_response = llm.chat_completion([
        {
            "role": "assistant",
            "content": context
        },
        {
            "role": "user",
            "content": cpc_prompt +
                       "\n\nWhichever point you have decided on an answer, indicate it by typing 'Yes' or 'No'. But first, take a deep breath "
                       "and think step by step to answer the question"
        }
    ])
    return cot_response, llm.yesno_completion([
        {
            "role": "assistant",
            "content": cot_response
        },
        {
            "role": "user",
            "content": "Do your thoughts in the previous message suggest a Yes or No answer? answer either"
                       "'Yes' or "
                       "'No'"
        }
    ])


