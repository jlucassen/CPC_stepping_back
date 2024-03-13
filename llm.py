from typing import Union

from openai import OpenAI, AsyncOpenAI


class LLM:
    def __init__(self, model_name, openai: Union[OpenAI, AsyncOpenAI] = None):
        self.model_name = model_name
        self.openai = openai or OpenAI()

    def chat_completion(self, prompt):
        chat_completion = self.openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content

    def yesno_completion(self, prompt):
        """Use the logit bias feature to prompt a "Yes" or "No" completion"""
        return self.openai.chat.completions.create(
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            model=self.model_name,
            max_tokens=1,
            # Force Yes (9642) or No (2822)
            logit_bias={"9642": 100, "2822": 100}
        ).choices[0].message.content
