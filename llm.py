from openai import OpenAI


class LLM:
    def __init__(self, openai: OpenAI, model_name):
        self.model_name = model_name
        self.openai = openai

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
            logit_bias={5297: 100, 2949: 100}
        ).choices[0].message.content
