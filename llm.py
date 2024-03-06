from openai import OpenAI


class LLM:
    def __init(self, openai: OpenAI, model_name):
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
        return chat_completion.messages[-1].content

    def one_token_completion(self, prompt):
        return self.openai.completions.create(
            prompt=prompt,
            model=self.model_name,
            max_tokens=1
        ).choices[0].text