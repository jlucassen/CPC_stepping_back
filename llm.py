from openai import AsyncOpenAI


class RateLimiter:
    def __init__(self, rate_per_minute):
        self.rate_per_minute = rate_per_minute
        self.last_time = 0

    async def __aenter__(self):
        if self.last_time == 0:
            self.last_time = time.time()
        else:
            elapsed = time.time() - self.last_time
            if elapsed < 60 / self.rate_per_minute:
                await asyncio.sleep(60 / self.rate_per_minute - elapsed)
            self.last_time = time.time()


class LLM:
    def __init__(self, model_name, openai: AsyncOpenAI = None):
        self.model_name = model_name
        self.openai = openai or AsyncOpenAI()

    async def chat_completion(self, prompt):
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        chat_completion = await self.openai.chat.completions.create(
            messages=prompt,
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content

    async def yesno_completion(self, prompt):
        """Use the logit bias feature to prompt a "Yes" or "No" completion"""
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        chat_completion = await self.openai.chat.completions.create(
            messages=prompt,
            model=self.model_name,
            max_tokens=1,
            # Force Yes (9642) or No (2822)
            logit_bias={"9642": 100, "2822": 100}
        )
        return chat_completion.choices[0].message.content

    def chat_completion_sync(self, prompt):
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        chat_completion = self.openai.chat.completions.create(
            messages=prompt,
            model=self.model_name,
        )
        return chat_completion.choices[0].message.content

    def yesno_completion_sync(self, prompt):
        """Use the logit bias feature to prompt a "Yes" or "No" completion"""
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        chat_completion = self.openai.chat.completions.create(
            messages=prompt,
            model=self.model_name,
            max_tokens=1,
            # Force Yes (9642) or No (2822)
            logit_bias={"9642": 100, "2822": 100}
        )
        return chat_completion.choices[0].message.content
