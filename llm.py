import threading
import time
import tiktoken

from openai import OpenAI


class RateLimiter:
    """
    Rate limiter which allows you to limit the number of calls executed per minute on this llm instance.
    Will block the current thread until it's ok to proceed.
    Guaranteed not to exceed the rate_per_minute. Will probably be slightly slower than the rate_per_minute.
    """

    def __init__(self, rate_per_minute, encoding):
        self.rate_per_minute = rate_per_minute
        self.tpm = 300000
        self.last_time = 0
        self.lock = threading.Lock()
        self.encoding = encoding
        self.last_time_tpm = 0

    def __enter__(self):
        if self.last_time == 0:
            self.last_time = time.time()
        else:
            tick_duration_seconds = 60 / self.rate_per_minute
            with self.lock:
                # if we're too fast, sleep for the rest of the tick
                time_since_last = time.time() - self.last_time
                if time_since_last < tick_duration_seconds:
                    time.sleep(tick_duration_seconds - time_since_last)
                self.last_time = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def wait_for_tpm(self, prompt):
        if self.last_time_tpm == 0:
            self.last_time_tpm = time.time()
        else:
            n_toks = len(self.encoding.encode(prompt))
            tick_duration_seconds = 60 * n_toks / self.tpm
            with self.lock:
                # if we're too fast, sleep for the rest of the tick
                time_since_last = time.time() - self.last_time_tpm
                if time_since_last < tick_duration_seconds:
                    time.sleep(tick_duration_seconds - time_since_last)
                self.last_time_tpm = time.time()


class LLM:
    def __init__(self, model_name, openai: OpenAI = None, rate_limiter: RateLimiter = None):
        self.model_name = model_name
        self.openai = openai or OpenAI()
        self.encoding = tiktoken.encoding_for_model(self.model_name)
        # https://platform.openai.com/docs/guides/rate-limits/usage-tiers?context=tier-three
        self.rate_limiter = RateLimiter(3500, self.encoding)
        self.yesno = {self.encoding.encode("Yes")[0]: 100, self.encoding.encode("No")[0]: 100}

    def chat_completion(self, prompt):
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        # self.rate_limiter.wait_for_tpm(prompt)
        with self.rate_limiter:
            chat_completion = self.openai.chat.completions.create(
                messages=prompt,
                model=self.model_name,
            )
            return chat_completion.choices[0].message.content

    def yesno_completion(self, prompt, t=1):
        """Use the logit bias feature to prompt a "Yes" or "No" completion"""
        if isinstance(prompt, str):
            prompt = [{"role": "user", "content": prompt}]
        # self.rate_limiter.wait_for_tpm(prompt)
        with self.rate_limiter:
            chat_completion = self.openai.chat.completions.create(
                messages=prompt,
                model=self.model_name,
                max_tokens=1,
                temperature=t, # t=1 by default, match openai settings
                logit_bias=self.yesno
            )
            return chat_completion.choices[0].message.content

    def chat_completion_false_start(self, prompt, false_start):
        # self.rate_limiter.wait_for_tpm(prompt)
        with self.rate_limiter:
            chat_completion = self.openai.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                    {
                        "role": "assistant",
                        "content": false_start,
                    }
                ],
                model=self.model_name,
            )
        return chat_completion.choices[0].message.content
    
    def choice_completion(self, prompt, choices):
        choice_tokens = []
        for choice in choices:
            token = self.encoding.encode(choice)
            if len(token) > 1:
                raise ValueError(f"Choice \"{choice}\" is not encodable as a single token")
            choice_tokens += token
        logit_bias = {str(token): 100 for token in choice_tokens}
        # self.rate_limiter.wait_for_tpm(prompt)
        with self.rate_limiter:
            chat_completion = self.openai.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=self.model_name,
                max_tokens=1,
                logit_bias=logit_bias
            )
        return chat_completion.choices[0].message.content
