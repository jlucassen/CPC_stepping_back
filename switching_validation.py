from llm import LLM
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock

llm = LLM("gpt-3.5-turbo")

tot = [0,0]
n = 100
requests = range(n+1)
pbar = tqdm(total=n)
lock = Lock()
def request_func(i, tot, lock, pbar):
    out = llm.chat_completion_false_start("Please find the roots of the quadratic equation x^2 + 1 = 0. If you switch your strategy from factoring to using the quadratic formula, say SWITCHING in all caps.",
                                           "Sure, for my first strategy I'll try factoring the equation.")
    if "switch" in out.lower():
        with lock:
            tot[0] += 1
            pbar.update(1)
            if "SWITCHING" in out:
                tot[1] += 1
    else:
        with lock:
            pbar.update(1)
    return 0

with ThreadPoolExecutor(max_workers=50) as executor:
    executor.map(partial(request_func, tot=tot, lock=lock, pbar=pbar), requests)
print(tot[0]/n, tot[1]/n)