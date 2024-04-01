from llm import LLM
from tqdm import tqdm

llm = LLM("gpt-3.5-turbo")

tot = 0
n = 100
for _ in tqdm(range(n+1)):
    out = llm.chat_completion_false_start("Please find the roots of the quadratic equation x^2 + 1 = 0. If you ever change your strategy while working on this problem, say SWITCHING in all caps.",
                                           "Sure, for my first strategy I'll try factoring the equation.")
    print(out)
    if "switch" in out.lower():
        tot += 1
print(tot/n)