# %%
import re

from llm import LLM

llm = LLM("gpt-3.5-turbo")

factoring_contexts = []
formula_contexts = []

with open('data/quadratic_problems/quadratic_problems_5_True.jsonl', 'r') as readfile:
    for line in readfile.readlines()[:10]:
        problem = line[:-1]
        factoring_contexts.append(llm.chat_completion_false_start("Find the roots of the quadratic equation "+problem+" using factoring. Do not use the quadratic equation.", "Sure, I'll use factoring to find the roots of the quadratic equation "+problem+"."))
        formula_contexts.append(llm.chat_completion_false_start("Find the roots of the quadratic equation "+problem+" using the quadratic formula. Do not use factoring.", "Sure, I'll use the quadratic formula to find the roots of the quadratic equation "+problem+"."))

for i in range(1):
    factor = factoring_contexts[i]
    formula = formula_contexts[i]
    factor_split = re.split('\n|\.', factor)
    formula_split = re.split('\n|\.', formula)
    print('\n'.join(factor_split[:int(len(factor_split)/2)])+'\n'+'\n'.join(formula_split[int(len(formula_split)/2):]))



# %%import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

context = """To find the roots of the quadratic equation 21x^2 - 1302x + 19845 = 0, we can either factor the equation or use the quadratic formula.

Let's first try factoring:

21x^2 - 1302x + 19845 = 0
3(7x^2 - 434x + 6615) = 0
3(7x - 195)(x - 45) = 0

Therefore, the roots of the equation are:
x = 195/7 or x = 45

If factoring is no longer practical, we will switch to the quadratic formula.

The quadratic formula is: x = (-b ± √(b^2 - 4ac)) / 2a

For the equation 21x^2 - 1302x + 19845 = 0, a = 21, b = -1302, and c = 19845.

Plugging these values into the formula:

x = (1302 ± √((-1302)^2 - 4*21*19845)) / 2*21
x = (1302 ± √(1694404 - 1673820)) / 42
x = (1302 ± √20584) / 42
x = (1302 ± 143.49) / 42

Therefore, the roots of the equation using the quadratic formula are approximately:
x = 45.76 or x = 39.61","21x^2 - 1302x + 19845 = 0"""

#context = "To find the roots of the quadratic equation 3x^2 - 12x + 12 = 0, we can use the quadratic formula.\n\nThe quadratic formula is given by:\nx = (-b \u00b1 \u221a(b^2 - 4ac)) / 2a\n\nIn this case, the coefficients are a = 3, b = -12, and c = 12.\n\nPlugging these values into the formula, we get:\nx = (12 \u00b1 \u221a((-12)^2 - 4*3*12)) / 2*3\nx = (12 \u00b1 \u221a(144 - 144)) / 6\nx = (12 \u00b1 \u221a0) / 6\nx = 12 / 6\nx = 2\n\nTherefore, the roots of the quadratic equation 3x^2 - 12x + 12 = 0 are x = 2."

switching_prompt = "This is a reasoning transcript of an agent trying to find the roots of a quadratic equation. The agent will start by attempting to factor the quadratic, and may switch over to using the quadratic formula instead. First you will be shown the complete transcript, then just a prefix of the transcript. At the current point in the prefix transcript, has the agent switched from factoring to using the quadratic formula yet?"
r = 50
contexts = [context[:x] for x in range(r, len(context), r)]

llm = LLM("gpt-4")
n = 1
switching = switching = np.zeros(len(contexts))
one_token = []
for i, context_prefix in tqdm(list(enumerate(contexts))):
    switching[i] = sum([llm.yesno_completion(switching_prompt+'\n\nFULL TRANSCRIPT:\n'+context+'\n\nPREFIX TRANSCRIPT:\n'+context_prefix+"\n\nANSWER:\n") == 'Yes' for _ in range(n)])/n
switching_errs = np.sqrt(switching*(1-switching)/n)
plt.errorbar(np.array(range(len(switching))), switching, switching_errs, fmt='o')
plt.show()



# def make_non_decreasing(arr):
#     '''
#     Takes a binary array, returns the index or list of indices at which it takes the fewest flips to make the array non-decreasing.
#     '''
#     table = np.zeros([2, len(arr)+1])
#     table[:, 0] = [0, 0]
#     for i, element in list(enumerate(arr)):
#         if element == 1:
#             table[0, i+1] = table[0, i] + 1 # if you don't want to commit to all 1's yet, need to pay 1 to flip
#             table[1, i+1] = min(table[0, i], table[1, i]) # no flip needed to commit to 1's now
#         else:
#             table[0, i+1] = table[0, i] # no flip needed
#             table[1, i+1] = table[1, i] + 1 # can  commit to 1's now, if you want

#     max_flips = int(min(table[0, -1], table[1, -1]))
#     return max_flips