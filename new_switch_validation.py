from llm import LLM
import numpy as np
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

switching_prompt = "This is a reasoning transcript of an agent trying to find the roots of a quadratic equation. The agent will start by attempting to factor the quadratic, and may switch over to using the quadratic formula instead. First you will be shown the complete transcript, then the transcript up to a point. At the current point in the transcript, has the agent switched from factoring to using the quadratic formula?"
cpc_prompt = "This is a reasoning transcript of an agent trying to find the roots of a quadratic equation. There are two strategies for this task: factoring and the quadratic formula. At the current point in the transcript, should the agent switch strategies?"
r = 50
contexts = [context[:x] for x in range(r, len(context), r)]

llm = LLM("gpt-4")
n = 3
switching = switching = np.zeros(len(contexts))
one_token = []
for i, context_prefix in tqdm(list(enumerate(contexts))):
    switching[i] = sum([llm.yesno_completion(switching_prompt+'\n\nFULL TRANSCRIPT:\n'+context+'\n\nPARTIAL TRANSCRIPT:\n'+context_prefix+"\n\nANSWER:\n") == 'Yes' for _ in range(n)])/n
switching_errs = np.sqrt(switching*(1-switching)/n)
plt.errorbar(np.array(range(len(switching))), switching, switching_errs, fmt='o')
plt.show()