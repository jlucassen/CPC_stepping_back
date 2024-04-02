# %%
from llm import LLM
import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock

# %%
llm = LLM("gpt-3.5-turbo")
n = 100
tot = np.empty([n, 4])
requests = range(n+1)
pbar = tqdm(total=n)
lock = Lock()
def request_func(i, tot, lock, pbar):
    out = llm.chat_completion_false_start("Please find the roots of the quadratic equation x^2 + 9 = 0. If you ever change your strategy to a different approach for solving quadratics, say SWITCHING.",
                                           "Sure, for my first strategy I'll try factoring the equation.")
    to_add = [0]*4
    if "switch" in out.lower():
        to_add[0] = 1
    if "SWITCHING" in out:
        to_add[1] = 1
    if "formula" in out.lower():
        to_add[2] = 1
    if "3i" in out:
        to_add[3] = 1
    with lock:
        tot[i,:] = to_add
        pbar.update(1)
    return 0

with ThreadPoolExecutor(max_workers=50) as executor:
    executor.map(partial(request_func, tot=tot, lock=lock, pbar=pbar), requests)
# %%
def confusion_matrix(col1, col2):
    z = list(zip(col1, col2))
    a = sum([1 for (x,y) in z if x and y])
    b = sum([1 for (x,y) in z if x and not y])
    c = sum([1 for (x,y) in z if not x and y])
    d = sum([1 for (x,y) in z if not x and not y])
    print(f"Both: {a}")
    print(f"Col1: {b}")
    print(f"Col2: {c}")
    print(f"Neither: {d}")
    return a,b,c,d

print("\nSwitching vs formula")
svf = confusion_matrix(tot[:,0], tot[:,2])
print("\nSwitching vs 3i")
sv3 = confusion_matrix(tot[:,0], tot[:,3])
print("\nFormula vs 3i")
fv3 = confusion_matrix(tot[:,2], tot[:,3])
print(f"Switching vs formula accuracy: {(svf[0]+svf[3])/n*100}%")
print(f"Switching vs 3i accuracy: {(sv3[0]+sv3[3])/n*100}%")
print(f"Formula vs 3i accuracy: {(fv3[0]+fv3[3])/n*100}%")
# %%
