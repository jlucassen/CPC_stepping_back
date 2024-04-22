# %%
from llm import LLM
import numpy as np
from tqdm import tqdm
import os

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock

# %%
llm = LLM("gpt-4")
n = 1000
request_range = range(n+1)
lock = Lock()
def request_func(i, prompt, false_start, data, lock, pbar):
    out = llm.chat_completion_false_start(prompt, false_start)
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
        data[i,:] = to_add
        pbar.update(1)
    return 0

# %%
factorable = "x^2 - 9 = 0"
nonfactorable = "x^2 + 9 = 0"

max_direct_prompt_switch = f"Please find the roots of the quadratic equation {nonfactorable}. If you ever change your strategy from factoring to using the quadratic formula, say SWITCHING."
direct_prompt_switch = f"Please find the roots of the quadratic equation {nonfactorable}. If you ever change your strategy to using the quadratic formula, say SWITCHING."
mixed_prompt_switch = f"Please find the roots of the quadratic equation {nonfactorable}. If you ever change your strategy to a different approach for solving quadratics, say SWITCHING."
indirect_prompt_switch = f"Please find the roots of the quadratic equation {nonfactorable}. If you ever change your strategy, say SWITCHING."

max_direct_prompt_stay = f"Please find the roots of the quadratic equation {factorable}. If you ever change your strategy from factoring to using the quadratic formula, say SWITCHING."
direct_prompt_stay = f"Please find the roots of the quadratic equation {factorable}. If you ever change your strategy to using the quadratic formula, say SWITCHING."
mixed_prompt_stay = f"Please find the roots of the quadratic equation {factorable}. If you ever change your strategy to a different approach for solving quadratics, say SWITCHING."
indirect_prompt_stay = f"Please find the roots of the quadratic equation {factorable}. If you ever change your strategy, say SWITCHING."

false_start = "Sure, for my first strategy I'll try factoring the equation."

# order used for gpt-4 tests at n=1000, others clearly worse at n=100
prompts = [
    max_direct_prompt_switch,
    direct_prompt_switch,
    max_direct_prompt_stay,
    direct_prompt_stay,
]

# order used for gpt-3.5-turbo tests
#prompts = [direct_prompt_switch, mixed_prompt_switch, indirect_prompt_switch, direct_prompt_stay, mixed_prompt_stay, indirect_prompt_stay, max_direct_prompt_switch, max_direct_prompt_stay]

# %%
def confusion_matrix(col1, col2):
    z = list(zip(col1, col2))
    a = sum([1 for (x,y) in z if x and y])
    b = sum([1 for (x,y) in z if x and not y])
    c = sum([1 for (x,y) in z if not x and y])
    d = sum([1 for (x,y) in z if not x and not y])
    #print(f"Both: {a}")
    #print(f"Col1: {b}")
    #print(f"Col2: {c}")
    #print(f"Neither: {d}")
    return a,b,c,d

def run_experiment(i, prompt):
    if f"switching_validation_data_{i}.csv" not in os.listdir('gpt4'):
        data = np.empty([n, 4])
        pbar = tqdm(total=n)
        with ThreadPoolExecutor(max_workers=50) as executor:
            executor.map(partial(request_func, prompt=prompt, false_start=false_start, data=data, lock=lock, pbar=pbar), request_range)
        pbar.close()
        np.savetxt(f"gpt4/switching_validation_data_{i}.csv", data, delimiter=",")
    else:
        data = np.loadtxt(f"gpt4/switching_validation_data_{i}.csv", delimiter=",")

    print(f"\nPROMPT {i}: {prompt}")
    #print("\nSwitching vs formula")
    svf = confusion_matrix(data[:,0], data[:,2])
    #print("\nSwitching vs 3i")
    #sv3 = confusion_matrix(data[:,0], data[:,3])
    #print("\nFormula vs 3i")
    #fv3 = confusion_matrix(data[:,2], data[:,3])
    print(f"Switching vs formula accuracy: {(svf[0]+svf[3])/n*100}%")
    #print(f"Switching vs 3i accuracy: {(sv3[0]+sv3[3])/n*100}%")
    #print(f"Formula vs 3i accuracy: {(fv3[0]+fv3[3])/n*100}%")    
    qf = sum(data[:,2])/n
    print(f"QF frequency: {qf*100:.2f} +- {2*np.sqrt(qf*(1-qf)/n)*100}%")
# %%
    
for i, prompt in enumerate(prompts):
    run_experiment(i, prompt)