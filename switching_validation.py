# %%
from llm import LLM
import numpy as np
from tqdm import tqdm

from concurrent.futures import ThreadPoolExecutor
from functools import partial
from threading import Lock

# %%
llm = LLM("gpt-3.5-turbo")
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
direct_prompt = "Please find the roots of the quadratic equation x^2 + 9 = 0. If you ever change your strategy to using the quadratic formula, say SWITCHING."
indirect_prompt = "Please find the roots of the quadratic equation x^2 + 9 = 0. If you ever change your strategy to a different approach for solving quadratics, say SWITCHING."
false_start = "Sure, for my first strategy I'll try factoring the equation."

data_direct = np.empty([n, 4])
pbar1 = tqdm(total=n)
with ThreadPoolExecutor(max_workers=50) as executor:
    executor.map(partial(request_func, prompt=direct_prompt, false_start=false_start, data=data_direct, lock=lock, pbar=pbar1), request_range)
np.savetxt("switching_validation_data_direct.csv", data_direct, delimiter=",")

data_indirect = np.empty([n, 4])
pbar2 = tqdm(total=n)
with ThreadPoolExecutor(max_workers=50) as executor:
    executor.map(partial(request_func, prompt=indirect_prompt, false_start=false_start, data=data_indirect, lock=lock, pbar=pbar2), request_range)
np.savetxt("switching_validation_data_indirect.csv", data_indirect, delimiter=",")
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
# %%
print("Direct:")
print("\nSwitching vs formula")
svf_d = confusion_matrix(data_direct[:,0], data_direct[:,2])
print("\nSwitching vs 3i")
sv3_d = confusion_matrix(data_direct[:,0], data_direct[:,3])
print("\nFormula vs 3i")
fv3_d = confusion_matrix(data_direct[:,2], data_direct[:,3])
print(f"Switching vs formula accuracy: {(svf_d[0]+svf_d[3])/n*100}%")
print(f"Switching vs 3i accuracy: {(sv3_d[0]+sv3_d[3])/n*100}%")
print(f"Formula vs 3i accuracy: {(fv3_d[0]+fv3_d[3])/n*100}%")
# %%
print("Indirect:")
print("\nSwitching vs formula")
svf_i = confusion_matrix(data_indirect[:,0], data_indirect[:,2])
print("\nSwitching vs 3i")
sv3_i = confusion_matrix(data_indirect[:,0], data_indirect[:,3])
print("\nFormula vs 3i")
fv3_i = confusion_matrix(data_indirect[:,2], data_indirect[:,3])
print(f"Switching vs formula accuracy: {(svf_i[0]+svf_i[3])/n*100}%")
print(f"Switching vs 3i accuracy: {(sv3_i[0]+sv3_i[3])/n*100}%")
print(f"Formula vs 3i accuracy: {(fv3_i[0]+fv3_i[3])/n*100}%")
# %%
print(f"Direct switching vs formula accuracy: {(svf_d[0]+svf_d[3])/n*100:.2f}%")
print(f"2sd: {2*np.sqrt((svf_d[0]+svf_d[3])/n*(1-(svf_d[0]+svf_d[3])/n)/n)*100:.2f}% (percentage points)")
print(f"Indirect switching vs formula accuracy: {(svf_i[0]+svf_i[3])/n*100:.2f}%")
print(f"2sd: {2*np.sqrt((svf_i[0]+svf_i[3])/n*(1-(svf_i[0]+svf_i[3])/n)/n)*100:.2f}% (percentage points)")
# %%
