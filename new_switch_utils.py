import numpy as np
from llm import LLM

def make_non_decreasing(arr):
        '''
        Takes a binary array, returns 
        '''
        if all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)): return [arr]

        table = np.zeros([2, len(arr)+1])
        table[:, 0] = [0, 0]
        for i, element in list(enumerate(arr)):
            if element == 1:
                table[0, i+1] = table[0, i] + 1 # if you don't want to commit to all 1's yet, need to pay 1 to flip
                table[1, i+1] = min(table[0, i], table[1, i]) # no flip needed to commit to 1's now
            else:
                table[0, i+1] = table[0, i] # no flip needed
                table[1, i+1] = table[1, i] + 1 # can  commit to 1's now, if you want

        max_flips = int(min(table[0, -1], table[1, -1]))

        out = []
        # backtracking to find actual solution
        def backtrack(index, current_flips, arr):
            nonlocal out
            # Base case: If we've processed the entire array. Time to make a decision
            if index == len(arr):
                if all(arr[i] <= arr[i + 1] for i in range(len(arr) - 1)):
                    out.append(arr[:]) # record all valid sols, not just first
                    return True
                else:
                    return False
            # Recursion: Try without flipping
            no_flip = backtrack(index + 1, current_flips, arr)
            if no_flip:
                return no_flip
            # Try flipping if within flip limit
            if current_flips < max_flips:
                arr[index] = 1 -  arr[index]  # Flip the bit
                backtrack(index + 1, current_flips + 1, arr)
                arr[index] = 1 - arr[index]  # Unflip the bit back

        backtrack(0, 0, arr[:]) # Start the recursion with a copy of the array
        return out

original_prompt = "This is a reasoning transcript of an agent trying to find the roots of a quadratic equation. The agent will start by attempting to factor the quadratic, and may switch over to using the quadratic formula instead. First you will be shown the full transcript, then just a prefix of the transcript. By the end of the prefix transcript, has the agent switched from factoring to using the quadratic formula yet?"
gpt4 = LLM("gpt-4")
def original_4(context, prefix):
    return 1 if gpt4.yesno_completion(original_prompt+'\n\nFULL TRANSCRIPT:\n'+context+'\n\nPREFIX TRANSCRIPT:\n'+prefix+"\n\nANSWER:\n") == 'Yes' else 0