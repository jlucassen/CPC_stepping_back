import random
import json

def make_quadratic_problem(max, factorable):
    root1 = random.randint(1, max)
    root2 = random.randint(1, max)
    scale = random.randint(1, max)
    shift = 0 if factorable else random.randint(1, max)
    a = scale
    b = -scale*(root1+root2)
    c = scale*root1*root2 + shift
    actual_roots = [-b/(2*a)+(b**2 - 4*a*c)**0.5/(2*a), -b/(2*a)-(b**2 - 4*a*c)**0.5/(2*a)]
    if not factorable:
        if not any([isinstance(root, complex) for root in actual_roots]): # if all actual roots are real
            if all([root.is_integer() for root in actual_roots]): # and all actual roots are integers
                return make_quadratic_problem(max, factorable) # try again
    return f"{a}x^2 - {-b}x + {c} = 0"

for max in range(5, 50, 5):
    for factorable in [True, False]:
        with open(f'data/quadratic_problems/quadratic_problems_{max}_{factorable}.jsonl', 'w') as outfile:
            for _ in range(100):
                outfile.write(make_quadratic_problem(max, factorable) + '\n')