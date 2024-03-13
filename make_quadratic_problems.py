import random
import json

def make_quadratic_problem(max = 10, factorable=True):
    a = random.randint(1, max)
    b = random.randint(1, max)
    c = random.randint(1, max)
    d = random.randint(1, max)
    if factorable:
        return f"{a*b}x^2 + {a*d + b*c}x + {c*d} = 0"
    else:
        e = random.randint(-max, max)
        return f"{a*b}x^2 + {a*d + b*c}x + {c*d+e} = 0"

for max in range(5, 50, 5):
    for factorable in [True, False]:
        with open(f'quadratic_problems/quadratic_problems_{max}_{factorable}.jsonl', 'w') as outfile:
            for _ in range(100):
                outfile.write(json.dumps({
                    'equation': make_quadratic_problem(max, factorable)
                    }) + '\n')