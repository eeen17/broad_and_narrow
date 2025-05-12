model_name = "unsloth/Phi-4"
base = 10
cot = False

from arithmetic.query import main as q_main
from arithmetic.eval import main as e_main

q_main(f'arithmetic/data/0shot/base{base}.txt', base, model_name, 'output.txt', cot=cot, n_shots=0)
e_main('output.txt', base)

