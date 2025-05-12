
from arithmetic.query import main as q_main
from arithmetic.eval import main as e_main

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="unsloth/Phi-4")
    args = parser.parse_args()

    q_main(f'arithmetic/data/0shot/base{args.base}.txt', args.base, args.model_name, 'output.txt', cot=args.cot, n_shots=0)
    e_main('output.txt', args.base)
main()