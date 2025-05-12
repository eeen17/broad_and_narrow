
from arithmetic.query import main as q_main
from arithmetic.eval import main as e_main

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="unsloth/Phi-4")
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()

    q_main(f'arithmetic/data/0shot/base{args.base}.txt', args.base, args.model_name, f'output_{args.base}.txt', cot=args.cot, n_shots=0, size=args.size)
    e_main(f'output_{args.base}.txt', args.base)
main()