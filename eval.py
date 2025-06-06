
from arithmetic.query import main as q_main
from arithmetic.eval import main as e_main

import argparse
import os


done = ['outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 8 False 2',
 'outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 8 False 2',
 'outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 8 False 2',
 'outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 8 False 3',
 'unsloth/Phi-4 8 False 2',
 'outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 8 False 2',
 'outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 8 False 2',
 'outputs/test_run_8_cot_False_n_digits_2/checkpoint-60 8 False 3',
 'outputs/test_run_10_cot_False_n_digits_2/checkpoint-60 8 False 3',
 'outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 8 False 2',
 'outputs/test_run_11_cot_False_n_digits_2/checkpoint-60 8 False 2',
 'outputs/test_run_8_cot_True_n_digits_2/checkpoint-60 8 False 3',
 'outputs/test_run_9_cot_False_n_digits_2/checkpoint-60 8 False 3',
 'outputs/test_run_9_cot_True_n_digits_2/checkpoint-60 8 False 2',
 'outputs/test_run_11_cot_True_n_digits_2/checkpoint-60 8 False 3',
 'outputs/test_run_10_cot_True_n_digits_2/checkpoint-60 8 False 3']

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="unsloth/Phi-4")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--n_digits", type=int, default=2)
    args = parser.parse_args()

    if f'{args.model_name} {args.base} {args.cot} {args.n_digits}' in done:
        return

    if args.n_digits == 2:
        q_main(f'arithmetic/data/0shot/base{args.base}.txt', args.base, args.model_name, f'output.txt', cot=args.cot, n_shots=0, size=args.size)
    elif args.n_digits == 3:
        q_main(f'arithmetic/data/0shot_3digits/base{args.base}.txt', args.base, args.model_name, f'output.txt', cot=args.cot, n_shots=0, size=args.size)
    elif args.n_digits == 4:
        q_main(f'arithmetic/data/0shot_4digits/base{args.base}.txt', args.base, args.model_name, f'output.txt', cot=args.cot, n_shots=0, size=args.size)
    acc = e_main(f'output.txt', args.base)
    with open('results.txt', 'a') as f:
        f.write(f'{args.model_name} {args.base} {args.cot} {args.n_digits}: {acc}\n')
    
main()