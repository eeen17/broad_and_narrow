from arithmetic.query import main as q_main
from arithmetic.eval import main as e_main

import argparse


def main():

    
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--cot", type=bool, default=False)
    parser.add_argument("--model_name", type=str, default="unsloth/Phi-4")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--n_digits", type=int, default=2)
    parser.add_argument("--n_shots", type=int, default=0)
    parser.add_argument("--icl_cot", type=bool, default=True)
    args = parser.parse_args() 

    if args.n_digits == 2:
        q_main(f'arithmetic/data/0shot/base{args.base}.txt', args.base, args.model_name, f'output.txt', cot=args.cot, n_shots=args.n_shots, size=args.size, icl_cot=args.icl_cot)
    elif args.n_digits == 3:
        q_main(f'arithmetic/data/0shot_3digits/base{args.base}.txt', args.base, args.model_name, f'output.txt', cot=args.cot, n_shots=args.n_shots, size=args.size, icl_cot=args.icl_cot)
    elif args.n_digits == 4:
        q_main(f'arithmetic/data/0shot_4digits/base{args.base}.txt', args.base, args.model_name, f'output.txt', cot=args.cot, n_shots=args.n_shots, size=args.siz, icl_cot=args.icl_cote)
    acc = e_main(f'output.txt', args.base)
    with open('results.txt', 'a') as f:
        f.write(f'{args.model_name} {args.base} {args.cot} {args.n_digits} {args.n_shots}: {acc}\n')
    
main()
