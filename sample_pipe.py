
from arithmetic.sample import main as s_main

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--n_digits", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()

    s_main(f'data_ft_{args.base}.txt', args.n_samples, args.n_digits, args.base)
main()

