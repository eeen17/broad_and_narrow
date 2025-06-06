
from arithmetic.sample import main as s_main

import argparse


def main(base, n_digits, n_samples):

    s_main(f'ft_data/data_ft_{base}_{n_digits}.txt', n_samples, n_digits, base)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=int, default=10)
    parser.add_argument("--n_digits", type=int, default=2)
    parser.add_argument("--n_samples", type=int, default=1000)
    args = parser.parse_args()
    main(args.base, args.n_digits, args.n_samples)
