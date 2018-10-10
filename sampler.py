import os
import sys

if __name__ == '__main__':
    # set python path
    sys.path.insert(0, os.getcwd())

from random import random
from argparse import ArgumentParser


def sample(line, prob):
    if random() > prob:
        return None
    return line


def main(**kwargs):
    parser = ArgumentParser(**kwargs)
    parser.add_argument('--input_file_path', type=str, default=None)
    parser.add_argument('--output_file_path', type=str, default=None)
    parser.add_argument('--prob', type=int, default=1)
    args = parser.parse_args()

    for line in sys.stdin:
        line = line.strip()
        if line:
            line = sample(line, args.prob/100)
            if line:
                sys.stdout.write(line)
                sys.stdout.write('\n')


if __name__ == '__main__':
    main()
