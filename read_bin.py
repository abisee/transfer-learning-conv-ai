"""
Usage: python read_bin.py file.bin
"""


import torch
from argparse import ArgumentParser


def main():
    parser = ArgumentParser()
    parser.add_argument("-f", "--bin_file", type=str, required=True, help="Path to the bin file you want to read")
    args = parser.parse_args()

    data = torch.load(args.bin_file)
    if isinstance(data, dict):
        for k,v in data.items():
            print(k,v)
    else:
        print(data)

if __name__=="__main__":
    main()
