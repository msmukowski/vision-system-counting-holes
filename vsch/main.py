import argparse


def main():
    args = parse_args()

    print(args.obp)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('im', metavar='Images directory', type=str, help='Directory path where images are located.', nargs='?')
    parser.add_argument('obp', metavar='Objects data directory', type=str, help='Directory path where image objects description data is located.', nargs='?')
    parser.add_argument('out', metavar='Output directory', type=str, help='Path to the directory where the output file is to be written.', nargs='?')
    args, unknown = parser.parse_known_args()

    return args


if __name__ == '__main__':
    main()