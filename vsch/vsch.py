from helpers.handling import loadData
from helpers.processing import Picture
import argparse
import glob


def main():
    args = parse_args()
    
    # Method auto-imported from helpers.py to load images data
    #image_data, image_list = loadData("C:\\projects\\vision-system-counting-holes\\data\\input.json")
    image_data, image_list = loadData("data\\input.json")

    # Output dictionary declaration
    output_dict = {}

    # Update dictionary with image list
    for i, img in enumerate(image_list):
        output_dict.update({img : None})


    for img in image_list:
        image_path = glob.glob(f'{args.im}{img}.jpg')
        print(image_path)
        image = Picture(image_path[0])
        image.preprocessing()
        image.count_objects()
        print(image.objects)

    print("BELOW")
    print(output_dict)


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('im', metavar='Images directory', type=str, help='Directory path where images are located.', nargs='?')
    parser.add_argument('obp', metavar='Objects data directory', type=str, help='Directory path where image objects description data is located.', nargs='?')
    parser.add_argument('out', metavar='Output directory', type=str, help='Path to the directory where the output file is to be written.', nargs='?')
    args, unknown = parser.parse_known_args()

    return args


if __name__ == '__main__':
    main()