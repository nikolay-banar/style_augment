from cut_images import square_image
import argparse
import glob
from PIL import Image
import os


def arguments():
    parser = argparse.ArgumentParser(description='CharNMT arguments')

    parser.add_argument('-path', type=str, help='path to data', default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = arguments()
    files = glob.glob(args.path + '/*/*.*', recursive=True)
    for file in files:
        img = Image.open(os.path.join(file))
        img = square_image(img)
        img.save(file)
