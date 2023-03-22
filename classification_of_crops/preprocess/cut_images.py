import torchvision.transforms as transforms
from PIL import Image
from os.path import join as jp
import os
import pandas as pd
import argparse


def arguments():
    parser = argparse.ArgumentParser(description='CharNMT arguments')

    parser.add_argument('-data', type=str, help='path to data', default=None)
    parser.add_argument('-splits', type=str, help='path to splits', default=None)
    parser.add_argument('-save', type=str, help='path to save processed images', default=None)
    parser.add_argument("--crop", default=False, action="store_true", help="Flag to crop")
    parser.add_argument("--square", default=False, action="store_true", help="Flag to square")

    return parser.parse_args()


def square_image(im, size=(224, 224)):
    width, height = im.size

    delta = width - height
    if delta > 0:
        transform = transforms.Compose([transforms.Pad((0, delta // 2, 0, delta - delta // 2)),
                                        transforms.Resize(size)])
        im = transform(im)
    elif delta < 0:
        transform = transforms.Compose([transforms.Pad((-delta // 2, 0, -delta + delta // 2, 0)),
                                        transforms.Resize(size)])
        im = transform(im)
    else:
        transform = transforms.Compose([transforms.Resize(size)])
        im = transform(im)

    return im


def map_images(data_path, splits_path, part, save_path, crop=True, square=True):
    df = pd.read_csv(os.path.join(splits_path, part + '.txt'))

    for row in df.iterrows():
        file_name = row[1]['image_filename']

        instrument = row[1]['lev1'] if 'hypernym' in splits_path else row[1]['instrument_name']

        if crop:
            bb = row[1]['bounding_box_coordinates'].split(",")
            bb = list(map(int, bb))
            x_min, y_min, x_max, y_max = bb

            if min(x_min, y_min) < 0:
                print(f"The annotation is corrupted {x_min} and {x_min}:", file_name)
                continue

        try:
            # img = Image.open(os.path.join(data_path, file_name.replace('%22', '"')))
            img = Image.open(os.path.join(data_path, file_name))
        except:
            print("The annotation is corrupted:", file_name)
            continue

        cropped = img.crop((x_min, y_min, x_max, y_max)) if crop else img
        cropped = square_image(cropped) if square else cropped

        class_path = jp(jp(save_path, part), instrument)

        if not os.path.isdir(class_path):
            os.makedirs(class_path)

        cropped.save(jp(class_path, str(row[0]) + '_' + file_name.replace('/', '_')))


if __name__ == '__main__':
    args = arguments()

    for p in ("train", "dev", "test"):
        map_images(data_path=args.data,
                   splits_path=args.splits,
                   part=p,
                   save_path=args.save,
                   crop=args.crop, square=args.square)
