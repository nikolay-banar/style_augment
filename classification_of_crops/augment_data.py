from shutil import copytree, copy
import glob
import os
from os.path import dirname, basename, join
from random import shuffle
from keras.preprocessing.image import ImageDataGenerator
import argparse
from utils import multiple_path_flow

def arguments():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('-source', type=str, help='path to train data', default=None)
    parser.add_argument('-save', type=str, help='path to dev data', default=None)

    return parser.parse_args()


def augment_from_path(source_path, augment_path, n_samples):
    prefix = basename(dirname(augment_path))
    print("PREFIX ", prefix)
    classes = set(map(lambda x: basename(dirname(x)), glob.glob(augment_path + '/**/*.*', recursive=True)))
    print(classes)
    aug_prefix = augment_path.split('/')[-2]
    for n in n_samples:
        new_path = join(source_path.replace('0', str(n)), f'{aug_prefix}')
        print("SAVE TO PATH", new_path)
        copytree(source_path, new_path)

    for class_name in classes:
        augment_files = glob.glob(join(augment_path, class_name) + '/*.*', recursive=True)
        shuffle(augment_files)
        # print(augment_files[:3])
        for n in n_samples:
            new_path = join(source_path.replace('0', str(n)), f'{aug_prefix}')
            sampled_list = augment_files[:n]
            for item in sampled_list:
                file_name = prefix + basename(item)
                # print("FILE NAME ", join(*[new_path, class_name, file_name]))
                copy(item, join(*[new_path, class_name, file_name]))


def std_augmentation(source_path, save_to):
    distortions = {'rot5': dict(rotation_range=5),
                   'rot10': dict(rotation_range=10),
                   'rot15': dict(rotation_range=15),
                   'zoom5': dict(zoom_range=0.05),
                   'zoom10': dict(zoom_range=0.1),
                   'zoom15': dict(zoom_range=0.15),
                   'shift5': dict(width_shift_range=0.05, height_shift_range=0.05),
                   'shift10': dict(width_shift_range=0.1, height_shift_range=0.1),
                   'shift15': dict(width_shift_range=0.15, height_shift_range=0.15),
                   'shear5': dict(shear_range=5),
                   'shear10': dict(shear_range=10),
                   'shear15': dict(shear_range=15),
                   'flip': dict(horizontal_flip=True)
                   }

    df = multiple_path_flow([source_path])
    for key, distortion in distortions.items():
        save_to_dir = os.path.join(save_to, key)
        print("DISTORTION", save_to_dir)
        for gr_key, group in df.groupby(['class']):

            if not os.path.isdir(os.path.join(save_to_dir, gr_key)):
                os.makedirs(os.path.join(save_to_dir, gr_key))

            data_generator = ImageDataGenerator(**distortion)
            gen = data_generator.flow_from_dataframe(group,
                                                     target_size=(224, 224),
                                                     color_mode="rgb",
                                                     batch_size=1,
                                                     class_mode="categorical",
                                                     shuffle=False,
                                                     seed=42,
                                                     save_to_dir=os.path.join(save_to_dir, gr_key),
                                                     save_prefix=key,
                                                     save_format='jpeg'
                                                     )

            steps = gen.n // gen.batch_size

            for x_batch, y_batch in gen:
                steps -= 1
                if steps == 0:
                    break


#
#
if __name__ == '__main__':
    args = arguments()
    std_augmentation(source_path=args.source, save_to=args.save)

# augment('./images/outstyle/0', './images/minerva/0', [25, 50, 100, 200, 400])
# augment('./images/outstyle/0', './images/mimo/0', [25, 50, 100, 200, 400])

# augment('./images/mimo/0', './images/minerva/0', [25, 50, 100, 200, 400])
# augment('./images/mimo/0', './images/outstyle/0', [25, 50, 100, 200, 400])
#
# augment('./images/minerva/0', './images/mimo/0', [25, 50, 100, 200, 400])
# augment('./images/minerva/0', './images/outstyle/0', [25, 50, 100, 200, 400])
