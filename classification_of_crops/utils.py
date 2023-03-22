from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
import os
from keras.applications import VGG19, ResNet50, InceptionV3
import glob
import pandas as pd
import random


class DataGenerator:
    def __init__(self):
        self.data_generator = ImageDataGenerator()

    def get_training_generator(self, images_paths, amount=None, batch_size=32):
        df = multiple_path_flow(images_paths, amount)
        print("IMAGE PATH", images_paths)
        print("IMAGE PATH", amount)
        print(df.head(), df.shape)
        train_generator = self.data_generator.flow_from_dataframe(
            dataframe=df,
            target_size=(224, 224),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=True,
            seed=42
        )
        # train_generator = self.data_generator.flow_from_directory(
        #     directory=images_path,
        #     target_size=(224, 224),
        #     color_mode="rgb",
        #     batch_size=batch_size,
        #     class_mode="categorical",
        #     shuffle=True,
        #     seed=42
        # )

        return train_generator

    def get_validation_generator(self, images_paths, amount=None, batch_size=32):
        df = multiple_path_flow(images_paths, amount)
        valid_generator = self.data_generator.flow_from_dataframe(
            dataframe=df,
            target_size=(224, 224),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
            seed=42
        )

        return valid_generator

    def get_testing_generator(self, images_paths, amount=None, batch_size=1):
        df = multiple_path_flow(images_paths, amount)
        testing_generator = self.data_generator.flow_from_dataframe(
            dataframe=df,
            target_size=(224, 224),
            color_mode="rgb",
            batch_size=batch_size,
            class_mode="categorical",
            shuffle=False,
            seed=42
        )

        return testing_generator


def multiple_path_flow(paths, amount=None):
    dfs = []
    for i, p in enumerate(paths):
        if amount is not None and amount[i] == 0:
            continue

        files = glob.glob(p + '/*/*.*', recursive=True)
        classes = map(lambda x: x.split('/')[-2], files)
        random.Random(4).shuffle(files)
        df = pd.DataFrame(zip(classes, files), columns=['class', 'filename'])
        if amount is not None and amount[i] is not None:
            df = df.groupby(["class"]).head(amount[i])
        dfs.append(df)

    dfs = pd.concat(dfs)
    return dfs
    # shuffle(augment_files)


def download_pre_trained_model(save_path='./ImageNet/', net_name='ResNet'):
    """
    Choose one of the different pre-trained models from the ECCV paper
    :return: one of the models + weights
    """
    models = {"VGG": VGG19, "ResNet": ResNet50, "V3": InceptionV3}

    pre_trained_model = models[net_name](include_top=False, weights='imagenet',
                                         input_tensor=None, input_shape=None,
                                         pooling='avg')
    # input_shape = (224, 224, 3)
    model_path = os.path.join(save_path, net_name)
    # print("here")

    if not os.path.exists(model_path):
        os.makedirs(model_path)

    pre_trained_model.save(os.path.join(model_path, 'model.h5'))
    pre_trained_model.save_weights(os.path.join(model_path, 'weights.h5'))
    # print(pre_trained_model.summary())

    # return pre_trained_model


def get_pre_trained_model(model_path='./ECCVModels/'):
    """
    Choose one of the different pre-trained models from the ECCV paper
    :return: one of the models + weights
    """
    structure_path = glob.glob(os.path.join(model_path, f'*model.h5'))[0]
    weights_path = glob.glob(os.path.join(model_path, f'*weights.h5'))[0]
    pre_trained_model = load_model(structure_path)
    pre_trained_model.load_weights(weights_path)

    if "Softmax" in pre_trained_model.output.name:
        return pre_trained_model.input, pre_trained_model.layers[-2].output
    else:
        return pre_trained_model.input, pre_trained_model.output


if __name__ == '__main__':
    # get_pre_trained_model(model_path='./pretrained_models/ECCVModels/V3')
    # download_pre_trained_model(net_name='ResNet')
    # download_pre_trained_model(net_name='VGG')
    # download_pre_trained_model(save_path='./pretrained_models/ImageNet/', net_name='V3')
    # download_pre_trained_model(save_path='./pretrained_models/ImageNet/', net_name='V3')
    # get_pre_trained_model(model_path='./pretrained_models/ImageNet/V3')
    # get_pre_trained_model(model_path='./pretrained_models/ECCVModels/V3')

    gen = DataGenerator()
    print(gen.get_training_generator(['./images/mimo/0', './images/minerva/0'], [400, 25]))
    print(gen.get_validation_generator(['./images/mimo/dev']))
    print(gen.get_testing_generator(['./images/mimo/test']))
