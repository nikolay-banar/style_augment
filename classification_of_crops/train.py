from utils import DataGenerator, get_pre_trained_model
import os
import tensorflow as tf
from keras.layers import Dense
from keras.models import Model
from keras.callbacks import CSVLogger, EarlyStopping
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import pickle
from keras.optimizers import Adam
import argparse
import numpy as np


def arguments():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('-train', type=str, help='path to train data', default=None)
    parser.add_argument('-dev', type=str, help='path to dev data', default=None)
    parser.add_argument('-model_path', type=str, help='path where the model is', default=None)
    # parser.add_argument('-net', type=str, help='ResNet or V3 or VGG19', default=None)
    parser.add_argument('-save', type=str, help='path to save results', default=None)
    # parser.add_argument('-lr', type=float, help='learning rate', default=None)

    return parser.parse_args()


def run_experiment(model_path, train_generator, valid_generator, save_path, lr=0.0001, save_model=False):
    nb_epochs = 200  # adjust the number of epochs
    # results_path = f'./results_up/{data_path}/{net_name}/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # data_generator = DataGenerator()

    csv_logger_callback = CSVLogger(os.path.join(save_path, 'results_file.csv'), append=True, separator=';')
    early_stopping_callback = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto',
                                            restore_best_weights=True
                                            )

    # train_generator = data_generator.get_training_generator(images_path=train_path, batch_size=32)
    # valid_generator = data_generator.get_validation_generator(images_path=dev_path, batch_size=32)
    # test_generator = data_generator.get_testing_generator()

    # print("Classes:", test_generator.class_indices)
    # print("Number of examples:", train_generator.n)

    pre_trained_input, pre_trained_output = get_pre_trained_model(model_path)
    # print(pre_trained_model.summary())
    # pre_trained_output = pre_trained_model.output

    predictions = Dense(len(valid_generator.class_indices), activation=tf.nn.softmax, name='final_output')(
        pre_trained_output)

    model = Model(input=pre_trained_input, output=predictions)
    # print(model.summary())
    adam = Adam(lr)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['accuracy'])

    model.fit_generator(generator=train_generator,
                        steps_per_epoch=train_generator.n // train_generator.batch_size,
                        validation_data=valid_generator,
                        validation_steps=valid_generator.n // valid_generator.batch_size,
                        epochs=nb_epochs,
                        callbacks=[csv_logger_callback, early_stopping_callback]
                        )
    if save_model:
        model.save(os.path.join(save_path, 'model.h5'))
        model.save_weights(os.path.join(save_path, 'weights.h5'))
    return model


if __name__ == "__main__":
    args = arguments()
    data_generator = DataGenerator()
    train_gen = data_generator.get_training_generator(images_paths=[args.train], batch_size=32)
    valid_gen = data_generator.get_validation_generator(images_paths=[args.dev], batch_size=25)
    run_experiment(model_path=args.model_path, train_generator=train_gen,
                   valid_generator=valid_gen, save_path=args.save, save_model=True)
