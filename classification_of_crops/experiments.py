from train import run_experiment
from predict import predict_from_generator
import argparse
import os
from keras import backend as K
import pandas as pd
from utils import DataGenerator
import pickle
import sys


def arguments():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('-train', type=str, help='name of  train data', default=None)
    parser.add_argument('-dev', type=str, help='name of dev data', default=None)
    parser.add_argument('-exp', type=int, help='exp', default=None)

    return parser.parse_args()


def get_test_set():
    test = [["mimo", 'base', "./images/mimo/test"], ["minerva", 'base', "./images/minerva/test"]]

    for p in os.listdir("./images/mimo/art_test"):
        test.append(["mimo", p, os.path.join("./images/mimo/art_test", p)])

    for p in os.listdir("./images/minerva/art_test"):
        test.append(["minerva", p, os.path.join("./images/minerva/art_test", p)])

    return pd.DataFrame(test, columns=['origin', 'name', 'path'])


def get_add_train(train_set):
    add_train = []
    for p in os.listdir("./images/mimo/augmented"):
        add_train.append(["mimo", p, os.path.join("./images/mimo/augmented", p)])

    for p in os.listdir("./images/minerva/augmented"):
        add_train.append(["minerva", p, os.path.join("./images/minerva/augmented", p)])

    if train_set == 'mimo':
        add_train.append(["minerva", 'no', "./images/minerva/train"])
    elif train_set == 'minerva':
        add_train.append(["mimo", 'no', "./images/mimo/train"])

    return pd.DataFrame(add_train, columns=['origin', 'distortion', 'path'])


def get_hash_set(df):
    columns = ['exp', 'base_train', 'n_base', 'dev', 'weights',
               'dist_origin', 'distortion', 'n_add',
               'test_origin', 'test_name']

    output = ''
    for col in columns:
        output += df[col].astype(str) + '/'
    return set(output)


def get_hashed_string(my_list):
    return '/'.join(my_list) + '/'


def main(train_set, dev_set, exp):
    dev = {"mimo": "./images/mimo/dev", "minerva": "./images/minerva/dev"}
    train_base = {"mimo": "./images/mimo/train", "minerva": "./images/minerva/train"}
    add_train = get_add_train(train_set)

    test = get_test_set()

    source_weights = {'ECCVModels': './pretrained_models/ECCVModels/V3', 'ImageNet': './pretrained_models/ImageNet/V3'}
    # source_weights = {'ImageNet': './pretrained_models/ImageNet/V3'}
    n_samples = list(zip(6 * [400], [0, 25, 50, 100, 200, 400])) + list(zip([0, 25, 50, 100, 200], 5 * [400]))\
                + list(zip([25, 50, 100, 200], 4 * [0]))
    result_file = f'./csv/{exp}_{train_set}_{dev_set}_results.csv'
    results_frame = pd.read_csv(result_file, sep='\t') if os.path.isfile(result_file) else None
    hash_set = set() if results_frame is None else get_hash_set(results_frame)

    for n_base, n_add in n_samples:
        for dist_row in add_train.iterrows():
            distortion, distortion_path, distortion_origin = dist_row[1]['distortion'], dist_row[1]['path'], \
                                                             dist_row[1]['origin']
            for weight_name, weights in source_weights.items():

                data_gen = None
                model = None
                dev_results = None
                should_be_model_trained = True
                for test_row in test.iterrows():

                    test_name, test_path, test_origin = test_row[1]['name'], test_row[1]['path'], \
                                                        test_row[1]['origin']

                    if n_add == 0:
                        distortion_origin = 'No'
                        distortion = 'No'

                    ref_to_result = hash((exp, train_set, n_base, dev_set, weight_name, distortion_origin,
                                          distortion, n_add, test_origin, test_name))

                    hash_string = get_hashed_string(
                        [str(exp), train_set, str(n_base), dev_set, weight_name, distortion_origin,
                         distortion, str(n_add), test_origin, test_name])
                    # 0 / mimo / 400 / mimo / ImageNet / minerva / no / 400 / minerva / art6 /

                    if hash_string in hash_set:
                        print(hash_string)
                    else:
                        logs = f'./logs/{exp}_{train_set}_{n_add}+{dev_set}_{weight_name}_{distortion_origin}_{distortion}_{n_add}'

                        if should_be_model_trained:
                            print("TRAIN GENERATOR:", hash_string)
                            data_gen = DataGenerator()
                            train_gen = data_gen.get_training_generator(images_paths=[train_base[train_set],
                                                                                      distortion_path],
                                                                        amount=[n_base, n_add],
                                                                        batch_size=32)
                            valid_gen = data_gen.get_validation_generator(images_paths=[dev[dev_set]],
                                                                          batch_size=25)

                            model = run_experiment(model_path=weights, train_generator=train_gen,
                                                   valid_generator=valid_gen,
                                                   save_path=logs)

                            dev_results = predict_from_generator(model,
                                                                 data_gen.get_testing_generator(
                                                                     images_paths=[dev[dev_set]],
                                                                     batch_size=25))
                            should_be_model_trained = False

                        test_gen = data_gen.get_testing_generator(images_paths=[test_path], batch_size=25)

                        test_results = predict_from_generator(model, test_gen)

                        cur_results_frame = pd.DataFrame([[exp, train_set, n_base, dev_set, weight_name,
                                                           distortion_origin, distortion, n_add,
                                                           test_origin, test_name, dev_results['accuracy'],
                                                           dev_results['f1_macro'], test_results['accuracy'],
                                                           test_results['f1_macro'], ref_to_result]],
                                                         columns=['exp', 'base_train', 'n_base', 'dev', 'weights',
                                                                  'dist_origin', 'distortion', 'n_add',
                                                                  'test_origin', 'test_name', 'dev_acc',
                                                                  'dev_f1', 'test_acc',
                                                                  'test_f1', 'hash'])

                        if not os.path.isdir(f'./results/{ref_to_result}'):
                            os.makedirs(f'./results/{ref_to_result}')

                        if not os.path.isdir(f'./csv/'):
                            os.makedirs(f'./csv/')

                        if results_frame is None:
                            cur_results_frame.to_csv(result_file, sep='\t')
                            results_frame = cur_results_frame
                        else:
                            with open(result_file, 'a') as f:
                                cur_results_frame.to_csv(f, header=False, sep='\t')

                        with open(os.path.join(f'./results/{ref_to_result}', 'dev.pickle'), 'wb') as handle:
                            pickle.dump(dev_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        with open(os.path.join(f'./results/{ref_to_result}', 'test.pickle'), 'wb') as handle:
                            pickle.dump(test_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

                        hash_set.add(hash_string)
                K.clear_session()


if __name__ == '__main__':
    args = arguments()
    # for e in range(5*args.exp, 5*(args.exp + 1)):
    main(train_set=args.train, dev_set=args.dev, exp=args.exp)
