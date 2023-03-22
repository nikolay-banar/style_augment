from train import run_experiment
from predict import predict_from_generator
import argparse
import os
from keras import backend as K
import pandas as pd
from utils import DataGenerator
import pickle
import sys

from experiments import get_hash_set, get_hashed_string

def get_test_set():
    test = [["mimo", 'base', "./images/mimo/test"], ["minerva", 'base', "./images/minerva/test"]]
    return pd.DataFrame(test, columns=['origin', 'name', 'path'])


def get_add_train():
    # add base

    aug_mimo = ['shear10', 'shift10', 'rot15', 'zoom5', 'flip', 'art2']
    aug_mimo = ["./images/minerva/train"] + list(map(lambda x: os.path.join("./images/mimo/augmented", x), aug_mimo))
    n_aug_mimo = [400, 50, 50, 50, 50, 100, 50]
    common_mimo = [aug_mimo[:-1], n_aug_mimo[:-1]]
    art_mimo = [aug_mimo, n_aug_mimo]
    aug_minerva = ['shift15', 'shear15', 'rot15', 'zoom5', 'flip', 'art4']
    aug_minerva = ["./images/minerva/train"] + list(map(lambda x: os.path.join("./images/minerva/augmented", x), aug_minerva))
    n_aug_minerva = [400, 400, 50, 400, 50, 400, 25]
    common_minerva = [aug_minerva[:-1], n_aug_minerva[:-1]]
    art_minerva = [aug_minerva, n_aug_minerva]
    all_common = [aug_minerva[:-1] + aug_mimo[1:-1], n_aug_minerva[:-1] + n_aug_mimo[1:-1]]
    all_art = [aug_minerva + aug_mimo[1:], n_aug_minerva + n_aug_mimo[1:]]
    common_minerva_art_mimo = [aug_minerva[:-1] + [aug_mimo[-1]], n_aug_minerva[:-1] + [n_aug_mimo[-1]]]
    common_minerva_art_both = [aug_minerva + [aug_mimo[-1]], n_aug_minerva + [n_aug_mimo[-1]]]

    all_common_art_minerva = [aug_minerva + aug_mimo[1:-1], n_aug_minerva + n_aug_mimo[1:-1]]
    all_common_art_mimo = [aug_minerva[:-1] + aug_mimo[1:], n_aug_minerva[:-1] + n_aug_mimo[1:]]


    return {'common_mimo': common_mimo, 'art_mimo': art_mimo, 'common_minerva': common_minerva, 'art_minerva': art_minerva,
            'all_common': all_common, 'all_art': all_art, 'common_minerva_art_mimo': common_minerva_art_mimo,
            'common_minerva_art_both': common_minerva_art_both,
            'all_common_art_minerva': all_common_art_minerva, 'all_common_art_mimo': all_common_art_mimo}



def main():
    dev_set = 'minerva'
    train_set = 'minerva'
    dev = {"minerva": "./images/minerva/dev"}
    # train_base = {"mimo": "./images/mimo/train", "minerva": "./images/minerva/train"}
    # add_train = get_add_train(train_set)
    train_voc =  get_add_train()

    test = get_test_set()
    n_base=400

    weight_name, weights = 'ImageNet', './pretrained_models/ImageNet/V3'
    # n_samples = list(zip(6 * [400], [0, 25, 50, 100, 200, 400])) + list(zip([0, 25, 50, 100, 200], 5 * [400]))

    for exp in range(10):
        result_file = f'./add_csv/{exp}_{train_set}_{dev_set}_results.csv'
        results_frame = pd.read_csv(result_file, sep='\t') if os.path.isfile(result_file) else None
        hash_set = set() if results_frame is None else get_hash_set(results_frame)
        for distortion, train_data in train_voc.items():

            distortion_origin = distortion.split('_')[-1]

            data_gen = None
            model = None
            dev_results = None
            should_be_model_trained = True
            n_add = sum(train_data[1])
            for test_row in test.iterrows():
                test_name, test_path, test_origin = test_row[1]['name'], test_row[1]['path'], \
                                                    test_row[1]['origin']

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
                        train_gen = data_gen.get_training_generator(images_paths=train_data[0],
                                                                    amount=train_data[1],
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

                    if not os.path.isdir(f'./add_csv/'):
                        os.makedirs(f'./add_csv/')

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

    main()
