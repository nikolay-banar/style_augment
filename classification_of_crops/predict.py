from keras.models import load_model
from utils import DataGenerator
import os
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
import numpy as np
import argparse
import pickle


def arguments():
    parser = argparse.ArgumentParser(description='arguments')

    parser.add_argument('-test', type=str, help='path to data', default=None)
    parser.add_argument('-dev', type=str, help='path to data', default=None)
    parser.add_argument('-model', type=str, help='model', default=None)
    parser.add_argument('-save', type=str, help='model', default=None)

    return parser.parse_args()


def predict_from_generator(model, generator):
    output = model.predict_generator(generator=generator, steps=generator.n // generator.batch_size,
                                     pickle_safe=True)

    output = np.argmax(output, axis=1)
    accuracy = accuracy_score(generator.classes, output)
    f1_macro = f1_score(generator.classes, output, average='macro')
    f1_micro = f1_score(generator.classes, output, average='micro')
    error_matrix = confusion_matrix(generator.classes, output)

    return {'accuracy': accuracy, 'f1_macro': f1_macro, 'f1_micro': f1_micro, 'error_matrix': error_matrix,
            'class_indices': generator.class_indices}


def predict(dev_path, test_path, model, save_path, batch_size=1):
    data_generator = DataGenerator()
    test_generator = data_generator.get_testing_generator([test_path], None, batch_size)
    dev_generator = data_generator.get_testing_generator([dev_path],  None, batch_size)

    test_results = predict_from_generator(model, test_generator)
    dev_results = predict_from_generator(model, dev_generator)

    print(dev_results)
    print(test_results)

    with open(os.path.join(save_path, 'dev.pickle'), 'wb') as handle:
        pickle.dump(dev_results, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open(os.path.join(save_path, 'test.pickle'), 'wb') as handle:
        pickle.dump(test_results, handle, protocol=pickle.HIGHEST_PROTOCOL)


def predict_from_path(dev_path, test_path, model_path, save_path, batch_size=1):
    model = load_model(os.path.join(model_path, 'model.h5'))
    model.load_weights(os.path.join(model_path, 'weights.h5'))
    predict(dev_path, test_path, model, save_path, batch_size)


if __name__ == '__main__':
    args = arguments()
    predict_from_path(dev_path=args.dev, test_path=args.test, model_path=args.model, save_path=args.save, batch_size=25)
