# Load dataset
import os

# Load dataset
import numpy as np
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import itnt.Constants as Constants


def get_count_errors(testing_labels, predicts):
    cnt = 0
    errors_count = 0
    for predict in predicts:
        if predict != testing_labels[cnt]:
            errors_count += 1
        cnt += 1
    return errors_count


def train_model(training_features, training_labels, model):
    model.fit(training_features, training_labels)
    return model


def test_samples_and_labels(testing_features, testing_labels, model):
    preds = model.predict(testing_features)
    count_errors = get_count_errors(testing_labels, preds)
    return preds, count_errors


def get_samples_and_labels(dataset_name, model_folder_name, folder_name, count_samples, name_embed):
    training_features = []
    training_labels = []
    for i in range(0, count_samples):
        path_right = os.path.join("", dataset_name, f"{model_folder_name}/{folder_name}", "right",
                                  f"{name_embed}{i}.npy")
        np_arr = np.load(path_right)
        np_arr = np.ravel(np_arr)
        training_features.append(np_arr)
        training_labels.append(1)
        path_wrong = os.path.join("",dataset_name, f"{model_folder_name}/{folder_name}", "wrong",
                                  f"{name_embed}{i}.npy")
        np_arr = np.load(path_wrong)
        np_arr = np.ravel(np_arr)
        training_features.append(np_arr)
        training_labels.append(0)
    return np.array(training_features), np.array(training_labels)


def test_model(dataset_name, model_folder_name, training_folder_name, testing_folder_name, count_training_samples,
               count_testing_samples, name_embed):
    training_features, training_labels = get_samples_and_labels(dataset_name, model_folder_name, training_folder_name,
                                                                count_training_samples, name_embed)
    count_test_pairs = count_testing_samples * 2
    testing_features, testing_labels = get_samples_and_labels(dataset_name, model_folder_name, testing_folder_name,
                                                              count_testing_samples, name_embed)

    # clf = GaussianNB()
    # clf = train_model(training_features, training_labels, clf)

    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(training_features, training_labels)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('svc', SVC(gamma='auto'))])

    predicts, count_errors = test_samples_and_labels(testing_features, testing_labels, clf)
    accuracy = 1 - count_errors / count_test_pairs
    return count_errors, accuracy


if __name__ == "__main__":
    # """
    # Testing TDCS
    # """
    # count_errors, accuracy = test_model(Constants.TDCS_DATASET_NAME, Constants.ARCFACE_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
    #                                     Constants.TESTING_FOLDER_NAME, Constants.COUNT_TRAINING_SAMPLES,
    #                                     Constants.COUNT_TESTING_SAMPLES, Constants.ARCFACE_EMBED_CONC_NAME)
    # print(f"Testing arcface without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")
    #
    # count_errors, accuracy = test_model(Constants.TDCS_DATASET_NAME, Constants.ARCFACE_WITH_ST_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
    #                                     Constants.TESTING_FOLDER_NAME, Constants.COUNT_TRAINING_SAMPLES,
    #                                     Constants.COUNT_TESTING_SAMPLES, Constants.ARCFACE_EMBED_CONC_NAME)
    # print(f"Testing arcface with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")
    #
    # count_errors, accuracy = test_model(Constants.TDCS_DATASET_NAME, Constants.TENSORFLOW_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
    #                                     Constants.TESTING_FOLDER_NAME, Constants.COUNT_TRAINING_SAMPLES,
    #                                     Constants.COUNT_TESTING_SAMPLES, Constants.TENSORFLOW_EMBED_CONC_NAME)
    # print(f"Testing tensorflow without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")
    #
    # count_errors, accuracy = test_model(Constants.TDCS_DATASET_NAME, Constants.TENSORFLOW_WITH_ST_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
    #                                     Constants.TESTING_FOLDER_NAME, Constants.COUNT_TRAINING_SAMPLES,
    #                                     Constants.COUNT_TESTING_SAMPLES, Constants.TENSORFLOW_EMBED_CONC_NAME)
    # print(f"Testing tensorflow with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")
    #
    # """
    # Testing vectors (euclidean(u,v), chebyshev(u, v), distance.cosine(u, v))
    # """
    # count_errors, accuracy = test_model(Constants.TDCS_DATASET_NAME, Constants.ARCFACE_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
    #                                     Constants.TESTING_FOLDER_NAME, Constants.COUNT_TRAINING_SAMPLES,
    #                                     Constants.COUNT_TESTING_SAMPLES, Constants.ARCFACE_EMBED_DISTANCE_NAME)
    # print(f"Testing arcface(using distances) without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")
    #
    # count_errors, accuracy = test_model(Constants.TDCS_DATASET_NAME, Constants.ARCFACE_WITH_ST_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
    #                                     Constants.TESTING_FOLDER_NAME, Constants.COUNT_TRAINING_SAMPLES,
    #                                     Constants.COUNT_TESTING_SAMPLES, Constants.ARCFACE_EMBED_DISTANCE_NAME)
    # print(f"Testing arcface(using distances) with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")
    #
    # count_errors, accuracy = test_model(Constants.TDCS_DATASET_NAME, Constants.TENSORFLOW_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
    #                                     Constants.TESTING_FOLDER_NAME, Constants.COUNT_TRAINING_SAMPLES,
    #                                     Constants.COUNT_TESTING_SAMPLES, Constants.TENSORFLOW_EMBED_DISTANCE_NAME)
    # print(f"Testing tensorflow(using distances) without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")
    #
    # count_errors, accuracy = test_model(Constants.TDCS_DATASET_NAME, Constants.TENSORFLOW_WITH_ST_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
    #                                     Constants.TESTING_FOLDER_NAME, Constants.COUNT_TRAINING_SAMPLES,
    #                                     Constants.COUNT_TESTING_SAMPLES, Constants.TENSORFLOW_EMBED_DISTANCE_NAME)
    # print(f"Testing tensorflow(using distances) with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    """
    Testing CUHK
    """
    count_errors, accuracy = test_model(Constants.CUHK_DATASET_NAME, Constants.ARCFACE_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
                                        Constants.TESTING_FOLDER_NAME, Constants.COUNT_TESTING_SAMPLES_CUHK,
                                        Constants.COUNT_TESTING_SAMPLES_CUHK, Constants.ARCFACE_EMBED_CONC_NAME)
    print(f"Testing arcface without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    count_errors, accuracy = test_model(Constants.CUHK_DATASET_NAME, Constants.ARCFACE_WITH_ST_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
                                        Constants.TESTING_FOLDER_NAME, Constants.COUNT_TESTING_SAMPLES_CUHK,
                                        Constants.COUNT_TESTING_SAMPLES_CUHK, Constants.ARCFACE_EMBED_CONC_NAME)
    print(f"Testing arcface with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    count_errors, accuracy = test_model(Constants.CUHK_DATASET_NAME, Constants.TENSORFLOW_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
                                        Constants.TESTING_FOLDER_NAME, Constants.COUNT_TESTING_SAMPLES_CUHK,
                                        Constants.COUNT_TESTING_SAMPLES_CUHK, Constants.TENSORFLOW_EMBED_CONC_NAME)
    print(f"Testing tensorflow without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    count_errors, accuracy = test_model(Constants.CUHK_DATASET_NAME, Constants.TENSORFLOW_WITH_ST_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
                                        Constants.TESTING_FOLDER_NAME, Constants.COUNT_TESTING_SAMPLES_CUHK,
                                        Constants.COUNT_TESTING_SAMPLES_CUHK, Constants.TENSORFLOW_EMBED_CONC_NAME)
    print(f"Testing tensorflow with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    """
    Testing vectors (euclidean(u,v), chebyshev(u, v), distance.cosine(u, v))
    """
    count_errors, accuracy = test_model(Constants.CUHK_DATASET_NAME, Constants.ARCFACE_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
                                        Constants.TESTING_FOLDER_NAME, Constants.COUNT_TESTING_SAMPLES_CUHK,
                                        Constants.COUNT_TESTING_SAMPLES_CUHK, Constants.ARCFACE_EMBED_DISTANCE_NAME)
    print(f"Testing arcface(using distances) without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    count_errors, accuracy = test_model(Constants.CUHK_DATASET_NAME, Constants.ARCFACE_WITH_ST_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
                                        Constants.TESTING_FOLDER_NAME, Constants.COUNT_TESTING_SAMPLES_CUHK,
                                        Constants.COUNT_TESTING_SAMPLES_CUHK, Constants.ARCFACE_EMBED_DISTANCE_NAME)
    print(f"Testing arcface(using distances) with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    count_errors, accuracy = test_model(Constants.CUHK_DATASET_NAME, Constants.TENSORFLOW_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
                                        Constants.TESTING_FOLDER_NAME, Constants.COUNT_TESTING_SAMPLES_CUHK,
                                        Constants.COUNT_TESTING_SAMPLES_CUHK, Constants.TENSORFLOW_EMBED_DISTANCE_NAME)
    print(f"Testing tensorflow(using distances) without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    count_errors, accuracy = test_model(Constants.CUHK_DATASET_NAME, Constants.TENSORFLOW_WITH_ST_FOLDER_NAME, Constants.TRAINING_FOLDER_NAME,
                                        Constants.TESTING_FOLDER_NAME, Constants.COUNT_TESTING_SAMPLES_CUHK,
                                        Constants.COUNT_TESTING_SAMPLES_CUHK, Constants.TENSORFLOW_EMBED_DISTANCE_NAME)
    print(f"Testing tensorflow(using distances) with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")