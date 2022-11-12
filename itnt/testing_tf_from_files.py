import os
# Load dataset
import os

# Load dataset
import numpy as np
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score


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


def get_samples_and_labels(model_folder_name, folder_name, count_samples, name_embed):
    training_features = []
    training_labels = []
    for i in range(0, count_samples):
        path_right = os.path.join("../itnt", f"{model_folder_name}/{folder_name}", "right",
                                  f"{name_embed}{i}.npy")
        np_arr = np.load(path_right)
        np_arr = np.ravel(np_arr)
        training_features.append(np_arr)
        training_labels.append(1)
        path_wrong = os.path.join("../itnt", f"{model_folder_name}/{folder_name}", "wrong",
                                  f"{name_embed}{i}.npy")
        np_arr = np.load(path_wrong)
        np_arr = np.ravel(np_arr)
        training_features.append(np_arr)
        training_labels.append(0)
    return np.array(training_features), np.array(training_labels)


def test_model(model_folder_name, training_folder_name, testing_folder_name, count_training_samples,
               count_testing_samples, name_embed):
    training_features, training_labels = get_samples_and_labels(model_folder_name, training_folder_name,
                                                                count_training_samples, name_embed)
    count_test_pairs = count_testing_samples * 2
    testing_features, testing_labels = get_samples_and_labels(model_folder_name, testing_folder_name,
                                                              count_testing_samples, name_embed)
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf = train_model(training_features, training_labels, clf)
    Pipeline(steps=[('standardscaler', StandardScaler()),
                    ('svc', SVC(gamma='auto'))])
    predicts, count_errors = test_samples_and_labels(testing_features, testing_labels, clf)
    accuracy = 1 - count_errors / count_test_pairs
    print(testing_labels)
    print(predicts)
    print(f"errors count = {count_errors} from {count_test_pairs}")
    return count_errors, accuracy


if __name__ == "__main__":
    model_folder_name = "for_arc"
    training_folder_name = "training"
    count_training_samples = 78
    name_embed = "arc_embed_"
    testing_folder_name = "test"
    count_testing_samples = 32
    count_errors, accuracy = test_model(model_folder_name, training_folder_name, testing_folder_name,
                                        count_training_samples, count_testing_samples, name_embed)
    print(f"Testing arcface without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    model_folder_name = "for_arc_with_st"
    count_errors, accuracy = test_model(model_folder_name, training_folder_name, testing_folder_name,
                                        count_training_samples, count_testing_samples, name_embed)
    print(f"Testing arcface with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    model_folder_name = "for_tf"
    name_embed = "tf_embed_"
    count_errors, accuracy = test_model(model_folder_name, training_folder_name, testing_folder_name,
                                        count_training_samples, count_testing_samples, name_embed)
    print(f"Testing tensorflow without style transfer\ncount errors = {count_errors} accuracy = {accuracy}")

    model_folder_name = "for_tf_with_st"
    count_errors, accuracy = test_model(model_folder_name, training_folder_name, testing_folder_name,
                                        count_training_samples, count_testing_samples, name_embed)
    print(f"Testing tensorflow with style transfer\ncount errors = {count_errors} accuracy = {accuracy}")
