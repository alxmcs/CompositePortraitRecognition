import sklearn
import os
from sklearn.datasets import load_breast_cancer
# Load dataset
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


if __name__ == "__main__":
    training_features = []
    training_labels = []
    folder_name = "training"
    # todo вынести в функцию
    for i in range(0, 78):
        path_right = os.path.join("../itnt", f"for_tf/{folder_name}", "right",
                            f"tf_embed_{i}.npy")
        np_arr = np.fromfile(path_right)
        if np.isnan(np_arr).any():
            print('we have nan')
        training_features.append(np.fromfile(path_right))
        training_labels.append(1)
        path_wrong = os.path.join("../itnt", f"for_tf/{folder_name}", "wrong",
                                                       f"tf_embed_{i}.npy")
        np_arr = np.fromfile(path_wrong)
        if np.isnan(np_arr).any():
            print('we have nan')
        training_features.append(np.fromfile(path_wrong))
        training_labels.append(0)

    for f in training_features:
        if np.isnan(f).any():
            print('we have nan')
    if np.isnan(training_features).any():
        print('we have nan')

    if np.isnan(training_labels).any():
        print('we have nan')

    # gnb = GaussianNB()
    # model = gnb.fit(np.array(training_features), np.array(training_labels))
    # todo ошибки возникают на этом этапе
    clf = make_pipeline(StandardScaler(), SVC(gamma='auto'))
    clf.fit(training_features, training_labels)

    testing_feautures = []
    folder_name = "test"
    for i in range(0, 32):
        path_right = os.path.join("../itnt", f"for_tf/{folder_name}", "right",
                            f"tf_embed_{i}.npy")
        testing_feautures.append(np.fromfile(path_right))
        path_wrong = os.path.join("../itnt", f"for_tf/{folder_name}", "wrong",
                                                       f"tf_embed_{i}.npy")
        testing_feautures.append(np.fromfile(path_wrong))
    # todo предикты все дают 0, что-то работает не так...
    # preds = gnb.predict(testing_feautures)
    # print(preds)
    preds = clf.predict(testing_feautures)
    print(preds)