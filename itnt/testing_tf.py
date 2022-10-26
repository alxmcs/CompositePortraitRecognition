import sklearn
import os
from sklearn.datasets import load_breast_cancer
# Load dataset
import numpy as np
from sklearn.naive_bayes import GaussianNB

if __name__ == "__main__":
    training_features = []
    training_labels = []
    folder_name = "training"
    # todo вынести в функцию
    for i in range(0, 78):
        path_right = os.path.join("../itnt", f"for_tf/{folder_name}", "right",
                            f"tf_embed_{i}.npy")
        training_features.append(np.fromfile(path_right))
        training_labels.append(1)
        path_wrong = os.path.join("../itnt", f"for_tf/{folder_name}", "wrong",
                                                       f"tf_embed_{i}.npy")
        training_features.append(np.fromfile(path_wrong))
        training_labels.append(0)

    gnb = GaussianNB()
    model = gnb.fit(training_features, training_labels)

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
    preds = gnb.predict(testing_feautures)
    print(preds)