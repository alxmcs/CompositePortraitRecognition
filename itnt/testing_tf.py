import sklearn
import os
import sqlite3
from sklearn.datasets import load_breast_cancer
# Load dataset
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import json

if __name__ == "__main__":
    training_features = []
    training_labels = []
    folder_name = "training"

    conn = sqlite3.connect("C:\\CompositePortraitRecongnition\\db\\database.db")
    cursor = conn.cursor()

    res = cursor.execute("select id from person")
    ids_array = np.array(res.fetchall())
    ids_array_size = ids_array.size
    test_index = int(ids_array_size * 0.7)

    photo_true_str = 'photo_true_tf'
    sketch_true_str = 'sketch_true_tf'
    sketch_false_str = 'sketch_false_tf'

    for i in range(0, test_index):
        person_id = int(ids_array[i])
        photo_true_embed_json = cursor.execute("select value from embedding where embedding.person_id = ? "
                                          "and embedding.info = ?",
                                          [person_id, photo_true_str]).fetchone()
        photo_true_embed = json.loads(photo_true_embed_json)

        sketch_true_embed_json = cursor.execute("select value from embedding where embedding.person_id = ? "
                                           "and embedding.info = ?",
                                           [person_id, sketch_true_str]).fetchone()
        sketch_true_embed = json.loads(sketch_true_embed_json)

        sketch_false_embed_json = cursor.execute("select value from embedding where embedding.person_id = ? "
                                            "and embedding.info = ?",
                                            [person_id, sketch_false_str]).fetchone()
        sketch_false_embed = json.loads(sketch_false_embed_json)

        right_pair = np.concatenate((photo_true_embed, sketch_true_embed))
        wrong_pair = np.concatenate((photo_true_embed, sketch_false_embed))
        training_features.append(right_pair)
        training_labels.append(1)
        training_features.append(wrong_pair)
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

    testing_features = []
    folder_name = "test"
    for i in range(test_index, ids_array_size):
        person_id = ids_array[i]
        photo_true_embed = cursor.execute("select value from embedding where embedding.person_id = ? "
                                          "and embedding.info = ?",
                                          [person_id, photo_true_str])
        sketch_true_embed = cursor.execute("select value from embedding where embedding.person_id = ? "
                                           "and embedding.info = ?",
                                           [person_id, sketch_true_str])
        sketch_false_embed = cursor.execute("select value from embedding where embedding.person_id = ? "
                                            "and embedding.info = ?",
                                            [person_id, sketch_false_str])
        right_pair = np.concatenate((photo_true_embed, sketch_true_embed))
        wrong_pair = np.concatenate((photo_true_embed, sketch_false_embed))
        testing_features.append(right_pair)
        testing_features.append(wrong_pair)


    # preds = gnb.predict(testing_feautures)
    # print(preds)
    preds = clf.predict(testing_features)
    print(preds)
