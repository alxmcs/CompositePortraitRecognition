import sqlite3
import numpy as np
import openpyxl
from matplotlib import pyplot as plt
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA


from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

names = [
    "Nearest Neighbors",
    "Linear SVM",
    "RBF SVM",
    "Gaussian Process",
    "Decision Tree",
    "Random Forest",
    "Neural Net",
    "AdaBoost",
    "Naive Bayes",
    "QDA",
]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=10000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(),
]

MODEL_ID = 1

QUERIES = {
    'same': """select value_p, value_s, pt_id, 1 as class  
        from 
        (select value as value_p, person_id as pt_id from embedding where model_id = 1 and info = 
        \'image_with_st_tf_tdcs\')
        inner join
        (select value as value_s, person_id as st_id from embedding where model_id = 1 and info = 
        \'sketch_true_tf_tdcs\')
        on pt_id = st_id 
        where value_p is not null and value_s is not null 
        order by pt_id asc""",
    'different': """select value_p, value_s, pt_id, 0 as class 
        from 
        (select value as value_p, person_id as pt_id from embedding where model_id = 1 and info = 
        \'image_with_random_st_tf_tdcs\')
        inner join
        (select value as value_s, person_id as st_id from embedding where model_id = 1 and info = 
        \'sketch_false_tf_tdcs\')
        on pt_id = st_id 
        where value_p is not null and value_s is not null 
        order by pt_id asc"""
}


def get_tf_data_from_db(key):
    conn = sqlite3.connect("..\\db\\database.db")
    cursor = conn.cursor()
    db_data = [list(item) for item in cursor.execute(QUERIES[key]).fetchall()]
    for row in db_data:
        row[0] = (np.array([float(x) for x in ''.join(row[0]).strip('[]').replace('\n', '').split(' ') if x]) -
                  np.array([float(x) for x in ''.join(row[1]).strip('[]').replace('\n', '').split(' ') if x]))
    return [np.array([db_data[x][0] for x in range(0, len(db_data))]),
            np.array([db_data[x][2] for x in range(0, len(db_data))], dtype=int),
            np.array([db_data[x][3] for x in range(0, len(db_data))], dtype=int)]


def get_arc_data_from_db(key):
    conn = sqlite3.connect("..\\db\\database.db")
    cursor = conn.cursor()
    db_data = [list(item) for item in cursor.execute(QUERIES[key]).fetchall()]
    for row in db_data:
        row[0] = row[0].replace('tf.Tensor(\n', '')
        row[0] = row[0].replace('\'[[', '\'[')
        row[0] = row[0].replace(', shape=(1, 512), dtype=float32)', '')
        row[1] = row[1].replace('tf.Tensor(\n', '')
        row[1] = row[1].replace('\'[[', '\'[')
        row[1] = row[1].replace(', shape=(1, 512), dtype=float32)', '')
        row[0] = (np.array([float(x) for x in ''.join(row[0]).strip('[]').replace('\n', '').split(' ') if x]) -
                  np.array([float(x) for x in ''.join(row[1]).strip('[]').replace('\n', '').split(' ') if x]))
    return [np.array([db_data[x][0] for x in range(0, len(db_data))]),
            np.array([db_data[x][2] for x in range(0, len(db_data))], dtype=int),
            np.array([db_data[x][3] for x in range(0, len(db_data))], dtype=int)]


def testing_PCA(test_n_components_array, data, target):
    vect_accuracy_n_comp = []
    tmp_data = data
    for n_comp in test_n_components_array:
        data = tmp_data
        pca = PCA(n_comp)
        pca.fit(data)
        data = pca.transform(data)
        print(data.shape)

        x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0,
                                                            stratify=target)
        clf = svm.NuSVC(gamma="auto")
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        print(f"n_component = {n_comp} NuSVC: {accuracy_score(y_test, y_pred)}")
        vect_accuracy_n_comp.append([n_comp, accuracy_score(y_test, y_pred)])
    return vect_accuracy_n_comp


if __name__ == "__main__":
    tf_true = get_tf_data_from_db('same')
    tf_false = get_tf_data_from_db('different')
    data = np.concatenate((tf_true[0], tf_false[0]))
    target = np.concatenate((tf_true[2], tf_false[2]))
    data = StandardScaler().fit_transform(data)
    print(data.shape)
    print(target.shape)
    tmp_data = data

    pca = PCA(.60)
    pca.fit(data)
    data = pca.transform(data)
    print(data.shape)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0, stratify=target)
    accuracy_array = []
    precision_array = []
    recall_array = []
    for name, clf in zip(names, classifiers):
        clf.fit(x_train, y_train)
        y_pred = clf.predict(x_test)
        accuracy_array.append(accuracy_score(y_test, y_pred))
        precision_array.append(precision_score(y_test, y_pred))
        recall_array.append(recall_score(y_test, y_pred))

    book = openpyxl.Workbook()
    sheet_1 = book.create_sheet("results", 0)
    headers = ['clf name', 'accuracy', 'precision', 'recall']
    sheet_1.append(headers)
    for name, ac_sc, pr, rec in zip(names, accuracy_array, precision_array, recall_array):
        sheet_1.append([name, ac_sc, pr, rec])
    book.save("results.xlsx")

