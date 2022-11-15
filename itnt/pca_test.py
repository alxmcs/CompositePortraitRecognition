import sqlite3
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.decomposition import PCA

QUERIES = {
    'same': """select value_p, value_s, pt_id, 1 as class  
        from 
        (select value as value_p, person_id as pt_id from embedding where model_id = 1 and info = \'photo_true_tf\')
        inner join
        (select value as value_s, person_id as st_id from embedding where model_id = 1 and info = \'sketch_true_tf\')
        on pt_id = st_id 
        where value_p is not null and value_s is not null 
        order by pt_id asc""",
    'different': """select value_p, value_s, pt_id, 0 as class 
        from 
        (select value as value_p, person_id as pt_id from embedding where model_id = 1 and info = \'photo_true_tf\')
        inner join
        (select value as value_s, person_id as st_id from embedding where model_id = 1 and info = \'sketch_false_tf\')
        on pt_id = st_id 
        where value_p is not null and value_s is not null 
        order by pt_id asc"""
}


def get_tf_data_from_db(key):
    conn = sqlite3.connect("..\\db\\database.db")
    cursor = conn.cursor()
    db_data = [list(item) for item in cursor.execute(QUERIES[key]).fetchall()]
    for row in db_data:
        # row[0] = np.concatenate((np.array([float(x) for x in ''.join(row[0]).strip('[]').replace('\n', '').split(' ') if x]),
        #                        np.array([float(x) for x in ''.join(row[1]).strip('[]').replace('\n', '').split(' ') if x])))
        # вместо конкатенации ебанул поэлементную разность
        row[0] = (np.array([float(x) for x in ''.join(row[0]).strip('[]').replace('\n', '').split(' ') if x]) -
                  np.array([float(x) for x in ''.join(row[1]).strip('[]').replace('\n', '').split(' ') if x]))
    return [np.array([db_data[x][0] for x in range(0, len(db_data))]),
            np.array([db_data[x][2] for x in range(0, len(db_data))], dtype=int),
            np.array([db_data[x][3] for x in range(0, len(db_data))], dtype=int)]


if __name__ == "__main__":
    tf_true = get_tf_data_from_db('same')
    tf_false = get_tf_data_from_db('different')
    data = np.concatenate((tf_true[0], tf_false[0]))
    target = np.concatenate((tf_true[2], tf_false[2]))
    data = StandardScaler().fit_transform(data)
    print(data.shape)
    print(target.shape)

    pca = PCA(.90) # при таких параметрах вроде получаются наилучшие значения - но попробуй сам
    pca.fit(data)
    data = pca.transform(data)
    print(data.shape)

    x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=0, stratify=target)
    clf = svm.NuSVC(gamma="auto")
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(f"NuSVC: {accuracy_score(y_test, y_pred)}")

    mlp = MLPClassifier(alpha=0.5, max_iter=10000)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(x_test)
    print(f"MLPClassifier: {accuracy_score(y_test, y_pred)}")

    rfc = RandomForestClassifier(max_depth=20, n_estimators=10, max_features=10)
    rfc.fit(x_train, y_train)
    y_pred = rfc.predict(x_test)
    print(f"RandomForestClassifier: {accuracy_score(y_test, y_pred)}")

    qda = QuadraticDiscriminantAnalysis()
    qda.fit(x_train, y_train)
    y_pred = qda.predict(x_test)
    print(f"QuadraticDiscriminantAnalysis: {accuracy_score(y_test, y_pred)}")
