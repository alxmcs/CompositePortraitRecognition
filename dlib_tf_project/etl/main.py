import datetime
import json
import os
import sqlite3

from PIL import Image

import dlib_tf_project.utils.tensorflow.face_encoding
import dlib_tf_project.utils.my_arcface.main

QUERIES = {
    'select_from_preprocessing': """Select name, comment from preprocessing""",
    'person_insert': """insert into person(name, patronymic, surname, comment, date_added) VALUES(?, ?, ?, ?, ?)""",
    'person_select': """select * from person""",
    'tf_embedding_insert': """insert into embedding(value, date_added, model_id, preprocessing_id, person_id, info) 
    VALUES(?, ?, ?, ?, ?, ?)""",
    'arc_embedding_insert': """insert into embedding(value, date_added, model_id, preprocessing_id, person_id, info) 
    VALUES(?, ?, ?, ?, ?, ?)"""

}


def get_photo_path_and_preprocessing_id(cursor, photo_path):
    cursor.execute(QUERIES['select_from_preprocessing'])
    rows = cursor.fetchall()

    for row in rows:
        print(row)

    inp = input("enter 1 if you want thumbnail or 0 if you don't want")
    if int(inp) == 1:
        size = int(input("enter the size you want to convert the image to: "))
        portrait_image = Image.open(photo_path)
        portrait_image.thumbnail((size, size))
        portrait_image.save("tmp_image.png")
        photo_path = "tmp_image.png"
        return photo_path, 1
    return photo_path, None


def get_person_data(json_data):
    name = json_data['name']
    patronymic = json_data['patronymic']
    surname = json_data['surname']
    comment = json_data['comment']

    date_added = datetime.datetime.now()
    data = [(name, patronymic, surname, comment, str(date_added))]
    return data


if __name__ == "__main__":
    with open('settings.json') as info_data:
        json_data = json.load(info_data)

    photo_path = json_data['photo_path']
    db_path = os.path.join('../../common/db', 'database.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    photo_path, preprocessing_id = get_photo_path_and_preprocessing_id(cursor, photo_path)

    tensorflow_embedding = dlib_tf_project.utils.tensorflow.face_encoding.get_encoding(photo_path)

    arcface_embedding = dlib_tf_project.utils.my_arcface.main.calculate_embedding(photo_path)

    data = get_person_data(json_data)

    cursor.executemany(QUERIES['person_insert'], data)
    cursor.execute(QUERIES['person_select'])

    print(cursor.fetchall())
    last_id_int = cursor.lastrowid
    print(last_id_int)

    embedding_data_tensorflow = [(str(tensorflow_embedding)), str(datetime.datetime.now()), 1, preprocessing_id,
                                 last_id_int, 'tf_photo_etl']
    cursor.execute(QUERIES['tf_embedding_insert'], embedding_data_tensorflow)
    embedding_data_arcface = [(str(arcface_embedding)), str(datetime.datetime.now()), 2, preprocessing_id, last_id_int,
                              'arc_photo_etl']
    cursor.execute(QUERIES['arc_embedding_insert'], embedding_data_tensorflow)
    conn.commit()
    conn.close()
