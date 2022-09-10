import datetime
import json
import sqlite3

from PIL import Image

import utils.tensorflow.face_encoding
import utils.my_arcface.main
if __name__ == "__main__":
    with open('settings.json') as info_data:
        json_data = json.load(info_data)

    photo_path = json_data['photo_path']

    conn = sqlite3.connect("C:\\CompositePortraitRecongnition\\db\\database.db")
    cursor = conn.cursor()
    cursor.execute("Select name, comment from preprocessing")
    rows = cursor.fetchall()

    for row in rows:
        print(row)

    inp = input("enter 1 if you want thumbnail or 0 if you don't want")
    preprocessing_id = 0
    if int(inp) == 1:
        size = int(input("enter the size you want to convert the image to: "))
        portrait_image = Image.open(photo_path)
        portrait_image.thumbnail((size, size))
        portrait_image.save("tmp_image.png")
        photo_path = "tmp_image.png"
        preprocessing_id = 1

    tensorflow_embedding = utils.tensorflow.face_encoding.get_encoding(photo_path)

    arcface_embedding = utils.my_arcface.main.calculate_embedding(photo_path)

    name = json_data['name']
    patronymic = json_data['patronymic']
    surname = json_data['surname']
    comment = json_data['comment']

    date_added = datetime.datetime.now()
    data = [(name, patronymic, surname, comment, str(date_added))]
    cursor.executemany("insert into person(name, patronymic, surname, comment, date_added) VALUES(?, ?, ?, ?, ?)",
                       data)
    cursor.execute("select * from person")
    print(cursor.fetchall())
    last_id_int = cursor.lastrowid
    print(last_id_int)
    embedding_data_tensorflow = [(str(tensorflow_embedding)), str(datetime.datetime.now()), 1, preprocessing_id,
                                 last_id_int]
    cursor.execute(
        "insert into embedding(value, date_added, model_id, preprocessing_id, person_id) VALUES(?, ?, ?, ?, ?",
        embedding_data_tensorflow)
    embedding_data_arcface = [(str(arcface_embedding)), str(datetime.datetime.now()), 2, preprocessing_id, last_id_int]
    cursor.execute(
        "insert into embedding(value, date_added, model_id, preprocessing_id, person_id) VALUES(?, ?, ?, ?, ?",
        embedding_data_tensorflow)
    conn.commit()
    conn.close()
