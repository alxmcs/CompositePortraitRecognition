import datetime
import json
import sqlite3
from PIL import Image
import cv2
from arcface.lib import ArcFaceModel

import utils.tensorflow.face_encoding

if __name__ == "__main__":
    with open('settings.json') as info_data:
        json_data = json.load(info_data)

    photo_path = json_data['photo_path']

    conn = sqlite3.connect("database.db")
    cursor = conn.cursor()
    cursor.execute("Select name, comment from model")
    rows = cursor.fetchall()

    for row in rows:
        print(f"Name:{row.name}, comment:{row.comment}")

    inp = input("select preprocessing(enter 0 if you don't want)")
    if inp != 0:
        if inp == 1:
            size = int(input("enter the size you want to convert the image to: "))
            portrait_image = Image.open(photo_path)
            portrait_image.thumbnail((size, size))
            portrait_image.save("tmp_image.png")
            photo_path = "tmp_image.png"

    tensorflow_embedding = utils.tensorflow.face_encoding.get_encoding(photo_path)

    arcface_embedding = utils.my_arcface.calculate_embedding(photo_path)

    name = json_data['name']
    patronymic = json_data['patronymic']
    surname = json_data['surname']
    comment = json_data['comment']

    date_added = datetime.datetime.now()
    data = [(name, patronymic, surname, comment, date_added)]
    cursor.executemany("insert into person(name, patronymic, surname, comment, date_added) VALUES(?, ?, ?, ?, ?)",
                       data)
