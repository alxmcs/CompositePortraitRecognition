import datetime
import sqlite3
import io
import numpy as np
import json


def adapt_array(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())


def convert_array(text):
    out = io.BytesIO(text)
    out.seek(0)
    return np.load(out)





def insert_person(cursor, name, patronymic, surname, comment, date_added):
    person_data = [(name, patronymic, surname, comment, str(date_added))]
    cursor.executemany(
        "insert into person(name, patronymic, surname, comment, date_added) VALUES(?, ?, ?, ?, ?)",
        person_data)
    cursor.execute("select * from person")
    print(cursor.fetchall())
    print(cursor.lastrowid)
    return cursor.lastrowid


def insert_embedding(cursor, embedding, date_added, model_id, person_id, preprocessing_id, info):
    embedding_data = [embedding, str(date_added), model_id, preprocessing_id,
                      person_id, info]
    cursor.execute(
        "insert into embedding(value, date_added, model_id, preprocessing_id, person_id, info) VALUES(?, ?, ?, ?, ?, ?)",
        embedding_data)


def clear_db(cursor, table_names):
    for table_name in table_names:
        if table_name == 'embedding':
            cursor.execute('delete from embedding')
        if table_name == 'model':
            cursor.execute('delete from model')
        if table_name == 'person':
            cursor.execute('delete from person')
        if table_name == 'preprocessing':
            cursor.execute('delete from preprocessing')


if __name__ == "__main__":
    # Converts np.array to TEXT when inserting
    sqlite3.register_adapter(np.ndarray, adapt_array)

    # Converts TEXT to np.array when selecting
    sqlite3.register_converter("array", convert_array)

    conn = sqlite3.connect("C:\\CompositePortraitRecongnition\\db\\database.db")
    cursor = conn.cursor()
    clear_db(cursor, ['embedding', 'person'])
    conn.commit()
    conn.close()
