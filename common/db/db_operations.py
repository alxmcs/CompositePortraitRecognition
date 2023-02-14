import datetime
import os.path
import sqlite3

QUERIES = {
    'insert_person': """insert into person(name, patronymic, surname, comment, date_added) VALUES(?, ?, ?, ?, ?)""",
    'select_person': """select * from person""",
    'insert_embedding': """insert into embedding(value, date_added, model_id, preprocessing_id, person_id, info) 
    VALUES(?, ?, ?, ?, ?, ?)""",
    'delete_from_embedding': """delete from embedding""",
    'delete_from_model': """delete from model""",
    'delete_from_person': """delete from person""",
    'delete_from_preprocessing': """delete from preprocessing""",
    'insert_model': """insert into model(name, comment, date_added) VALUES (?, ?, ?)""",
    'get_model_id_by_name': """select id from model where name = ?"""
}


def init_models(cursor):
    models = [
        "VGG-Face",
        "Facenet",
        "Facenet512",
        "OpenFace",
        "DeepFace",
        "DeepID",
        "ArcFace",
    ]
    date_added = datetime.datetime.now()
    for model_name in models:
        cursor.execute(QUERIES['insert_model'], [model_name, '', str(date_added)])


def insert_person(cursor, name, patronymic, surname, comment, date_added):
    person_data = [(name, patronymic, surname, comment, str(date_added))]
    cursor.executemany(
        QUERIES['insert_person'],
        person_data)
    cursor.execute(QUERIES['select_person'])
    return cursor.lastrowid


def insert_embedding(cursor, embedding, date_added, model_id, person_id, preprocessing_id, info):
    embedding_data = [str(embedding), str(date_added), model_id, preprocessing_id,
                      person_id, info]
    cursor.execute(
        QUERIES['insert_embedding'],
        embedding_data)


def clear_tables_from_db(cursor, table_names):
    for table_name in table_names:
        if table_name == 'embedding':
            cursor.execute(QUERIES['delete_from_embedding'])
        if table_name == 'model':
            cursor.execute(QUERIES['delete_from_model'])
        if table_name == 'person':
            cursor.execute(QUERIES['delete_from_person'])
        if table_name == 'preprocessing':
            cursor.execute(QUERIES['delete_from_preprocessing'])


if __name__ == "__main__":
    db_path = os.path.join('', 'database.db')
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    clear_tables_from_db(cursor, ['embedding', 'person'])
    conn.commit()
    conn.close()
