import os
import sqlite3
import datetime
from get_embeddings import get_embedding, insert_person
from common.db.db_operations import insert_embedding

model_name = "Facenet512"

db_path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'db', 'database.db')
with sqlite3.connect(db_path) as conn:
    cursor = conn.cursor()
    date_added = datetime.datetime.now()
    for i in range(1, 11):
        print(f'it {i}')
        path = os.path.join('C:\\CompositePortraitRecongnition', 'common', 'dataset', 'sketches_to_pipeline',
                            f'{i}.jpg')
        embed = get_embedding(path, model_name)
        person_id = insert_person(cursor, i, date_added)
        insert_embedding(cursor, embed, date_added, 5, person_id, None, f'sketch_to_pipeline_{i}')
    conn.commit()
