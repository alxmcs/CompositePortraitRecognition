import sqlite3
import datetime

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

cursor.execute("CREATE TABLE preprocessing(id integer PRIMARY KEY, name text, comment text, date_added date)")

cursor.execute(
    "CREATE TABLE person(id integer PRIMARY KEY autoincrement, name text, patronymic text, surname text, "
    "comment text, date_added date)")

cursor.execute("CREATE TABLE model(id integer PRIMARY KEY, name text, comment text, date_added date)")

cursor.execute("PRAGMA foreign_keys=on;")
cursor.execute(
    "CREATE TABLE embedding(id integer PRIMARY KEY autoincrement, value varbinary(100), date_added date, "
    "model_id INTEGER NOT NULL, FOREIGN KEY (model_id) REFERENCES model(id), person_id INTEGER NOT NULL, FOREIGN KEY "
    "(person_id) REFERENCES "
    "person(id), preprocessing_id INTEGER NOT NULL, FOREIGN KEY(preprocessing_id) REFERENCES preprocessing(id)")

preprocessing_data = [
    (1, "thumbnail", "resize an image using PIL and maintain its aspect ratio", datetime.date(2022, 8, 19))]
cursor.executemany("INSERT INTO preprocessing VALUES(?, ?, ?, ?)", preprocessing_data)

model_data = [(1, "tensorflow", "tensorflow model", datetime.date(2022, 8, 19)),
              (2, "arcface", "arcface model", datetime.date(2022, 8, 19))]
cursor.executemany("INSERT INTO model VALUES(?, ?, ?, ?)", model_data)

cursor.execute("SELECT name from model where id = 1")
print(cursor.fetchall())

conn.commit()
conn.close()
