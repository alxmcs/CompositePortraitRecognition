import os.path
import sqlite3
from db_operations import init_preprocessings, init_models

def get_database_path():
    return os.path.abspath('mydb.sql')


if __name__ == "__main__":
    database_path = get_database_path()
    with sqlite3.connect(database_path) as conn:
        cursor = conn.cursor()

        path_fd = get_database_path()
        with open(path_fd, 'r', encoding='utf-8') as fd:
            sqlFile = fd.read()

        sqlCommands = sqlFile.split(';')

        for command in sqlCommands:
            cursor.execute(command)
        # init tables
        init_preprocessings(cursor)
        init_models(cursor)

        conn.commit()
