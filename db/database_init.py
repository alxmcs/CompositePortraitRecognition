import os.path
import sqlite3


def get_database_path():
    return os.path.abspath('mydb.sql')


if __name__ == "__main__":
    database_path = get_database_path()
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    path_fd = get_database_path()
    with open(path_fd, 'r', encoding='utf-8') as fd:
        sqlFile = fd.read()

    sqlCommands = sqlFile.split(';')

    for command in sqlCommands:
        cursor.execute(command)

    conn.commit()
    conn.close()
