import sqlite3
import codecs
conn = sqlite3.connect("database.db")
cursor = conn.cursor()

fd = open('mydb.sql', 'r', encoding='utf-8')
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(';')

for command in sqlCommands:
    cursor.execute(command)

conn.commit()
conn.close()
