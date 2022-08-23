import sqlite3

conn = sqlite3.connect("database.db")
cursor = conn.cursor()

fd = open('mydb.sql', 'r')
sqlFile = fd.read()
fd.close()

sqlCommands = sqlFile.split(';')

for command in sqlCommands:
    cursor.execute(command)

conn.commit()
conn.close()
