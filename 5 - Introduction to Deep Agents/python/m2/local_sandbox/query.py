import sqlite3
conn = sqlite3.connect('chinook.db')
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM Genre')
print('Genre count:', cursor.fetchone()[0])
cursor.execute('SELECT COUNT(*) FROM Track')
print('Track count:', cursor.fetchone()[0])
cursor.execute('SELECT COUNT(*) FROM InvoiceLine')
print('InvoiceLine count:', cursor.fetchone()[0])