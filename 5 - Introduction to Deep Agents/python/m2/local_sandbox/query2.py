import sqlite3
conn = sqlite3.connect('chinook.db')
cursor = conn.cursor()
cursor.execute('''
    SELECT g.Name, SUM(il.UnitPrice * il.Quantity) as revenue
    FROM Genre g
    JOIN Track t ON g.GenreId = t.GenreId
    JOIN InvoiceLine il ON t.TrackId = il.TrackId
    GROUP BY g.GenreId, g.Name
    ORDER BY revenue DESC
''')
results = cursor.fetchall()
for r in results:
    print(f'{r[0]}: {r[1]:.2f}')