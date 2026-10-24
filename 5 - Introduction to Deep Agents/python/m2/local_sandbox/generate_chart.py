#!/usr/bin/env python3
import sqlite3
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Connect to the Chinook database
conn = sqlite3.connect('chinook.db')
cur = conn.cursor()

# Query total revenue by genre
query = '''
SELECT g.Name AS genre,
       SUM(il.UnitPrice * il.Quantity) AS revenue
FROM InvoiceLine il
JOIN Track t ON il.TrackId = t.TrackId
JOIN Genre g ON t.GenreId = g.GenreId
GROUP BY g.Name
ORDER BY revenue DESC
'''
cur.execute(query)
rows = cur.fetchall()

# Compute total revenue
total_revenue = sum(row[1] for row in rows)

# Prepare data for chart, grouping small genres into 'Other'
labels = []
values = []
other_revenue = 0.0
for genre, rev in rows:
    percent = rev / total_revenue * 100
    if percent < 3:
        other_revenue += rev
    else:
        labels.append(genre)
        values.append(rev)

if other_revenue > 0:
    labels.append('Other')
    values.append(other_revenue)

# Create donut chart
fig, ax = plt.subplots(figsize=(8, 8))
wedges, texts = ax.pie(values, wedgeprops=dict(width=0.3), startangle=-40)
# Add labels with percentages
ax.legend(wedges, labels, title="Genres", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))
ax.set_title('Revenue Share by Genre')

# Save the figure
plt.savefig('genre_revenue.png', bbox_inches='tight')
plt.close()

print('Chart saved to /genre_revenue.png')
