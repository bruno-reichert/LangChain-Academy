import sqlite3
import matplotlib.pyplot as plt
import os

# Connect to the Chinook SQLite database
db_path = '/chinook.db'
conn = sqlite3.connect(db_path)
cur = conn.cursor()

# Query total revenue per genre
query = '''
SELECT g.Name as Genre, SUM(il.UnitPrice * il.Quantity) as Revenue
FROM InvoiceLine il
JOIN Track t ON il.TrackId = t.TrackId
JOIN Genre g ON t.GenreId = g.GenreId
GROUP BY g.GenreId, g.Name
ORDER BY Revenue DESC;
'''
cur.execute(query)
rows = cur.fetchall()

# Compute total revenue
total_revenue = sum(row[1] for row in rows)

# Build lists, grouping small genres into 'Other'
labels = []
revenues = []
other_revenue = 0.0
for genre, rev in rows:
    pct = rev / total_revenue * 100
    if pct < 3:
        other_revenue += rev
    else:
        labels.append(genre)
        revenues.append(rev)
# Add 'Other' if needed
if other_revenue > 0:
    labels.append('Other')
    revenues.append(other_revenue)

# Compute percentages for labeling
percentages = [rev/total_revenue*100 for rev in revenues]
# Create labels with name and percentage
pie_labels = [f"{name} ({pct:.1f}%)" for name, pct in zip(labels, percentages)]

# Plot donut chart
fig, ax = plt.subplots(figsize=(8, 8))
# Choose a distinct color palette
cmap = plt.get_cmap('tab20c')
colors = cmap.colors[:len(revenues)]

wedges, texts = ax.pie(
    revenues,
    labels=pie_labels,
    startangle=140,
    colors=colors,
    wedgeprops=dict(width=0.3, edgecolor='w')
)

# Add white circle in the centre to make it a donut
centre_circle = plt.Circle((0, 0), 0.70, fc='white')
ax.add_artist(centre_circle)

# Title with top padding
ax.set_title('Revenue Share by Genre', fontsize=16, pad=20)

# Ensure equal aspect ratio
ax.axis('equal')

# Save the figure
output_path = '/genre_revenue.png'
plt.savefig(output_path, bbox_inches='tight')
print('Chart saved to', output_path)

# Close connection
conn.close()
