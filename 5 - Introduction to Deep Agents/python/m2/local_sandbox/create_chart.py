import sqlite3
import matplotlib.pyplot as plt
import numpy as np

# Get data from database
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

# Calculate total revenue
total_revenue = sum(r[1] for r in results)

# Calculate percentages and group small genres
genre_data = []
other_revenue = 0
for name, revenue in results:
    pct = (revenue / total_revenue) * 100
    if pct < 3:
        other_revenue += revenue
    else:
        genre_data.append((name, revenue, pct))

# Add "Other" if there are small genres
if other_revenue > 0:
    other_pct = (other_revenue / total_revenue) * 100
    genre_data.append(('Other', other_revenue, other_pct))

# Sort by revenue descending
genre_data.sort(key=lambda x: x[1], reverse=True)

# Prepare data for plotting
labels = [f'{name} ({pct:.1f}%)' for name, _, pct in genre_data]
revenues = [rev for _, rev, _ in genre_data]

# Create color palette - visually distinct colors
colors = plt.cm.Set3(np.linspace(0, 1, len(genre_data)))

# Create donut chart
fig, ax = plt.subplots(figsize=(10, 8))

# Create the donut chart
wedges, texts, autotexts = ax.pie(
    revenues,
    labels=labels,
    colors=colors,
    autopct='',
    startangle=90,
    pctdistance=0.85,
    wedgeprops=dict(width=0.5, edgecolor='white', linewidth=2),
    textprops={'fontsize': 11}
)

# Add percentage labels manually to avoid overlap
for i, (wedge, (name, rev, pct)) in enumerate(zip(wedges, genre_data)):
    ang = (wedge.theta2 - wedge.theta1) / 2. + wedge.theta1
    x = np.cos(np.deg2rad(ang)) * 0.7
    y = np.sin(np.deg2rad(ang)) * 0.7
    ax.text(x, y, f'{pct:.1f}%', ha='center', va='center', fontsize=10, fontweight='bold', color='white')

# Add white center hole
centre_circle = plt.Circle((0, 0), 0.35, fc='white')
fig.gca().add_artist(centre_circle)

# Title with padding
ax.set_title('Revenue Share by Genre', fontsize=16, fontweight='bold', pad=30)

# Equal aspect ratio ensures circular pie
ax.axis('equal')

# Adjust layout to prevent label overlap
plt.tight_layout()

# Save
plt.savefig('/genre_revenue.png', dpi=300, bbox_inches='tight', facecolor='white')
print("Chart saved to /genre_revenue.png")

# Print summary
print(f"\nTotal Revenue: ${total_revenue:.2f}")
print("\nGenre Breakdown:")
for name, rev, pct in genre_data:
    print(f"  {name}: ${rev:.2f} ({pct:.1f}%)")