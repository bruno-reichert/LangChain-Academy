import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Revenue by genre data
genres = [
    'Rock', 'Latin', 'Metal', 'Alternative & Punk', 'TV Shows', 'Jazz',
    'Sci Fi & Fantasy', 'Blues', 'Classical', 'R&B/Soul', 'Drama',
    'Reggae', 'Comedy', 'Alternative', 'Bossa Nova', 'Hip Hop/Rap',
    'Electronica/Dance', 'Science Fiction', 'Soundtrack', 'World',
    'Rock And Roll', 'Pop', 'Easy Listening'
]

revenues = [
    300.96, 137.61, 85.14, 70.29, 37.81, 33.66,
    19.90, 18.81, 18.81, 17.82, 15.92,
    12.87, 11.94, 9.90, 8.91, 7.92,
    5.94, 3.98, 3.96, 3.96,
    2.97, 1.98, 1.98
]

# Create pie chart - only show top genres, group rest as "Other"
# Let's show top 10 and group the rest
top_n = 10
top_genres = genres[:top_n]
top_revenues = revenues[:top_n]
other_revenue = sum(revenues[top_n:])

chart_genres = top_genres + ['Other']
chart_revenues = top_revenues + [other_revenue]

# Colors
colors = plt.cm.Set3(range(len(chart_genres)))

fig, ax = plt.subplots(figsize=(10, 8))
wedges, texts, autotexts = ax.pie(
    chart_revenues,
    labels=chart_genres,
    autopct='%1.1f%%',
    startangle=90,
    colors=colors,
    textprops={'fontsize': 10}
)

# Make percentage text bold
for autotext in autotexts:
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

ax.set_title('Revenue by Genre — Jane Peacock\'s Territory', fontsize=14, fontweight='bold', pad=20)

# Add total revenue annotation
ax.text(0, -1.3, f'Total Revenue: $833.04', ha='center', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig('/outputs/territory_chart-2026-11-03T18-05-39.png', dpi=150, bbox_inches='tight')
plt.close()
print("Chart saved successfully!")