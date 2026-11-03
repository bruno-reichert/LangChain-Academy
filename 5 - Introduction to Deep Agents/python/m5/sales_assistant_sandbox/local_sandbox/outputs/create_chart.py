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

# Create pie chart
fig, ax = plt.subplots(figsize=(12, 10))

# Only show labels for genres with > 2% of total
total = sum(revenues)
labels = [g if r/total > 0.02 else '' for g, r in zip(genres, revenues)]

wedges, texts, autotexts = ax.pie(
    revenues,
    labels=labels,
    autopct=lambda p: f'{p:.1f}%' if p > 2 else '',
    startangle=90,
    textprops={'fontsize': 9}
)

# Make percentage labels more readable
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(8)

ax.set_title('Revenue by Genre — Jane Peacock\'s Territory', fontsize=14, fontweight='bold', pad=20)

# Add legend for all genres
ax.legend(wedges, genres, title='Genres', loc='center left', bbox_to_anchor=(1, 0, 0.5, 1), fontsize=8)

plt.tight_layout()
plt.savefig('/outputs/territory_chart-2026-11-03T18-14-05.png', dpi=150, bbox_inches='tight')
plt.close()
print('Chart saved successfully')