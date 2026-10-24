#!/usr/bin/env python3
import json
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Load telemetry data
with open('telemetry.json', 'r') as f:
    data = json.load(f)

# Extract laps and times
laps = [entry['lap'] for entry in data]
times = [entry['time'] for entry in data]

# Create line chart
plt.figure(figsize=(8, 5))
plt.plot(laps, times, marker='o', linestyle='-')
plt.title('Lap Times Across 10 Laps')
plt.xlabel('Lap Number')
plt.ylabel('Time (seconds)')
plt.grid(True)
plt.xticks(laps)  # ensure each lap is shown

# Save the chart
plt.savefig('homework_chart.png')
print('Chart saved as homework_chart.png')
