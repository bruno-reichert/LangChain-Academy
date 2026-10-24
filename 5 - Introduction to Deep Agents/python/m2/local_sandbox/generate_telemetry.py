import json, random, statistics

# Generate lap times for 10 laps of a Nissan Skyline GT-R around Suzuka Circuit
# Suzuka is 5.807 km; typical lap times for a GT-R might be ~105-110 seconds
lap_times = [round(random.uniform(104, 108), 2) for _ in range(10)]
# Build data structure
telemetry_data = [{'lap': i + 1, 'time': t} for i, t in enumerate(lap_times)]
# Write to telemetry.json
with open('telemetry.json', 'w') as f:
    json.dump(telemetry_data, f, indent=4)

# Print summary
print('Generated lap times for 10 laps of a Nissan Skyline GT-R around Suzuka Circuit:')
for entry in telemetry_data:
    print(f"Lap {entry['lap']}: {entry['time']} s")

# Compute statistics
times = [entry['time'] for entry in telemetry_data]
print(f"Min: {min(times):.2f}s, Max: {max(times):.2f}s, Avg: {statistics.mean(times):.2f}s")
