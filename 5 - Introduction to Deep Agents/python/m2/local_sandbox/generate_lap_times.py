import json, random, statistics

# Generate lap times for 10 laps of a Nissan Skyline GT-R around Suzuka Circuit
# Lap times in seconds, realistic range 90-100 seconds
lap_times = [round(random.uniform(90, 100), 2) for _ in range(10)]

# Prepare data structure
laps_data = {
    "laps": [
        {
            "lap_number": i + 1,
            "lap_time_sec": t
        }
        for i, t in enumerate(lap_times)
    ]
}

# Write to telemetry.json
with open("telemetry.json", "w") as f:
    json.dump(laps_data, f, indent=4)

# Compute summary statistics
avg_time = statistics.mean(lap_times)
min_time = min(lap_times)
max_time = max(lap_times)
std_time = statistics.stdev(lap_times)

print("Lap time data generated and saved to 'telemetry.json'.")
print(f"Average lap time: {avg_time:.2f} s")
print(f"Fastest lap: {min_time:.2f} s")
print(f"Slowest lap: {max_time:.2f} s")
print(f"Standard deviation: {std_time:.2f} s")
