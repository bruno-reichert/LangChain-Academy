#!/usr/bin/env python3
import json, random, statistics

# Generate lap time data for 10 laps of a Nissan Skyline GT-R around Suzuka Circuit
num_laps = 10
lap_data = []
for lap in range(1, num_laps + 1):
    # Typical lap times around Suzuka for a Skyline GT-R: ~110-120 seconds
    lap_time = round(random.uniform(110.0, 120.0), 2)
    lap_data.append({"lap": lap, "time": lap_time})

# Write to telemetry.json
with open("telemetry.json", "w") as f:
    json.dump(lap_data, f, indent=2)

# Compute summary statistics
times = [entry["time"] for entry in lap_data]
average = statistics.mean(times)
minimum = min(times)
maximum = max(times)
std_dev = statistics.stdev(times) if len(times) > 1 else 0.0

print("Lap Time Data Summary:"
      f"\nTotal Laps: {num_laps}"
      f"\nAverage Lap Time: {average:.2f} s"
      f"\nMinimum Lap Time: {minimum:.2f} s"
      f"\nMaximum Lap Time: {maximum:.2f} s"
      f"\nStandard Deviation: {std_dev:.2f} s")
