import json, statistics

with open('telemetry.json') as f:
    data = json.load(f)

lap_times = [entry['time'] for entry in data]

print('Lap Time Summary:')
print(f'  Total laps: {len(lap_times)}')
print(f'  Average lap time: {statistics.mean(lap_times):.2f} s')
print(f'  Fastest lap: {min(lap_times):.2f} s')
print(f'  Slowest lap: {max(lap_times):.2f} s')
print(f'  Standard deviation: {statistics.stdev(lap_times):.2f} s')
