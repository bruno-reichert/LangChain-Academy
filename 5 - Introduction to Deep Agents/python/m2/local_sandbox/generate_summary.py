import json, statistics

with open('telemetry.json') as f:
    data = json.load(f)

lap_times = [lap['lap_time_sec'] for lap in data['laps']]

avg = statistics.mean(lap_times)
min_time = min(lap_times)
max_time = max(lap_times)
std = statistics.stdev(lap_times)

print('Data summary:')
print(f'Average lap time: {avg:.2f} s')
print(f'Fastest lap: {min_time:.2f} s')
print(f'Slowest lap: {max_time:.2f} s')
print(f'Standard deviation: {std:.2f} s')
