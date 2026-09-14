import csv
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root = Path(__file__).resolve().parents[1]
results = root / 'profiling' / 'results'
images = root / 'images'
images.mkdir(exist_ok=True)

# Load individual runs, rather than rounding the data before averaging.
with (results / 'timings.csv').open(encoding='utf-8-sig') as file:
    rows = list(csv.DictReader(file))
labels = {
    'cpu': 'CPU', 'naive': 'Naive', 'efficient': 'Work-efficient', 'thrust': 'Thrust',
    'cpu-compact': 'CPU without scan', 'cpu-scan-compact': 'CPU with scan',
    'gpu-compact': 'GPU with scan'
}
colors = {'cpu': '#555555', 'naive': '#d9822b', 'efficient': '#2878a5', 'thrust': '#34844b',
          'cpu-compact': '#555555', 'cpu-scan-compact': '#d9822b', 'gpu-compact': '#8656a3'}
plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

def plot(experiment, modes, xkey, xlabel, title, filename, logx=False, logy=False):
    fig, ax = plt.subplots(figsize=(9, 5.2))

    # Each implementation gets its own line in the graph.
    for mode in modes:
        # Group repeated runs by their x-axis value.
        groups = defaultdict(list)
        for row in rows:
            if row['experiment'] == experiment and row['mode'] == mode:
                groups[int(row[xkey])].append(float(row['ms']))

        # Sort the x values so lines connect points in the right order.
        xs = sorted(groups)
        means = [statistics.mean(groups[x]) for x in xs]
        # Error bars show variation between runs, not between individual steps.
        errors = [statistics.stdev(groups[x]) if len(groups[x]) > 1 else 0 for x in xs]
        ax.errorbar(xs, means, yerr=errors, marker='o', capsize=3,
                    label=labels[mode], color=colors[mode])

    # Log scales make the small and large input sizes visible together.
    if logx:
        ax.set_xscale('log', base=2)
    if logy:
        ax.set_yscale('log')
    ax.set(xlabel=xlabel, ylabel='Milliseconds per operation', title=title)
    ax.grid(alpha=.22)
    ax.legend()
    fig.text(.5, .015, 'Release build | 20 warmups + 100 operations per run | 3 runs | error bars: 1 SD',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .035, 1, 1))

    # Save a static image for the README.
    fig.savefig(images / filename, dpi=160)
    plt.close(fig)

plot('array_size', ['cpu', 'naive', 'efficient', 'thrust'], 'size', 'Array length',
     'Exclusive scan', 'profile_scan.png', True, True)
plot('array_size', ['cpu-compact', 'cpu-scan-compact', 'gpu-compact'], 'size', 'Array length',
     'Stream compaction (50% keep probability)', 'profile_compaction.png', True, True)
plot('block_size', ['naive', 'efficient', 'gpu-compact'], 'block_size', 'Threads per block',
     'Block size (4,194,304 elements)', 'profile_block_size.png', True)
plot('keep_percent', ['cpu-compact', 'cpu-scan-compact', 'gpu-compact'], 'keep_percent',
     'Probability of keeping an element (%)', 'Stream compaction (1,048,576 elements)',
     'profile_keep_percent.png')

# Use all the experiment settings when grouping the summary table.
# Different block sizes or keep percentages should not be averaged together.
summary = defaultdict(list)
for row in rows:
    key = tuple(row[k] for k in ('experiment', 'mode', 'size', 'block_size', 'keep_percent'))
    summary[key].append(float(row['ms']))

with (results / 'summary.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['experiment', 'mode', 'size', 'block_size', 'keep_percent', 'mean_ms', 'sd_ms'])
    for key, values in summary.items():
        writer.writerow([*key, statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0])
print('Wrote four graphs and summary.csv')
