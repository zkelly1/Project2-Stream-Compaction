import csv
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

root = Path(__file__).resolve().parents[1]
results = root / 'profiling' / 'results'
fig, axes = plt.subplots(3, 1, figsize=(11, 7.5))
export = []

# Give each implementation a separate timeline from its own capture.
for ax, mode in zip(axes, ['naive', 'efficient', 'thrust']):
    with sqlite3.connect(results / f'nsight_systems_{mode}.sqlite') as db:
        # Join the recorded name IDs to their readable CUDA function names.
        api = db.execute('SELECT r.start, r.end, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r '
                         'JOIN StringIds s ON r.nameId=s.id ORDER BY r.start').fetchall()

        # The final pair of event records belongs to the last measured scan.
        events = [row for row in api if 'cudaEventRecord' in row[2]]
        start, stop = events[-2:]

        # Include the wait after recording the end event; launches are asynchronous.
        base = start[0]
        finish = next(row[1] for row in api if row[0] >= stop[1] and 'cudaEventSynchronize' in row[2])

        kernels = db.execute('SELECT k.start, k.end, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k '
                             'JOIN StringIds s ON k.shortName=s.id ORDER BY k.start').fetchall()

        # The trace uses nanoseconds. Divide by 1000 for microseconds.
        # Draw host API calls and GPU kernels on separate rows.
        for begin, end, name in api:
            if base <= begin < finish:
                color = '#d9822b' if 'Malloc' in name or 'Free' in name else '#999999'
                ax.broken_barh([((begin-base)/1000, (end-begin)/1000)], (1.1, .55), facecolors=color)
                export.append([mode, 'API', (begin-base)/1000, (end-begin)/1000, name])

        for begin, end, name in kernels:
            if base <= begin < finish:
                ax.broken_barh([((begin-base)/1000, (end-begin)/1000)], (.15, .55), facecolors='#2878a5')
                export.append([mode, 'GPU', (begin-base)/1000, (end-begin)/1000, name])

        ax.set_xlim(0, (finish-base)/1000)
        ax.set(yticks=[.42, 1.37], yticklabels=['GPU', 'CUDA API'], title=mode.capitalize(),
               xlabel='Microseconds from the final start-event API call')
        ax.grid(axis='x', alpha=.2)

fig.legend(handles=[Patch(color='#2878a5', label='Kernel'), Patch(color='#999999', label='CUDA API'),
                    Patch(color='#d9822b', label='Allocation / free')], loc='upper center', bbox_to_anchor=(.5, .95), ncol=3)
fig.suptitle('Nsight Systems: one scan of 1,048,576 elements', y=.99)
fig.tight_layout(rect=(0, 0, 1, .90))
fig.savefig(root / 'images/nsight_systems.png', dpi=160, bbox_inches='tight')

# Keep the exact rows used in the figure so the plot can be checked.
with (results / 'timeline.csv').open('w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['mode', 'track', 'start_us', 'duration_us', 'name'])
    writer.writerows(export)

# Compare memory throughput, compute throughput, and occupancy.
metrics = [('gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed', 'DRAM throughput'),
           ('sm__throughput.avg.pct_of_peak_sustained_elapsed', 'SM throughput'),
           ('sm__warps_active.avg.pct_of_peak_sustained_active', 'Achieved occupancy')]
fig, ax = plt.subplots(figsize=(9, 4.6))
for offset, mode, label, color in [(-.18, 'naive', 'Naive: first add pass', '#d9822b'),
                                  (.18, 'efficient', 'Work-efficient: first upsweep', '#2878a5')]:
    with (results / f'nsight_compute_{mode}.csv').open(encoding='utf-8-sig') as file:
        rows = list(csv.DictReader(file))

    # The export includes a units row before the actual kernel measurements.
    row = rows[-1]
    values = [float(row[key]) for key, _ in metrics]

    # Offset the bars so both implementations are visible for each metric.
    bars = ax.bar([i + offset for i in range(len(metrics))], values, .36, label=label, color=color)
    ax.bar_label(bars, fmt='%.1f%%', padding=3)
ax.set(xticks=range(len(metrics)), xticklabels=[label for _, label in metrics],
       ylabel='Percent', ylim=(0, 110), title='Nsight Compute: one kernel, 1,048,576 elements')
ax.legend()
ax.grid(axis='y', alpha=.2)
fig.tight_layout()
fig.savefig(root / 'images/nsight_compute.png', dpi=160)
print('Wrote Nsight graphs and timeline.csv')
