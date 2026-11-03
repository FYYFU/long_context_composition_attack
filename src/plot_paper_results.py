#!/usr/bin/env python3
"""Build the historical main table and a figure from computed per-cell CSV data."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONTEXTS = ('0k', '16k', '64k')
STRUCTURES = ('Direct', 'Single-hop', 'Chain', 'Multi-hop')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-dir', type=Path, default=ROOT / 'outputs/paper100')
    args = parser.parse_args()
    with (args.input_dir / 'cell_safety_ratio_wilson.csv').open(newline='') as f:
        rows = list(csv.DictReader(f))
    models = list(dict.fromkeys(r['model'] for r in rows))
    lookup = {(r['model'], r['context'], int(r['level'])): r for r in rows}
    headers = ['Model'] + [f'{context} / {name}' for context in CONTEXTS for name in STRUCTURES]
    table = []
    for model in models:
        table.append([model] + [float(lookup[model, context, level]['safety_ratio_percent'])
                               for context in CONTEXTS for level in range(4)])
    table.append(['Average'] + [float(np.mean([row[i] for row in table])) for i in range(1, 13)])
    with (args.input_dir / 'main_table.csv').open('w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(table)
    markdown = ['# Historical main table', '',
                'Safety Ratio (%), using the historical `score < 5` convention, including `-1`.',
                'All records enter the denominator. See `invalid_score_audit.csv` for failures.', '',
                '| ' + ' | '.join(headers) + ' |',
                '| ' + ' | '.join(['---'] + ['---:'] * 12) + ' |']
    markdown += ['| ' + row[0] + ' | ' + ' | '.join(f'{v:.1f}' for v in row[1:]) + ' |' for row in table]
    (args.input_dir / 'main_table.md').write_text('\n'.join(markdown) + '\n')
    fig, axes = plt.subplots(1, 3, figsize=(15, 7), sharey=True, constrained_layout=True)
    for ax, context in zip(axes, CONTEXTS):
        values = np.array([[float(lookup[model, context, level]['safety_ratio_percent'])
                            for level in range(4)] for model in models])
        picture = ax.imshow(values, vmin=0, vmax=100, cmap='YlGnBu', aspect='auto')
        ax.set_title(f'{context} context')
        ax.set_xticks(range(4))
        ax.set_xticklabels(STRUCTURES, rotation=35, ha='right')
        ax.set_yticks(range(len(models)))
        ax.set_yticklabels(models)
        for i in range(len(models)):
            for j in range(4):
                ax.text(j, i, f'{values[i, j]:.0f}', ha='center', va='center', fontsize=8,
                        color='white' if values[i, j] >= 65 else 'black')
    fig.colorbar(picture, ax=axes, label='Historical Safety Ratio (%)', shrink=0.8)
    fig.suptitle('Compositional safety benchmark — historical 100-request results')
    for suffix in ('png', 'pdf'):
        fig.savefig(args.input_dir / f'main_table.{suffix}', dpi=180)
    plt.close(fig)
    print(f'[complete] table and figure: {args.input_dir}')


if __name__ == '__main__':
    main()
