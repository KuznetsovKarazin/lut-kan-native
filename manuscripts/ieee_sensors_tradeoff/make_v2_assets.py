"""Common-fixture plots and tables from audited device captures, no fitting."""
import hashlib
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from make_assets import DATA, TABLES, save

METHODS = ['lut32_float32', 'lut60_int16', 'cubic32_unrolled', 'horner8',
           'polynomial', 'polynomial_horner', 'physics']
LABELS = ['LUT32 f32 / 144 B', 'LUT60 i16 / 144 B', 'Compact cubic32 / 144 B',
          'Piecewise Horner8 / 144 B', 'Poly. Clenshaw / 56 B', 'Poly. Horner / 56 B', 'Physics / 32 B']
COLORS = ['#0072B2', '#56B4E9', '#009E73', '#D55E00', '#AA4499', '#6A3D9A', '#9D7028']
MARKERS = ['o', '^', 's', 'D', 'P', '*', 'X']
BOARDS = ['mega2560', 'esp32c3', 'esp32s3']


def generate():
    source = DATA / 'v2_hardware.json'
    manifest = json.loads((DATA / 'v2_hardware_sha256.json').read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == manifest['v2_hardware.json']
    data = json.loads(source.read_text())
    fig, axes = plt.subplots(2, 3, figsize=(7.05, 3.9))
    for j, board in enumerate(BOARDS):
        methods = data['boards'][board]['methods']
        for row in range(2):
            ax = axes[row, j]
            for m, label, color, marker in zip(METHODS, LABELS, COLORS, MARKERS):
                r = methods[m]
                x, y = r['median_of_session_medians_us'], r['mae_C']
                if row and y > .057:
                    continue
                ax.plot(x, y, marker=marker, color=color, ms=8.4 if marker == '*' else 4.7,
                        ls='none', label=label, zorder=7 if marker == '*' else 4,
                        markeredgewidth=.4, markeredgecolor='white' if marker == '*' else color)
                ax.hlines(y, min(r['session_medians_us']), max(r['session_medians_us']), color=color, lw=1)
            frontier = sorted((methods[m]['median_of_session_medians_us'], methods[m]['mae_C'])
                              for m in data['boards'][board]['frontier'])
            ax.plot(*zip(*frontier), color='#7C8790', lw=.65, ls='--', zorder=1)
            ax.grid(alpha=.18)
            ax.set_xlim([(130, 390), (9, 33), (.65, 2.4)][j])
            if row == 0:
                ax.set_yscale('log'); ax.set_ylim(.039, .43)
                ax.yaxis.set_major_locator(FixedLocator([.04, .05, .1, .2, .4]))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
                ax.yaxis.set_minor_formatter(NullFormatter())
                ax.set_title(['(a) Mega2560 / 16 MHz', '(b) ESP32-C3 / 160 MHz', '(c) ESP32-S3 / 240 MHz'][j], loc='left', fontsize=8)
            else:
                ax.set_ylim(.0434, .057)
                ax.set_yticks([.044, .048, .052, .056])
                if j == 2: ax.set_xlim(.70, 1.28)
                ax.set_title(['(d) Error detail', '(e) Error detail', '(f) Error / time detail'][j], loc='left', fontsize=8)
                ax.set_xlabel('Gross time (µs/call)')
        axes[0, j].tick_params(axis='x', labelbottom=True)
    axes[0, 0].set_ylabel('Fixed-case MAE (°C; log)')
    axes[1, 0].set_ylabel('Fixed-case MAE (°C)')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=3, loc='upper center', bbox_to_anchor=(.5, 1.015),
               frameon=False, fontsize=7, handletextpad=.4, columnspacing=1.1)
    fig.subplots_adjust(left=.085, right=.99, bottom=.12, top=.77, wspace=.36, hspace=.62)
    save(fig, 'hardware_complete')
    rows, sessions = [], []
    for m, label in zip(METHODS, LABELS):
        records = [data['boards'][b]['methods'][m] for b in BOARDS]
        r = records[0]
        rows.append(label.replace(' / ', ' / ') + ' & ' + ' & '.join(
            f"{v['median_of_session_medians_us']:.3g}" for v in records) +
            f" & {r['mae_C']:.3g} & {r['max_error_C']:.3g}" + r' \\')
    (TABLES / 'v2_hardware.tex').write_text('\n'.join(rows) + '\n')
    for b in BOARDS:
        for m in ['overhead', *METHODS]:
            r = data['boards'][b]['methods'][m]
            label = 'Overhead' if m == 'overhead' else LABELS[METHODS.index(m)].split(' / ')[0]
            sessions.append(b.replace('mega2560', 'Mega').replace('esp32c3', 'C3').replace('esp32s3', 'S3') +
                ' & ' + label + ' & ' + ' & '.join(f'{v:.6f}' for v in r['session_medians_us']) +
                f" & {r['batch_min_us']:.6f} & {r['batch_max_us']:.6f}" + r' \\')
    (TABLES / 'v2_sessions.tex').write_text('\n'.join(sessions) + '\n')


if __name__ == '__main__':
    generate()
