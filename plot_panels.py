import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from synthetic_oracle import oracle_lengths
from metrics import mean_ci
from metrics import subset_metrics
LABELS = {'cond': 'Ours', 'exact': 'Exact', 'nested': 'Nested', 'weightedcp': 'Weighted-CP', 'arccp': 'ARC-CP'}
COLOR = dict(zip(LABELS, ['#4c78a8', '#9fc1b3', '#c5b0d5', '#9aa5b1', '#d8b59d']))
COLORS = list(COLOR.values())

def render(summary, seeds, out, include_baselines=True):
    stats = summary['configs']
    out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'serif', 'font.size': 9, 'mathtext.fontset': 'cm', 'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    written, skipped, oracle_report, ablation_report = ([], [], {}, {})
    subset_report, observed_report = ({}, None)

    def ready(ids):
        return all((c in stats and stats[c]['complete'] for c in ids))

    def save(fig, name):
        fig.savefig(out / f'{name}.pdf', bbox_inches='tight')
        fig.savefig(out / f'{name}.png', bbox_inches='tight', dpi=180)
        plt.close(fig)
        written.append(name)

    def comparison(prefix, mechanisms, methods, name):

        def key(mech, method):
            return f'{prefix}_{mech.lower()}_{method}' if mech else f'{prefix}_{method}'
        ids = [key(mech, method) for mech in mechanisms for method in methods]
        if not ready(ids):
            skipped.append(name)
            return
        fig, axes = plt.subplots(2, 1, figsize=(5.0, 2.6), sharex=True)
        x = np.arange(len(mechanisms))
        width = 0.8 / len(methods)
        oracle_means = []
        if prefix == 'highdim':
            for mech in mechanisms:
                cid = key(mech, 'cond')
                masks = [p['mask'] for p in stats[cid]['pooled_masks']]
                lengths = oracle_lengths(masks, seeds[cid][0]['protocol']['args'])
                oracle_means.append(float(np.mean(list(lengths.values()))))
                oracle_report[cid] = dict(by_mask=lengths, mean=oracle_means[-1])
        for ax, metric, title in zip(axes, ['shortfall', 'length'], ['$\\max(0.9-\\mathrm{cov},0)$', 'Length']):
            finite = [float(stats[c]['paper_metrics'][metric]['mean']) for c in ids if np.isfinite(float(stats[c]['paper_metrics'][metric]['mean']))]
            ceiling = max(finite, default=1) * 1.3 or 1
            for j, method in enumerate(methods):
                values = [stats[key(mech, method)]['paper_metrics'][metric] for mech in mechanisms]
                means = np.array([float(v['mean']) for v in values])
                errors = np.array([v['se_masks'] or 0 for v in values])
                locations = x + (j - (len(methods) - 1) / 2) * width
                finite_mask = np.isfinite(means)
                ax.bar(locations[finite_mask], means[finite_mask], width, yerr=errors[finite_mask], capsize=2, color=COLOR[method], label=LABELS[method], error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'ecolor': '.3'}, zorder=2)
                for xx in locations[~finite_mask]:
                    ax.bar(xx, ceiling, width, facecolor='none', edgecolor=COLOR[method], hatch='///')
                    ax.text(xx, ceiling, 'unbounded', rotation=90, va='top', ha='center', fontsize=7)
            ax.set_xticks(x, [m or 'MIMIC-IV' for m in mechanisms])
            ax.set_ylabel('Shortfall risk' if prefix == 'highdim' and metric == 'shortfall' else title, fontsize=11)
            ax.tick_params(labelsize=10)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6, zorder=0)
            if metric == 'length' and oracle_means:
                for xx, value in zip(x, oracle_means):
                    ax.hlines(value, xx - 0.41, xx + 0.41, colors='.4', linestyles='--', linewidth=0.8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in methods]
        labels = [LABELS[m] for m in methods]
        if oracle_means:
            handles.append(plt.Line2D([0], [0], color='.4', ls='--', lw=0.8))
            labels.append('Oracle length')
        fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.54, 1.0), ncol=3 if oracle_means else len(methods), frameon=False, fontsize=10, columnspacing=0.8, handlelength=1.2)
        fig.subplots_adjust(left=0.16, right=0.99, bottom=0.13, top=0.77 if oracle_means else 0.83, hspace=0.15)
        save(fig, name)
    methods = ['cond', 'exact', 'nested', 'weightedcp', 'arccp'] if include_baselines else ['cond']
    for d in (3, 6):
        comparison(f'synthetic_d{d}', ['MCAR', 'MAR', 'PMIX'], methods, f'd{d}_MCAR_MAR_PMIX')
    comparison('highdim', ['MCAR', 'PMIX'], [m for m in methods if m != 'nested'], 'd100_MCAR_PMIX')
    comparison('concrete', ['MCAR', 'MAR', 'MNAR'], methods, 'concrete_MCAR_MAR_MNAR')
    ablation = [f'ablation_d6_mar_nt{n}' for n in (0, 4, 16, 64)] + ['synthetic_d6_mar_cond']
    if ready(ablation):
        fig, ax = plt.subplots(figsize=(4.6, 2.7))
        masks = [p['mask'] for p in stats[ablation[0]]['pooled_masks']]
        x = np.arange(len(masks))
        for n, config, color in zip((0, 4, 16, 64, 256), ablation, plt.cm.viridis(np.linspace(0, 0.8, 5))):
            intervals = [mean_ci([next((p['coverage'] for p in row['patterns'] if p['mask'] == m)) for row in seeds[config]]) for m in masks]
            means = np.array([p['mean'] for p in intervals])
            halfwidths = np.array([p['ci95_halfwidth'] for p in intervals])
            ax.plot(x, means, color=color, linewidth=1, label=f'$n_{{\\mathrm{{tree}}}}={n}$')
            ax.fill_between(x, means - halfwidths, means + halfwidths, color=color, alpha=0.12, linewidth=0)
            ablation_report[config] = dict(masks=masks, coverage=intervals, best_lambda=[r['protocol']['best_lambda'] for r in seeds[config]], has_intercept=seeds[config][0]['protocol']['has_intercept'])
        ax.axhline(0.9, color='red', linestyle=':', linewidth=1, label='Target')
        ax.set_xticks(x, masks, rotation=35, ha='right', fontsize=10)
        ax.tick_params(axis='y', labelsize=10)
        ax.set_xlabel('Mask patterns', fontsize=11)
        ax.set_ylabel('Coverage', fontsize=11)
        ax.grid(axis='y', linewidth=0.5, alpha=0.25)
        ax.legend(frameon=False, fontsize=10, ncol=3, loc='lower center', bbox_to_anchor=(0.5, 1.01), columnspacing=0.9, handlelength=1.4)
        fig.subplots_adjust(left=0.14, right=0.99, bottom=0.29, top=0.77)
        save(fig, 'ntree_sensitivity')
        fig, ax = plt.subplots(figsize=(5, 3))
        for i, (n, config) in enumerate(zip((4, 16, 64, 256), ablation[1:])):
            values = ablation_report[config]['best_lambda']
            ax.scatter(i + np.linspace(-0.12, 0.12, len(values)), values, s=18, color=COLORS[0], alpha=0.8)
        ax.set_xticks(range(4), ['4', '16', '64', '256'])
        ax.set_yscale('log')
        ax.set_xlabel('Number of trees (MAR, d=6)')
        ax.set_ylabel('Selected $\\lambda$')
        ax.grid(axis='y', linewidth=0.5, alpha=0.25)
        fig.tight_layout()
        save(fig, 'best_lambda')
    else:
        skipped.extend(['ntree_sensitivity', 'best_lambda'])
    config = 'observed_d6_mcar'
    if ready([config]):
        observed_report = subset_metrics(seeds[config], observed=True)
        fig, ax = plt.subplots(figsize=(4.4, 2.8))
        for p in observed_report['pooled_masks']:
            half = p['coverage']['ci95_halfwidth']
            ax.errorbar(p['cal_count'], p['coverage']['mean'], yerr=half, fmt='o', ms=3, mfc=COLORS[0] if half is not None else 'white', mec=COLORS[0], color=COLORS[0], ecolor='.6', elinewidth=0.7, capsize=1.5, alpha=0.8)
        ax.axhline(0.9, color='red', linestyle='--', linewidth=0.8, label='Target (0.9)')
        ax.set_xscale('log')
        ax.set_xlabel('Mean calibration count (when present)', fontsize=10)
        ax.set_ylabel('Empirical coverage', fontsize=11)
        ax.tick_params(labelsize=10)
        ax.set_ylim(0.7, 1.04)
        ax.legend(loc='lower right', frameon=False, fontsize=9)
        fig.tight_layout()
        save(fig, 'coverage_vs_freq')
    else:
        skipped.append('coverage_vs_freq')
    pmix_ids = [f'synthetic_d6_pmix_{m}' for m in methods]
    for present, name in [(True, 'd6_seen_PMIX'), (False, 'd6_unseenmask_pmix')]:
        if not ready(pmix_ids):
            skipped.append(name)
            continue
        metrics = {m: subset_metrics(seeds[c], present) for m, c in zip(methods, pmix_ids)}
        subset_key = lambda report: [(p['mask'], p['seeds']) for p in report['pooled_masks']]
        assert all((subset_key(metrics[m]) == subset_key(metrics['cond']) for m in methods))
        oracle = oracle_lengths([p['mask'] for p in metrics['cond']['pooled_masks']], seeds[pmix_ids[0]][0]['protocol']['args'])
        oracle_mean = float(np.mean(list(oracle.values())))
        subset_report[name] = dict(methods=metrics, oracle=oracle, oracle_mean=oracle_mean)
        fig, axes = plt.subplots(1, 2, figsize=(6, 2.8))
        for ax, metric, label in zip(axes, ['shortfall', 'length'], ['$\\max(0.9-\\mathrm{cov},0)$', 'Length']):
            finite = [metrics[m]['paper_metrics'][metric]['mean'] for m in methods if np.isfinite(metrics[m]['paper_metrics'][metric]['mean'])]
            ceiling = max(finite, default=1) * 1.3 or 1
            for j, m in enumerate(methods):
                value = metrics[m]['paper_metrics'][metric]
                if np.isfinite(value['mean']):
                    ax.bar(j, value['mean'], yerr=value['se_masks'], width=0.8, color=COLOR[m], capsize=2, error_kw={'elinewidth': 0.7, 'capthick': 0.7, 'ecolor': '.3'})
                else:
                    ax.bar(j, ceiling, width=0.8, facecolor='none', edgecolor=COLOR[m], hatch='///')
                    ax.text(j, ceiling, 'unbounded', rotation=90, va='top', ha='center', fontsize=7)
            ax.set_xticks([(len(methods)-1)/2], ['PMIX'])
            ax.set_ylabel(label)
            ax.set_ylim(bottom=0)
            ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.4)
            if metric == 'length':
                ax.axhline(oracle_mean, color='.4', ls='--', lw=0.8)
        handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m]) for m in methods]
        handles.append(plt.Line2D([0], [0], color='.4', ls='--', lw=0.8))
        fig.legend(handles, [LABELS[m] for m in methods] + ['Oracle length'], loc='upper center', ncol=3, frameon=False, fontsize=9)
        fig.subplots_adjust(left=0.09, right=0.99, bottom=0.14, top=0.76, wspace=0.3)
        save(fig, name)
    return dict(written=written, skipped=skipped, oracle_values=oracle_report, ablation_values=ablation_report, observed_values=observed_report, subset_values=subset_report)
