import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from metrics import collect, json_safe, mean_ci
from plot_panels import render, COLOR, LABELS
from project import CONFIGURATION, ROOT, read_json, selected_configs, write_json
from summarize import write_tables
from synthetic_oracle import oracle_lengths


def save(fig, output, name):
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"{name}.{suffix}", dpi=300, bbox_inches="tight", pad_inches=.04)
    plt.close(fig)


def mimic(summary, rows, output, include_baselines=True):
    methods = ("cond", "exact", "weightedcp", "arccp") if include_baselines else ("cond",)
    stats = summary["configs"]
    if not all(f"mimic_{m}" in stats for m in methods):
        return None
    fig, axes = plt.subplots(1, 2, figsize=(4.4, 1.8))
    for ax, key, label in zip(axes, ("shortfall", "length"), (r"$\max(0.9-\mathrm{cov},0)$", "Length")):
        values = [stats[f"mimic_{m}"]["paper_metrics"][key] for m in methods]
        tops = [v["mean"]+(v["se_masks"] or 0) for v in values if np.isfinite(v["mean"])]
        ymax = max(tops+[.01 if key == "shortfall" else 1])*1.25
        for i, (method, value) in enumerate(zip(methods, values)):
            if np.isfinite(value["mean"]):
                ax.bar(i, value["mean"], width=.95, color=COLOR[method], zorder=2)
                ax.errorbar(i, value["mean"], yerr=value["se_masks"], fmt="none", ecolor=".3", elinewidth=.7, capsize=2, zorder=3)
            else:
                ax.annotate("", xy=(i, .81*ymax), xytext=(i, .03*ymax), arrowprops=dict(arrowstyle="->", color=COLOR[method], lw=2.5))
                ax.text(i, .86*ymax, r"$\infty$", ha="center", va="bottom")
        ax.set(xlim=(-.65, len(methods)-.35), ylim=(0, ymax), ylabel=label, xticks=[])
        ax.grid(axis="y", linestyle=":", linewidth=.6, alpha=.6, zorder=0)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLOR[m], label=LABELS[m]) for m in methods]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, 1.03), ncol=4, frameon=False, fontsize=8, columnspacing=.7, handlelength=1.1)
    fig.subplots_adjust(left=.14, right=.99, bottom=.08, top=.79, wspace=.62)
    save(fig, output, "MIMIC_results")
    records = rows["mimic_cond"]
    masks = [p["mask"] for p in stats["mimic_cond"]["pooled_masks"]]
    intervals = [mean_ci([next(p["coverage"] for p in r["patterns"] if p["mask"] == m) for r in records]) for m in masks]
    means = np.asarray([r["mean"] for r in intervals])
    half = np.asarray([r["ci95_halfwidth"] for r in intervals])
    diagnostic = np.asarray([min(r["diagnostic"][m] for r in records) for m in masks])
    fig, ax = plt.subplots(figsize=(4.4, 2.4))
    x = np.arange(len(masks))
    ax.errorbar(x, means, yerr=np.vstack([np.minimum(half, means), np.minimum(half, 1-means)]),
        fmt="o-", color="#1f77b4", ecolor="#1f77b4", capsize=2, markersize=2.5,
        linewidth=1.1, elinewidth=.8, label="Empirical coverage")
    ax.plot(x, diagnostic, "^-", color="#d62728", markersize=2.7, linewidth=1.1, label="Residual diagnostic")
    ax.axhline(.9, color=".5", linestyle="--", linewidth=.9, zorder=0)
    lower = min(.7, max(0, np.floor(min((means-half).min(), diagnostic.min())*10)/10))
    ax.set(xlabel="Mask index", ylabel="Coverage", ylim=(lower, 1.02), xlim=(-1, len(masks)))
    ax.set_yticks(np.arange(lower, 1.01, .1))
    ax.tick_params(axis="x", bottom=False, labelbottom=False)
    ax.grid(axis="y", linestyle="--", alpha=.35, linewidth=.6)
    ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), frameon=False, ncol=2, fontsize=8.5, columnspacing=1.4, handlelength=1.2)
    fig.subplots_adjust(left=.15, right=.99, bottom=.19, top=.81)
    save(fig, output, "mimic_representation_diagnostic")
    return dict(masks=masks, coverage=intervals, diagnostic=diagnostic.tolist())


def embeddings(summary, rows, output):
    models = (("xgboost", "XGBoost", "#1f77b4"), ("lightgbm", "LightGBM", "#ff7f0e"), ("catboost", "CatBoost", "#2ca02c"))
    ids = [f"embedding_d6_mcar_{model}" for model, _, _ in models]
    if not all(cid in rows for cid in ids):
        return None
    masks = [p["mask"] for p in summary["configs"][ids[0]]["pooled_masks"]]
    oracle = list(oracle_lengths(masks, rows[ids[0]][0]["protocol"]["args"]).values())
    x = np.arange(len(masks))
    report = {}
    for metric, label in (("coverage", "Coverage"), ("length", "Length")):
        fig, ax = plt.subplots(figsize=(5.4, 2.75))
        lows, highs = [], []
        for cid, (model, title, color) in zip(ids, models):
            intervals = [mean_ci([next(p[metric] for p in r["patterns"] if p["mask"] == m) for r in rows[cid]]) for m in masks]
            mean = np.asarray([v["mean"] for v in intervals])
            half = np.asarray([v["ci95_halfwidth"] if v["ci95_halfwidth"] is not None else 0 for v in intervals])
            finite = np.isfinite(mean)
            low, high = mean-half, mean+half
            if metric == "coverage":
                low, high = np.clip(low, 0, 1), np.clip(high, 0, 1)
            ax.plot(x, np.where(finite, mean, np.nan), label=title, color=color, lw=1)
            ax.fill_between(x, low, high, where=finite, color=color, alpha=.15, linewidth=0)
            lows.extend(low[finite].tolist())
            highs.extend(high[finite].tolist())
            report[f"{model}_{metric}"] = intervals
        if metric == "coverage":
            ax.axhline(.9, color=".25", ls="--", lw=.9)
            ax.set_ylim(max(0, min(.7, min(lows)-.025)), 1.01)
        else:
            ax.plot(x, oracle, label="Oracle", color=".25", ls="--", lw=1)
            ax.set_ylim(max(0, min(lows+oracle)-.5), max(highs+oracle)+.5)
            for model, _, color in models:
                for j, value in enumerate(report[f"{model}_{metric}"]):
                    if not np.isfinite(value["mean"]):
                        ax.text(j, ax.get_ylim()[1], r"$\infty$", color=color, ha="center", va="top")
        ax.set_xlim(-1, len(masks))
        ax.set_xticks(x)
        ax.set_xticklabels([])
        ax.tick_params(axis="x", length=2, width=.4)
        ax.set(xlabel="Mask", ylabel=label)
        ax.grid(axis="y", ls=":", lw=.5, alpha=.65)
        handles, labels = ax.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.56, .995), ncol=4, frameon=False)
        fig.subplots_adjust(left=.12, right=.99, bottom=.17, top=.84)
        save(fig, output, f"tree_comparison_{metric}")
    return dict(masks=masks, oracle=oracle, intervals=report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=ROOT / "results")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--group", nargs="+", choices=[f"{i:02d}" for i in range(1, 10)], required=True)
    parser.add_argument("--configuration", type=Path, default=CONFIGURATION)
    parser.add_argument("--include-baselines", action="store_true")
    args = parser.parse_args()
    configuration = selected_configs(read_json(args.configuration), args.group, args.include_baselines)
    summary, rows = collect(args.results, configuration)
    output = args.output or args.results / ("figures_"+"_".join(args.group)+("_with_baselines" if args.include_baselines else ""))
    output.mkdir(parents=True, exist_ok=True)
    report = render(summary, rows, output, args.include_baselines)
    report["mimic"] = mimic(summary, rows, output, args.include_baselines)
    report["embeddings"] = embeddings(summary, rows, output)
    write_tables(output, summary)
    write_json(output / "summary.json", json_safe(summary))
    write_json(output / "figure_values.json", json_safe(report))
    print(f"Figures and tables written to {output}")


if __name__ == "__main__":
    main()
