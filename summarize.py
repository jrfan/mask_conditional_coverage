import argparse
from pathlib import Path

from metrics import collect, json_safe
from project import CONFIGURATION, ROOT, read_json, selected_configs, write_json


def write_tables(output, summary):
    coordinate = []
    mice = []
    for cid, stats in summary["configs"].items():
        if cid.startswith("coordinate_"):
            cells = []
            for value in stats["coordinate_coverage_plugin_weighted"].values():
                cells.append("N/A" if value is None else f"${100*value['mean']:.1f}\\pm{100*value['ci95_halfwidth']:.1f}$")
            coordinate.append(cid.split("_")[-1].upper()+" & "+" & ".join(cells)+r" \\")
        if cid.startswith("mice_"):
            p = stats["paper_metrics"]
            name = "MIMIC-IV" if cid == "mice_mimic" else cid.split("_")[-1].upper()
            mice.append(f"{name} & {p['coverage']['mean']:.4f} & {p['shortfall']['mean']:.4f} & {p['length']['mean']:.3f} & ${stats['masks_below_nominal_after_seed_averaging']}/{len(stats['pooled_masks'])}$"+r" \\")
    for name, lines in (("coordinate_coverage", coordinate), ("mice_comparison", mice)):
        if lines:
            (output / f"{name}.tex").write_text("\n".join(lines)+"\n")


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
    output = args.output or args.results / ("summary_"+"_".join(args.group)+("_with_baselines" if args.include_baselines else ""))
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "summary.json", json_safe(summary))
    write_json(output / "per_seed.json", json_safe(rows))
    write_tables(output, summary)
    print(f"Summarized {len(configuration)} complete configurations into {output}")


if __name__ == "__main__":
    main()
