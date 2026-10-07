import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import os
from pathlib import Path
import subprocess
import sys
import urllib.request

from project import CONFIGURATION, ROOT, file_hash, identity, parse_seeds, read_json, selected_configs, source_hashes, write_json


def prepare_data(args, entries, output):
    paths = {}
    names = {e["parameters"]["dataset_name"] for e in entries.values()}
    if "mimic" in names:
        for name in ("mimic_train", "mimic_test"):
            value = getattr(args, name)
            if not value or not Path(value).is_file():
                raise FileNotFoundError(f"Provide the existing preprocessed file with --{name.replace('_', '-')}")
            paths[name] = str(Path(value).resolve())
    if "concrete_compression" in names:
        if args.concrete:
            path = Path(args.concrete).resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
        else:
            path = output / "datasets" / "Concrete_Data.xls"
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                temporary = path.with_suffix(".download")
                urllib.request.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls", temporary)
                os.replace(temporary, path)
        paths["concrete"] = str(path)
    return {k: dict(path=v, sha256=file_hash(v)) for k, v in paths.items()}


def execute(job, workers, retry):
    directory, specification, seed = job
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return f"Already running: {specification['id']} seed {seed}", False
        status_path = directory / "status.json"
        if status_path.exists():
            previous = read_json(status_path)
            if previous["identity"] != specification["identity"]:
                raise ValueError(f"Configuration or source mismatch: {directory}; choose a new output directory")
            if previous["status"] == "completed":
                for name, digest in previous["artifacts"].items():
                    if file_hash(directory / previous["attempt"] / name) != digest:
                        raise ValueError(f"Changed completed artifact: {directory / name}")
                return f"Reused: {specification['id']} seed {seed}", True
            if not retry:
                return f"Incomplete: {directory}; use --retry-failed to retain this attempt and start another", False
        index = len(list(directory.glob("attempt_*"))) + 1
        attempt = directory / f"attempt_{index:03d}"
        attempt.mkdir()
        request = dict(specification, seed=seed, prediction_workers=workers, output=str(attempt))
        write_json(attempt / "request.json", request)
        status = dict(status="running", seed=seed, identity=specification["identity"], attempt=attempt.name,
                      started=datetime.now(timezone.utc).isoformat())
        write_json(status_path, status)
        environment = os.environ.copy()
        for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "EXPERIMENT_THREADS", "MOSEK_NUM_THREADS"):
            environment[key] = "1"
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        with (attempt / "run.log").open("w") as stream:
            process = subprocess.run([sys.executable, "-B", str(ROOT / "worker.py"), str(attempt / "request.json")],
                                     cwd=attempt, env=environment, stdout=stream, stderr=subprocess.STDOUT)
        status["returncode"] = process.returncode
        status["finished"] = datetime.now(timezone.utc).isoformat()
        if process.returncode == 0:
            products = ["predictions.npz", "patterns.csv", "protocol.json"]
            products.extend(n for n in ("representation.csv", "numerical.jsonl") if (attempt / n).exists())
            missing = [name for name in products if not (attempt / name).is_file()]
            if missing:
                status["status"] = "failed"
                status["error"] = "Missing output artifacts: " + ", ".join(missing)
            else:
                status["artifacts"] = {name: file_hash(attempt / name) for name in products}
                status["status"] = "completed"
        else:
            status["status"] = "failed"
        write_json(status_path, status)
        return f"{status['status']}: {specification['id']} seed {seed}", status["status"] == "completed"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", nargs="+", choices=[f"{i:02d}" for i in range(1, 10)], required=True)
    parser.add_argument("--configuration", type=Path, default=CONFIGURATION)
    parser.add_argument("--only", nargs="+")
    parser.add_argument("--seeds", default="0-9")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--prediction-workers", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--concrete", default=os.environ.get("CONCRETE_PATH"))
    parser.add_argument("--mimic-train", default=os.environ.get("MIMIC_TRAIN_PATH"))
    parser.add_argument("--mimic-test", default=os.environ.get("MIMIC_TEST_PATH"))
    parser.add_argument("--retry-failed", action="store_true")
    args = parser.parse_args()
    if args.jobs < 1 or args.prediction_workers < 1:
        parser.error("Worker counts must be positive")
    entries = selected_configs(read_json(args.configuration), args.group)
    if args.only:
        if not set(args.only) <= entries.keys():
            parser.error("--only must name our method or ablation configurations in the selected groups; see README.md for external baseline repositories")
        entries = {key: entries[key] for key in args.only}
    seeds = parse_seeds(args.seeds)
    if not entries or any(max(seeds) >= e["parameters"]["reps"] for e in entries.values()):
        parser.error("The selected groups or replication indices are invalid")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = prepare_data(args, entries, output)
    jobs = []
    for key, entry in entries.items():
        dataset = entry["parameters"]["dataset_name"]
        inputs = {k: v for k, v in data.items() if (dataset == "mimic" and k.startswith("mimic_")) or (dataset == "concrete_compression" and k == "concrete")}
        specification = dict(id=key, engine=entry["engine"], parameters=entry["parameters"],
                             data=inputs, sources=source_hashes(entry["engine"]))
        specification["identity"] = identity(specification)
        for seed in seeds:
            jobs.append((output / key / f"seed{seed}", specification, seed))
    passed = True
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(execute, job, args.prediction_workers, args.retry_failed) for job in jobs]
        for future in as_completed(futures):
            message, ok = future.result()
            print(message, flush=True)
            passed &= ok
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
