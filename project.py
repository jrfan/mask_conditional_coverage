import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CONFIGURATION = ROOT / "configs" / "experiments.json"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_hashes(engine):
    paths = [ROOT / n for n in ("worker.py", "numerical.py", "uci_dataloader.py")]
    paths.extend(sorted((ROOT / "engines" / engine).rglob("*.py")))
    return {str(p.relative_to(ROOT)): file_hash(p) for p in paths}


def selected_configs(configuration, groups, include_baselines=False):
    return {key: entry for key, entry in configuration.items()
            if set(entry["groups"]) & set(groups)
            and (include_baselines or entry["parameters"]["method"] == "cond")}


def parse_seeds(value):
    result = []
    for token in value.split(","):
        if "-" in token:
            lower, upper = map(int, token.split("-"))
            result.extend(range(lower, upper + 1))
        else:
            result.append(int(token))
    if not result or len(set(result)) != len(result) or min(result) < 0:
        raise ValueError("Seeds must be distinct nonnegative replication indices")
    return sorted(result)
