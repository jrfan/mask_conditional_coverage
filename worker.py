import os
from pathlib import Path
import sys
from types import SimpleNamespace
import warnings

from project import ROOT, read_json, write_json


def run(request):
    engine = request["engine"]
    sys.path.insert(0, str(ROOT / "engines" / engine))
    os.environ["SKLEARN_ALLOW_INVALID_FEATURE_NAMES"] = "1"
    os.environ["TEST_PARALLEL_JOBS"] = str(request["prediction_workers"] if request["prediction_workers"] > 1 else 0)
    os.environ["NUMERICAL_REPORT_PATH"] = str(Path(request["output"]) / "numerical.jsonl")
    os.environ["REPLICATION_INDEX"] = str(request["seed"])
    os.environ.pop("REPLICATION_START", None)
    for name, item in request["data"].items():
        os.environ[{"concrete": "CONCRETE_PATH", "mimic_train": "MIMIC_TRAIN_PATH", "mimic_test": "MIMIC_TEST_PATH"}[name]] = item["path"]
    warnings.simplefilter("ignore", FutureWarning)
    warnings.simplefilter("ignore", DeprecationWarning)
    import numpy as np
    from dataloader import Synthetic_linear_data, Real_data, PatternMixtureLinearData, PreMaskedNpzData
    np.random.seed(42)
    values = dict(var_missing=None, mechanism="MCAR")
    values.update(request["parameters"])
    args = SimpleNamespace(**values, replication_index=request["seed"], output_dir=request["output"])
    if isinstance(args.var_missing, str):
        args.var_missing = [int(value) for value in args.var_missing.split(",")]
    if engine == "part2":
        advance_size = 3000 if args.dataset_name == "mimic" else 1000
        for _ in range(request["seed"]):
            np.random.permutation(advance_size)
    if args.dataset_name == "mimic":
        data = PreMaskedNpzData(train_path=request["data"]["mimic_train"]["path"],
            test_path=request["data"]["mimic_test"]["path"], n_rep=args.reps,
            name="mimic", seed=42, train_size=args.train_size, cal_size=args.cal_size, test_size=None)
        if data.d != args.dim or data.test_size_iid != args.test_size_iid:
            raise ValueError("MIMIC feature dimension or test size does not match the configuration")
        if args.num_test_patterns is not None and len(np.unique(np.isnan(data.X_test_full), axis=0)) != args.num_test_patterns:
            raise ValueError("MIMIC test-mask count does not match the configuration")
        args.train_size, args.cal_size = data.train_size, data.cal_size
        args.test_size_iid, args.test_size_per_pattern = data.test_size_iid, data.test_size_per_pattern
    else:
        mask = np.ones(args.dim, dtype=int) if args.var_missing is None else np.asarray(args.var_missing, dtype=int)
        common = dict(n_rep=args.reps, train_size=args.train_size, cal_size=args.cal_size,
            test_size_iid=args.test_size_iid, test_size_per_pattern=args.test_size_per_pattern,
            missing_prob=args.missingproba, var_missing=mask, var_dependent=[], num_test_patterns=args.num_test_patterns)
        if args.dataset_name == "synthetic":
            common.update(dim=args.dim, rho=args.phi, split_train=args.split_train)
            if args.mechanism == "PMIX":
                data = PatternMixtureLinearData(seed=0, **common)
            else:
                data = Synthetic_linear_data(mechanism=args.mechanism, **common)
        elif args.dataset_name == "concrete_compression":
            data = Real_data(name=args.dataset_name, mechanism=args.mechanism, missing_function=None, seed=0, **common)
        else:
            raise ValueError(f"Unsupported dataset: {args.dataset_name}")
    split_state = np.random.get_state()
    data.generate_multiple_data()
    if engine == "part2":
        np.random.set_state(split_state)
    from experiments import ConformalExperiment
    experiment = ConformalExperiment(args, "patterns", data, output_dir=request["output"])
    if args.method == "cond":
        experiment.run()
    else:
        experiment.run_sota(args.method)
    if engine == "part2":
        protocol_path = Path(request["output"]) / "protocol.json"
        protocol = read_json(protocol_path)
        protocol["split_rng"] = dict(seed=42, preadvance_count=request["seed"],
                                     preadvance_size=advance_size, data_generation_isolated=True)
        write_json(protocol_path, protocol)
    with np.load(Path(request["output"]) / "predictions.npz", allow_pickle=False) as z:
        for key in ("y", "lower", "upper", "masks", "training_masks", "calibration_masks"):
            if np.isnan(z[key]).any():
                raise ValueError(f"Invalid prediction output: {key}")
        if z["y"].size != z["lower"].size or z["y"].size != z["upper"].size:
            raise ValueError("Prediction output dimensions do not agree")


if __name__ == "__main__":
    run(read_json(sys.argv[1]))
