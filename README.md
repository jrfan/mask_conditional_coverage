# Beyond Imputation — NeurIPS 2026

This repository accompanies the NeurIPS 2026 paper **Beyond Imputation: Mask-Adaptive Conformal Prediction via Tree Embeddings on General Missing Data Mechanisms**.

The experiment configurations select the appropriate implementation in `engines/part1/` or `engines/part2/` automatically.

## Setup

The reference environment is Linux with Python 3.10.20. Use Python 3.10 and the pinned dependencies for numerical reproduction. Other platforms can differ in floating-point details.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Entry points

| Entry | Experiment and output |
| --- | --- |
| `run_01.sh` | Our method on low-dimensional synthetic data; PMIX seen/unseen panels |
| `run_02.sh` | Coordinate-wise coverage table |
| `run_03.sh` | Observed-pattern coverage versus calibration count |
| `run_04.sh` | Tree-count ablation and selected-lambda plot |
| `run_05.sh` | Our method on Concrete |
| `run_06.sh` | Our method on MIMIC and representation-error diagnostic |
| `run_07.sh` | MICE-plus-mask table |
| `run_08.sh` | LightGBM, XGBoost and CatBoost comparison |
| `run_09.sh` | Our method on high-dimensional synthetic data |

```bash
bash run_01.sh
bash run_04.sh --seeds 0-9 --jobs 2 --prediction-workers 4
bash run_05.sh --concrete /path/to/Concrete_Data.xls
bash run_06.sh --mimic-train /path/to/train.npz --mimic-test /path/to/test_top50.npz
bash run_07.sh --mimic-train /path/to/train.npz --mimic-test /path/to/test_top50.npz
```

Defaults are ten replications (indices 0–9), one concurrent job, one prediction worker and one library thread. `--seeds 0,2,4-6` runs a subset without changing the ten-replication data generator. Set `PYTHON_BIN` to choose a Python executable for Bash entrypoints. `--only CONFIG_ID` selects our method or ablation configurations within an entrypoint. To run all groups, use `python -B main.py --group 01 02 03 04 05 06 07 08 09` with the required data arguments. Neither the numbered entrypoints nor `main.py` launches Exact, Nested, Weighted CP or ARC CP.

By default, outputs are created under this project's `results/`, independently of the caller's working directory. `--output /path/to/results` selects another directory. Identical configurations share the same output, including the 256-tree MAR run shared by groups 01 and 04. Completed runs are reused only if their configuration, source identity and saved artifacts match. After correcting a failed run, use `--retry-failed`; earlier attempts remain intact. Use a new output directory after changing code or configurations.

Synthetic data are generated locally. Concrete is downloaded from the UCI repository when `--concrete` is omitted. MIMIC requires the already-preprocessed training file and corresponding top-50-mask test file, each with numeric `X` and `y` arrays; missing covariates are NaNs. The original experiment uses 300 feature columns, a 3000-row training pool, 1000 calibration rows and all 2331 test rows from the supplied top-50-mask file. No patient data or preprocessing code is included. `CONCRETE_PATH`, `MIMIC_TRAIN_PATH` and `MIMIC_TEST_PATH` are alternatives to the path arguments.

## External comparison methods

| Methods | Paper | Repository |
| --- | --- | --- |
| CP-MDA-Exact and CP-MDA-Nested | Zaffran et al., *Conformal Prediction with Missing Values* (ICML 2023) | [mzaffran/ConformalPredictionMissingValues](https://github.com/mzaffran/ConformalPredictionMissingValues) |
| Weighted CP and ARC CP | Fan et al., *Mask-Conditional Conformal Prediction: Valid Uncertainty For All Missing Data Mechanisms* (AISTATS 2026) | [jrfan/Mask-Conditional-Conformal-Prediction](https://github.com/jrfan/Mask-Conditional-Conformal-Prediction) |
