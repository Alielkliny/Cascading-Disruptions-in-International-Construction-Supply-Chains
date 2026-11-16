# Incentive and timing model for supply recovery

Decides what recovery payment a contractor offers a supplier, and when to act, when disruptions spread through the supply pipeline of a construction project. A discrete-event simulation drives the supply pipeline and the project network; Stackelberg solvers compute the supplier's best response and the contractor's leader decision, and a centralised plan serves as a reference. Instances come from DSLIB construction projects, and disruption persistence is calibrated on the Global Supply Chain Pressure Index.

## Scope of this repository

This repository holds the code of the method, the comparison methods and the evaluation, with the split,
protocol and settings files that the scripts read. It does not hold the input data, run outputs, figures or tables.

## Repository layout

| Path | Contents |
|---|---|
| `code/core.py` | Supply-pipeline dynamics shared by the simulation and the planning model |
| `code/engine.py` | Instances, disruption scenarios, policies and the simulation driver |
| `code/solvers.py`, `fastplan.py` | Stackelberg solvers and the exact fast path of the planning searches |
| `code/dslib.py`, `admit_instances.py`, `admit_fresh.py`, `build_pool.py` | DSLIB parsing, instance admission and splits |
| `code/runner.py`, `run_stage.py`, `run_fast.py`, `run_chunks.py`, `run_test_blocks.py`, `run_test_chunks.py` | Runners with common random numbers |
| `code/tune_dev.py`, `select_kappa.py`, `select_a7.py` | Tuning and selection rules on development projects |
| `code/score.py` | Normalised metrics and paired tests with Holm adjustment |
| `code/experiments.py` | Case study, thresholds, compound shocks and sensitivity |
| `code/gscpi_analysis.py` | Disruption persistence from the GSCPI |
| `code/check_fast.py`, `code/tests/` | Checks and verification tests |
| `data/processed/*.json`, `results/design/*.json` | Splits, the selected design and the tuned comparator settings |

## Environment

The code was run with Python 3.13 on Windows 11. Install the packages with

```
python -m pip install -r requirements.txt
```

The versions in `requirements.txt` are those of the Python environment the code was run in.

## Input data

The input data are public and are not redistributed here. Place them as follows.

- DSLIB 3.4 as `data/raw/dslib/DSLIB3.4.zip` (extracted to `data/raw/dslib/extracted/DSLIB 3.4/`).
- NY Fed Global Supply Chain Pressure Index as `data/raw/gscpi/gscpi_data.xlsx`.

## Running the code

```
cd code
python admit_instances.py
python admit_fresh.py
python build_pool.py
python gscpi_analysis.py
python tune_dev.py
python select_kappa.py
python select_a7.py
python run_stage.py ...   (see header)
python run_test_blocks.py
python score.py
python experiments.py <name>
python -m pytest -q tests
```

The usage line in each script header gives its arguments.
