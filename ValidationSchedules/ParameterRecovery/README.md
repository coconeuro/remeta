# Parameter Recovery Validation Pipeline

This module provides a comprehensive parameter recovery validation pipeline for ReMeta. It allows you to verify whether ReMeta can successfully recover the true parameters used to generate synthetic data.

## Overview

The parameter recovery pipeline consists of three main components:

1. **`ParameterRecovery` class** (`parameter_recovery.py`): Core pipeline that generates synthetic data, fits ReMeta, computes parameter differences, and saves results
2. **Interface script** (`param_recovery_interface.py`): Front-end for running multiple parameter recovery experiments in parallel with two modes:
   - **Independent Data Mode**: Each run generates its own dataset (studies data variance)
   - **Shared Data Mode**: All runs use the same dataset (studies fitting variance)
3. **Analysis tools**: Jupyter notebook and Streamlit webapp for visualizing and analyzing results

## Directory Structure

```
ValidationSchedules/ParameterRecovery/
├── __init__.py
├── parameter_recovery.py          # Core ParameterRecovery class
├── param_recovery_interface.py    # Parallel execution interface
├── notebooks/
│   └── ParameterRecoveryResultsDashboard.ipynb
└── webapp/
    ├── dashboard_app.py           # Streamlit dashboard
    └── utils.py                   # Shared utilities

Experimentations/
├── ParameterRecovery/
│   └── <experiment_id>/           # Experiment results
│       ├── run_<run_id>/          # Individual run results
│       └── experiment_summary.txt
└── SyntheticData/
    └── <experiment_id>/           # Generated datasets
        ├── shared_data.pkl        # Shared dataset (if applicable)
        ├── shared_data_metadata.pkl
        ├── data_run_<run_id>.pkl  # Independent datasets
        └── data_run_<run_id>_metadata.pkl
```

## Quick Start

### 1. Environment Setup

Activate the conda environment:

```bash
conda activate remeta_env
```

### 2. Choose Your Experiment Mode

#### Mode A: Independent Data (Default)

Each run generates its own dataset. Use this to study data variance from the generative model.

Edit `param_recovery_interface.py`:

```python
# Set mode flag
USE_SHARED_DATA = False

# Define experiment ID
experiment_id = "my_experiment_001"

# Define parameter sets for each run
remeta_param_sets = {
    'run_001': {
        'enable_noise_sens': 2,
        'noise_sens': [0.5, 0.7],
        'bias_sens': -0.1,
        'noise_meta': 0.1
    },
    'run_002': {
        'noise_sens': 0.6,
        'bias_sens': 0.0,
        'noise_meta': 0.15
    },
    'run_003': None  # Baseline with defaults (None or {} both supported)
}

# Define dataset specifications (one per run)
data_param_sets = {
    'run_001': {'nsubjects': 1, 'nsamples': 1000, 'stimuli_stepsize': 0.25},
    'run_002': {'nsubjects': 1, 'nsamples': 1000, 'stimuli_stepsize': 0.25},
    'run_003': {'nsubjects': 1, 'nsamples': 1000, 'stimuli_stepsize': 0.25}
}
```

#### Mode B: Shared Data

All runs use the same dataset. Use this to study fitting variance on identical data.

Edit `param_recovery_interface.py`:

```python
# Set mode flag
USE_SHARED_DATA = True

# Define experiment ID
experiment_id = "my_shared_data_exp"

# Define parameter sets for each run (can vary model parameters)
remeta_param_sets = {
    'run_001': {'noise_meta': 0.1, 'bias_sens': -0.1},
    'run_002': {'noise_meta': 0.15, 'bias_sens': 0.0},
    'run_003': {'noise_meta': 0.2, 'bias_sens': 0.1}
}

# Define SINGLE dataset specification (shared across all runs)
shared_data_params = {
    'nsubjects': 1,
    'nsamples': 1000,
    'stimuli_stepsize': 0.25
}
```

### 3. Run the Experiment

```bash
cd ValidationSchedules/ParameterRecovery
python param_recovery_interface.py
```

This will:
- Create an experiment directory in `/Experimentations/ParameterRecovery/<experiment_id>`
- Execute all runs in parallel
- Save results, figures, and logs for each run
- Save datasets to `/Experimentations/SyntheticData/<experiment_id>`

### 4. Analyze Results

#### Option A: Jupyter Notebook

```bash
jupyter notebook notebooks/ParameterRecoveryResultsDashboard.ipynb
```

#### Option B: Streamlit Webapp

```bash
streamlit run webapp/dashboard_app.py
```

## Experiment Modes Explained

### Independent Data Mode

**When to use**: Study robustness across different random datasets

**Characteristics**:
- Each run generates its own dataset using `remeta.simu_data()`
- Datasets saved as `/Experimentations/SyntheticData/<exp_id>/data_run_<run_id>.pkl`
- Studies **data variance** from the generative model
- Runs can have different `remeta_param_dict` and `data_param_dict`

**Use cases**:
- Testing parameter recovery under different data conditions
- Studying robustness to different sample sizes
- Analyzing impact of data variance on parameter estimation

### Shared Data Mode

**When to use**: Study fitting variance on identical datasets

**Characteristics**:
- Dataset generated once before parallel execution
- All runs use the same Simulation object
- Dataset saved as `/Experimentations/SyntheticData/<exp_id>/shared_data.pkl`
- Studies **fitting variance** on identical data
- Runs can have different `remeta_param_dict` (same `data_param_dict`)

**Use cases**:
- Testing model sensitivity to different parameter configurations
- Studying convergence behavior with different initializations
- Isolating fitting variance from data variance

## Usage Details

### ParameterRecovery Class

The core class that implements the parameter recovery pipeline:

```python
from ValidationSchedules.ParameterRecovery import ParameterRecovery

# Create instance
pr = ParameterRecovery(
    primary_storage_dir="/path/to/Experimentations/ParameterRecovery/experiment_001",
    run_id="run_001",
    remeta_param_dict={'noise_sens': [0.5, 0.7], 'bias_sens': -0.1, ...},
    data_param_dict={'nsubjects': 1, 'nsamples': 1000, 'stimuli_stepsize': 0.25}
)

# Run the pipeline
result = pr.run()
```

**Pipeline steps:**
1. Creates Configuration object from `remeta_param_dict`
2. Generates synthetic data using `remeta.simu_data()`
3. Fits ReMeta model to the data
4. Computes parameter differences (Delta)
5. Saves all figures (PNG and PDF)
6. Saves results as `run_dict.pkl`
7. Creates SUCCESS marker

### Output Structure

Each run creates a directory with the following structure:

```
run_<ID>/
├── run_dict.pkl                  # Results dictionary (P, D, P_hat, Delta, negll)
├── <ID>_out.txt                  # Log file with terminal output
├── SUCCESS                       # Marker file (present if successful)
├── psychometric.png              # Psychometric curve
├── confidence.png                # Confidence plot
├── link_function.png             # Link function plot
├── confidence_dist.png           # Confidence distribution
└── figpdfs/                      # PDF versions for papers
    ├── psychometric.pdf
    ├── confidence.pdf
    ├── link_function.pdf
    └── confidence_dist.pdf
```

### Results Dictionary Structure

The `run_dict.pkl` file contains:

```python
{
    'P': params_true,              # True parameters used for data generation
    'D': data,                     # Generated synthetic dataset
    'P_hat': params_estimated,     # Estimated parameters from fitting
    'Delta': delta,                # Differences (P_hat - P)
    'negll': {                     # Negative log-likelihoods
        'negll_sens_true': ...,
        'negll_sens_fitted': ...,
        'negll_meta_true': ...,
        'negll_meta_fitted': ...
    },
    'cfg': cfg,                    # Configuration object
    'run_id': run_id               # Run identifier
}
```

### Parallel Execution

The `parallel_execution()` function uses Python's `multiprocessing` to run multiple parameter recovery pipelines in parallel:

```python
from ValidationSchedules.ParameterRecovery.param_recovery_interface import parallel_execution

# Create list of ParameterRecovery instances
jobs = [pr1, pr2, pr3, ...]

# Execute in parallel (defaults to all available CPU cores)
results = parallel_execution(jobs, n_workers=4)
```

Features:
- Progress bar with `tqdm`
- Error handling (logs errors and continues with other runs)
- Scalable to cluster environments

### Analysis Utilities

The `utils.py` module provides functions for loading and analyzing results:

```python
from ValidationSchedules.ParameterRecovery.webapp import utils

# Load results from a run
results_df, extras_dict = utils.load_param_recov_results(run_dir)

# Load figures
fig_paths = utils.load_param_recov_figs(run_dir, format='png')

# Load log file
log_content = utils.load_param_recov_history(run_dir)

# List available runs
available_runs = utils.list_available_runs(primary_storage_dir, experiment_id)

# Check run status
status = utils.check_run_status(run_dir)  # 'SUCCESS', 'FAILED', or 'IN_PROGRESS'
```

## Parameter Specifications

### ReMeta Parameters (`remeta_param_dict`)

This dictionary can contain:

**Configuration parameters** (prefix `enable_*`):
- `enable_noise_sens`: 0, 1, or 2 (disabled, enabled, duplex mode)
- `enable_bias_sens`: 0 or 1
- `enable_noise_meta`: 0 or 1
- `enable_evidence_bias_mult_meta`: 0 or 1
- `enable_evidence_bias_add_meta`: 0 or 1
- And others (see ReMeta Configuration docs)

**Actual parameter values**:
- `noise_sens`: float or [float, float] for duplex mode
- `bias_sens`: float
- `noise_meta`: float
- `evidence_bias_mult_meta`: float
- `evidence_bias_add_meta`: float

**Special case**: `None` or an empty dictionary `{}` uses the default Configuration with default values (baseline).

### Data Parameters (`data_param_dict`)

- `nsubjects`: Number of participants (typically 1 for synthetic data)
- `nsamples`: Number of trials per participant
- `stimuli_stepsize`: Step size for stimulus intensities (e.g., 0.25 creates stimuli at ±0.25, ±0.5, ±0.75, ±1.0)

## Tips and Best Practices

1. **Start with baseline**: Always include a run with empty `remeta_param_dict` to establish baseline performance

2. **Use small samples for testing**: Set `nsamples=100` during development, increase to 1000+ for real experiments

3. **Monitor logs**: Check `<ID>_out.txt` files to ensure fitting converged successfully

4. **Compare negll**: Lower negative log-likelihood for fitted vs true parameters indicates successful parameter recovery

5. **Visualize results**: Use the Streamlit dashboard for quick interactive exploration

6. **Check SUCCESS markers**: Only analyze runs with SUCCESS marker files

## Troubleshooting

### Run fails with import errors
- Ensure `remeta_env` conda environment is activated
- Check that all dependencies are installed: `pip install -r requirements.txt`

### Parallel execution uses too many resources
- Reduce `n_workers` parameter in `parallel_execution()`
- Run jobs sequentially by setting `n_workers=1`

### Figures not displaying in webapp
- Check that PNG files exist in run directory
- Verify file permissions

### Results loading fails
- Check that `run_dict.pkl` exists
- Verify run completed successfully (SUCCESS marker present)

## Development

### Running Unit Tests

```bash
cd unit_tests/test_parameter_recovery
python -m unittest test_parameter_recovery.py
python -m unittest test_interface.py
```

### Adding New Features

1. Add TODOs in skeleton files
2. Implement functionality
3. Write unit tests
4. Update this README

## References

- [ReMeta GitHub Repository](https://github.com/m-guggenmos/remeta/)
- [ReMeta Basic Usage](https://github.com/m-guggenmos/remeta/blob/master/demo/basic_usage.ipynb)
- ReMeta paper: Guggenmos, M. (2021). Reverse engineering of metacognition. *bioRxiv*.
