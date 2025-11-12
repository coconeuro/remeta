# Shared Data Feature Implementation

## Summary

Implemented a dual-mode system for parameter recovery experiments that allows users to choose between:
1. **Independent Data Mode**: Each run generates its own dataset (studies data variance)
2. **Shared Data Mode**: All runs use the same dataset (studies fitting variance)

## Motivation

The original implementation had a systematic issue: multiple runs with identical `data_param_dict` would generate different datasets due to the non-deterministic generative model. This conflated two sources of variance:
- **Data variance**: Different datasets from the same generative model
- **Fitting variance**: Different fits on identical data

The new implementation separates these concerns, enabling both types of studies.

## Key Changes

### 1. ParameterRecovery Class (`parameter_recovery.py`)

#### Modified `__init__()`
Added new parameters:
- `experiment_id` (str): Experiment identifier for data storage organization
- `shared_data` (Optional): Pre-generated shared dataset (Simulation object)
- `is_data_saver` (bool): Designates one run to save shared data

```python
def __init__(
    self,
    primary_storage_dir: str,
    run_id: str,
    remeta_param_dict: Dict[str, Any],
    data_param_dict: Dict[str, Any],
    experiment_id: str,
    shared_data: Optional[Any] = None,
    is_data_saver: bool = False
):
```

#### Modified `_generate_synthetic_data()`
Now supports two modes:
- If `shared_data is not None`: Uses provided data (shared mode)
- If `shared_data is None`: Generates new data (independent mode)

Both modes save datasets to `/Experimentations/SyntheticData/<experiment_id>/`:
- Shared mode: `shared_data.pkl` (saved by designated data saver)
- Independent mode: `data_run_<run_id>.pkl` (each run saves its own)

#### Added `_save_dataset()`
New method to handle dataset persistence:
- Saves Simulation objects as pickle files
- Saves metadata (data_param_dict) for reproducibility
- Creates `/Experimentations/SyntheticData/<experiment_id>/` directory structure

### 2. Interface Script (`param_recovery_interface.py`)

#### Added Helper Functions

**`create_shared_data_experiment()`**
- Generates shared dataset once before parallel execution
- Creates ParameterRecovery instances with `shared_data` parameter
- Designates first run as data saver
- Returns list of configured instances

**`create_independent_data_experiment()`**
- Creates ParameterRecovery instances without shared data
- Each instance will generate its own dataset
- Returns list of configured instances

#### Modified `main()`
- Added `USE_SHARED_DATA` flag for mode selection
- Conditional logic to call appropriate helper function
- Updated experiment summary to include mode information

### 3. Directory Structure

New directory added to project:
```
Experimentations/
├── ParameterRecovery/
│   └── <experiment_id>/           # Results storage (unchanged)
│       ├── run_<run_id>/
│       └── experiment_summary.txt
└── SyntheticData/                 # NEW: Dataset storage
    └── <experiment_id>/
        ├── shared_data.pkl        # Shared dataset
        ├── shared_data_metadata.pkl
        ├── data_run_<run_id>.pkl  # Independent datasets
        └── data_run_<run_id>_metadata.pkl
```

### 4. Unit Tests

Updated both test files to accommodate new parameters:

#### `test_parameter_recovery.py`
- Added `experiment_id` parameter to all test fixtures
- Added `test_initialization_with_shared_data()` test case
- Updated all ParameterRecovery instantiations

#### `test_interface.py`
- Added `test_experiment_id` to setUp()
- Updated all ParameterRecovery instantiations with `experiment_id`

### 5. Documentation

Updated `README.md` with:
- Explanation of both modes
- When to use each mode
- Code examples for both modes
- Updated directory structure diagram
- Use case descriptions

## Usage Examples

### Independent Data Mode (Default)
```python
USE_SHARED_DATA = False

remeta_param_sets = {
    'run_001': {'noise_meta': 0.1},
    'run_002': {'noise_meta': 0.15},
    'run_003': {'noise_meta': 0.2}
}

data_param_sets = {
    'run_001': {'nsubjects': 1, 'nsamples': 1000, 'stimuli_stepsize': 0.25},
    'run_002': {'nsubjects': 1, 'nsamples': 1000, 'stimuli_stepsize': 0.25},
    'run_003': {'nsubjects': 1, 'nsamples': 1000, 'stimuli_stepsize': 0.25}
}

parameter_recovery_runs = create_independent_data_experiment(
    experiment_id=experiment_id,
    primary_storage_dir=str(primary_storage_dir),
    remeta_param_sets=remeta_param_sets,
    data_param_sets=data_param_sets
)
```

### Shared Data Mode
```python
USE_SHARED_DATA = True

remeta_param_sets = {
    'run_001': {'noise_meta': 0.1, 'bias_sens': -0.1},
    'run_002': {'noise_meta': 0.15, 'bias_sens': 0.0},
    'run_003': {'noise_meta': 0.2, 'bias_sens': 0.1}
}

shared_data_params = {
    'nsubjects': 1,
    'nsamples': 1000,
    'stimuli_stepsize': 0.25
}

parameter_recovery_runs = create_shared_data_experiment(
    experiment_id=experiment_id,
    primary_storage_dir=str(primary_storage_dir),
    remeta_param_sets=remeta_param_sets,
    data_param_dict=shared_data_params,
    data_generation_run_id='run_001'  # Optional: specify which run's params to use
)
```

## Technical Implementation Details

### Data Passing Strategy
- **Memory**: Shared data passed as objects in memory to all ParameterRecovery instances
- **Persistence**: Datasets saved to disk for later reference and reproducibility
- **No redundancy**: Only one designated run saves shared data to avoid race conditions

### Dataset Storage
- Format: Pickle (`.pkl`) for Simulation objects
- Metadata: Separate pickle files with `data_param_dict` for reproducibility
- Location: Centralized in `/Experimentations/SyntheticData/` for easy management

### Backward Compatibility
- Default behavior: `shared_data=None` maintains independent data generation
- Existing code: Works without modification (experiment_id required, but can be added easily)
- Tests: Updated but maintain same test coverage

## Use Cases Enabled

1. **Data Variance Study** (Independent Mode)
   - How much do parameters vary across different random datasets?
   - Is parameter recovery robust to different data realizations?

2. **Fitting Variance Study** (Shared Mode)
   - How sensitive is fitting to different model parameters on the same data?
   - Does the fitting process converge to the same solution?
   - What is the impact of different initializations?

3. **Model Sensitivity Analysis** (Shared Mode)
   - How do different parameter configurations affect the same dataset?
   - Which parameters have the most impact on fit quality?

4. **Convergence Testing** (Shared Mode)
   - Do multiple fits on the same data converge to the same parameters?
   - Is the optimization process stable?

## Benefits

1. **Flexibility**: Users can choose the appropriate mode for their research question
2. **Clarity**: Explicitly separates data variance from fitting variance
3. **Reproducibility**: All datasets saved with metadata for future reference
4. **Efficiency**: Shared data mode generates data once, reducing computation time
5. **Organization**: Centralized dataset storage in `/Experimentations/SyntheticData/`

## Files Modified

1. `ValidationSchedules/ParameterRecovery/parameter_recovery.py`
   - Modified `__init__()` signature
   - Modified `_generate_synthetic_data()` logic
   - Added `_save_dataset()` method

2. `ValidationSchedules/ParameterRecovery/param_recovery_interface.py`
   - Added `create_shared_data_experiment()` function
   - Added `create_independent_data_experiment()` function
   - Modified `main()` to support mode switching
   - Added `import remeta` and `import pickle`

3. `ValidationSchedules/ParameterRecovery/README.md`
   - Added mode explanations
   - Updated directory structure
   - Added usage examples for both modes

4. `unit_tests/test_parameter_recovery/test_parameter_recovery.py`
   - Added `experiment_id` to test fixtures
   - Added `test_initialization_with_shared_data()` test
   - Updated all test instantiations

5. `unit_tests/test_parameter_recovery/test_interface.py`
   - Added `test_experiment_id` to setUp()
   - Updated all test instantiations

## Testing Recommendations

1. Run unit tests to verify basic functionality:
   ```bash
   python -m unittest discover unit_tests/test_parameter_recovery
   ```

2. Test independent mode with small dataset:
   ```python
   USE_SHARED_DATA = False
   # Run with nsamples=100 for quick test
   ```

3. Test shared mode with small dataset:
   ```python
   USE_SHARED_DATA = True
   # Run with nsamples=100 for quick test
   ```

4. Verify dataset files are created in `/Experimentations/SyntheticData/`

5. Check experiment_summary.txt includes mode information

## Future Enhancements (Optional)

1. Add dataset loading utilities to reload saved datasets
2. Add dataset comparison tools to verify datasets are identical in shared mode
3. Add visualization of dataset properties (stimulus distribution, etc.)
4. Support for seeded random generation in independent mode for reproducibility
5. Dashboard updates to distinguish shared vs independent experiments
