# AIC Comparison Feature - Implementation Summary

## Overview
Added Akaike Information Criterion (AIC) comparison functionality to the Parameter Recovery pipeline as a sanity check for model fitting quality.

## What is AIC?
**AIC (Akaike Information Criterion)** = 2 × negLL + 2 × k

Where:
- negLL = negative log-likelihood  
- k = number of free parameters

**Lower AIC = better model fit** (balances goodness-of-fit with model complexity)

## Purpose
The fitted parameters should have **lower (better) AIC** than the true parameters used for data generation. This serves as a sanity check that:
1. The fitting procedure is working correctly
2. The fitted model explains the data better than the ground truth (as expected for parameter recovery)
3. The optimization converged to a reasonable solution

## Implementation Details

### 1. Core Method: `_compute_aic()`
Location: `parameter_recovery.py` (lines 457-525)

**Functionality:**
- Computes AIC for both true and fitted parameters
- Breaks down AIC by component (sensitivity and metacognition)
- Performs sanity check (fitted ≤ true)
- Logs comprehensive comparison results

**Returns:**
```python
{
    'aic_sens_true': float,      # AIC for true sensory params
    'aic_sens_fitted': float,    # AIC for fitted sensory params  
    'aic_meta_true': float,      # AIC for true metacog params
    'aic_meta_fitted': float,    # AIC for fitted metacog params
    'aic_total_true': float,     # Total AIC for true params
    'aic_total_fitted': float    # Total AIC for fitted params
}
```

### 2. Pipeline Integration
**Updated run() method to include AIC computation:**

Previous pipeline (6 steps):
1. Create configuration
2. Generate synthetic data
3. Fit model
4. Compute delta
5. Save figures
6. Save results

New pipeline (7 steps):
1. Create configuration
2. Generate synthetic data
3. Fit model
4. Compute delta
5. **Compute AIC** ← NEW
6. Save figures
7. Save results

### 3. Data Storage
**Added to `run_dict.pkl`:**
- New key: `'aic'` containing the AIC dictionary
- Persists alongside P, D, P_hat, Delta, negll, cfg

**Updated _save_results() method:**
- Includes `aic_dict` in saved pickle
- Documented in docstring

### 4. Attribute Initialization
**Added to `__init__` method:**
```python
self.aic_dict = {}  # Dictionary storing AIC values for true and fitted params
```

## Testing

### Test File: `test_aic_method.py`
- **Purpose:** Unit test for `_compute_aic()` method with mock data
- **Coverage:**
  - Mock configuration with 2 sens + 1 meta parameters
  - Mock negll values for components
  - Verifies all 6 AIC values computed correctly
  - Validates sanity check logic

### Test Results:
```
✓ All AIC Method Tests PASSED!
  ✓ Has aic_sens_true key
  ✓ Has aic_sens_fitted key
  ✓ Has aic_meta_true key
  ✓ Has aic_meta_fitted key
  ✓ Has aic_total_true key
  ✓ Has aic_total_fitted key
  ✓ All computed values match expected
  ✓ Sanity check: fitted <= true
```

## Interpretation

### Expected Behavior:
**aic_total_fitted < aic_total_true** → ✓ GOOD
- Fitted parameters explain data better
- Parameter recovery successful
- Optimization converged properly

**aic_total_fitted > aic_total_true** → ⚠ WARNING  
- May indicate fitting issues
- Could be due to local minima
- Warrants investigation

### Logging Output Example:
```
================================================================================
COMPUTING AIC COMPARISON  
================================================================================
Number of parameters: sens=2, meta=1, total=3

AIC Results:
  AIC (true params):   1506.40
  AIC (fitted params): 1496.20
    - Sensitivity:     794.00
    - Metacognition:   702.20
  Delta AIC:           -10.20
  ✓ Sanity check PASSED: AIC_fitted <= AIC_true
================================================================================
```

## Usage

### Access AIC Results:
```python
# From run() return value
result = pr.run()
aic_dict = result['aic']

# From saved pickle
with open('run_dict.pkl', 'rb') as f:
    data = pickle.load(f)
aic_dict = data['aic']

# Component breakdown
aic_sens_true = aic_dict['aic_sens_true']
aic_meta_fitted = aic_dict['aic_meta_fitted']
aic_total_true = aic_dict['aic_total_true']
aic_total_fitted = aic_dict['aic_total_fitted']

# Check if fitted is better
if aic_total_fitted < aic_total_true:
    print("✓ Fitted model has better AIC")
```

### Dashboard Integration (Future):
Can add AIC comparison table to Streamlit dashboard showing:
- Component-wise AIC values  
- Total AIC comparison
- Delta AIC with visual indicator (✓/⚠)

## Files Modified

1. **parameter_recovery.py**
   - Added `_compute_aic()` method (lines 457-525)
   - Updated `run()` to call `_compute_aic()` (Step 5/7)
   - Added `self.aic_dict = {}` to `__init__`
   - Updated `_save_results()` to save AIC dict
   - Updated return dict in `run()` to include AIC

2. **test_aic_method.py** (NEW)
   - Unit test for AIC computation
   - Mock-based testing without data generation
   - Validates all computations and sanity checks

## Benefits

1. **Quality Assurance:** Automatic sanity check for every run
2. **Debugging Aid:** Helps identify fitting failures early
3. **Documentation:** AIC values persist in run_dict.pkl  
4. **Model Selection:** Can compare different configurations using AIC
5. **Reproducibility:** Standardized metric across all experiments

## Next Steps

1. **Dashboard Visualization:** Add AIC comparison table to Streamlit app
2. **Notebook Integration:** Update analysis notebook with AIC section
3. **Batch Analysis:** Create utility to summarize AIC across multiple runs
4. **Failure Detection:** Flag runs where AIC sanity check fails

## References

- Akaike, H. (1974). "A new look at the statistical model identification"
- AIC formula: 2k - 2ln(L) = 2k + 2·negLL
- Lower AIC indicates better model (penalizes both poor fit and complexity)
