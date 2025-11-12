"""
Utility functions for loading and displaying Parameter Recovery results.

This module provides shared utilities for both the Jupyter notebook dashboard
and the Streamlit webapp.
"""

import os
import pickle
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import pandas as pd
import numpy as np


def unpack_array_params(params_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Unpack array/list parameters into separate scalar entries.
    
    For example:
        {'noise_sens': [0.5, 0.7]} -> {'noise_sens1': 0.5, 'noise_sens2': 0.7}
    
    Parameters
    ----------
    params_dict : Dict[str, Any]
        Dictionary of parameters that may contain arrays/lists
        
    Returns
    -------
    Dict[str, Any]
        Dictionary with unpacked parameters
    """
    # Initialize unpacked dictionary
    unpacked = {}
    
    # Iterate through parameters
    for param_name, param_value in params_dict.items():
        # Check if value is list or numpy array
        if isinstance(param_value, (list, np.ndarray)):
            # Unpack each element
            for i, val in enumerate(param_value):
                unpacked[f'{param_name}{i+1}'] = val
        else:
            # Keep scalar value as-is
            unpacked[param_name] = param_value
    
    # Return unpacked dictionary
    return unpacked


def load_param_recov_results(run_dir: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load parameter recovery results from a run directory.
    
    Loads run_dict.pkl and converts to a DataFrame plus extras dictionary.
    
    Parameters
    ----------
    run_dir : str
        Path to the run directory (e.g., /path/to/experiment/run_001)
        
    Returns
    -------
    results_df : pd.DataFrame
        DataFrame with columns: ['Parameter Name', 'True', 'Estimated', 'Delta']
        One row per parameter (with array parameters unpacked)
    extras_dict : Dict[str, Any]
        Dictionary containing data that couldn't fit in the DataFrame:
        - 'negll': negative log-likelihood values
        - 'run_id': run identifier
        - 'cfg': Configuration object
        - 'data': Generated dataset
    """
    # Load run_dict.pkl
    pkl_path = os.path.join(run_dir, 'run_dict.pkl')
    if not os.path.exists(pkl_path):
        raise FileNotFoundError(f"run_dict.pkl not found in {run_dir}")
    with open(pkl_path, 'rb') as f:
        run_dict = pickle.load(f)
    
    # Extract components from run_dict
    params_true = run_dict.get('P', {})
    params_estimated = run_dict.get('P_hat', {})
    delta = run_dict.get('Delta', {})
    
    # Unpack array parameters
    params_true_unpacked = unpack_array_params(params_true)
    params_estimated_unpacked = unpack_array_params(params_estimated)
    # delta is already unpacked in ParameterRecovery._compute_delta()
    
    # Create DataFrame rows
    rows = []
    for param_name in params_true_unpacked.keys():
        row = {
            'Parameter Name': param_name,
            'True': params_true_unpacked[param_name],
            'Estimated': params_estimated_unpacked.get(param_name, None),
            'Delta': delta.get(param_name, None)
        }
        rows.append(row)
    
    # Create DataFrame
    results_df = pd.DataFrame(rows)
    
    # Create extras dictionary
    extras_dict = {
        'negll': run_dict.get('negll', {}),
        'aic': run_dict.get('aic', {}),
        'run_id': run_dict.get('run_id', 'unknown'),
        'cfg': run_dict.get('cfg', None),
        'data': run_dict.get('D', None)
    }
    
    # Return DataFrame and extras
    return results_df, extras_dict


def load_param_recov_figs(run_dir: str, format: str = 'png') -> Dict[str, str]:
    """
    Load figure paths from a run directory.
    
    Parameters
    ----------
    run_dir : str
        Path to the run directory
    format : str
        Figure format to load ('png' or 'pdf')
        
    Returns
    -------
    Dict[str, str]
        Dictionary mapping figure names to file paths
        Keys: 'psychometric', 'confidence', 'link_function', 'confidence_dist'
    """
    # Validate format
    if format not in ['png', 'pdf']:
        raise ValueError(f"format must be 'png' or 'pdf', got {format}")
    
    # Determine directory based on format
    if format == 'png':
        fig_dir = run_dir
    else:  # pdf
        fig_dir = os.path.join(run_dir, 'figpdfs')
    
    # Define figure names
    figure_names = ['psychometric', 'confidence', 'link_function', 'confidence_dist']
    
    # Build dictionary of figure paths
    fig_paths = {}
    for fig_name in figure_names:
        fig_path = os.path.join(fig_dir, f'{fig_name}.{format}')
        if os.path.exists(fig_path):
            fig_paths[fig_name] = fig_path
        else:
            fig_paths[fig_name] = None  # Mark as missing
    
    # Return figure paths dictionary
    return fig_paths


def load_param_recov_history(run_dir: str) -> str:
    """
    Load fitting history text from the run's log file.
    
    Parameters
    ----------
    run_dir : str
        Path to the run directory
        
    Returns
    -------
    str
        Content of the <run_id>_out.txt file
    """
    # Find the log file
    # Log file should be named <run_id>_out.txt
    run_id = os.path.basename(run_dir).replace('run_', '')
    log_file = os.path.join(run_dir, f'{run_id}_out.txt')
    
    # Read and return log content
    if not os.path.exists(log_file):
        return f"Log file not found: {log_file}"
    with open(log_file, 'r') as f:
        return f.read()


def list_available_runs(primary_storage_dir: str, experiment_id: str) -> List[str]:
    """
    List all available run IDs in an experiment directory.
    
    Parameters
    ----------
    primary_storage_dir : str
        Base path to Experimentations/ParameterRecovery
    experiment_id : str
        The experiment ID
        
    Returns
    -------
    List[str]
        List of run IDs (e.g., ['run_001', 'run_002', 'run_003'])
    """
    # Build experiment path
    experiment_path = os.path.join(primary_storage_dir, experiment_id)
    
    # Check if experiment directory exists
    if not os.path.exists(experiment_path):
        return []
    
    # Find all run directories
    run_dirs = []
    for item in os.listdir(experiment_path):
        item_path = os.path.join(experiment_path, item)
        if os.path.isdir(item_path) and item.startswith('run_'):
            run_dirs.append(item)
    
    # Sort run directories
    run_dirs.sort()
    
    # Return list of run IDs
    return run_dirs


def check_run_status(run_dir: str) -> str:
    """
    Check the status of a parameter recovery run.
    
    Parameters
    ----------
    run_dir : str
        Path to the run directory
        
    Returns
    -------
    str
        Status: 'SUCCESS', 'FAILED', or 'IN_PROGRESS'
    """
    # Check for SUCCESS marker
    success_file = os.path.join(run_dir, 'SUCCESS')
    if os.path.exists(success_file):
        return 'SUCCESS'
    
    # Check for ERROR marker
    error_file = os.path.join(run_dir, 'ERROR')
    if os.path.exists(error_file):
        return 'FAILED'
    
    # Check if run_dict.pkl exists
    pkl_file = os.path.join(run_dir, 'run_dict.pkl')
    if os.path.exists(pkl_file):
        return 'SUCCESS'  # Completed but no marker
    
    # Otherwise, assume in progress or incomplete
    return 'IN_PROGRESS'


def format_negll_table(negll_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Format negative log-likelihood values as a DataFrame.
    
    Parameters
    ----------
    negll_dict : Dict[str, float]
        Dictionary containing negll values from the run
        
    Returns
    -------
    pd.DataFrame
        Formatted table with columns: ['Component', 'True', 'Fitted']
    """
    # Extract negll values
    # Expected keys: 
    # - 'negll_sens_true', 'negll_sens_fitted'
    # - 'negll_meta_true', 'negll_meta_fitted'
    
    # Create DataFrame rows
    rows = [
        {
            'Component': 'Sensory',
            'True': negll_dict.get('negll_sens_true', None),
            'Fitted': negll_dict.get('negll_sens_fitted', None)
        },
        {
            'Component': 'Metacognitive',
            'True': negll_dict.get('negll_meta_true', None),
            'Fitted': negll_dict.get('negll_meta_fitted', None)
        }
    ]
    
    # Create and return DataFrame
    return pd.DataFrame(rows)


def format_aic_table(aic_dict: Dict[str, float]) -> pd.DataFrame:
    """
    Format AIC values as a DataFrame for easy display.
    
    Parameters
    ----------
    aic_dict : Dict[str, float]
        Dictionary containing AIC values from the run
        Expected keys:
        - 'aic_sens_true', 'aic_sens_fitted'
        - 'aic_meta_true', 'aic_meta_fitted'
        - 'aic_total_true', 'aic_total_fitted'
        
    Returns
    -------
    pd.DataFrame
        Formatted table with columns: ['Component', 'True', 'Fitted', 'Delta']
        where Delta = Fitted - True (negative means fitted is better)
    """
    # Create DataFrame rows
    rows = []
    
    # Sensory component
    aic_sens_true = aic_dict.get('aic_sens_true', None)
    aic_sens_fitted = aic_dict.get('aic_sens_fitted', None)
    if aic_sens_true is not None and aic_sens_fitted is not None:
        rows.append({
            'Component': 'Sensory',
            'True': aic_sens_true,
            'Fitted': aic_sens_fitted,
            'Delta': aic_sens_fitted - aic_sens_true
        })
    
    # Metacognitive component
    aic_meta_true = aic_dict.get('aic_meta_true', None)
    aic_meta_fitted = aic_dict.get('aic_meta_fitted', None)
    if aic_meta_true is not None and aic_meta_fitted is not None:
        rows.append({
            'Component': 'Metacognitive',
            'True': aic_meta_true,
            'Fitted': aic_meta_fitted,
            'Delta': aic_meta_fitted - aic_meta_true
        })
    
    # Total
    aic_total_true = aic_dict.get('aic_total_true', None)
    aic_total_fitted = aic_dict.get('aic_total_fitted', None)
    if aic_total_true is not None and aic_total_fitted is not None:
        rows.append({
            'Component': 'Total',
            'True': aic_total_true,
            'Fitted': aic_total_fitted,
            'Delta': aic_total_fitted - aic_total_true
        })
    
    # Create and return DataFrame
    return pd.DataFrame(rows)
