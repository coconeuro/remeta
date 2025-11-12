"""
Parameter Recovery Interface Script

This script provides a front-end interface for running multiple parameter recovery
pipelines in parallel using the ParameterRecovery class.

Supports two modes:
1. Independent Data Mode: Each run generates its own dataset
2. Shared Data Mode: All runs use the same dataset (to study fitting variance)
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any, Optional
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import logging
import pickle

# Add parent directory to path to import ParameterRecovery and remeta
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from ValidationSchedules.ParameterRecovery.parameter_recovery import ParameterRecovery
import remeta


def create_shared_data_experiment(
    experiment_id: str,
    primary_storage_dir: str,
    remeta_param_sets: Dict[str, Optional[Dict[str, Any]]],
    data_param_dict: Dict[str, Any],
    data_generation_run_id: Optional[str] = None
) -> List[ParameterRecovery]:
    """
    Create ParameterRecovery instances for a SHARED DATA experiment.
    
    All runs will use the same dataset. This mode is useful for studying:
    - Fitting variance on identical data
    - Sensitivity to different model parameters on the same data
    - Convergence behavior with different initializations
    
    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier
    primary_storage_dir : str
        Path to experiment directory
    remeta_param_sets : Dict[str, Optional[Dict[str, Any]]]
        Dictionary mapping run_id -> remeta_param_dict
        Use None to run with the default ReMeta parameter configuration
        Each run can have different model parameters
    data_param_dict : Dict[str, Any]
        Single dataset specification used for all runs
        Example: {'nsubjects': 1, 'nsamples': 1000, 'stimuli_stepsize': 0.25}
    data_generation_run_id : Optional[str]
        Run ID to use for generating shared data. If None, uses first run's parameters.
        
    Returns
    -------
    List[ParameterRecovery]
        List of ParameterRecovery instances configured for shared data mode
    """
    logging.info(f"Creating SHARED DATA experiment: {experiment_id}")
    logging.info(f"Number of runs: {len(remeta_param_sets)}")
    logging.info(f"Data parameters: {data_param_dict}")
    
    # Determine which run's parameters to use for data generation
    if data_generation_run_id is None:
        data_generation_run_id = list(remeta_param_sets.keys())[0]
        logging.info(f"Using first run '{data_generation_run_id}' parameters for data generation")
    
    # Generate shared dataset
    logging.info("Generating shared dataset...")
    data_gen_params = remeta_param_sets[data_generation_run_id]
    
    # Create configuration for data generation
    cfg = remeta.Configuration()
    param_overrides: Dict[str, Any] = {}

    if data_gen_params is not None:
        for key, value in data_gen_params.items():
            if key.startswith('enable_'):
                setattr(cfg, key, value)
            else:
                param_overrides[key] = value

    params_true = ParameterRecovery.get_default_parameter_values(cfg)
    if param_overrides:
        params_true.update(param_overrides)

    logging.info("Parameters used for shared data generation:")
    for key, value in sorted(params_true.items()):
        logging.info(f"  {key} = {value}")

    # Generate shared data using defaults merged with overrides
    shared_data = remeta.simu_data(
        nsubjects=data_param_dict.get('nsubjects', 1),
        nsamples=data_param_dict.get('nsamples', 1000),
        params=params_true,
        squeeze=True,
        stimuli_stepsize=data_param_dict.get('stimuli_stepsize', 0.25),
        cfg=cfg
    )
    logging.info("Shared dataset generated successfully")
    
    # Create ParameterRecovery instances with shared data
    parameter_recovery_runs = []
    for idx, (run_id, remeta_params) in enumerate(remeta_param_sets.items()):
        # First run is designated as data saver
        is_data_saver = (idx == 0)
        
        pr_instance = ParameterRecovery(
            primary_storage_dir=primary_storage_dir,
            run_id=run_id,
            remeta_param_dict=remeta_params,
            data_param_dict=data_param_dict,
            experiment_id=experiment_id,
            shared_data=shared_data,
            is_data_saver=is_data_saver
        )
        parameter_recovery_runs.append(pr_instance)
    
    logging.info(f"Created {len(parameter_recovery_runs)} ParameterRecovery instances (SHARED DATA mode)")
    return parameter_recovery_runs


def create_independent_data_experiment(
    experiment_id: str,
    primary_storage_dir: str,
    remeta_param_sets: Dict[str, Optional[Dict[str, Any]]],
    data_param_sets: Dict[str, Dict[str, Any]]
) -> List[ParameterRecovery]:
    """
    Create ParameterRecovery instances for an INDEPENDENT DATA experiment.
    
    Each run generates its own dataset. This mode is useful for studying:
    - Data variance from the generative model
    - Robustness across different random datasets
    - Parameter recovery under different data conditions
    
    Parameters
    ----------
    experiment_id : str
        Unique experiment identifier
    primary_storage_dir : str
        Path to experiment directory
    remeta_param_sets : Dict[str, Optional[Dict[str, Any]]]
        Dictionary mapping run_id -> remeta_param_dict
        Use None for baseline runs with default ReMeta parameters
    data_param_sets : Dict[str, Dict[str, Any]]
        Dictionary mapping run_id -> data_param_dict
        Each run can have different data specifications
        
    Returns
    -------
    List[ParameterRecovery]
        List of ParameterRecovery instances configured for independent data mode
    """
    logging.info(f"Creating INDEPENDENT DATA experiment: {experiment_id}")
    logging.info(f"Number of runs: {len(remeta_param_sets)}")
    
    # Validate matching run IDs
    if set(remeta_param_sets.keys()) != set(data_param_sets.keys()):
        raise ValueError("remeta_param_sets and data_param_sets must have the same run IDs")
    
    # Create ParameterRecovery instances without shared data
    parameter_recovery_runs = []
    for run_id in remeta_param_sets.keys():
        pr_instance = ParameterRecovery(
            primary_storage_dir=primary_storage_dir,
            run_id=run_id,
            remeta_param_dict=remeta_param_sets[run_id],
            data_param_dict=data_param_sets[run_id],
            experiment_id=experiment_id,
            shared_data=None,  # No shared data
            is_data_saver=False  # Each run saves its own data
        )
        parameter_recovery_runs.append(pr_instance)
    
    logging.info(f"Created {len(parameter_recovery_runs)} ParameterRecovery instances (INDEPENDENT DATA mode)")
    return parameter_recovery_runs


def _run_single_job(pr_instance: ParameterRecovery) -> Dict[str, Any]:
    """
    Wrapper function to run a single ParameterRecovery instance.
    
    This function is defined at module level (not nested) so it can be pickled
    by multiprocessing.Pool.
    
    Parameters
    ----------
    pr_instance : ParameterRecovery
        The ParameterRecovery instance to run
        
    Returns
    -------
    Dict[str, Any]
        Result dictionary with status and run_id
    """
    try:
        result = pr_instance.run()
        return result
    except Exception as e:
        logging.error(f"Job {pr_instance.run_id} failed: {str(e)}")
        return {'run_id': pr_instance.run_id, 'status': 'FAILED', 'error': str(e)}


def parallel_execution(
    jobs: List[ParameterRecovery],
    n_workers: Optional[int] = None
) -> List[Dict[str, Any]]:
    """
    Execute multiple ParameterRecovery pipelines in parallel.
    
    Parameters
    ----------
    jobs : List[ParameterRecovery]
        List of ParameterRecovery instances to execute
    n_workers : Optional[int]
        Number of worker processes. Defaults to total CPU cores available.
        
    Returns
    -------
    List[Dict[str, Any]]
        List of results from each parameter recovery run
    """
    # Set default n_workers
    if n_workers is None:
        n_workers = cpu_count()
    logging.info(f"Using {n_workers} worker processes")
    
    # Execute jobs in parallel with progress bar
    logging.info(f"Starting parallel execution of {len(jobs)} jobs")
    results = []
    with Pool(processes=n_workers) as pool:
        # Use tqdm for progress bar
        # Use _run_single_job which is defined at module level for pickling
        for result in tqdm(pool.imap(_run_single_job, jobs), total=len(jobs), desc="Processing runs"):
            results.append(result)
    
    # Log completion summary
    successful = sum(1 for r in results if r.get('status') == 'SUCCESS')
    failed = len(results) - successful
    logging.info(f"Parallel execution completed: {successful} successful, {failed} failed")
    
    # Return results
    return results


def main():
    """
    Main execution function for parameter recovery experiments.
    
    This function demonstrates both modes:
    1. INDEPENDENT DATA mode: Each run generates its own dataset
    2. SHARED DATA mode: All runs use the same dataset
    
    Set USE_SHARED_DATA flag to switch between modes.
    """
    # Setup logging for the main script
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # ==============================================================
    # USER CONFIGURATION
    # ==============================================================
    
    # Define experiment_id
    experiment_id = "EXP0000.num_runs50"  # User should modify this
    
    # MODE SWITCH: Set to True for shared data, False for independent data
    USE_SHARED_DATA = False
    num_runs = 50
    
    # ==============================================================
    # EXPERIMENT SETUP
    # ==============================================================
    
    logging.info(f"Starting experiment: {experiment_id}")
    logging.info(f"Mode: {'SHARED DATA' if USE_SHARED_DATA else 'INDEPENDENT DATA'}")
    
    # Create primary_storage_dir
    project_root = Path(__file__).parent.parent.parent
    experimentations_dir = project_root / "Experimentations" / "ParameterRecovery"
    primary_storage_dir = experimentations_dir / experiment_id
    primary_storage_dir.mkdir(parents=True, exist_ok=True)
    logging.info(f"Experiment directory: {primary_storage_dir}")
    
    # Define remeta_param_sets dictionary
    # Dictionary mapping run IDs to ReMeta parameter dictionaries (or None for defaults)
    #
    # Provide None for runs that should use the default ReMeta parameters (baseline).
    # Dictionaries can still override enable_* flags and/or specific parameter values.
    #remeta_param_sets = { # run_id: params dict
        #'001': {
        #    'enable_noise_sens': 2,
        #    'enable_evidence_bias_add_meta': 1,
        #    'noise_sens': [0.5, 0.7],
        #    'bias_sens': -0.1,
        #    'noise_meta': 0.1,
        #    'evidence_bias_mult_meta': 1.3,
        #   'evidence_bias_add_meta': -0.1
        #},
     #   '000': None,  # Baseline with default configuration
     #   '001': None,
     #   '002': None,
     #   '003': None,
     #   '004': None,
    #}
    remeta_param_sets = {str(i):None for i in range(num_runs)}
    
    # ==============================================================
    # CREATE PARAMETER RECOVERY INSTANCES
    # ==============================================================
    
    if USE_SHARED_DATA:
        # SHARED DATA MODE
        # All runs use the same dataset
        # Define a single data_param_dict for shared data generation
        shared_data_params = {
            'nsubjects': 1,
            'nsamples': 1000,
            'stimuli_stepsize': 0.25
        }
        
        parameter_recovery_runs_list = create_shared_data_experiment(
            experiment_id=experiment_id,
            primary_storage_dir=str(primary_storage_dir),
            remeta_param_sets=remeta_param_sets,
            data_param_dict=shared_data_params,
            data_generation_run_id='run_001'  # Use run_001's params for data generation
        )
        
    else:
        # INDEPENDENT DATA MODE
        # Each run generates its own dataset
        # Define data_param_sets dictionary (one per run)
        basic_data_cfg = {
                'nsubjects': 1,
                'nsamples': int(10e3),
                'stimuli_stepsize': 0.25
            }
        #data_param_sets = { # run_id : data cfg dict
        #    '000': basic_data_cfg,
        #    '001': basic_data_cfg,
        #    '002': basic_data_cfg,
        #    '003': basic_data_cfg,
        #    '004': {basic_data_cfg,
        #}
        data_param_sets = {str(i):basic_data_cfg for i in range(num_runs)}
        
        parameter_recovery_runs_list = create_independent_data_experiment(
            experiment_id=experiment_id,
            primary_storage_dir=str(primary_storage_dir),
            remeta_param_sets=remeta_param_sets,
            data_param_sets=data_param_sets
        )
    
    # ==============================================================
    # EXECUTE PARAMETER RECOVERY
    # ==============================================================
    
    logging.info("Starting parallel execution")
    results = parallel_execution(parameter_recovery_runs_list, n_workers=4)
    
    # ==============================================================
    # SAVE EXPERIMENT SUMMARY
    # ==============================================================
    
    summary_file = primary_storage_dir / "experiment_summary.txt"
    with open(summary_file, 'w') as f:
        f.write(f"Experiment ID: {experiment_id}\n")
        f.write(f"Mode: {'SHARED DATA' if USE_SHARED_DATA else 'INDEPENDENT DATA'}\n")
        f.write(f"Run configs:\n\n{remeta_param_sets}\n")
        f.write(f"Data configs:\n\n{data_param_sets}\n")
        f.write(f"Total runs: {len(results)}\n\n")
        for result in results:
            status = result.get('status', 'UNKNOWN')
            run_id = result.get('run_id', 'UNKNOWN')
            f.write(f"Run {run_id}: {status}\n")
            if status == 'FAILED':
                f.write(f"  Error: {result.get('error', 'Unknown error')}\n")
    
    logging.info(f"Experiment {experiment_id} completed")
    logging.info(f"Results saved in: {primary_storage_dir}")


if __name__ == "__main__":
    main()
