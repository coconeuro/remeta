"""
ParameterRecovery class for validating ReMeta parameter estimation.

This module implements a parameter recovery pipeline that:
1. Generates synthetic data from known parameters
2. Fits ReMeta to recover the parameters
3. Computes similarity metrics between true and estimated parameters
4. Saves all results, plots, and logs
"""

import os
import sys
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

# Add parent directories to path to allow importing remeta
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import remeta


class ParameterRecovery:
    """
    Parameter recovery validation pipeline for ReMeta.
    
    This class packages the entire parameter recovery analysis from 
    demo/basic_usage.ipynb into a coherent, reusable pipeline.
    """
    
    def __init__(
        self,
        primary_storage_dir: str = os.getcwd(),
        run_id: str = '0001',
        remeta_param_dict: Optional[Dict[str, Any]] = {},
        data_param_dict: Dict[str, Any]={},
        experiment_id: str = 'EXP0001',
        shared_data: Optional[Any] = None,
        is_data_saver: bool = False
    ):
        """
        Initialize the ParameterRecovery pipeline.
        
        Parameters
        ----------
        primary_storage_dir : str
            Path to the experiment directory (e.g., /Experimentations/ParameterRecovery/<experiment_id>)
        run_id : str
            Unique identifier for this specific run
        remeta_param_dict : Optional[Dict[str, Any]]
            Dictionary of ReMeta parameters to use for generating synthetic data.
            If None, uses default Configuration() with default parameters (baseline).
            If empty {}, uses default Configuration() values.
            Missing parameter entries automatically fall back to the current Configuration defaults.
            Example: {
                'enable_noise_sens': 2,
                'enable_evidence_bias_add_meta': 1,
                'noise_sens': [0.5, 0.7],
                'bias_sens': -0.1,
                'noise_meta': 0.1,
                'evidence_bias_mult_meta': 1.3,
                'evidence_bias_add_meta': -0.1
            }
        data_param_dict : Dict[str, Any]
            Dictionary of dataset specifications.
            Example: {
                'nsubjects': 1,
                'nsamples': 1000,
                'stimuli_stepsize': 0.25
            }
        experiment_id : str
            Experiment identifier for organizing synthetic data storage
        shared_data : Optional[Any]
            Pre-generated shared dataset (Simulation object). If None, generates new data.
            Used for studying fitting variance on identical datasets.
        is_data_saver : bool
            If True, this run will save the dataset to disk. Only one run should be
            designated as the data saver to avoid race conditions in parallel execution.
        """
        # Store initialization parameters
        self.primary_storage_dir = primary_storage_dir
        self.run_id = run_id
        self.remeta_param_dict = remeta_param_dict
        self.data_param_dict = data_param_dict
        self.experiment_id = experiment_id
        self.shared_data = shared_data
        self.is_data_saver = is_data_saver
        
        # Create run directory path
        self.run_dir = os.path.join(primary_storage_dir, f"run_{run_id}")
        self.figpdfs_dir = os.path.join(self.run_dir, "figpdfs")
        
        # Create synthetic data directory path
        # Path structure: /Experimentations/SyntheticData/<experiment_id>/
        project_root = Path(primary_storage_dir).parent.parent
        self.synthetic_data_dir = os.path.join(project_root, "SyntheticData", experiment_id)
        
        # Initialize placeholders for results
        self.cfg = None  # Configuration object (will be created in _create_configuration)
        self.params_true = {}  # True parameters used for data generation
        self.data = None  # Generated synthetic data
        self.rem = None  # Fitted ReMeta instance
        self.result = None  # Fit result from rem.summary()
        self.params_estimated = {}  # Estimated parameters from fitting
        self.delta = {}  # Similarity/difference metrics between true and estimated params
        self.negll_dict = {}  # Dictionary storing negative log-likelihoods
        self.aic_dict = {}  # Dictionary storing AIC values for true and fitted params
        
        # Will setup logging later in run() after directory creation
    
    @staticmethod
    def get_default_parameter_values(cfg = remeta.Configuration()) -> Dict[str, Any]:
        """Return numeric defaults for all parameters enabled in the configuration."""
        original_print_flag = getattr(cfg, "print_configuration", True)
        cfg.print_configuration = False
        cfg.setup()
        cfg.print_configuration = original_print_flag

        defaults: Dict[str, Any] = {}

        def _collect(paramset):
            if paramset is None:
                return
            idx = 0
            for name, length in zip(paramset.base_names, paramset.base_len):
                if length == 0:
                    continue
                values = paramset.guess[idx: idx + length]
                idx += length
                if length == 1:
                    defaults[name] = float(values[0])
                else:
                    defaults[name] = [float(v) for v in np.atleast_1d(values).tolist()]

        _collect(cfg.paramset_sens)
        _collect(cfg.paramset_meta)

        return defaults

    @staticmethod
    def _log_parameter_values(params: Dict[str, Any], header: str) -> None:
        if not params:
            logging.info(f"{header}: <none>")
            return
        logging.info(header)
        for key in sorted(params):
            value = params[key]
            if isinstance(value, np.ndarray):
                value = value.tolist()
            logging.info(f"  {key} = {value}")

    def _setup_logging(self):
        """
        Setup logging and output redirection.
        
        Creates a log file <run_id>_out.txt and configures:
        - Stdout/stderr redirection to the log file
        - Console logging for progress updates
        - File logging for detailed output
        """
        # Create log file path
        self.log_file_path = os.path.join(self.run_dir, f"{self.run_id}_out.txt")
        
        # Open log file
        self.log_file = open(self.log_file_path, 'w', buffering=1)  # Line buffered
        
        # Store original stdout/stderr for restoration
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr
        
        # Create custom writer that writes to both file and console
        class TeeWriter:
            def __init__(self, file_handle, original_stream):
                self.file_handle = file_handle
                self.original_stream = original_stream
            
            def write(self, message):
                self.file_handle.write(message)
                self.original_stream.write(message)
            
            def flush(self):
                self.file_handle.flush()
                self.original_stream.flush()
        
        # Redirect stdout and stderr to both file and console
        sys.stdout = TeeWriter(self.log_file, self.original_stdout)
        sys.stderr = TeeWriter(self.log_file, self.original_stderr)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s: %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ],
            force=True  # Override any existing configuration
        )
    
    def _create_configuration(self) -> remeta.Configuration:
        """
        Create Configuration object from remeta_param_dict.
        
        If remeta_param_dict is None, returns default Configuration() with default parameters.
        If remeta_param_dict is empty {}, returns default Configuration() with empty parameters.
        Otherwise, creates Configuration() and overrides only the specified parameters.
        
        Returns
        -------
        remeta.Configuration
            Configured ReMeta Configuration object
        """
        # Create base Configuration
        cfg = remeta.Configuration()
        
        # Handle None case (baseline with defaults)
        if self.remeta_param_dict is None:
            logging.info("Using default configuration (baseline run with default parameters)")
            self.params_true = self.get_default_parameter_values(cfg)
            self._log_parameter_values(self.params_true, "True parameters for data generation (defaults)")
            return cfg
        
        # Separate configuration params from actual parameter values
        config_params = {}  # enable_* parameters
        param_values = {}  # actual parameter values
        
        for key, value in (self.remeta_param_dict or {}).items():
            if key.startswith('enable_'):
                config_params[key] = value
            else:
                param_values[key] = value
        
        # Set configuration parameters
        for key, value in config_params.items():
            if hasattr(cfg, key):
                setattr(cfg, key, value)
                logging.info(f"Configuration: {key} = {value}")
            else:
                logging.warning(f"Configuration parameter '{key}' not found in Configuration object")

        defaults = self.get_default_parameter_values(cfg)
        final_params = defaults.copy()

        # Handle empty dict case (defaults after potential enable_* overrides)
        if not self.remeta_param_dict:
            logging.info("Using default configuration (empty parameter dictionary)")
            self.params_true = final_params
            self._log_parameter_values(self.params_true, "True parameters for data generation (defaults)")
            return cfg
        
        if param_values:
            final_params.update(param_values)
            self._log_parameter_values(param_values, "Parameter overrides for data generation")

        # Store parameter values for data generation (defaults + overrides)
        self.params_true = final_params
        self._log_parameter_values(self.params_true, "True parameters for data generation")
        
        return cfg
    
    def _generate_synthetic_data(self) -> Any:
        """
        Generate synthetic data using remeta.simu_data() or use shared data.
        
        If shared_data is provided, uses it directly (shared data mode).
        Otherwise, generates new data (independent data mode).
        
        Saves dataset to /Experimentations/SyntheticData/<experiment_id>/:
        - shared_data.pkl for shared data mode
        - data_run_<run_id>.pkl for independent data mode
        
        Returns
        -------
        data
            Generated or provided synthetic dataset (Simulation object)
        """
        # Check if using shared data
        if self.shared_data is not None:
            logging.info("Using provided shared dataset")
            data = self.shared_data
            
            # Save shared data if this is the designated data saver
            if self.is_data_saver:
                self._save_dataset(data, shared=True)
            
            return data
        
        # Generate new data (independent mode)
        # Extract data generation parameters
        nsubjects = self.data_param_dict.get('nsubjects', 1)
        nsamples = self.data_param_dict.get('nsamples', 1000)
        stimuli_stepsize = self.data_param_dict.get('stimuli_stepsize', 0.25)
        
        # Log data generation start
        logging.info(f"Generating synthetic data with nsubjects={nsubjects}, nsamples={nsamples}, stimuli_stepsize={stimuli_stepsize}")
        
        # Generate data
        data = remeta.simu_data(
            nsubjects=nsubjects,
            nsamples=nsamples,
            params=self.params_true,
            squeeze=True,
            stimuli_stepsize=stimuli_stepsize,
            cfg=self.cfg
        )
        
        # Log data generation completion
        logging.info("Synthetic data generated successfully")
        
        # Save independently generated data
        self._save_dataset(data, shared=False)
        
        return data
    
    def _save_dataset(self, data: Any, shared: bool = False):
        """
        Save dataset to /Experimentations/SyntheticData/<experiment_id>/.
        
        Parameters
        ----------
        data : Any
            Simulation object to save
        shared : bool
            If True, saves as shared_data.pkl. Otherwise, saves as data_run_<run_id>.pkl
        """
        # Create synthetic data directory
        os.makedirs(self.synthetic_data_dir, exist_ok=True)
        
        # Determine filename
        if shared:
            filename = "shared_data.pkl"
            logging.info("Saving shared dataset")
        else:
            filename = f"data_run_{self.run_id}.pkl"
            logging.info(f"Saving independent dataset for run {self.run_id}")
        
        # Full path
        data_path = os.path.join(self.synthetic_data_dir, filename)
        
        # Save data as pickle
        with open(data_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Also save metadata (data_param_dict) for reproducibility
        metadata_filename = filename.replace('.pkl', '_metadata.pkl')
        metadata_path = os.path.join(self.synthetic_data_dir, metadata_filename)
        with open(metadata_path, 'wb') as f:
            pickle.dump(self.data_param_dict, f)
        
        logging.info(f"Dataset saved to: {data_path}")
        logging.info(f"Metadata saved to: {metadata_path}")
    
    def _fit_model(self):
        """
        Fit ReMeta model to the generated synthetic data.
        
        Creates ReMeta instance and fits it to self.data.
        Stores the result in self.result.
        """
        # Store true params in cfg for comparison
        self.cfg.true_params = self.params_true
        
        # Create ReMeta instance
        logging.info("Creating ReMeta instance")
        self.rem = remeta.ReMeta(cfg=self.cfg)
        
        # Fit the model
        logging.info("Fitting ReMeta model to synthetic data...")
        self.rem.fit(self.data.stimuli, self.data.choices, self.data.confidence)
        
        # Get fit results
        self.result = self.rem.summary()
        logging.info("Model fitting completed")
        
        # Extract estimated parameters (combine sensory and metacognitive params)
        self.params_estimated = {}
        self.params_estimated.update(self.result.model.params_sens)
        self.params_estimated.update(self.result.model.params_meta)
        
        # Log estimated parameters
        logging.info("Estimated parameters:")
        for key, value in self.params_estimated.items():
            logging.info(f"  {key} = {value}")
        
        # Extract negative log-likelihoods
        self.negll_dict = {
            'negll_sens_true': self.result.model.evidence_sens.get('negll_true', None),
            'negll_sens_fitted': self.result.model.evidence_sens.get('negll', None),
            'negll_meta_true': self.result.model.evidence_meta.get('negll_true', None),
            'negll_meta_fitted': self.result.model.evidence_meta.get('negll', None)
        }
        
        # Log negll comparison
        logging.info("Negative log-likelihood comparison:")
        logging.info(f"  Sensory - True: {self.negll_dict['negll_sens_true']:.2f}, Fitted: {self.negll_dict['negll_sens_fitted']:.2f}")
        logging.info(f"  Metacognitive - True: {self.negll_dict['negll_meta_true']:.2f}, Fitted: {self.negll_dict['negll_meta_fitted']:.2f}")
    
    def _compute_delta(self) -> Dict[str, Any]:
        """
        Compute similarity metrics between true and estimated parameters.
        
        For numeric parameters: delta = estimated - true (negative means underestimation)
        For boolean parameters: delta = (estimated == true)
        For array parameters: compute element-wise differences
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing delta values for each parameter
        """
        logging.info("Computing parameter differences (Delta = P_hat - P)")
        delta = {}
        
        # Iterate through all parameters in params_true
        for key in self.params_true.keys():
            true_val = self.params_true[key]
            est_val = self.params_estimated.get(key, None)
            
            if est_val is None:
                logging.warning(f"Parameter '{key}' not found in estimated parameters, skipping")
                continue
            
            # Handle array/list parameters
            if isinstance(true_val, (list, tuple, np.ndarray)):
                true_arr = np.array(true_val)
                est_arr = np.array(est_val)
                
                if true_arr.shape != est_arr.shape:
                    logging.warning(f"Shape mismatch for parameter '{key}': true={true_arr.shape}, estimated={est_arr.shape}")
                    continue
                
                # Compute element-wise differences
                delta_arr = est_arr - true_arr
                
                # Store individual elements with indexed keys (e.g., noise_sens1, noise_sens2)
                for i, delta_val in enumerate(delta_arr):
                    delta_key = f"{key}{i+1}"
                    delta[delta_key] = delta_val
                    logging.info(f"  {delta_key}: {est_arr[i]:.4f} - {true_arr[i]:.4f} = {delta_val:.4f}")
                
                # Also store the array itself
                delta[key] = delta_arr
            
            # Handle boolean parameters
            elif isinstance(true_val, bool):
                delta[key] = (est_val == true_val)
                logging.info(f"  {key}: Match = {delta[key]} (true={true_val}, estimated={est_val})")
            
            # Handle scalar numeric parameters (int, float)
            else:
                delta[key] = est_val - true_val
                logging.info(f"  {key}: {est_val:.4f} - {true_val:.4f} = {delta[key]:.4f}")
        
        logging.info("Parameter delta computation completed")
        
        return delta
    
    def _compute_aic(self) -> Dict[str, float]:
        """
        Compute AIC values for true and estimated parameters.
        
        AIC (Akaike Information Criterion) = 2*k + 2*NegLL
        where k is the number of parameters.
        
        A good sanity check: AIC_fitted should be <= AIC_true
        (fitted parameters should have equal or better fit than true parameters)
        
        Returns
        -------
        Dict[str, float]
            Dictionary containing AIC values:
            - aic_sens_true: AIC for true sensory parameters
            - aic_sens_fitted: AIC for fitted sensory parameters
            - aic_meta_true: AIC for true metacognitive parameters
            - aic_meta_fitted: AIC for fitted metacognitive parameters
            - aic_total_true: Total AIC for true parameters
            - aic_total_fitted: Total AIC for fitted parameters
        """
        logging.info("Computing AIC values for true and estimated parameters")
        
        aic_dict = {}
        
        # Get number of parameters
        nparams_sens = self.cfg.paramset_sens.nparams if self.cfg.paramset_sens else 0
        nparams_meta = self.cfg.paramset_meta.nparams if self.cfg.paramset_meta else 0
        
        # Sensory AIC
        if self.negll_dict.get('negll_sens_true') is not None:
            aic_dict['aic_sens_true'] = 2 * nparams_sens + 2 * self.negll_dict['negll_sens_true']
            logging.info(f"  AIC (sensory, true): {aic_dict['aic_sens_true']:.2f}")
        
        if self.negll_dict.get('negll_sens_fitted') is not None:
            aic_dict['aic_sens_fitted'] = 2 * nparams_sens + 2 * self.negll_dict['negll_sens_fitted']
            logging.info(f"  AIC (sensory, fitted): {aic_dict['aic_sens_fitted']:.2f}")
        
        # Metacognitive AIC
        if self.negll_dict.get('negll_meta_true') is not None:
            aic_dict['aic_meta_true'] = 2 * nparams_meta + 2 * self.negll_dict['negll_meta_true']
            logging.info(f"  AIC (metacognitive, true): {aic_dict['aic_meta_true']:.2f}")
        
        if self.negll_dict.get('negll_meta_fitted') is not None:
            aic_dict['aic_meta_fitted'] = 2 * nparams_meta + 2 * self.negll_dict['negll_meta_fitted']
            logging.info(f"  AIC (metacognitive, fitted): {aic_dict['aic_meta_fitted']:.2f}")
        
        # Total AIC (combined sensory + metacognitive)
        if 'aic_sens_true' in aic_dict and 'aic_meta_true' in aic_dict:
            aic_dict['aic_total_true'] = aic_dict['aic_sens_true'] + aic_dict['aic_meta_true']
            logging.info(f"  AIC (total, true): {aic_dict['aic_total_true']:.2f}")
        
        if 'aic_sens_fitted' in aic_dict and 'aic_meta_fitted' in aic_dict:
            aic_dict['aic_total_fitted'] = aic_dict['aic_sens_fitted'] + aic_dict['aic_meta_fitted']
            logging.info(f"  AIC (total, fitted): {aic_dict['aic_total_fitted']:.2f}")
        
        # Sanity check
        if 'aic_total_true' in aic_dict and 'aic_total_fitted' in aic_dict:
            if aic_dict['aic_total_fitted'] <= aic_dict['aic_total_true']:
                logging.info("  ✓ Sanity check PASSED: AIC_fitted <= AIC_true")
            else:
                logging.warning("  ⚠ Sanity check FAILED: AIC_fitted > AIC_true")
                logging.warning(f"    Difference: {aic_dict['aic_total_fitted'] - aic_dict['aic_total_true']:.2f}")
        
        logging.info("AIC computation completed")
        return aic_dict
    
    def _save_figures(self):
        """
        Generate and save all plots from the parameter recovery analysis.
        
        Saves both PNG (in run_dir) and PDF (in run_dir/figpdfs) versions.
        
        Plots generated:
        - psychometric.png/pdf: Psychometric curve
        - confidence.png/pdf: Confidence vs stimulus intensity
        - link_function.png/pdf: Confidence link function
        - confidence_dist.png/pdf: Confidence distribution
        """
        # Create figpdfs directory if it doesn't exist
        os.makedirs(self.figpdfs_dir, exist_ok=True)
        logging.info(f"Saving figures to {self.run_dir} and {self.figpdfs_dir}")
        
        # Generate and save psychometric plot
        logging.info("Generating psychometric plot")
        remeta.plot_psychometric_sim(self.data)
        plt.savefig(os.path.join(self.run_dir, 'psychometric.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.figpdfs_dir, 'psychometric.pdf'), bbox_inches='tight')
        plt.close()
        
        # Generate and save confidence plot
        logging.info("Generating confidence plot")
        remeta.plot_confidence_sim(self.data)
        plt.savefig(os.path.join(self.run_dir, 'confidence.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.figpdfs_dir, 'confidence.pdf'), bbox_inches='tight')
        plt.close()
        
        # Generate and save link function plot
        logging.info("Generating link function plot")
        self.rem.plot_link_function()
        plt.savefig(os.path.join(self.run_dir, 'link_function.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.figpdfs_dir, 'link_function.pdf'), bbox_inches='tight')
        plt.close()
        
        # Generate and save confidence distribution plot
        logging.info("Generating confidence distribution plot")
        plt.figure(figsize=(10, 5))
        self.rem.plot_confidence_dist()
        plt.savefig(os.path.join(self.run_dir, 'confidence_dist.png'), dpi=150, bbox_inches='tight')
        plt.savefig(os.path.join(self.figpdfs_dir, 'confidence_dist.pdf'), bbox_inches='tight')
        plt.close()
        
        logging.info("All figures saved successfully")
    
    def _save_results(self):
        """
        Save run results as run_dict.pkl.
        
        Saves a dictionary containing:
        - P: True parameters (params_true)
        - D: Generated dataset (data)
        - P_hat: Estimated parameters (params_estimated)
        - Delta: Similarity metrics (delta)
        - negll: Negative log-likelihood values (negll_dict)
        - aic: AIC values (aic_dict)
        - cfg: Configuration object
        """
        # Construct run_dict with all results
        run_dict = {
            'P': self.params_true,
            'D': self.data,
            'P_hat': self.params_estimated,
            'Delta': self.delta,
            'negll': self.negll_dict,
            'aic': self.aic_dict,
            'cfg': self.cfg,
            'run_id': self.run_id
        }
        
        # Save as pickle file
        pickle_path = os.path.join(self.run_dir, 'run_dict.pkl')
        logging.info(f"Saving results to {pickle_path}")
        with open(pickle_path, 'wb') as f:
            pickle.dump(run_dict, f)
        
        logging.info("Results saved successfully")
    
    def _create_success_marker(self):
        """
        Create a SUCCESS marker file to indicate successful completion.
        
        Creates an empty file named 'SUCCESS' in the run directory.
        """
        success_file = os.path.join(self.run_dir, 'SUCCESS')
        Path(success_file).touch()
        logging.info("Created SUCCESS marker file")
    
    def run(self) -> Dict[str, Any]:
        """
        Execute the complete parameter recovery pipeline.
        
        Pipeline steps:
        1. Setup logging
        2. Create configuration
        3. Generate synthetic data
        4. Fit model
        5. Compute delta
        6. Save figures
        7. Save results
        8. Create success marker
        
        Returns
        -------
        Dict[str, Any]
            Dictionary containing run results (same as run_dict.pkl)
        """
        # Log pipeline start
        logging.info(f"=" * 80)
        logging.info(f"Starting Parameter Recovery Pipeline for Run ID: {self.run_id}")
        logging.info(f"=" * 80)
        
        try:
            # Step 1 - Create directories
            os.makedirs(self.run_dir, exist_ok=True)
            os.makedirs(self.figpdfs_dir, exist_ok=True)
            logging.info(f"Created run directory: {self.run_dir}")
            
            # Step 2 - Setup logging (must be after directory creation)
            self._setup_logging()
            
            # Step 3 - Create configuration
            logging.info("Step 1/6: Creating configuration")
            self.cfg = self._create_configuration()
            
            # Step 4 - Generate synthetic data
            logging.info("Step 2/6: Generating synthetic data")
            self.data = self._generate_synthetic_data()
            
            # Step 5 - Fit model
            logging.info("Step 3/6: Fitting model")
            self._fit_model()
            
            # Step 6 - Compute delta
            logging.info("Step 4/7: Computing parameter differences")
            self.delta = self._compute_delta()
            
            # Step 7 - Compute AIC
            logging.info("Step 5/7: Computing AIC comparison")
            self.aic_dict = self._compute_aic()
            
            # Step 8 - Save figures
            logging.info("Step 6/7: Generating and saving figures")
            self._save_figures()
            
            # Step 9 - Save results
            logging.info("Step 7/7: Saving results")
            self._save_results()
            
            # Step 9 - Create success marker
            self._create_success_marker()
            
            # Log pipeline completion
            logging.info(f"=" * 80)
            logging.info(f"Parameter Recovery Pipeline completed successfully for Run ID: {self.run_id}")
            logging.info(f"=" * 80)
            
            # Return results
            return {
                'P': self.params_true,
                'D': self.data,
                'P_hat': self.params_estimated,
                'Delta': self.delta,
                'negll': self.negll_dict,
                'aic': self.aic_dict,
                'run_id': self.run_id,
                'status': 'SUCCESS'
            }
            
        except Exception as e:
            # Handle errors
            logging.error(f"Error in Parameter Recovery Pipeline: {str(e)}")
            logging.exception("Full traceback:")
            
            # Create error marker file
            error_file = os.path.join(self.run_dir, 'ERROR')
            with open(error_file, 'w') as f:
                f.write(str(e))
            
            # Re-raise exception
            raise
