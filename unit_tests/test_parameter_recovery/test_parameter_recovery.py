"""
Unit tests for the ParameterRecovery class.
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ValidationSchedules.ParameterRecovery.parameter_recovery import ParameterRecovery


class TestParameterRecovery(unittest.TestCase):
    """Test cases for the ParameterRecovery class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create temporary directory for test outputs
        self.test_dir = tempfile.mkdtemp()
        self.primary_storage_dir = os.path.join(self.test_dir, "test_experiment")
        os.makedirs(self.primary_storage_dir, exist_ok=True)
        
        # Define test parameters
        self.test_experiment_id = "TEST_EXP_001"
        self.test_run_id = "test_run_001"
        self.test_remeta_params = {
            'enable_noise_sens': 2,
            'noise_sens': [0.5, 0.7],
            'bias_sens': -0.1,
            'noise_meta': 0.1,
            'evidence_bias_mult_meta': 1.3
        }
        self.test_data_params = {
            'nsubjects': 1,
            'nsamples': 100,  # Small for fast testing
            'stimuli_stepsize': 0.25
        }
    
    def tearDown(self):
        """Clean up test fixtures after each test method."""
        # Remove temporary directory
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_initialization(self):
        """Test ParameterRecovery initialization."""
        # Test initialization with valid parameters
        pr = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id=self.test_run_id,
            remeta_param_dict=self.test_remeta_params,
            data_param_dict=self.test_data_params,
            experiment_id=self.test_experiment_id
        )
        self.assertEqual(pr.run_id, self.test_run_id)
        self.assertEqual(pr.experiment_id, self.test_experiment_id)
        self.assertIsNone(pr.shared_data)
        self.assertFalse(pr.is_data_saver)
        self.assertIsNotNone(pr.run_dir)
    
    def test_initialization_with_empty_params(self):
        """Test initialization with empty parameter dictionary (baseline case)."""
        # Test with empty remeta_param_dict
        pr = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id="test_baseline",
            remeta_param_dict={},  # Empty = use defaults
            data_param_dict=self.test_data_params,
            experiment_id=self.test_experiment_id
        )
        # Should not raise an error
        self.assertEqual(pr.run_id, "test_baseline")
    
    def test_initialization_with_shared_data(self):
        """Test initialization with shared data mode."""
        # Mock shared data
        mock_shared_data = "mock_simulation_object"
        
        pr = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id=self.test_run_id,
            remeta_param_dict=self.test_remeta_params,
            data_param_dict=self.test_data_params,
            experiment_id=self.test_experiment_id,
            shared_data=mock_shared_data,
            is_data_saver=True
        )
        
        self.assertEqual(pr.shared_data, mock_shared_data)
        self.assertTrue(pr.is_data_saver)
    
    def test_create_configuration(self):
        """Test Configuration creation with parameters."""
        # Basic smoke test - just ensure it doesn't crash
        pr = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id=self.test_run_id,
            remeta_param_dict=self.test_remeta_params,
            data_param_dict=self.test_data_params,
            experiment_id=self.test_experiment_id
        )
        cfg = pr._create_configuration()
        # Check that configuration has expected values
        self.assertEqual(cfg.enable_noise_sens, 2)
    
    def test_create_configuration_default(self):
        """Test Configuration creation with empty params (defaults)."""
        pr = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id=self.test_run_id,
            remeta_param_dict={},
            data_param_dict=self.test_data_params,
            experiment_id=self.test_experiment_id
        )
        cfg = pr._create_configuration()
        # Should return a valid Configuration with defaults
        self.assertIsNotNone(cfg)
    
    def test_compute_delta_scalars(self):
        """Test delta computation for scalar parameters."""
        pr = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id=self.test_run_id,
            remeta_param_dict=self.test_remeta_params,
            data_param_dict=self.test_data_params,
            experiment_id=self.test_experiment_id
        )
        # Mock params_true and params_estimated
        pr.params_true = {'noise_meta': 0.1, 'bias_sens': -0.1}
        pr.params_estimated = {'noise_meta': 0.12, 'bias_sens': -0.08}
        delta = pr._compute_delta()
        # Check delta values
        self.assertAlmostEqual(delta['noise_meta'], 0.02, places=5)
        self.assertAlmostEqual(delta['bias_sens'], 0.02, places=5)
    
    def test_compute_delta_arrays(self):
        """Test delta computation for array parameters."""
        pr = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id=self.test_run_id,
            remeta_param_dict=self.test_remeta_params,
            data_param_dict=self.test_data_params,
            experiment_id=self.test_experiment_id
        )
        # Mock params with arrays
        pr.params_true = {'noise_sens': [0.5, 0.7]}
        pr.params_estimated = {'noise_sens': [0.52, 0.68]}
        delta = pr._compute_delta()
        # Check unpacked delta values
        self.assertAlmostEqual(delta['noise_sens1'], 0.02, places=5)
        self.assertAlmostEqual(delta['noise_sens2'], -0.02, places=5)
    
    def test_run_directory_creation(self):
        """Test that run directory path is correctly set."""
        pr = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id=self.test_run_id,
            remeta_param_dict=self.test_remeta_params,
            data_param_dict=self.test_data_params,
            experiment_id=self.test_experiment_id
        )
        expected_run_dir = os.path.join(self.primary_storage_dir, f"run_{self.test_run_id}")
        self.assertEqual(pr.run_dir, expected_run_dir)


if __name__ == '__main__':
    unittest.main()
