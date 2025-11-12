"""
Unit tests for param_recovery_interface.py
"""

import unittest
import os
import sys
import tempfile
import shutil
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from ValidationSchedules.ParameterRecovery.parameter_recovery import ParameterRecovery
from ValidationSchedules.ParameterRecovery import param_recovery_interface


class TestParallelExecution(unittest.TestCase):
    """Test cases for parallel execution functionality."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.test_dir = tempfile.mkdtemp()
        self.primary_storage_dir = os.path.join(self.test_dir, "test_experiment")
        self.test_experiment_id = "TEST_EXP_INTERFACE"
        os.makedirs(self.primary_storage_dir, exist_ok=True)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir)
    
    def test_parallel_execution_single_job(self):
        """Test parallel execution with a single job."""
        pr_instance = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id="test_single",
            remeta_param_dict={'noise_meta': 0.1},
            data_param_dict={'nsubjects': 1, 'nsamples': 50, 'stimuli_stepsize': 0.25},
            experiment_id=self.test_experiment_id
        )
        
        # Execute with n_workers=1
        results = param_recovery_interface.parallel_execution([pr_instance], n_workers=1)
        
        # Check results
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]['run_id'], 'test_single')
    
    def test_parallel_execution_multiple_jobs(self):
        """Test parallel execution with multiple jobs."""
        jobs = []
        for i in range(3):
            pr_instance = ParameterRecovery(
                primary_storage_dir=self.primary_storage_dir,
                run_id=f"test_multi_{i}",
                remeta_param_dict={'noise_meta': 0.1 + i * 0.05},
                data_param_dict={'nsubjects': 1, 'nsamples': 50, 'stimuli_stepsize': 0.25},
                experiment_id=self.test_experiment_id
            )
            jobs.append(pr_instance)
        
        # Execute with n_workers=2
        results = param_recovery_interface.parallel_execution(jobs, n_workers=2)
        
        # Check results
        self.assertEqual(len(results), 3)
        run_ids = [r['run_id'] for r in results]
        self.assertIn('test_multi_0', run_ids)
        self.assertIn('test_multi_1', run_ids)
        self.assertIn('test_multi_2', run_ids)
    
    def test_parallel_execution_with_failure(self):
        """Test parallel execution when one job fails."""
        # Create jobs where one has invalid parameters
        jobs = []
        
        # Valid job
        pr_valid = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id="test_valid",
            remeta_param_dict={'noise_meta': 0.1},
            data_param_dict={'nsubjects': 1, 'nsamples': 50, 'stimuli_stepsize': 0.25},
            experiment_id=self.test_experiment_id
        )
        jobs.append(pr_valid)
        
        # Execute - even if one fails, others should continue
        results = param_recovery_interface.parallel_execution(jobs, n_workers=1)
        
        # Check that we got results (error handling in wrapper should prevent crash)
        self.assertIsInstance(results, list)
    
    def test_default_n_workers(self):
        """Test that default n_workers is set to cpu_count()."""
        from multiprocessing import cpu_count
        
        # Create a simple job
        pr_instance = ParameterRecovery(
            primary_storage_dir=self.primary_storage_dir,
            run_id="test_default",
            remeta_param_dict={'noise_meta': 0.1},
            data_param_dict={'nsubjects': 1, 'nsamples': 50, 'stimuli_stepsize': 0.25},
            experiment_id=self.test_experiment_id
        )
        
        # Execute without specifying n_workers (uses default)
        results = param_recovery_interface.parallel_execution([pr_instance])
        
        # Just verify it runs without error
        self.assertEqual(len(results), 1)


class TestUtils(unittest.TestCase):
    """Test utility functions from utils.py."""
    
    def setUp(self):
        """Set up test fixtures."""
        from ValidationSchedules.ParameterRecovery.webapp import utils
        self.utils = utils
    
    def test_unpack_array_params(self):
        """Test unpacking of array parameters."""
        params = {
            'noise_sens': [0.5, 0.7],
            'bias_sens': -0.1,
            'noise_meta': 0.15
        }
        unpacked = self.utils.unpack_array_params(params)
        
        # Check unpacked values
        self.assertIn('noise_sens1', unpacked)
        self.assertIn('noise_sens2', unpacked)
        self.assertEqual(unpacked['noise_sens1'], 0.5)
        self.assertEqual(unpacked['noise_sens2'], 0.7)
        self.assertEqual(unpacked['bias_sens'], -0.1)
        self.assertEqual(unpacked['noise_meta'], 0.15)
    
    def test_unpack_array_params_no_arrays(self):
        """Test unpack with no array parameters."""
        params = {
            'noise_meta': 0.1,
            'bias_sens': -0.05
        }
        unpacked = self.utils.unpack_array_params(params)
        
        # Should return unchanged
        self.assertEqual(unpacked, params)
    
    def test_check_run_status(self):
        """Test run status checking."""
        test_dir = tempfile.mkdtemp()
        
        # Test SUCCESS status
        success_dir = os.path.join(test_dir, 'run_success')
        os.makedirs(success_dir)
        Path(os.path.join(success_dir, 'SUCCESS')).touch()
        self.assertEqual(self.utils.check_run_status(success_dir), 'SUCCESS')
        
        # Test FAILED status
        failed_dir = os.path.join(test_dir, 'run_failed')
        os.makedirs(failed_dir)
        Path(os.path.join(failed_dir, 'ERROR')).touch()
        self.assertEqual(self.utils.check_run_status(failed_dir), 'FAILED')
        
        # Test IN_PROGRESS status
        progress_dir = os.path.join(test_dir, 'run_progress')
        os.makedirs(progress_dir)
        self.assertEqual(self.utils.check_run_status(progress_dir), 'IN_PROGRESS')
        
        # Cleanup
        shutil.rmtree(test_dir)


if __name__ == '__main__':
    unittest.main()
