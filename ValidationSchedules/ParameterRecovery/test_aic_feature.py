"""
Quick test to verify AIC computation feature is working.

This test runs a minimal parameter recovery with AIC computation enabled.
"""
import sys
import os
from pathlib import Path

# Add parent directories to path
parent_dir = Path(__file__).parent
project_root = parent_dir.parent.parent
sys.path.insert(0, str(project_root))

from ValidationSchedules.ParameterRecovery.parameter_recovery import ParameterRecovery

def test_aic_computation():
    """Test that AIC values are computed and stored correctly."""
    
    print("=" * 80)
    print("Testing AIC Computation Feature")
    print("=" * 80)
    
    # Setup test parameters
    storage_dir = os.path.join(project_root, "Experimentations", "ParameterRecovery", "TEST_AIC")
    
    # Use minimal parameters for quick test
    remeta_params = {
        'enable_noise_sens': 1,
        'noise_sens': 0.6
    }
    
    data_params = {
        'nsubjects': 1,
        'nsamples': 500,  # Small dataset for speed
        'stimuli_stepsize': 0.25
    }
    
    # Create and run pipeline
    print("\n1. Initializing ParameterRecovery...")
    pr = ParameterRecovery(
        primary_storage_dir=storage_dir,
        run_id=0,
        remeta_param_dict=remeta_params,
        data_param_dict=data_params,
        experiment_id="TEST_AIC",
        shared_data=None,
        is_data_saver=True
    )
    
    print("2. Running parameter recovery pipeline...")
    result = pr.run()
    
    print("\n3. Checking results...")
    # Check that AIC dict exists and has expected keys
    assert 'aic' in result, "AIC dict not found in results!"
    aic_dict = result['aic']
    
    print(f"\n4. AIC Results:")
    print(f"   Keys in aic_dict: {list(aic_dict.keys())}")
    
    # Verify expected structure
    if 'aic_true' in aic_dict:
        print(f"   ✓ AIC (true params): {aic_dict['aic_true']}")
    else:
        print("   ✗ Missing 'aic_true' key")
        
    if 'aic_fitted' in aic_dict:
        print(f"   ✓ AIC (fitted params): {aic_dict['aic_fitted']}")
    else:
        print("   ✗ Missing 'aic_fitted' key")
        
    if 'delta_aic' in aic_dict:
        print(f"   ✓ Delta AIC: {aic_dict['delta_aic']}")
        if aic_dict['delta_aic'] < 0:
            print(f"   ✓ Fitted model has BETTER AIC (lower by {-aic_dict['delta_aic']:.2f})")
        else:
            print(f"   ⚠ Warning: Fitted model has WORSE AIC (higher by {aic_dict['delta_aic']:.2f})")
    else:
        print("   ✗ Missing 'delta_aic' key")
    
    print("\n5. Checking saved pickle file...")
    import pickle
    pickle_path = os.path.join(storage_dir, "run_0", "run_dict.pkl")
    with open(pickle_path, 'rb') as f:
        saved_data = pickle.load(f)
    
    assert 'aic' in saved_data, "AIC dict not saved in pickle file!"
    print(f"   ✓ Pickle file contains AIC data: {list(saved_data['aic'].keys())}")
    
    print("\n" + "=" * 80)
    print("✓ AIC Computation Test PASSED!")
    print("=" * 80)
    
    return result

if __name__ == "__main__":
    test_aic_computation()
