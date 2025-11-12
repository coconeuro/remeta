"""
Verify AIC computation by testing the _compute_aic method on an existing run.
"""
import sys
import os
from pathlib import Path
import pickle

# Add parent directories to path
parent_dir = Path(__file__).parent
project_root = parent_dir.parent.parent
sys.path.insert(0, str(project_root))

def test_aic_method():
    """Test _compute_aic method with mocked result object."""
    
    print("=" * 80)
    print("Testing AIC Computation Method")
    print("=" * 80)
    
    # Create a mock configuration
    class MockParamSet:
        def __init__(self, nparams):
            self.nparams = nparams
    
    class MockConfig:
        def __init__(self):
            self.paramset_sens = MockParamSet(nparams=2)  # 2 sensitivity params
            self.paramset_meta = MockParamSet(nparams=1)  # 1 metacognition param
    
    # Create a mock result object similar to what ReMeta returns
    class MockModel:
        def __init__(self):
            self.evidence_sens = {'aic': 1500.5}
            self.evidence_meta = {'aic': 1520.3}
    
    class MockResult:
        def __init__(self):
            self.model = MockModel()
    
    # Create minimal ParameterRecovery instance
    from ValidationSchedules.ParameterRecovery.parameter_recovery import ParameterRecovery
    
    pr = ParameterRecovery(
        primary_storage_dir="/tmp/test",
        run_id=0,
        remeta_param_dict={},
        data_param_dict={},
        experiment_id="TEST",
        shared_data=None,
        is_data_saver=False
    )
    
    # Set mock objects
    pr.cfg = MockConfig()
    pr.result = MockResult()
    
    # Set mock negll_dict with component values
    pr.negll_dict = {
        'negll_true': 750.2,
        'negll_fitted': 745.1,
        'negll_sens_true': 400.0,
        'negll_sens_fitted': 395.0,
        'negll_meta_true': 350.2,
        'negll_meta_fitted': 350.1
    }
    
    print("\n1. Mock Data:")
    print(f"   Config: {pr.cfg.paramset_sens.nparams} sens params, {pr.cfg.paramset_meta.nparams} meta params")
    print(f"   negll_sens_true: {pr.negll_dict['negll_sens_true']}")
    print(f"   negll_sens_fitted: {pr.negll_dict['negll_sens_fitted']}")
    print(f"   negll_meta_true: {pr.negll_dict['negll_meta_true']}")
    print(f"   negll_meta_fitted: {pr.negll_dict['negll_meta_fitted']}")
    
    print("\n2. Computing AIC...")
    aic_dict = pr._compute_aic()
    
    print("\n3. AIC Results:")
    print(f"   Keys: {list(aic_dict.keys())}")
    for key, value in aic_dict.items():
        print(f"   {key}: {value:.2f}")
    
    # Compute expected values
    k_sens = 2
    k_meta = 1
    
    expected_aic_sens_true = 2 * k_sens + 2 * pr.negll_dict['negll_sens_true']
    expected_aic_sens_fitted = 2 * k_sens + 2 * pr.negll_dict['negll_sens_fitted']
    expected_aic_meta_true = 2 * k_meta + 2 * pr.negll_dict['negll_meta_true']
    expected_aic_meta_fitted = 2 * k_meta + 2 * pr.negll_dict['negll_meta_fitted']
    expected_aic_total_true = expected_aic_sens_true + expected_aic_meta_true
    expected_aic_total_fitted = expected_aic_sens_fitted + expected_aic_meta_fitted
    
    print("\n4. Expected Values:")
    print(f"   AIC_sens_true = 2*{k_sens} + 2*{pr.negll_dict['negll_sens_true']} = {expected_aic_sens_true:.2f}")
    print(f"   AIC_sens_fitted = 2*{k_sens} + 2*{pr.negll_dict['negll_sens_fitted']} = {expected_aic_sens_fitted:.2f}")
    print(f"   AIC_meta_true = 2*{k_meta} + 2*{pr.negll_dict['negll_meta_true']} = {expected_aic_meta_true:.2f}")
    print(f"   AIC_meta_fitted = 2*{k_meta} + 2*{pr.negll_dict['negll_meta_fitted']} = {expected_aic_meta_fitted:.2f}")
    print(f"   AIC_total_true = {expected_aic_total_true:.2f}")
    print(f"   AIC_total_fitted = {expected_aic_total_fitted:.2f}")
    
    print("\n5. Validation:")
    checks = [
        ('aic_sens_true' in aic_dict, "Has aic_sens_true key"),
        ('aic_sens_fitted' in aic_dict, "Has aic_sens_fitted key"),  
        ('aic_meta_true' in aic_dict, "Has aic_meta_true key"),
        ('aic_meta_fitted' in aic_dict, "Has aic_meta_fitted key"),
        ('aic_total_true' in aic_dict, "Has aic_total_true key"),
        ('aic_total_fitted' in aic_dict, "Has aic_total_fitted key"),
        (abs(aic_dict.get('aic_sens_true', 0) - expected_aic_sens_true) < 0.01, "AIC_sens_true correct"),
        (abs(aic_dict.get('aic_sens_fitted', 0) - expected_aic_sens_fitted) < 0.01, "AIC_sens_fitted correct"),
        (abs(aic_dict.get('aic_meta_true', 0) - expected_aic_meta_true) < 0.01, "AIC_meta_true correct"),
        (abs(aic_dict.get('aic_meta_fitted', 0) - expected_aic_meta_fitted) < 0.01, "AIC_meta_fitted correct"),
        (abs(aic_dict.get('aic_total_true', 0) - expected_aic_total_true) < 0.01, "AIC_total_true correct"),
        (abs(aic_dict.get('aic_total_fitted', 0) - expected_aic_total_fitted) < 0.01, "AIC_total_fitted correct"),
        (aic_dict.get('aic_total_fitted', float('inf')) <= aic_dict.get('aic_total_true', 0), "Sanity check: fitted <= true")
    ]
    
    all_passed = True
    for check, desc in checks:
        status = "✓" if check else "✗"
        print(f"   {status} {desc}")
        all_passed = all_passed and check
    
    print("\n" + "=" * 80)
    if all_passed:
        print("✓ All AIC Method Tests PASSED!")
    else:
        print("✗ Some tests FAILED!")
    print("=" * 80)
    
    return all_passed

if __name__ == "__main__":
    success = test_aic_method()
    sys.exit(0 if success else 1)
