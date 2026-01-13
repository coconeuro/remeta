from remeta_v1 import remeta
import numpy as np
import warnings
warnings.filterwarnings('error')

np.random.seed(2)

cfg = remeta.Configuration()
# cfg.type2_noise_dist = 'censored_norm'
# cfg.type2_noise_dist = 'beta'
# cfg.type2_noise_dist = 'truncated_norm'
# cfg.type2_noise_type = 'noisy_report'
cfg.type2_fitting_type = 'criteria'
# cfg.type2_fitting_type = 'continuous'
# cfg.type2_noise_type = 'noisy_temperature'
# cfg.type2_noise_type = 'noisy_report'
cfg.type2_noise_type = 'noisy_readout'
cfg.type2_noise_dist = 'gamma_mode_std'
# cfg.type2_noise_dist = 'beta'
# cfg.type2_noise_dist = 'truncated_norm'
# cfg.enable_type1_param_bias = 1
# cfg.enable_type2_param_evidence_bias_mult = 0
# cfg.enable_type2_param_evidence_bias_add = 0
cfg.experimental_disable_type2_binsize = False
cfg.enable_type2_param_criteria = 1
# cfg.experimental_disable_type2_binsize = True
# cfg.experimental_discret_censored_confidence_binsize = 0.1
# cfg.n_discrete_confidence_levels = 5
# cfg.gradient_method = 'slsqp'
# cfg.gridsearch = True
# cfg.slsqp_epsilon = 1e-2
# cfg.initilialize_fitting_at_true_params = True
# cfg.y_decval_range_nbins = 1001
# cfg.experimental_include_incongruent_y_decval = True
# cfg.slsqp_epsilon = np.exp(-np.arange(0, 25, 3))

# cfg.type2_param_noise.grid_range = np.arange(0.01, 0.2, 0.01)
# cfg.type2_param_noise.bounds = (1e-6, 1)
# cfg.type2_param_noise.guess = 1e-7
params_true = dict(
    type1_noise=0.5,
    type1_bias=0,
    type2_noise=0.25,
    type2_criteria=[0.3, 0.4, 0.1, 0.1]
)

# params_true = dict(
#     type1_noise=0.2,
#     type1_bias=0,
#     type2_noise=0.4,
#     type2_criteria=[0.3, 0.4, 0.1, 0.1]
# )

# params_true = {'type1_noise': 0.5, 'type2_noise': np.float64(0.40210581740035417),
#  'type2_evidence_bias_add': np.float64(-0.3440299923679495),
#  'type2_criteria': [np.float64(0.06149078415957187),
#   np.float64(0.14872069632906548),
#   np.float64(0.20868551461574664),
#   np.float64(0.2641113899493955)]}

cfg.true_params = params_true

data = remeta.simu_data(nsubjects=1, nsamples=2000, params=params_true, squeeze=True, x_stim_stepsize=0.25,
                        cfg=cfg)


rem = remeta.ReMeta(cfg=cfg)
rem.fit(data.x_stim, data.d_dec, data.c_conf)
result = rem.summary()
# import matplotlib.pyplot as plt
# plt.figure()
# plt.hist(data.c_conf)