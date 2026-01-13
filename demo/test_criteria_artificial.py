from remeta_v1 import remeta
import numpy as np

np.random.seed(1)

cfg = remeta.Configuration()
# cfg.type2_noise_dist = 'censored_norm'
cfg.type2_noise_dist = 'truncated_norm'
cfg.type2_noise_type = 'noisy_report'
cfg.enable_type1_param_bias = 0
cfg.enable_type2_param_evidence_bias_mult = 0
cfg.enable_type2_param_evidence_bias_add = 0
# cfg.enable_type2_param_criteria = 1
cfg.experimental_discrete_type2_fitting = True
# cfg.experimental_disable_type2_binsize = True
# cfg.experimental_discret_censored_confidence_binsize = 0.1
cfg.gridsearch = False
cfg.n_discrete_confidence_levels = 5

params_true = dict(
    type1_noise=0.5,
    # type1_bias=0,
    type2_noise=0.2,
    type2_criteria=[0.2, 0.4, 0.6, 0.8]
    # type2_evidence_bias_mult=1,
    # type2_evidence_bias_add=0.1
)

data = remeta.simu_data(nsubjects=1, nsamples=5000, params=params_true, squeeze=True, x_stim_stepsize=0.25, cfg=cfg)

cfg.true_params = params_true
rem = remeta.ReMeta(cfg=cfg)
rem.fit(data.x_stim, data.d_dec, data.c_conf)
result = rem.summary()