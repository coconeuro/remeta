import pickle
import numpy as np
import remeta
import os
import pathlib
import gzip

mode = 'default'
# mode = 'type1_only'
# mode = 'type1_complex'
# mode = 'type2_multiplicative_bias'
# mode = 'noisy_readout'
# mode = 'noisy_temperature'


bounds = np.arange(0, 0.81, 0.2)

skip_type2 = False
if mode == 'default':
    nsamples = 2000
    seed = 1
    x_stim_stepsize = 0.25
    params = dict(
        type1_noise=0.5,
        type1_bias=-0.1,
        type2_noise=0.3,
        type2_criteria=[0.2, 0.2, 0.2, 0.2]
    )
    cfg = remeta.Configuration()
    cfg.type2_noise_dist = 'truncated_norm_mode'
elif mode == 'type1_only':
    nsamples = 2000
    seed = 1
    x_stim_stepsize = 0.25
    params = dict(
        type1_noise=0.7,
        type1_bias=0.2
    )
    cfg = remeta.Configuration()
    cfg.skip_type2 = True
elif mode == 'type1_complex':
    nsamples = 2000
    seed = 1
    x_stim_stepsize = 0.02
    params = dict(
        type1_noise=[0.5, 0.7],
        type1_thresh=0.1,
        type1_bias=[0.6, 0.1],
    )
    cfg = remeta.Configuration()
    cfg.enable_type1_param_noise = 2
    cfg.enable_type1_param_thresh = 1
    cfg.enable_type1_param_bias = 2
    cfg.skip_type2 = True
elif mode == 'type2_multiplicative_bias':
    nsamples = 2000
    seed = 7
    x_stim_stepsize = 0.25
    params = dict(
        type1_noise=0.6,
        type1_bias=0,
        type2_noise=0.2,
        type2_evidence_bias_mult=0.8,
    )
    cfg = remeta.Configuration()
    # cfg.type2_fitting_type = 'continuous'
    cfg.enable_type2_param_criteria = 0
    cfg.enable_type2_param_evidence_bias_mult = 1
    cfg.type2_noise_dist = 'truncated_norm_mode'
elif mode == 'noisy_readout':
    nsamples = 2000
    seed = 5
    x_stim_stepsize = 0.25
    params = dict(
        type1_noise=0.4,
        type1_bias=0,
        type2_noise=0.4,
        type2_criteria=[0.3, 0.4, 0.1, 0.1]
    )
    cfg = remeta.Configuration()
    cfg.type2_noise_type = 'noisy_readout'
    cfg.type2_noise_dist = 'lognorm_mode'
elif mode == 'noisy_temperature':
    nsamples = 2000
    seed = 1
    x_stim_stepsize = 0.25
    params = dict(
        type1_noise=0.5,
        type1_bias=0,
        type2_noise=0.25,
        type2_criteria=[0.3, 0.4, 0.1, 0.1]
    )
    cfg = remeta.Configuration()
    cfg.type2_noise_type = 'noisy_temperature'
    cfg.type2_noise_dist = 'lognorm_mode'


np.random.seed(seed)
data = remeta.simu_data(nsubjects=1, nsamples=nsamples, params=params, cfg=cfg, x_stim_external=None, verbosity=True,
                        x_stim_stepsize=x_stim_stepsize, squeeze=True, compute_stats=True)

path = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'remeta/demo_data', f'example_data_{mode}.pkl.gz')
# save = (data.x_stim, data.d_dec, data.c_conf, params, data.cfg, data.y_decval_mode, stats)
with gzip.open(path, "wb") as f:
    pickle.dump(data, f)
print(f'Saved to {path}')
