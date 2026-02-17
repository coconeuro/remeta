import pickle
import numpy as np
import remeta
import os
import pathlib
import gzip

mode = 'default'
# mode = 'group'
# mode = 'type1_only'
# mode = 'type1_complex'
# mode = 'type2_multiplicative_bias'
# mode = 'noisy_readout'
# mode = 'noisy_temperature'


skip_type2 = False
if mode == 'default':
    nsubjects = 1
    nsamples = 2000
    seed = 1
    stim_levels = 4
    params = dict(
        type1_noise=0.5,
        type1_bias=-0.1,
        type2_noise=0.3,
        type2_criteria=[0.25, 0.5, 0.75]
    )
    cfg = remeta.Configuration()
    # cfg.param_type2_noise.model = 'truncated_normal_mode'
if mode == 'group':
    nsubjects = 3
    nsamples = 1000
    seed = 1
    stim_levels = 4
    params = dict(
        type1_noise=0.5,
        type1_bias=-0.1,
        type2_noise=0.3,
        type2_criteria=[0.25, 0.5, 0.75]
    )
    cfg = remeta.Configuration()
    cfg.param_type1_bias.group = 'random'
    # cfg.param_type2_noise.model = 'truncated_normal_mode'
elif mode == 'type1_only':
    nsubjects = 1
    nsamples = 2000
    seed = 1
    stim_levels = 4
    params = dict(
        type1_noise=0.7,
        type1_bias=0.2
    )
    cfg = remeta.Configuration()
    cfg.skip_type2 = True
elif mode == 'type1_complex':
    nsubjects = 1
    nsamples = 2000
    seed = 1
    stim_levels = 50
    params = dict(
        type1_noise=[0.5, 0.7],
        type1_thresh=0.1,
        type1_bias=[0.6, 0.1],
    )
    cfg = remeta.Configuration()
    cfg.param_type1_noise.enable = 2
    cfg.param_type1_thresh.enable = 1
    cfg.param_type1_bias.enable = 2
    cfg.skip_type2 = True
elif mode == 'type2_multiplicative_bias':
    nsubjects = 1
    nsamples = 2000
    seed = 7
    stim_levels = 4
    params = dict(
        type1_noise=0.6,
        type1_bias=0,
        type2_noise=0.2,
        type2_evidence_bias=0.8,
    )
    cfg = remeta.Configuration()
    # cfg.type2_fitting_type = 'continuous'
    cfg.param_type2_criteria.enable = 0
    cfg.param_type2_evidence_bias.enable = 1
    # cfg.param_type2_noise.model = 'truncated_normal_mode'
elif mode == 'noisy_readout':
    nsubjects = 1
    nsamples = 2000
    seed = 5
    stim_levels = 4
    params = dict(
        type1_noise=0.4,
        type1_bias=0,
        type2_noise=0.4,
        type2_criteria=[0.3, 0.7, 0.9]
    )
    cfg = remeta.Configuration()
    cfg.type2_noise_type = 'readout'
    # cfg.param_type2_noise.model = 'lognormal_mode'
elif mode == 'noisy_temperature':
    nsubjects = 1
    nsamples = 2000
    seed = 1
    stim_levels = 4
    params = dict(
        type1_noise=0.5,
        type1_bias=0,
        type2_noise=0.25,
        type2_criteria=[0.3, 0.7, 0.9]
    )
    cfg = remeta.Configuration()
    cfg.type2_noise_type = 'temperature'
    # cfg.param_type2_noise.model = 'lognormal_mode'


np.random.seed(seed)
data = remeta.simulate(nsubjects=nsubjects, nsamples=nsamples, params=params, cfg=cfg, custom_stimuli=None, verbosity=True,
                       stim_levels=stim_levels, squeeze=True, compute_stats=True)

path = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'remeta/demo_data', f'example_data_{mode}.pkl.gz')
# save = (data.x_stim, data.d_dec, data.c_conf, params, data.cfg, data.y_decval_mode, stats)
with gzip.open(path, "wb") as f:
    pickle.dump(data, f)
print(f'Saved to {path}')
