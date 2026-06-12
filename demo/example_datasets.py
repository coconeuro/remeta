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
    n_subjects = 1
    n_samples = 2000
    n_ratings = 4
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
    n_subjects = 3
    n_samples = 1000
    n_ratings = 4
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
    n_subjects = 1
    n_samples = 2000
    n_ratings = None
    seed = 1
    stim_levels = 4
    params = dict(
        type1_noise=0.7,
        type1_bias=0.2
    )
    cfg = remeta.Configuration()
    cfg.skip_type2 = True
elif mode == 'type1_complex':
    n_subjects = 1
    n_samples = 2000
    n_ratings = None
    seed = 1
    stim_levels = 50
    params = dict(
        type1_noise=[0.5, 0.7],
        type1_thresh=0.1,
        type1_bias=[0.6, 0.1],
    )
    cfg = remeta.Configuration()
    cfg.param_type1_noise.asym = True
    cfg.param_type1_thresh.enable = True
    cfg.param_type1_bias.asym = True
    cfg.skip_type2 = True
elif mode == 'type2_multiplicative_bias':
    n_subjects = 1
    n_samples = 2000
    n_ratings = None
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
    cfg.param_type2_criteria.enable = False
    cfg.param_type2_evidence_bias.enable = True
    # cfg.param_type2_noise.model = 'truncated_normal_mode'
elif mode == 'noisy_readout':
    n_subjects = 1
    n_samples = 2000
    n_ratings = 4
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
    n_subjects = 1
    n_samples = 2000
    n_ratings = 4
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
data = remeta.simulate(n_subjects=n_subjects, n_samples=n_samples, n_ratings=n_ratings, params=params, cfg=cfg, custom_stimuli=None, verbosity=True,
                       stim_levels=stim_levels, squeeze=True, compute_stats=True)

path = os.path.join(pathlib.Path(__file__).parent.resolve(), '..', 'remeta/demo_data', f'example_data_{mode}.pkl.gz')
# save = (data.x_stim, data.d_dec, data.c_conf, params, data.cfg, data.y_decval_mode, stats)
with gzip.open(path, "wb") as f:
    pickle.dump(data, f)
print(f'Saved to {path}')
