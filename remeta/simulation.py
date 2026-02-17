from __future__ import annotations

import numpy as np
from scipy.stats import norm, logistic as logistic_dist

from remeta.configuration import Configuration
from remeta.modelspec import Parameter
from remeta.transform import compute_signal_dependent_type1_noise, type1_evidence_to_confidence, \
    compute_nonlinear_encoding
from remeta.type2_SDT import type2roc, type2_SDT_MLE
from remeta.type2_dist import get_type2_dist
from remeta.util import _check_param, discretize_confidence_with_bounds, print_dataset_characteristics, listlike


def simulate(
    params: dict = None,
    nsubjects: int = 1,
    nsamples: int = 1000,
    cfg: Configuration = None,
    stim_min: float | None = None,
    stim_max: float | None = 1,
    stim_levels: int = 10,
    custom_stimuli: list[float] | np.ndarray = None,
    squeeze: bool = False,
    compute_stats: bool = True,
    silence_warnings: bool = False,
    verbosity: int = 1,
    **kwargs
) -> Simulation:
    """
    Simulate data for ReMeta

    Usage:
        `sim = simulate(params=dict(type1_noise=0.5, type2_noise=0.3))`

        `sim = simulate(params=dict(type1_noise=0.5, type2_noise=0.3), nsamples=500)`

        `sim = simulate(params=dict(type1_noise=0.5, type2_noise=0.3), cfg=cfg)`


    Args:
        params: Parameter values (dictionary with {param_name: param_value} entries)
        nsubjects: Number of subjects
        nsamples: Number of samples per subject
        cfg: `remeta.configuration.Configuration` instance (for model specification)
        stim_min: minimum stimulus intensity
        stim_max: maximum stimulus intensity
        stim_levels: number of different stimulus intensity levels
        custom_stimuli: (optional) pass a custom array or list of signed stimulus intensities
        squeeze: if `True` and only 1 subject is simulated, remove the subject dimension
        compute_stats: if `True`, compute some descriptive statistics on the simulated dataset
        silence_warnings: if `True`, silences a few (custom) warnings
        verbosity: verbosity level (possible values: 0, 1, 2)
        **kwargs: extra arguments will be passed to the configuration

    Returns: a `remeta.simulation.Simulation` instance

    """

    params = params.copy()  # this variable can be modifed, better make a copy
    if cfg is None:
        # Set configuration attributes that match keyword arguments
        cfg_dict = Configuration.__dict__
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in cfg_dict}
        cfg = Configuration(**cfg_kwargs)
        for k, v in params.items():
            if f'param_{k}' in cfg_dict:
                setattr(getattr(cfg, f'param_{k}'), 'enable', len(v) if listlike(v) else 1)
            else:
                raise ValueError(f'Unknown parameter {k}')
        for k, v in cfg_dict.items():
            if isinstance(v, Parameter):
                if k.split('param_')[1] not in params:
                    setattr(getattr(cfg, k), 'enable', 0)

        # for setting in cfg.__dict__:
        #     if setting.startswith('param_'):
        #         if setting.split('param_')[1] not in params:
        #             setattr(getattr(cfg, setting), 'enable', 0)
    # if not cfg.setup_called:
    cfg.setup(generative_mode=True, silence_warnings=silence_warnings)

    if cfg.param_type2_noise.model is None:
        cfg.param_type2_noise.model = dict(report='truncated_norm_mode', readout='truncated_norm_mode', temperature='lognorm_mode')[cfg.type2_noise_type]

    # Make sure no unwanted parameters have been passed
    for p in ('thresh', 'bias', 'noise_heteroscedastic', 'nonlinear_gain', 'nonlinear_scale'):
        if not getattr(cfg, f'param_type1_{p}').enable:
            params.pop(f'type1_{p}', None)
    for p in ('evidence_bias', 'confidence_bias', 'criteria'):
        if not getattr(cfg, f'param_type2_{p}').enable:
            params.pop(f'type2_{p}', None)

    if custom_stimuli is None:
        x_stim = generate_stimuli(nsubjects, nsamples, stim_min=stim_min, stim_max=stim_max, stim_levels=stim_levels)
    else:
        custom_stimuli = np.array(custom_stimuli)
        if custom_stimuli.ndim == 1:
            custom_stimuli = custom_stimuli.reshape(1, -1)
        nsamples = custom_stimuli.shape[1]
        x_stim = custom_stimuli / np.max(np.abs(custom_stimuli))
        if (x_stim.shape[0] == 1) and (nsubjects > 1):
            x_stim = np.tile(x_stim, (nsubjects, 1))
    x_stim_category = (np.sign(x_stim) > 0).astype(int)
    y_decval_latent, y_decval, d_dec = stimulus_to_decision_value(x_stim, params, cfg)

    if not cfg.skip_type2:

        z1_type1_evidence_latent = np.abs(y_decval_latent)
        z1_type1_evidence_base = np.abs(y_decval)

        if cfg.type2_noise_type == 'readout':
            dist = get_type2_dist(cfg.param_type2_noise.model, type2_center=z1_type1_evidence_base, type2_noise=params['type2_noise'],
                                  type2_noise_type=cfg.type2_noise_type)

            z1_type1_evidence = np.maximum(0, dist.rvs((nsubjects, nsamples)))
        elif cfg.type2_noise_type == 'temperature':
            dist = get_type2_dist(cfg.param_type2_noise.model, type2_center=params['type1_noise'] * np.ones_like(z1_type1_evidence_base),
                                  type2_noise=params['type2_noise'], type2_noise_type=cfg.type2_noise_type)
            type1_noise_estimated = dist.rvs()
            z1_type1_evidence = z1_type1_evidence_base
        elif cfg.type2_noise_type == 'report':
            z1_type1_evidence = z1_type1_evidence_base
        else:
            raise ValueError('Unknown type 2 noise type')

        c_conf_latent = type1_evidence_to_confidence(
            z1_type1_evidence=z1_type1_evidence_latent, y_decval=y_decval,
            x_stim=x_stim,
            type1_noise_signal_dependency=cfg.param_type1_noise_heteroscedastic.model if cfg.param_type1_noise_heteroscedastic.enable else None,
            **({**params, **dict(type1_noise=type1_noise_estimated)} if cfg.type2_noise_type == 'temperature' else params)
        )

        c_conf_base = type1_evidence_to_confidence(
            z1_type1_evidence=z1_type1_evidence, y_decval=y_decval,
            x_stim=x_stim,
            type1_noise_signal_dependency=cfg.param_type1_noise_heteroscedastic.model if cfg.param_type1_noise_heteroscedastic.enable else None,
            **({**params, **dict(type1_noise=type1_noise_estimated)} if cfg.type2_noise_type == 'temperature' else params)
        )

        if cfg.type2_noise_type == 'report':
            dist = get_type2_dist(cfg.param_type2_noise.model, type2_center=c_conf_base, type2_noise=params['type2_noise'],
                                  type2_noise_type=cfg.type2_noise_type)
            c_conf = np.maximum(0, np.minimum(1, dist.rvs((nsubjects, nsamples))))
        else:
            c_conf = c_conf_base

        if cfg.param_type2_criteria.enable or cfg.param_type2_criteria.preset is not None:
            if cfg.param_type2_criteria.enable:
                if not np.all(np.diff(params['type2_criteria']) > 0) or (min(params['type2_criteria']) < 0) or (max(params['type2_criteria']) > 1):
                    raise ValueError('Type 2 criteria must be provided in an ascending manner and be between 0 and 1.\n'
                                     f'Provided criteria: {np.array2string(np.array(params['type2_criteria']), precision=3)}')

                # convert to criterion gaps
                # old_criteria = np.array(params['type2_criteria']).copy()
                # params['type2_criteria'] = np.diff(np.hstack((0, params['type2_criteria'])))
                # criteria_gaps = params['type2_criteria']
                # criteria = [v if i == 0 else np.sum(criteria_gaps[:i+1]) for i, v in enumerate(criteria_gaps)]
                criteria = params['type2_criteria']
            elif cfg.param_type2_criteria.preset is not None:
                criteria = np.arange(1 / cfg._n_conf_levels, 1-1e-10,  1 / cfg._n_conf_levels)

            c_conf = (np.digitize(c_conf, criteria) + 0.5) / cfg._n_conf_levels
            c_conf_base = (np.digitize(c_conf_base, criteria) + 0.5) / cfg._n_conf_levels

    if squeeze:
        x_stim_category = x_stim_category.squeeze()
        x_stim = x_stim.squeeze()
        d_dec = d_dec.squeeze()
        y_decval_latent = y_decval_latent.squeeze()
        y_decval = y_decval.squeeze()
        if not cfg.skip_type2:
            z1_type1_evidence_base = z1_type1_evidence_base.squeeze()  # noqa
            z1_type1_evidence = z1_type1_evidence.squeeze()  # noqa
            c_conf_base = c_conf_base.squeeze()  # noqa
            c_conf = c_conf.squeeze()  # noqa

    simargs = dict(
        nsubjects=nsubjects, nsamples=nsamples, params=params, cfg=cfg,
        x_stim_category=x_stim_category, x_stim=x_stim, d_dec=d_dec,
        y_decval=y_decval, y_decval_latent=y_decval_latent
    )
    if not cfg.skip_type2:
        simargs.update(
            z1_type1_evidence_latent=z1_type1_evidence_latent, z1_type1_evidence_base=z1_type1_evidence_base, z1_type1_evidence=z1_type1_evidence,
            c_conf_latent=c_conf_latent, c_conf_base=c_conf_base, c_conf=c_conf
        )

    if compute_stats:
        accuracy = (x_stim_category == d_dec).astype(int)
        type1_stats = dict(
            accuracy=np.mean(accuracy),
            dprime = norm.ppf(min(1 - 1e-3, max(1e-3, d_dec[x_stim_category == 1].mean()))) - \
                 norm.ppf(min(1 - 1e-3, max(1e-3, d_dec[x_stim_category == 0].mean().mean()))),
            choice_bias=d_dec.mean() - x_stim_category.mean(),
        )
        simargs.update(type1_stats=type1_stats)
        if not cfg.skip_type2:
            bounds = np.arange(0, 0.81, 0.2)
            fit = type2_SDT_MLE(x_stim_category.flatten(), d_dec.flatten(), discretize_confidence_with_bounds(c_conf.flatten(), bounds), len(bounds))
            type2_stats = dict(
                confidence=c_conf.mean(),
                auroc2=type2roc(accuracy.flatten(), c_conf.flatten()),
                mratio=fit.M_ratio
            )
            simargs.update(type2_stats=type2_stats)
            if 'type2_criteria' in params:
                params_extra = dict(
                    # type2_criteria_absolute=[np.sum(params['type2_criteria'][:i + 1]) for i in range(len(params['type2_criteria']))],
                    # type2_criteria_bias=np.mean(params['type2_criteria']) * (len(params['type2_criteria']) + 1) - 1
                    type2_criteria_bias=np.mean(params['type2_criteria']) - 0.5,
                    type2_criteria_confidence_bias=0.5 - np.mean(params['type2_criteria']),
                    # type2_criteria_bias_mult=np.mean(params['type2_criteria']) / 0.5,
                    # type2_criteria_confidence_bias_mult=0.5 / np.mean(params['type2_criteria']),
                    # type2_criteria_absdev=round(np.abs(np.array(params['type2_criteria']) -
                    #    np.arange(1/cfg._n_conf_levels, 1-1e-10, 1/cfg._n_conf_levels)).mean(), 10)
                )
                simargs.update(params_extra=params_extra)


    simulation = Simulation(**simargs)
    if verbosity:
        print_dataset_characteristics(simulation)

    return simulation



class Simulation:
    """Class to store simulated data.
    This class is created by the `remeta.simulation.simulate()`. Manual creation is discouraged.

    Args:
        nsubjects: Number of subjects
        nsamples: Number of samples per subject
        params: Dictionary of model parameters
        params_extra: Optional dictionary of extra parameters
        cfg: `remeta.configuration.Configuration` instance
        x_stim: stimuli (array or list of signed stimulus intensities) (x)
        x_stim_category: stimulus category
        d_dec: choices (D)
        y_decval_latent: latent, i.e. noise-free decision values
        y_decval: observed decision values (hat{y})
        z1_type1_evidence_latent: latent, i.e. noise-free type 1 evidence (z_1)
        z1_type1_evidence_base: absolute values of observed decision values (|hat{y}|)
        z1_type1_evidence: observed type 1 evidence (z2=hat{z_1})
        c_conf_latent: probability correct based on z1_type1_evidence_latent
        c_conf_base: probability correct based on z1_type1_evidence_base
        c_conf: probability correct based on z1_type1_evidence
        type1_stats: descriptive type 1 statistics
        type2_stats: descriptive type 2 statistics
    """

    def __init__(self,
        nsubjects: int = None,
        nsamples: int = None,
        params: dict = None,
        params_extra: dict | None = None,
        cfg: Configuration = None,
        x_stim: list[float] | np.ndarray = None,
        x_stim_category: list[int] | np.ndarray = None,
        d_dec: list[int] | np.ndarray = None,
        y_decval_latent: list[int] | np.ndarray = None,
        y_decval: list[int] | np.ndarray = None,
        z1_type1_evidence_latent: list[int] | np.ndarray = None,
        z1_type1_evidence_base: list[int] | np.ndarray = None,
        z1_type1_evidence: list[int] | np.ndarray = None,
        c_conf_latent: list[int] | np.ndarray = None,
        c_conf_base: list[int] | np.ndarray = None,
        c_conf: list[int] | np.ndarray = None,
        type1_stats: dict = None,
        type2_stats: dict = None
    ):

        self.nsubjects = nsubjects
        self.nsamples = nsamples
        self.params = params
        self.params_type1 = {k: v for k, v in self.params.items() if k.startswith('type1_')}
        self.params_type2 = {k: v for k, v in self.params.items() if k.startswith('type2_')}
        self.params_extra = params_extra
        self.cfg = cfg
        self.stimuli = x_stim
        self.stimuli_category = x_stim_category
        self.choices = d_dec
        self.accuracy = x_stim_category == d_dec
        self.y_decval_latent = y_decval_latent
        self.y_decval = y_decval
        self.z1_type1_evidence_latent = z1_type1_evidence_latent
        self.z1_type1_evidence_base = z1_type1_evidence_base
        self.z1_type1_evidence = z1_type1_evidence
        self.confidence_latent = c_conf_latent
        self.confidence_base = c_conf_base
        self.confidence = c_conf
        self.type1_stats = type1_stats
        self.type2_stats = type2_stats

    def squeeze(self):
        # for var in ('x_stim', 'x_stim_category', 'd_dec', 'accuracy'):
        #     if getattr(self, var) is not None:
        #         setattr(self, var, getattr(self, var).squeeze())
        for name, value in self.__dict__.items():
            if isinstance(value, np.ndarray):
                setattr(self, name, value.squeeze())
        return self

    def plot_psychometric(self, **kwargs):
        import remeta
        remeta.plot_psychometric(self, **kwargs)

    def plot_stimulus_versus_confidence(self, **kwargs):
        import remeta
        remeta.plot_stimulus_versus_confidence(self, **kwargs)

    def plot_confidence_histogram(self, **kwargs):
        import remeta
        remeta.plot_confidence_histogram(self, **kwargs)


def generate_stimuli(nsubjects, nsamples, stim_min=None, stim_max=1, stim_levels=10):
    stim_min = stim_max / stim_levels if stim_min is None else stim_min
    levels = np.hstack((-np.linspace(stim_min, stim_max, stim_levels)[::-1], np.linspace(stim_min, stim_max, stim_levels)))
    x_stim = np.array([np.random.permutation(np.tile(levels, int(np.ceil(nsamples / len(levels)))))[:nsamples] for _ in range(nsubjects)])
    return x_stim


def stimulus_to_latent_decision_value(x_stim, params, cfg=None):

    x_stim = np.array(x_stim)

    if ('type1_thresh' in params and not (cfg is not None and (cfg.param_type1_thresh.enable == 0))):
        type1_thresh = _check_param(params['type1_thresh'])
    else:
        type1_thresh = (0, 0)
    if ('type1_bias' in params and not (cfg is not None and (cfg.param_type1_bias.enable == 0))):
        type1_bias = _check_param(params['type1_bias'])
    else:
        type1_bias = (0, 0)

    if ('type1_nonlinear_gain' in params and not (cfg is not None and (cfg._param_type1_nonlinear_gain.enable == 0))):
        if ('type1_nonlinear_scale' in params and not (cfg is not None and (cfg.param_type1_nonlinear_scale.enable == 0))):
            scale = params['type1_nonlinear_scale'] * np.max(x_stim)
        else:
            scale = None
        x_stim_transform = compute_nonlinear_encoding(x_stim, params['type1_nonlinear_gain'], scale)
    else:
        x_stim_transform = x_stim

    y_decval_latent = np.full(x_stim_transform.shape, np.nan)
    y_decval_latent[x_stim_transform < 0] = (np.abs(x_stim_transform[x_stim_transform < 0]) > type1_thresh[0]) * \
                                   x_stim_transform[x_stim_transform < 0] + type1_bias[0]
    y_decval_latent[x_stim_transform >= 0] = (np.abs(x_stim_transform[x_stim_transform >= 0]) > type1_thresh[1]) * \
                                    x_stim_transform[x_stim_transform >= 0] + type1_bias[1]
    return y_decval_latent


def stimulus_to_decision_value(x_stim, params, cfg=None, return_only_decval=False):

    if ('type1_noise_heteroscedastic' in params and not (cfg is not None and (cfg.param_type1_noise_heteroscedastic.enable == 0))) or \
        (listlike(params['type1_noise']) and not (cfg is not None and (cfg.param_type1_noise.enable < 2))):
        type1_noise = compute_signal_dependent_type1_noise(
            x_stim=x_stim, type1_noise_signal_dependency=None if cfg is None or not cfg.param_type1_noise_heteroscedastic.enable else cfg.param_type1_noise_heteroscedastic.model, **params)
    else:
        type1_noise = params['type1_noise']

    y_decval_latent = stimulus_to_latent_decision_value(x_stim, params, cfg=cfg)

    if cfg is None or cfg.param_type1_noise.model == 'normal':
        y_decval = y_decval_latent + norm(scale=type1_noise).rvs(size=x_stim.shape)
    elif cfg is not None and cfg.param_type1_noise.model == 'logistic':
        y_decval = y_decval_latent + logistic_dist(scale=type1_noise * np.sqrt(3) / np.pi).rvs(size=x_stim.shape)

    d_dec = (y_decval >= 0).astype(int)

    if return_only_decval:
        return y_decval
    else:
        return y_decval_latent, y_decval, d_dec



if __name__ == '__main__':
    import remeta
    import warnings
    warnings.filterwarnings('error')
    params = dict(
        type1_noise=0.2,
        type1_thresh=0.2,
        type1_bias=0.2,
        type2_noise=0.2,
        # type2_evidence_bias=1.2
    )
    cfg = remeta.Configuration()
    cfg.meta_noise_type = 'report'
    cfg.param_type2_noise.model = 'beta_mode'
    cfg.param_type1_thresh.enable = 1
    cfg.param_type1_bias.enable = 1
    cfg.param_type2_evidence_bias.enable = 0
    m = simulate(params, nsubjects=1, nsamples=1000, cfg=cfg)

