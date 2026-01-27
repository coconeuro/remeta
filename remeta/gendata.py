import numpy as np
from scipy.stats import norm, logistic as logistic_dist
import warnings

from remeta.configuration import Configuration
from remeta.type2_dist import get_type2_dist
from remeta.transform import compute_signal_dependent_type1_noise, type1_evidence_to_confidence, check_criteria_sum, compute_nonlinear_encoding
from remeta.util import _check_param, TAB, discretize_confidence_with_bounds, print_dataset_characteristics
from remeta.type2_SDT import type2roc, type2_SDT_MLE


class Simulation:
    def __init__(self, nsubjects=None, nsamples=None, params=None, params_extra=None, cfg=None,
                 x_stim=None, x_stim_category=None, d_dec=None, y_decval_latent=None, y_decval=None,
                 z1_type1_evidence_latent=None, z1_type1_evidence_base=None, z1_type1_evidence=None,
                 c_conf_latent=None, c_conf_base=None, c_conf=None,
                 likelihood_dist=None, type1_stats=None, type2_stats=None):
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
        self.likelihood_dist = likelihood_dist
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


def generate_stimuli(nsubjects, nsamples, stepsize=0.02, warn_in_case_of_nondivisible_stepsize=False):
    levels = np.hstack((-np.arange(stepsize, 1.01, stepsize)[::-1], np.arange(stepsize, 1.01, stepsize)))
    if warn_in_case_of_nondivisible_stepsize and ((nsamples % (2/stepsize)) != 0):
        warnings.warn(f'At the chosen stepsize of {stepsize} there are {2/stepsize} stimulus levels,'
                      f'which is not a divisor of the chosen sample size {nsamples}', UserWarning)
    x_stim = np.array([np.random.permutation(np.tile(levels, int(np.ceil(nsamples / len(levels)))))[:nsamples] for _ in range(nsubjects)])
    return x_stim


def simu_type1_responses(x_stim, params, cfg):

    if (cfg.type1_noise_signal_dependency != 'none') or (cfg.enable_type1_param_noise == 2):
        type1_noise = compute_signal_dependent_type1_noise(
            x_stim=x_stim, type1_noise_signal_dependency=cfg.type1_noise_signal_dependency, **params)
    else:
        type1_noise = params['type1_noise']

    type1_param_thresh = _check_param(params['type1_thresh']) if cfg.enable_type1_param_thresh else (0, 0)
    type1_param_bias = _check_param(params['type1_bias']) if cfg.enable_type1_param_bias else (0, 0)

    if cfg.enable_type1_param_nonlinear_encoding_gain:
        x_stim_transform = compute_nonlinear_encoding(
            x_stim, params['type1_nonlinear_encoding_gain'],
            params['type1_nonlinear_encoding_transition'] if cfg.enable_type1_param_nonlinear_encoding_transition else None)
    else:
        x_stim_transform = x_stim

    y_decval_latent = np.full(x_stim_transform.shape, np.nan)
    y_decval_latent[x_stim_transform < 0] = (np.abs(x_stim_transform[x_stim_transform < 0]) > type1_param_thresh[0]) * \
                                   x_stim_transform[x_stim_transform < 0] + type1_param_bias[0]
    y_decval_latent[x_stim_transform >= 0] = (np.abs(x_stim_transform[x_stim_transform >= 0]) > type1_param_thresh[1]) * \
                                    x_stim_transform[x_stim_transform >= 0] + type1_param_bias[1]

    y_decval = y_decval_latent + logistic_dist(scale=type1_noise * np.sqrt(3) / np.pi).rvs(size=x_stim_transform.shape)
    d_dec = (y_decval >= 0).astype(int)

    return y_decval_latent, y_decval, d_dec


def simu_data(params, nsubjects=1, nsamples=1000, cfg=None, stimuli_external=None, verbose=True, stimuli_stepsize=0.02,
              squeeze=False, warn_in_case_of_nondivisible_stepsize=False,
              compute_stats=True, **kwargs):
    params = params.copy()  # this variable can be modifed, thus better to make a copy
    if cfg is None:
        # Set configuration attributes that match keyword arguments
        cfg_kwargs = {k: v for k, v in kwargs.items() if k in Configuration.__dict__}
        cfg = Configuration(**cfg_kwargs)
        for setting in cfg.__dict__:
            if setting.startswith('enable_'):
                if setting.split('enable_')[1].replace('_param_', '_') not in params:
                    setattr(cfg, setting, 0)
    # if not cfg.setup_called:
    cfg.setup(generative_mode=True)

    if cfg.type2_noise_dist is None:
        cfg.type2_noise_dist = dict(noisy_report='truncated_norm_mode', noisy_readout='truncated_norm_mode', noisy_temperature='lognorm_mode')[cfg.type2_noise_type]

    # Make sure no unwanted parameters have been passed
    for p in ('thresh', 'bias', 'noise_heteroscedastic', 'nonlinear_encoding_gain', 'nonlinear_encoding_transition'):
        if not getattr(cfg, f'enable_type1_param_{p}'):
            params.pop(f'type1_{p}', None)
    for p in ('evidence_bias_mult', 'criteria'):
        if not getattr(cfg, f'enable_type2_param_{p}'):
            params.pop(f'type2_{p}', None)

    if stimuli_external is None:
        x_stim = generate_stimuli(nsubjects, nsamples, stepsize=stimuli_stepsize,
                                  warn_in_case_of_nondivisible_stepsize=warn_in_case_of_nondivisible_stepsize)
    else:
        x_stim = stimuli_external / np.max(np.abs(stimuli_external))
        if stimuli_external.shape != (nsubjects, nsamples):
            x_stim = np.tile(x_stim, (nsubjects, 1))
    x_stim_category = (np.sign(x_stim) > 0).astype(int)
    y_decval_latent, y_decval, d_dec = simu_type1_responses(x_stim, params, cfg)

    if not cfg.skip_type2:

        z1_type1_evidence_latent = np.abs(y_decval_latent)
        z1_type1_evidence_base = np.abs(y_decval)

        if cfg.type2_noise_type == 'noisy_readout':
            dist = get_type2_dist(cfg.type2_noise_dist, type2_center=z1_type1_evidence_base, type2_noise=params['type2_noise'],
                                  type2_noise_type=cfg.type2_noise_type)

            z1_type1_evidence = np.maximum(0, dist.rvs((nsubjects, nsamples)))
        elif cfg.type2_noise_type == 'noisy_temperature':
            dist = get_type2_dist(cfg.type2_noise_dist, type2_center=params['type1_noise'] * np.ones_like(z1_type1_evidence_base),
                                  type2_noise=params['type2_noise'], type2_noise_type=cfg.type2_noise_type)
            type1_noise_estimated = dist.rvs()
            z1_type1_evidence = z1_type1_evidence_base
        elif cfg.type2_noise_type == 'noisy_report':
            z1_type1_evidence = z1_type1_evidence_base
        else:
            raise ValueError('Unknown type 2 noise type')

        c_conf_latent = type1_evidence_to_confidence(
            z1_type1_evidence=z1_type1_evidence_latent, y_decval=y_decval,
            x_stim=x_stim, type1_noise_signal_dependency=cfg.type1_noise_signal_dependency,
            **({**params, **dict(type1_noise=type1_noise_estimated)} if cfg.type2_noise_type == 'noisy_temperature' else params)
        )

        c_conf_base = type1_evidence_to_confidence(
            z1_type1_evidence=z1_type1_evidence, y_decval=y_decval,
            x_stim=x_stim, type1_noise_signal_dependency=cfg.type1_noise_signal_dependency,
            **({**params, **dict(type1_noise=type1_noise_estimated)} if cfg.type2_noise_type == 'noisy_temperature' else params)
        )

        if cfg.type2_noise_type == 'noisy_report':
            dist = get_type2_dist(cfg.type2_noise_dist, type2_center=c_conf_base, type2_noise=params['type2_noise'],
                                  type2_noise_type=cfg.type2_noise_type)
            c_conf = np.maximum(0, np.minimum(1, dist.rvs((nsubjects, nsamples))))
        else:
            c_conf = c_conf_base

        if cfg.enable_type2_param_criteria:
            if 'type2_criteria' in params:
                sum_criteria = np.sum(params['type2_criteria'])
                if sum_criteria > 1.001:
                    old_criteria = params['type2_criteria']
                    params['type2_criteria'] = check_criteria_sum(params['type2_criteria'])
                    warnings.warn(
                       '\nThe first entry of the criterion list is a criterion, whereas the subsequent entries encode\n'
                       'the gap to the respective previous criterion. Hence, the sum of all entries in the criterion\n'
                       f'list must be smaller than 1, but sum([{", ".join([f"{c:.3f}" for c in old_criteria])}]) = {sum_criteria:.3f}). '
                       f'Changing criteria to [{", ".join([f"{c:.3f}" for c in params['type2_criteria']])}].', UserWarning)
                first_criterion_and_gaps = params['type2_criteria']
                criteria = [v if i == 0 else np.sum(first_criterion_and_gaps[:i+1]) for i, v in enumerate(first_criterion_and_gaps)]
            else:
                first_criterion_and_gaps = np.ones(cfg.n_discrete_confidence_levels - 1) / cfg.n_discrete_confidence_levels
                criteria = [v if i == 0 else np.sum(first_criterion_and_gaps[:i+1]) for i, v in enumerate(first_criterion_and_gaps)]
                warnings.warn(
                    '\nType 2 criteria enabled, but type2_criteria have not been specified. Using default values\n'
                    f'of a Bayesian confidence observer for {cfg.n_discrete_confidence_levels} discrete ratings: [{', '.join([f"{v:.3g}" for v in first_criterion_and_gaps])}].\n'
                    'Note that the first entry of the criterion list is a criterion, whereas the subsequent\n'
                    f'entries encode the gap to the respective previous criterion.\n'
                    f'The final criteria are: [{', '.join([f"{v:.3g}" for v in criteria])}]', UserWarning)

            c_conf = (np.digitize(c_conf, criteria) + 0.5) / cfg.n_discrete_confidence_levels
            c_conf_base = (np.digitize(c_conf_base, criteria) + 0.5) / cfg.n_discrete_confidence_levels

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
            d1 = norm.ppf(min(1 - 1e-3, max(1e-3, d_dec[x_stim_category == 1].mean()))) - \
                 norm.ppf(min(1 - 1e-3, max(1e-3, d_dec[x_stim_category == 0].mean().mean()))),
            choice_bias=d_dec.mean(),
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
                    type2_criteria_absolute=[np.sum(params['type2_criteria'][:i + 1]) for i in range(len(params['type2_criteria']))],
                    type2_criteria_bias=np.mean(params['type2_criteria']) * (len(params['type2_criteria']) + 1) - 1
                )
                simargs.update(params_extra=params_extra)


    simulation = Simulation(**simargs)
    if verbose:
        print_dataset_characteristics(simulation)

    return simulation


if __name__ == '__main__':
    import warnings
    warnings.filterwarnings('error')
    params = dict(
        type1_noise=0.2,
        type1_thresh=0.2,
        type1_bias=0.2,
        type2_noise=0.2,
        # type2_evidence_bias_mult=1.2
    )
    options = dict(meta_noise_type='noisy_report', enable_type1_param_thresh=1, enable_type1_param_bias=1,
                   enable_type2_param_evidence_bias_mult=0, type2_noise_dist='beta_mode')
    m = simu_data(params, nsubjects=1, nsamples=1000, **options)

