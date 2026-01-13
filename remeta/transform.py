import warnings

import numpy as np
from scipy.special import expit

from .util import maxfloat
from .util import _check_param


def check_criteria_sum(criteria):
    """
    Ensure that criteria sum up to at most 1

    Parameters
    ----------
    criteria : array-like of dtype float
        Confidence criteria (typically provided as a list)

    Returns
    ----------
    criteria : array-like of dtype float
        Transformed confidence criteria
    """
    ind1 = np.where(np.cumsum(criteria) > 1)[0]
    if len(ind1):
        for i in ind1[1:]:
            criteria[i] = 0
        criteria[ind1[0]] = 1 - np.sum(criteria[:ind1[0]]) + 1e-8
        # Note: we add 1e-8 to avoid edge cases due to floating point precision
    return criteria


def logistic(x, type1_noise, squeeze=False):
    """
    Logistic function

    Parameters
    ----------
    x : array-like
        This is typically a stimulus array or a transformed stimulus array (e.g. decision values).
    type1_noise : float
        Type 1 noise parameter.

    Returns
    ----------
    posterior : array-like
        Posterior probability under a logistic model.
    """
    if squeeze:
        # posterior = 1 / (1 + np.exp(-((np.pi / np.sqrt(3)) * x / np.maximum(type1_noise, 0.001)).squeeze(),
        #                             dtype=maxfloat))
        posterior = expit(((np.pi / np.sqrt(3) * x) / np.maximum(type1_noise, 1e-8)).squeeze())
    else:
        # posterior = 1 / (1 + np.exp(-((np.pi / np.sqrt(3)) * x / np.maximum(type1_noise, 0.001)),
        #                             dtype=maxfloat))
        posterior = expit((np.pi / np.sqrt(3) * x) / np.maximum(type1_noise, 1e-8))

    return posterior.astype(np.float64)


def logistic_inv(posterior, type1_noise):
    """
    Inverse logistic function

    Parameters
    ----------
    posterior : array-like
        Posterior probability.
    type1_noise : float
        Type 1 noise parameter.

    Returns
    ----------
    x : array-like
        See docstring of the logistic method.
    """
    x = -(np.sqrt(3) * type1_noise / np.pi) * np.log(1 / posterior - 1)
    return x


def compute_signal_dependent_type1_noise(x_stim, type1_noise=None, type1_thresh=None, type1_noise_heteroscedastic=None,
                                         type1_noise_signal_dependency='none', **kwargs):  # noqa
    """
    Compute signal-dependent type 1 noise.

    Parameters
    ----------
    x_stim : array-like
        Array of signed stimulus intensity values, where the sign codes the stimulus category and the absolut value
        codes the intensity. Must be normalized to [-1; 1].
    type1_noise : float or array-like
        Type 1 noise parameter.
    type1_thresh: float or array-like
        Type 1 threshold.
    type1_noise_heteroscedastic : float or array-like
        Signal-dependent type 1 noise parameter.
    type1_noise_signal_dependency : str
        Define the signal dependency of type 1 noise. One of 'none', 'multiplicative', 'power', 'exponential', 'logarithm'.
    kwargs : dict
        Convenience parameter to avoid an error if irrelevant parameters are passed.

    Returns
    ----------
    type1_noise_heteroscedastic : array-like
        Signal-dependent (heteroscedastic) type 1 noise of shape stimuli.shape.
    """

    type1_noise_ = _check_param(type1_noise)
    type1_heteroscedastic_ = _check_param(type1_noise_heteroscedastic)
    type1_thresh_ = (0, 0) if type1_thresh is None else _check_param(type1_thresh)
    type1_noise_heteroscedastic = np.ones(x_stim.shape)
    neg, pos = x_stim < 0, x_stim >= 0
    if type1_noise_signal_dependency == 'none':
        type1_noise_heteroscedastic[neg] *= type1_noise_[0]
        type1_noise_heteroscedastic[pos] *= type1_noise_[1]
    elif type1_noise_signal_dependency == 'multiplicative':
        type1_noise_heteroscedastic[neg] = np.sqrt(type1_noise_[0] ** 2 +
                                                         ((np.abs(x_stim[neg]) - type1_thresh_[0]) * type1_heteroscedastic_[0]) ** 2)  # noqa
        type1_noise_heteroscedastic[pos] = np.sqrt(type1_noise_[1] ** 2 +
                                                         ((np.abs(x_stim[pos]) - type1_thresh_[1]) * type1_heteroscedastic_[1]) ** 2)  # noqa
    elif type1_noise_signal_dependency == 'power':
        type1_noise_heteroscedastic[neg] = np.sqrt(type1_noise_[0] ** 2 +
                                                         (np.abs(x_stim[neg]) - type1_thresh_[0]) ** (2 * type1_heteroscedastic_[0]))  # noqa
        type1_noise_heteroscedastic[pos] = np.sqrt(type1_noise_[1] ** 2 +
                                                         (np.abs(x_stim[pos]) - type1_thresh_[1]) ** (2 * type1_heteroscedastic_[1]))  # noqa
    elif type1_noise_signal_dependency == 'exponential':
        type1_noise_heteroscedastic[neg] = np.sqrt(type1_noise_[0] ** 2 +
                                                         (np.exp(type1_heteroscedastic_[0] * (np.abs(x_stim[neg]) - type1_thresh_[0])) - 1) ** 2)  # noqa
        type1_noise_heteroscedastic[pos] = np.sqrt(type1_noise_[1] ** 2 +
                                                         (np.exp(type1_heteroscedastic_[1] * (np.abs(x_stim[pos]) - type1_thresh_[1])) - 1) ** 2)  # noqa
    elif type1_noise_signal_dependency == 'logarithm':
        type1_noise_heteroscedastic[neg] = np.sqrt(type1_noise_[0] ** 2 +
                                                         np.log(type1_heteroscedastic_[0] * (np.abs(x_stim[neg]) - type1_thresh_[0]) + 1) ** 2)  # noqa
        type1_noise_heteroscedastic[pos] = np.sqrt(type1_noise_[1] ** 2 +
                                                         np.log(type1_heteroscedastic_[1] * (np.abs(x_stim[pos]) - type1_thresh_[1]) + 1) ** 2)  # noqa
    else:
        raise ValueError(f'{type1_noise_signal_dependency} is not a valid function for type1_noise_signal_dependency')

    return type1_noise_heteroscedastic


def type1_evidence_to_confidence(z1_type1_evidence,
                                 type2_evidence_bias_mult=1,
                                 type1_noise=None, type1_thresh=None,
                                 type1_noise_heteroscedastic=None, type1_noise_signal_dependency='none',
                                 y_decval=None, x_stim=None,
                                 **kwargs):  # noqa
    """
    Transformation from type 1 evidence (z1) to confidence (c).

    Parameters
    ----------
    z1_type1_evidence : array-like
        Evidence at the type 1 level (= absolute decision value).
    type2_evidence_bias_mult : float or array-like
        Multiplicative metacognitive bias parameter loading on evidence.
    type1_noise : float or array-like
        Type 1 noise parameter. Can be array-like in case of signal-dependent type 1 noise.
    type1_noise_heteroscedastic : float or array-like
        Signal-dependent type 1 noise parameter.
    type1_noise_signal_dependency : str
        Signal-dependent type 1 noise type. One of 'linear', 'power', 'exponential', 'logarithm'.
    y_decval : array-like
        Decision values.
    x_stim : array-like
        Array of signed stimulus intensity values, where the sign codes the stimulus category and the absolut value
        codes the intensity.
    kwargs : dict
        Convenience parameter to avoid an error if irrelevant parameters are passed.

    Returns
    ----------
    c_conf : array-like
        Model-predicted confidence.
    """
    z1_type1_evidence = np.atleast_1d(z1_type1_evidence)

    if ((type1_noise_signal_dependency != 'none') or (hasattr(type1_noise, '__len__') and len(type1_noise) == 2)):
        if x_stim is None:
            raise ValueError('Type 1 noise is signal-dependent, but stimuli (x_stim) have not been '
                             'passed.')
        type1_noise = compute_signal_dependent_type1_noise(
            x_stim.reshape(-1, 1) if (x_stim.ndim == 1) and (z1_type1_evidence.ndim == 2) else x_stim,
            type1_noise=type1_noise, type1_thresh=type1_thresh, type1_noise_heteroscedastic=type1_noise_heteroscedastic,
            type1_noise_signal_dependency=type1_noise_signal_dependency)

    z2_type2_evidence = type2_evidence_bias_mult * z1_type1_evidence
    c_conf = np.tanh(np.pi * z2_type2_evidence / (2 * np.sqrt(3) * type1_noise))

    return c_conf


def confidence_to_type1_evidence(c_conf,
                                 type2_evidence_bias_mult=1,
                                 type1_noise=None, type1_thresh=None,
                                 type1_noise_heteroscedastic=None, type1_noise_signal_dependency='none',
                                 y_decval=None, x_stim=None,
                                 tile_on_type1_uncertainty=True,
                                 **kwargs):  ## noqa
    """
    Transformation from confidence (c) to type 1 evidence (z1).

    Parameters
    ----------
    c_conf : array-like
        Confidence ratings (from behavioral or simulated data).
    type2_evidence_bias_mult : float or array-like
        Multiplicative metacognitive bias parameter loading on evidence.
    type1_noise : float or array-like
        Type 1 noise parameter. Can be array-like in case of signal-dependent type 1 noise.
    type1_noise_heteroscedastic : float or array-like
        Signal-dependent type 1 noise parameter.
    type1_noise_signal_dependency : str
        Signal-dependent type 1 noise type. One of 'linear', 'power', 'exponential', 'logarithm'.
    y_decval : array-like
        Decision values.
    x_stim : array-like
        Array of signed stimulus intensity values, where the sign codes the stimulus category and the absolut value
        codes the intensity.
    kwargs : dict
        Convenience parameter to avoid an error if irrelevant parameters are passed.

    Returns
    ----------
    z1_type1_evidence : array-like
        Absolute decision values ('z1_type1_evidence').
    """
    if tile_on_type1_uncertainty and y_decval is not None:
        c_conf = np.tile(c_conf, y_decval.shape[-1])

    if ((type1_noise_signal_dependency != 'none') or (hasattr(type1_noise, '__len__') and len(type1_noise) == 2)):
        if x_stim is None:
            raise ValueError('Type 1 noise is signal-dependent, but stimuli (x_stim) have not been '
                             'passed.')
        type1_noise = compute_signal_dependent_type1_noise(
            x_stim, type1_noise=type1_noise, type1_thresh=type1_thresh, type1_noise_heteroscedastic=type1_noise_heteroscedastic,
            type1_noise_signal_dependency=type1_noise_signal_dependency)

    c_conf = np.minimum(1 - 1e-8, c_conf)
    z2_type2_evidence = (2 * np.sqrt(3) * type1_noise / np.pi) * np.arctanh(c_conf)

    z1_type1_evidence = z2_type2_evidence / type2_evidence_bias_mult


    return z1_type1_evidence



def confidence_to_type1_noise(c_conf,
                              type2_evidence_bias_mult=1,
                              type1_noise=None, type1_thresh=None,
                              type1_noise_heteroscedastic=None, type1_noise_signal_dependency='none',
                              y_decval=None, x_stim=None,
                              tile_type1_uncertainty=False,
                              **kwargs):  ## noqa
    """
    Transformation from confidence (c) to type 1 evidence (z1).

    Parameters
    ----------
    c_conf : array-like
        Confidence ratings (from behavioral or simulated data).
    type2_evidence_bias_mult : float or array-like
        Multiplicative metacognitive bias parameter loading on evidence.
    type1_noise : float or array-like
        Type 1 noise parameter. Can be array-like in case of signal-dependent type 1 noise.
    type1_noise_heteroscedastic : float or array-like
        Signal-dependent type 1 noise parameter.
    type1_noise_signal_dependency : str
        Signal-dependent type 1 noise type. One of 'linear', 'power', 'exponential', 'logarithm'.
    y_decval : array-like
        Decision values.
    x_stim : array-like
        Array of signed stimulus intensity values, where the sign codes the stimulus category and the absolut value
        codes the intensity.
    kwargs : dict
        Convenience parameter to avoid an error if irrelevant parameters are passed.

    Returns
    ----------
    z1_type1_evidence : array-like
        Absolute decision values ('z1_type1_evidence').
    """
    if y_decval is None:
        y_decval = c_conf
    elif tile_type1_uncertainty:
        c_conf = np.tile(c_conf, y_decval.shape[-1])

    if ((type1_noise_signal_dependency != 'none') or (hasattr(type1_noise, '__len__') and len(type1_noise) == 2)):
        raise ValueError('Signal-dependency or dual parameter mode not yet supported for type2 noise type '
                         '"noisy_temperature"')
        # if x_stim is None:
        #     raise ValueError('Type 1 noise is signal-dependent, but stimuli (x_stim) have not been '
        #                      'passed.')
        # type1_noise = compute_signal_dependent_type1_noise(
        #     x_stim, type1_noise=type1_noise, type1_thresh=type1_thresh, type1_noise_heteroscedastic=type1_noise_heteroscedastic,
        #     type1_noise_signal_dependency=type1_noise_signal_dependency)

    c_conf = np.maximum(1e-8, np.minimum(1 - 1e-8, c_conf))

    z2_type2_evidence = np.abs(y_decval) * type2_evidence_bias_mult

    type1_noise = (np.pi / (2 * np.sqrt(3))) * (z2_type2_evidence / np.arctanh(c_conf))

    return type1_noise



def type1_noise_to_confidence(type1_noise,
                              type2_evidence_bias_mult=1,
                              type1_thresh=None,
                              type1_noise_heteroscedastic=None, type1_noise_signal_dependency='none',
                              y_decval=None, x_stim=None,
                              **kwargs):  # noqa
    """
    Transformation from type 1 evidence (z1) to confidence (c).

    Parameters
    ----------
    z1_type1_evidence : array-like
        Evidence at the type 1 level (= absolute decision value).
    type2_evidence_bias_mult : float or array-like
        Multiplicative metacognitive bias parameter loading on evidence.
    type1_noise : float or array-like
        Type 1 noise parameter. Can be array-like in case of signal-dependent type 1 noise.
    type1_noise_heteroscedastic : float or array-like
        Signal-dependent type 1 noise parameter.
    type1_noise_signal_dependency : str
        Signal-dependent type 1 noise type. One of 'linear', 'power', 'exponential', 'logarithm'.
    y_decval : array-like
        Decision values.
    x_stim : array-like
        Array of signed stimulus intensity values, where the sign codes the stimulus category and the absolut value
        codes the intensity.
    kwargs : dict
        Convenience parameter to avoid an error if irrelevant parameters are passed.

    Returns
    ----------
    c_conf : array-like
        Model-predicted confidence.
    """

    if ((type1_noise_signal_dependency != 'none') or (hasattr(type1_noise, '__len__') and len(type1_noise) == 2)):
        raise ValueError('Signal-dependency or dual parameter mode not yet supported for type2 noise type '
                         '"noisy_temperature"')
        # if x_stim is None:
        #     raise ValueError('Type 1 noise is signal-dependent, but stimuli (x_stim) have not been '
        #                      'passed.')
        # type1_noise = compute_signal_dependent_type1_noise(
        #     x_stim.reshape(-1, 1) if (x_stim.ndim == 1) and (z1_type1_evidence.ndim == 2) else x_stim,
        #     type1_noise=type1_noise, type1_thresh=type1_thresh, type1_noise_heteroscedastic=type1_noise_heteroscedastic,
        #     type1_noise_signal_dependency=type1_noise_signal_dependency)

    z2_type2_evidence = type2_evidence_bias_mult * np.abs(y_decval)
    c_conf = np.tanh(np.pi * z2_type2_evidence / (2 * np.sqrt(3) * type1_noise))

    return c_conf
