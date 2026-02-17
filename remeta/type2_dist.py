import warnings

import numpy as np
from scipy.stats import lognorm, beta, betaprime, gamma

from .fast_truncnorm import truncnorm
from ._distributions import (TruncatedGumbelMode, TruncatedGumbelModeSD, TruncatedLognormal, TruncatedLognormalMode,
                             TruncatedLognormalMean, TruncatedLognormalModeSD, _params_lognormal_mode_std,
                             _params_truncnorm_mode_std)

TYPE2_NOISE_DISTS = (
    'beta_mode_std', 'beta_mode', 'beta_mean_std',
    'lognormal_mean_std', 'lognormal_mean', 'lognormal_mode_std', 'lognormal_mode', 'lognormal_median_std',
    'gamma_mean_std', 'gamma_mean', 'gamma_mean_cv', 'gamma_mode_std', 'gamma_mode',
    'betaprime_mean_std',
    'truncated_normal_mode_std', 'truncated_normal_mode',
    'truncated_gumbel_mode_std', 'truncated_gumbel_mode',
    'truncated_lognormal_mode_std', 'truncated_lognormal', 'truncated_lognormal_mode', 'truncated_lognormal_mean'
)

TYPE2_NOISE_DISTS_REPORT_ONLY = (
    'beta_mode_std', 'beta_mode', 'beta_mean_std',
    'truncated_lognormal_mode_std', 'truncated_lognormal_mode', 'truncated_lognormal', 'truncated_lognormal_mean'
)
TYPE2_NOISE_DISTS_READOUT_TEMPERATURE_ONLY = (
    'lognormal_mean_std', 'lognormal_mean', 'lognormal_mode_std', 'lognormal_mode', 'lognormal_median_std',
    'gamma_mean_std', 'gamma_mean', 'gamma_mean_cv', 'gamma_mode_std', 'gamma_mode',
    'betaprime_mean_std'
)



def get_type2_dist(type2_dist, type2_center, type2_noise, type2_noise_type='report'):
    """
    Helper function to select appropriately parameterized type 2 noise distributions.

    Parameters:
    -----------
    type2_dist : str
        Name of the type 2 noise distribution. Check TYPE2_NOISE_DISTS for possible values.
    type2_noise : float or array-like of dtype float
        "True" value of a noisy type 2 variable. Corresponds to type 1 noise (noisy-temperature), metacognitve
        evidence (noisy-readout) or confidence (noisy-report).
        For mean- and mode preserving distributions, the center corresponds to the mean (suffix _mean) and mode (suffix
        _mode), respectively.
    type2_noise : float or array-like of dtype float with mode.shape
        Spread parameter that represents metacognitve noise. For standard-deviation-preserving distributions, this
        parameter corresponds to the standard deviation when sampling data from the distribution (suffix _std).
    type2_noise_type : str (default='report')
        Metacognitive noise type. Possible values: 'report', 'readout', 'temperature'.

    Returns:
    --------
    scipy.stats continuous distribution instance
    """

    if type2_dist not in TYPE2_NOISE_DISTS:
        raise ValueError(f"Unkonwn distribution '{type2_dist}'.")
    elif (type2_noise_type == 'report') and type2_dist in TYPE2_NOISE_DISTS_READOUT_TEMPERATURE_ONLY:
        raise ValueError(f"Distribution '{type2_dist}' is only valid for noisy-readout or noisy-temperature models.")
    elif (type2_noise_type in ('readout', 'temperature')) and type2_dist in TYPE2_NOISE_DISTS_REPORT_ONLY:
        raise ValueError(f"Distribution '{type2_dist}' is only valid for noisy-report models.")

    if type2_noise < 1e-10:
        # warnings.warn('Type 2 noise is smaller than 1e-10, which can lead to unstable numerical results. It is set '
        #               'to a hard minimum of 1e-10.')
        type2_noise = 1e-10

    if type2_dist == 'lognormal_median_std':
        # type2_center = median, type2_noise = SD
        s = np.maximum(1e-12, np.sqrt(np.log((1 + np.sqrt(1 + 4 * (type2_noise / type2_center) ** 2)) / 2)))
        dist = lognorm(s=s, scale=type2_center)
    elif type2_dist == 'lognormal_mean':
        # type2_center = mean, type2_noise = SD in log space
        dist = lognorm(s=type2_noise, scale=np.exp(-type2_noise ** 2 / 2) * type2_center)
    elif type2_dist == 'lognormal_mode':
        # type2_center = mode, type2_noise = SD in log space
        if type2_noise > 23:
            # warnings.warn(f'Type 2 noise for a {type2_dist} distribution is too high. Capping at 23')
            type2_noise = 23
        dist = lognorm(s=type2_noise, scale=np.exp(type2_noise ** 2) * type2_center)
    elif type2_dist == 'lognormal_mode_std':
        # type2_center = mode, type2_noise = SD
        shape, type2_noise = _params_lognormal_mode_std(np.maximum(1e-5, type2_center), type2_noise)
        dist = lognorm(loc=0, scale=type2_noise, s=shape)
    elif type2_dist == 'lognormal_mean_std':
        # Corresponds to the CASSANDRE/LogN setup
        # type2_center = mean, type2_noise = SD
        type2_center = np.maximum(type2_center, 1e-12)
        sigma2 = np.log1p((np.maximum(type2_noise, 0) / type2_center) ** 2)
        scale = np.exp(np.log(type2_center) - sigma2 / 2)
        dist = lognorm(loc=0, scale=scale, s=np.sqrt(sigma2))
    elif type2_dist == 'beta_mean_std':
        # Canonical "precision" parameterization of the beta distribution, where precision = 1 / noise**2, i.e.
        # inverse variance.
        # type2_center = mean, type2_noise = SD

        type2_center = np.clip(type2_center, 1e-12, 1 - 1e-12)          # avoid a=0 or b=0
        type2_noise = np.maximum(type2_noise, 1e-12)

        phi = type2_center * (1 - type2_center) / (type2_noise ** 2) - 1
        phi = np.maximum(phi, 1e-12)

        a = type2_center * phi
        b = (1 - type2_center) * phi

        dist = beta(loc=0, a=a, b=b, scale=1)
    elif type2_dist == 'beta_mode_std':
        # type2_center = mode, type2_noise = SD

        type2_center = np.clip(type2_center, 1e-8, 1 - 1e-8)

        var_target = np.maximum(type2_noise ** 2, 1e-8)
        # Initial closed-form approximation
        kappa = type2_center * (1 - type2_center) / var_target
        # One-step variance correction
        num = type2_center * (1 - type2_center) * kappa ** 2 + kappa + 1
        den = (kappa + 2)**2 * (kappa + 3)
        var_actual = num / den
        kappa *= var_actual / var_target

        a = type2_center * kappa + 1
        b = (1 - type2_center) * kappa + 1
        dist = beta(loc=0, a=a, b=b, scale=1)

    elif type2_dist == 'beta_mode':
        # type2_center = mode, type2_noise != SD
        type2_center = np.maximum(1e-5, np.minimum(1 - 1e-5, type2_center))
        kappa = 2 + (type2_center * (1 - type2_center)) / (type2_noise ** 2)   # concentration/precision (>2)
        a = 1 + type2_center * (kappa - 2)
        b = 1 + (1 - type2_center) * (kappa - 2)
        dist = beta(loc=0, a=a, b=b, scale=1)
    elif type2_dist == 'betaprime_mean_std':
        # type2_center = mean, type2_noise = SD
        b = 2 + (type2_center * (type2_center + 1)) / type2_noise ** 2
        a = type2_center * (b - 1)
        dist = betaprime(loc=0, a=a, b=b, scale=1)
    elif type2_dist == 'gamma_mode_std':
        # type2_center = mode, type2_noise = SD
        u = (type2_center + np.sqrt(type2_center * type2_center + 4 * type2_noise ** 2)) / (2 * type2_noise)
        dist = gamma(loc=0, a=u**2, scale=type2_noise / u)
    elif type2_dist == 'gamma_mean_std':
        # type2_center = mean, type2_noise = SD
        dist = gamma(loc=0, a=(type2_center / type2_noise) ** 2, scale=(type2_noise ** 2) / type2_center)
    elif type2_dist == 'gamma_mean':
        # type2_center = mean, type2_noise != SD
        dist = gamma(loc=0, a=type2_center / type2_noise, scale=type2_noise)
    elif type2_dist == 'gamma_mean_cv':
        # Corresponds to the CASSANDRE setup, i.e. standard multiplicative noise
        # type2_center = center, type2_noise = CV (relative noise)
        a = 1 / (type2_noise ** 2)
        dist = gamma(loc=0, a=a, scale=type2_center / a)
    elif type2_dist == 'gamma_mode':
        # type2_center = mode; type2_noise != SD
        dist = gamma(loc=0, a=(type2_center / type2_noise) + 1, scale=type2_noise)
    elif type2_dist.startswith('truncated_'):
        if type2_noise_type == 'report':
            if type2_dist == 'truncated_normal_mode_std':
                # type2_center = mode, type2_noise = SD of the truncated distribution
                sigma = _params_truncnorm_mode_std(low=0, high=1, x=type2_center, noise=type2_noise)
                dist = truncnorm(a=-type2_center / sigma, b=(1 - type2_center) / sigma, loc=type2_center, scale=sigma)
            elif type2_dist == 'truncated_normal_mode':
                # type2_center = mode, type2_noise = SD of the *un*truncated distribution
                dist = truncnorm(a=-type2_center / type2_noise, b=(1 - type2_center) / type2_noise, loc=type2_center, scale=type2_noise)
            elif type2_dist == 'truncated_gumbel_mode_std':
                # type2_center = mode, type2_noise = SD of the truncated distribution
                dist = TruncatedGumbelModeSD(mode=type2_center, noise=type2_noise, upper=1, n_newton=3)
            elif type2_dist == 'truncated_gumbel_mode':
                # type2_center = mode, type2_noise = SD of the *un*truncated distribution
                dist = TruncatedGumbelMode(mode=type2_center, noise=type2_noise * np.sqrt(6) / np.pi, upper=1)
            elif type2_dist == 'truncated_lognormal_mode_std':
                # type2_center = mode, type2_noise = SD
                dist = TruncatedLognormalModeSD(mode=type2_center, noise=type2_noise, upper=1, n_newton=5)
            elif type2_dist == 'truncated_lognormal':
                # type2_center = median of the untruncated lognormal, type2_noise = SD in log space
                dist = TruncatedLognormal(median_untrunc=type2_center, noise=type2_noise, upper=1)
            elif type2_dist == 'truncated_lognormal_mode':
                # type2_center = mode, type2_noise = SD in log space
                if type2_noise > 25:
                    # warnings.warn(f'Type 2 noise for a {type2_dist} distribution is too high. Capping at 25')
                    type2_noise = 25
                dist = TruncatedLognormalMode(mode=type2_center, noise=type2_noise, b=1)
            elif type2_dist == 'truncated_lognormal_mean':
                # type2_center = mean, type2_noise = SD in log space
                dist = TruncatedLognormalMean(mean=type2_center, noise=type2_noise, upper=1)
        elif type2_noise_type in ('readout', 'temperature'):
            if type2_dist == 'truncated_normal_mode_std':
                # type2_center = mode, type2_noise = SD
                sigma = _params_truncnorm_mode_std(low=0, high=np.inf, x=type2_center, noise=type2_noise)
                dist = truncnorm(a=-type2_center / sigma, b=np.inf, loc=type2_center, scale=sigma)
            elif type2_dist == 'truncated_normal_mode':
                # type2_center = mode, type2_noise != SD
                dist = truncnorm(a=-type2_center / type2_noise, b=np.inf, loc=type2_center, scale=type2_noise)
            elif type2_dist == 'truncated_gumbel_mode_std':
                # type2_center = mode, type2_noise = SD
                dist = TruncatedGumbelModeSD(mode=type2_center, noise=type2_noise, upper=np.inf, n_newton=3)
            elif type2_dist == 'truncated_gumbel_mode':
                # type2_center = mode, type2_noise != SD
                dist = TruncatedGumbelMode(mode=type2_center, noise=type2_noise * np.sqrt(6) / np.pi, upper=np.inf)

        else:
            raise ValueError(f"'{type2_noise_type}' is an unknown type 2 noise type")

    return dist  # noqa


def get_likelihood(x, type2_dist, type2_center, type2_noise, binsize_meta=1e-3, logarithm=False):
    """
    Helper function to get the likelihood mass within type2_center ± binsize_meta for a type 2 noise distribution.
    """
    dist = get_type2_dist(type2_dist=type2_dist, type2_center=type2_center, type2_noise=type2_noise)
    likelihood = dist.cdf(x + binsize_meta) - dist.cdf(x - binsize_meta)
    return np.log(np.maximum(1e-4, likelihood)) if logarithm else likelihood