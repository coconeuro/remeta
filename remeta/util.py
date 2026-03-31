import sys
import warnings

import numpy as np
from scipy.stats import rankdata
from scipy.linalg import cho_factor, cho_solve
from copy import deepcopy
from dataclasses import fields, MISSING


TAB = '    '
SP2 = '  '
maxfloat = np.float128 if hasattr(np, 'float128') else np.longdouble
_slsqp_epsilon = np.sqrt(np.finfo(float).eps)  # scipy's default value for the SLSQP epsilon parameter


class Struct:
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.__class__.__name__}[attributes: ' + ', '.join([k for k in self.__dict__.keys()]) + ']'


class ReprMixin:
    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f'{self.__class__.__name__}\n' + '\n'.join([f'\t{k}: {v}' for k, v in self.__dict__.items()])


# def create_struct_with_reprmixin(class_name):
#     return type(class_name, (ReprMixin,), {})()

class Stats(ReprMixin):
    pass

def print_class_instance(instance, attr_class_only=(), attr_replace_string=None):
    txt = f'{instance.__class__.__name__}'
    for k, v in instance.__dict__.items():
        if k in attr_class_only:
            txt += f"\n\t{k}: {v.__class__.__name__}"
        elif attr_replace_string is not None and k in attr_replace_string:
            txt += f"\n\t{k}: {attr_replace_string[k]}"
        elif isinstance(v, dict):
            txt += f"\n\t{k}: { {k_: np.array2string(np.array(v_), precision=4, threshold=20, separator=', ') for k_, v_ in v.items()} }"
        elif isinstance(v, list):
            if isinstance(v[0], dict):
                txt += f'\n\t{k}:[\n'
                for i in range(min(5, len(v))):
                    txt += '\t\t{' + ', '.join([f"'{k_}': {np.array2string(np.array(v_), precision=4, separator=', ')}" for k_, v_ in v[i].items()]) + '}\n'
                if len(v) > 10:
                    txt += '\t\t[...]\n'
                if len(v) > 5:
                    for i in range(max(-5, -len(v)), 0):
                        txt += '\t\t{' + ', '.join([f"'{k_}': {np.array2string(np.array(v_), precision=4, separator=', ')}" for k_, v_ in v[i].items()]) + '}\n'
                txt += '\t]'
            elif isinstance(v[0], float):
                txt += f"\n\t{k}: {np.array2string(np.array(v), precision=4, threshold=50, separator=', ')}"
        elif isinstance(v, np.ndarray):
            txt += f"\n\t{k}: {np.array2string(np.array(v), precision=4, threshold=50, separator=', ')}"
        else:
            txt += f"\n\t{k}: {v}"
    # print(txt)
    return txt

def reset_dataclass_on_init(cls):
    """
    For a @dataclass:
      - On each new instance, reset:
          * non-field class attributes
          * dataclass fields that have a direct default (not default_factory)
        back to the values they had at class definition time.
    """

    # ---- 1) Capture non-field class attributes (true class attrs) ----
    dataclass_field_names = set(cls.__dataclass_fields__.keys())

    original_class_attrs = {}
    for name, value in cls.__dict__.items():
        if name.startswith("__") and name.endswith("__"):
            continue
        if name in dataclass_field_names:
            # handled separately as dataclass fields
            continue
        if callable(value):
            continue

        original_class_attrs[name] = deepcopy(value)

    # ---- 2) Capture dataclass field defaults (for __init__ defaults + class attrs) ----
    dc_fields = list(fields(cls))

    # fields that appear as parameters with defaults in __init__
    fields_with_any_default = []
    field_default_kinds = []       # "value" or "factory"
    stored_field_defaults = []     # original defaults or factories

    for f in dc_fields:
        if f.default is not MISSING or f.default_factory is not MISSING:
            fields_with_any_default.append(f)
            if f.default_factory is not MISSING:
                field_default_kinds.append("factory")
                stored_field_defaults.append(f.default_factory)
            else:
                field_default_kinds.append("value")
                stored_field_defaults.append(deepcopy(f.default))

    # For fields with a *value* default, there is usually a class attribute with that name.
    # We'll reset that class attribute too.
    field_class_default_indices = {}
    for idx, (f, kind, stored) in enumerate(
        zip(fields_with_any_default, field_default_kinds, stored_field_defaults)
    ):
        if kind == "value" and hasattr(cls, f.name):
            field_class_default_indices[f.name] = idx

    # ---- 3) Wrap the original __init__ ----
    original_init = cls.__init__

    # Sanity: number of __init__ defaults should match number of fields with defaults
    orig_defaults_len = len(original_init.__defaults__ or ())
    if orig_defaults_len != len(fields_with_any_default):
        # You *can* turn this into an assert if you prefer it to fail loudly
        pass

    def __init__(self, *args, **kwargs):
        # 3a) reset non-field class attributes
        for name, value in original_class_attrs.items():
            setattr(cls, name, deepcopy(value))

        # 3b) reset class attributes for dataclass fields with direct defaults
        for fname, idx in field_class_default_indices.items():
            setattr(cls, fname, deepcopy(stored_field_defaults[idx]))

        # 3c) rebuild the original __init__ defaults for this call
        if fields_with_any_default:
            new_defaults = []
            for kind, stored in zip(field_default_kinds, stored_field_defaults):
                if kind == "factory":
                    # default_factory fields keep their original factory
                    new_defaults.append(stored)
                else:
                    # direct defaults get a fresh deep copy
                    new_defaults.append(deepcopy(stored))
            original_init.__defaults__ = tuple(new_defaults)

        # Call the real dataclass-generated __init__
        original_init(self, *args, **kwargs)

    cls.__init__ = __init__
    return cls


def listlike(x):
    return np.array(x).ndim > 0


def empty_list(n, *shapes_or_None):
    if (len(shapes_or_None) == 1) and shapes_or_None[0] is None:
        return [None] * n
    else:
        return [np.empty([shape[i] if listlike(shape) else shape for shape in shapes_or_None]) for i in range(n)]


def fmp(v, k=3):
    # 1. Format to exactly k decimal places
    s = f"{v:.{k}f}"

    # 2. If it ends in k zeros (e.g. .000 for k=3), strip it all (for cases like 1.000 -> 1)
    if s.endswith('.' + '0'*k):
        return s[:-(k+1)]

    # 1.110 -> 1.11
    orig_str = str(v)
    if '.' in orig_str:
        decimal_places = len(orig_str.split('.')[1])
        # If original has <k decimals, keep it as is.
        # If original has k+ decimals, truncate to k.
        precision = min(decimal_places, k)
        return f"{v:.{precision}f}"

    return str(int(v))


def _check_param(x):
    if hasattr(x, '__len__'):
        if len(x) == 2:
            return x
        elif len(x) == 1:
            return [x[0], x[0]]
        else:
            print(f'Something went wrong, parameter array has length {len(x)}')
    else:
        return [x, x]


def _check_criteria(x):
    if hasattr(x[0], '__len__'):
        return x
    else:
        return [x, x]


def pearson2d(x, y, axis=-1):
    assert axis in (-2, -1, 0, 1), 'axis must be one of -2, -1, 0, 1'
    x, y = np.asarray(x), np.asarray(y)
    mx, my = np.nanmean(x, axis=axis), np.nanmean(y, axis=axis)
    if axis in (0, -2):
        xm, ym = x - mx, y - my
    else:
        xm, ym = x - mx[..., None], y - my[..., None]
    r_num = np.nansum(xm * ym, axis=axis)
    r_den = np.sqrt(np.nansum(xm ** 2, axis=axis) * np.nansum(ym ** 2, axis=axis))
    r_den[np.isclose(r_den, 0)] = np.nan
    r = r_num / r_den
    return r


def spearman2d(x, y, axis=-1):
    assert axis in (-2, -1, 0, 1), 'axis must be one of -2, -1, 0, 1'
    x, y = np.asarray(x), np.asarray(y)
    xr, yr = rankdata(x, axis=axis), rankdata(y, axis=axis)
    mxr, myr = np.nanmean(xr, axis=axis), np.nanmean(yr, axis=axis)
    if axis in (0, -2):
        xmr, ymr = xr - mxr, yr - myr
    else:
        xmr, ymr = xr - mxr[..., None], yr - myr[..., None]
    r_num = np.nansum(xmr * ymr, axis=axis)
    r_den = np.sqrt(np.nansum(xmr ** 2, axis=axis) * np.nansum(ymr ** 2, axis=axis))
    r_den[np.isclose(r_den, 0)] = np.nan
    r = r_num / r_den
    return r


# def cov_from_hessian(H, symmetrize=True):
#     Hs = 0.5 * (H + H.T) + 1e-12 * np.eye(len(H)) if symmetrize else H + 1e-12 * np.eye(len(H))
#     c, lower = cho_factor(Hs, lower=True, check_finite=False)  # raises LinAlgError if not SPD
#     cov = cho_solve((c, lower), np.eye(Hs.shape[0]), check_finite=False)
#     # numerical cleanup: enforce symmetry
#     cov = 0.5 * (cov + cov.T)
#     return cov

def cov_from_hessian(H, symmetrize=True):
    Hs = 0.5 * (H + H.T) if symmetrize else np.asarray(H, dtype=float)
    Hs = Hs + 1e-12 * np.eye(Hs.shape[0])

    try:
        c, lower = cho_factor(Hs, lower=True, check_finite=False)
        cov = cho_solve((c, lower), np.eye(Hs.shape[0]), check_finite=False)
    except np.linalg.LinAlgError:
        from remeta.fit import ridge_hessian
        Hr = ridge_hessian(Hs)
        if Hr is None:
            return None
        c, lower = cho_factor(Hr, lower=True, check_finite=False)
        cov = cho_solve((c, lower), np.eye(Hr.shape[0]), check_finite=False)

    # numerical cleanup: enforce symmetry
    cov = 0.5 * (cov + cov.T)

    return cov


def se_from_cov(cov):
    var = np.diag(cov)
    se = np.sqrt(var)
    return se


def compute_cov_criteria(cov_full, idx_crit):
    cov_gaps = cov_full[np.ix_(idx_crit, idx_crit)]
    J = np.tril(np.ones((len(idx_crit), len(idx_crit))))
    cov_crit = J @ cov_gaps @ J.T
    return cov_crit


def compute_criterion_bias(criteria, cov_crit):
    """Idea: compute criterion bias as a weighted sum of differences from Bayes-optiomal criteria.
       The weights are the uncertainty estimates (SEs) of the criteria; we slightly improve on this
       by also considering the correlation structure between criterion uncertainties, i.e. the full covariance
       matrix.
    """

    k = len(criteria) + 1
    crit_bayes = np.arange(1/k, 1-1e-10, 1/k)
    diff = criteria - crit_bayes
    one = np.ones_like(criteria)

    c, lower = cho_factor(cov_crit)
    # numer = (multivariate) weighted sum of criterion differences
    numer = one @ cho_solve((c, lower), diff)
    # denom = (multivariate) sum of all weights
    denom = one @ cho_solve((c, lower), one)
    bias_crit = numer / denom
    bias_crit_se = np.sqrt(1 / denom)

    return bias_crit, bias_crit_se

def compute_choice_bias(stimuli, choices, smooth=0.5):
    # levels = np.sort(np.unique(np.abs(stimuli)))
    # ntrials = len(stimuli)

    # Unique signed levels
    x_levels = np.unique(stimuli)
    magnitudes = np.unique(np.abs(x_levels))
    magnitudes = magnitudes[magnitudes > 0]

    # Aggregate counts
    k = {}
    n = {}
    for x in x_levels:
        mask = (stimuli == x)
        n[x] = mask.sum()
        k[x] = choices[mask].sum()

    D_list, w_list = [], []

    for m in magnitudes:
        if m in k and -m in k:
            # empirical proportions
            p_plus = k[m] / n[m]
            p_minus = k[-m] / n[-m]
            # smoothed proportions for variance
            p_plus_var = (k[m] + smooth) / (n[m] + 2*smooth)
            p_minus_var = (k[-m] + smooth) / (n[-m] + 2*smooth)
            var_plus = p_plus_var * (1 - p_plus_var) / n[m]
            var_minus = p_minus_var * (1 - p_minus_var) / n[-m]
            D = p_plus + p_minus - 1
            var_D = var_plus + var_minus
            if var_D <= 0:
                continue
            w = 1.0 / var_D
            D_list.append(D)
            w_list.append(w)

    D_arr = np.array(D_list)
    w_arr = np.array(w_list)

    bias_hat = np.sum(w_arr * D_arr) / np.sum(w_arr)

    return bias_hat


def _pav_isotonic(y, w=None):
    """
    Pool-Adjacent-Violators (PAV) for isotonic regression (nondecreasing).
    Returns fitted values yhat with same length as y.
    """
    y = np.asarray(y, dtype=float)
    n = y.size
    if w is None:
        w = np.ones(n, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        if w.shape != y.shape:
            raise ValueError("w must have same shape as y")

    # Each point starts as its own block
    v = y.copy()
    ww = w.copy()
    start = np.arange(n)
    end = np.arange(n)

    # Stack of block indices
    m = 0
    for i in range(n):
        start[m] = i
        end[m] = i
        v[m] = y[i]
        ww[m] = w[i]
        m += 1

        # Merge while monotonicity violated
        while m >= 2 and v[m-2] > v[m-1]:
            new_w = ww[m-2] + ww[m-1]
            new_v = (ww[m-2]*v[m-2] + ww[m-1]*v[m-1]) / new_w
            ww[m-2] = new_w
            v[m-2] = new_v
            end[m-2] = end[m-1]
            m -= 1

    # Expand blocks
    yhat = np.empty(n, dtype=float)
    for b in range(m):
        yhat[start[b]:end[b]+1] = v[b]
    return yhat


def compute_choice_bias_horizontal_old(stimuli, choices, smooth=0.5):
    """
    Horizontal (stimulus-units) bias estimator using:
      - symmetry deviations across matched ±m levels
      - inverse-variance weighting from binomial sampling
      - nonparametric slope estimates from isotonic-smoothed proportions

    Parameters
    ----------
    stimuli : (T,) array-like of floats
        Signed stimulus intensities.
    choices : (T,) array-like of ints
        Binary choices: 0=negative category, 1=positive category.
    smooth : float
        Jeffreys-style smoothing for variance stability in Var(p).

    Returns
    -------
    mu_hat : float
        Estimated horizontal bias (PSE shift) in stimulus units.
        Positive mu_hat means the curve is shifted right (harder to choose 1).
        (Sign convention depends on defining p(x)=P(choice=1|x).)
    """

    # Aggregate by unique signed intensity
    x_levels = np.unique(stimuli)
    x_levels.sort()
    n = np.zeros_like(x_levels, dtype=int)
    k = np.zeros_like(x_levels, dtype=int)

    for i, x in enumerate(x_levels):
        mask = (stimuli == x)
        n[i] = int(mask.sum())
        k[i] = int(choices[mask].sum())

    p_hat = k / n

    # Smoothed p for variance stability
    p_var = (k + smooth) / (n + 2.0 * smooth)
    var_p = p_var * (1.0 - p_var) / n

    # Monotone smoothing (optional but very helpful) via isotonic regression
    # Weighted by n to respect unequal trial counts
    p_iso = _pav_isotonic(p_hat, w=n.astype(float))

    # Slope estimate at each x_level using finite differences on isotonic fit
    slope = np.empty_like(p_iso, dtype=float)
    if x_levels.size == 1:
        raise ValueError("Need more than one stimulus level to estimate slope.")

    # One-sided at edges, central inside
    slope[0] = (p_iso[1] - p_iso[0]) / (x_levels[1] - x_levels[0])
    slope[-1] = (p_iso[-1] - p_iso[-2]) / (x_levels[-1] - x_levels[-2])
    for i in range(1, x_levels.size - 1):
        slope[i] = (p_iso[i+1] - p_iso[i-1]) / (x_levels[i+1] - x_levels[i-1])

    # Build matched ±m pairs
    # Map x -> index for quick lookup
    idx = {float(x): i for i, x in enumerate(x_levels)}

    mags = np.unique(np.abs(x_levels))
    mags = mags[mags > 0]  # exclude 0

    D_list, s_list, varD_list, w_list, m_list = [], [], [], [], []

    for m in mags:
        mp = float(m)
        mm = float(-m)
        if mp in idx and mm in idx:
            ip = idx[mp]
            im = idx[mm]

            # Symmetry deviation uses empirical p_hat (could also use p_iso; choose empiric here)
            Dm = p_hat[ip] + p_hat[im] - 1.0

            # Sampling variance for Dm from (smoothed) binomial variance
            varDm = var_p[ip] + var_p[im]
            if varDm <= 0:
                continue

            # Slope estimate at ±m from isotonic fit; average them
            sm = 0.5 * (slope[ip] + slope[im])

            # If slope is ~0, this magnitude carries little info about a horizontal shift
            # (we can safely drop it)
            if sm == 0 or not np.isfinite(sm):
                continue

            wm = 1.0 / varDm

            D_list.append(Dm)
            s_list.append(sm)
            w_list.append(wm)

    D = np.asarray(D_list, dtype=float)
    s = np.asarray(s_list, dtype=float)
    w = np.asarray(w_list, dtype=float)

    # Weighted LS solution for Dm ≈ -2 s_m mu
    num = np.sum(w * s * D)
    den = np.sum(w * s * s)

    mu_hat = num / (2.0 * den)

    return float(mu_hat)



def compute_choice_bias_horizontal(stimuli, choices, smooth=0.5, eps=1e-12):
    """
    Stimulus-space choice bias estimated nonparametrically using ALL stimulus levels with inverse-variance weights.

    Parameters
    ----------
    stimuli : (T,) array-like
        Signed stimulus intensities (numeric).
    choices : (T,) array-like
        Binary choices: 0 = negative category, 1 = positive category.
    smooth : float, default 0.5
        Jeffreys-style smoothing for variance stability (used in Var estimates only).
    eps : float, default 1e-12
        Floors tiny variances to avoid infinite weights.

    Returns
    -------
    bias_hat : float
        Estimated bias in stimulus units (PSE = x where fitted p(x)=0.5).
        If your logistic is p(x)=sigmoid(beta*(x - mu)), then bias_hat ≈ mu.
        Returns +inf/-inf if 0.5 is not bracketed by the fitted curve.
    """
    stimuli = np.asarray(stimuli, dtype=float)
    choices = np.asarray(choices, dtype=int)

    if stimuli.ndim != 1 or choices.ndim != 1:
        raise ValueError("stimuli and choices must be 1D arrays.")
    if stimuli.size != choices.size:
        raise ValueError("stimuli and choices must have the same length.")
    if not np.all((choices == 0) | (choices == 1)):
        raise ValueError("choices must be binary (0/1).")

    # Aggregate by unique intensity levels
    x_levels, inv = np.unique(stimuli, return_inverse=True)
    order = np.argsort(x_levels)
    x_levels = x_levels[order]
    inv = order[inv]  # remap to sorted order indices

    n = np.bincount(inv)
    k = np.bincount(inv, weights=choices).astype(float)
    if x_levels.size < 2:
        raise ValueError("Need at least two distinct stimulus levels.")

    p_hat = k / n

    # Sampling variance (binomial), with smoothing to avoid var=0 at p=0 or 1
    p_var = (k + smooth) / (n + 2.0 * smooth)
    var_p = p_var * (1.0 - p_var) / n
    w = 1.0 / np.maximum(var_p, eps)  # inverse-variance weights

    # Nonparametric monotone psychometric using IV weights (uses ALL levels)
    p_iso = _pav_isotonic(p_hat, w=w)

    # If 0.5 not bracketed, PSE lies outside tested range
    if np.all(p_iso < 0.5):
        return float(np.inf)
    if np.all(p_iso > 0.5):
        return float(-np.inf)

    # If there's a plateau exactly at 0.5, return its midpoint in x
    eq = (p_iso == 0.5)
    if np.any(eq):
        xs = x_levels[eq]
        return float(0.5 * (xs.min() + xs.max()))

    # Find the crossing interval (monotone => should be unique)
    # We find the first index i where p_iso[i] >= 0.5, then interpolate with i-1.
    i_hi = int(np.argmax(p_iso >= 0.5))
    i_lo = i_hi - 1
    if i_lo < 0:
        return float(-np.inf)

    x0, x1 = x_levels[i_lo], x_levels[i_hi]
    p0, p1 = p_iso[i_lo], p_iso[i_hi]
    if p1 == p0:
        return float(0.5 * (x0 + x1))

    t = (0.5 - p0) / (p1 - p0)
    bias = float(x0 + t * (x1 - x0))
    return -bias



def print_warnings(w):
    if len(w):
        print('\tWarnings that occured during model estimation:')
    for el in set([w_.message.args[0] for w_ in w]):
        if 'delta_grad == 0.0' not in el:
            print('\t\tWarning: ' + el)


def raise_warning_in_catch_block(msg, category, w):
    warnings.warn(msg, category=category)
    if len(w):
        sys.stderr.write(warnings.formatwarning(
            w[-1].message, w[-1].category, w[-1].filename, w[-1].lineno, line=w[-1].line
        ))

# def type2roc(correct, conf, nbins=5):
#     # Calculate area under type 2 ROC
#     #
#     # correct - vector of 1 x ntrials, 0 for error, 1 for correct
#     # conf - vector of continuous confidence ratings between 0 and 1
#     # nbins - how many bins to use for discretization
#
#     bs = 1 / nbins
#     h2, fa2 = np.full(nbins, np.nan), np.full(nbins, np.nan)
#     for c in range(nbins):
#         if c:
#             h2[nbins - c - 1] = np.sum((conf > c*bs) & (conf <= (c+1)*bs) & correct.astype(bool)) + 0.5
#             fa2[nbins - c - 1] = np.sum((conf > c*bs) & (conf <= (c+1)*bs) & ~correct.astype(bool)) + 0.5
#         else:
#             h2[nbins - c - 1] = np.sum((conf >= c * bs) & (conf <= (c + 1) * bs) & correct.astype(bool)) + 0.5
#             fa2[nbins - c - 1] = np.sum((conf >= c * bs) & (conf <= (c + 1) * bs) & ~correct.astype(bool)) + 0.5
#
#     h2 /= np.sum(h2)
#     fa2 /= np.sum(fa2)
#     cum_h2 = np.hstack((0, np.cumsum(h2)))
#     cum_fa2 = np.hstack((0, np.cumsum(fa2)))
#
#     k = np.full(nbins, np.nan)
#     for c in range(nbins):
#         k[c] = (cum_h2[c+1] - cum_fa2[c])**2 - (cum_h2[c] - cum_fa2[c+1])**2
#
#     auroc2 = 0.5 + 0.25*np.sum(k)
#
#     return auroc2

def discretize_confidence_with_bounds(x, bounds):
    confidence = np.full(x.shape, np.nan)
    bounds = np.hstack((bounds, np.inf))
    for i, b in enumerate(bounds[:-1]):
        confidence[(bounds[i] <= x) & (x < bounds[i + 1])] = i + 1
    return confidence


def check_linearity(stimuli, choices, difficulty_levels=None, method=None, verbosity=0, **kwargs):

    """
    The `stimuli` variable passed to ReMeta should encode stimulus evidence *in interval scale*. Interval scale means
    that 1) identical increments of the stimulus variable anywhere along the stimulus axis should correspond to
    identical increments in evidence; and 2), the value 0 should indicate the absence of any evidence. This helper
    method helps visualize the linearity of the current data.

    It is recommended to do linearization on the entire group data and thus also to pass stimuli, choices (and
    optionally difficulty_levels) as flattened 1d group arrays or n_subjects x n_trials 2d arrays.

    Args:
        stimuli: 1d (n_samples) or 2d (n_subjects x n_samples) stimulus array
            It is recommended to use the data of the entire group!
            If stimuli is binary, difficulty_levels must be passed, otherwise the absolute value
            is assumed to encode difficulty / stimulus magnitude.
        choices: 1d (n_samples) or 2d (n_subjects x n_samples) choice array.
            It is recommended to use the data of the entire group!
            Choices must be encoded as 0/1 or -1/+1.
        difficulty_levels: 1d (n_samples) or 2d (n_subjects x n_samples) difficulty level array
            It is recommended to use the data of the entire group!
            Must be passed if the stimuli array is binary. Should encode difficulty or stimulus magnitude.
        method: method used for the call to `remeta.linearize_stimulus_evidence()`.
            It is recommended to keep this at None.
            None sets method='exact' if at most 10 difficulty levels and >=200 samples per difficulty level;
            otherwise method='discretize_linear'.
        verbosity: verbosity level passed to fit_type1() and linearize_stimulus_evidence()
        kwargs: parameters passed to linearize_stimulus_evidence()
    """

    stim_ids = sorted(np.unique(stimuli))

    if difficulty_levels is None:
        if len(stim_ids) <= 2:
            raise ValueError('Stimulus variable seems binary and no difficulty levels passed -> cannot compute gradual '
                             'stimulus values')
        difficulty_levels = np.abs(stimuli)
        stimuli = np.sign(stimuli)
    else:
        if len(stim_ids) > 2:
            raise ValueError('Stimuli should have exactly two values')
        if (stim_ids[0] != -1) or (stim_ids[1] != 1):
            warnings.warn('Stimuli are not in a -1/+1 format. Hence, the smaller stimulus value is converted to'
                          '-1 and the larger to +1.')
            stimuli[stimuli == stim_ids[0]] = -1
            stimuli[stimuli == stim_ids[1]] = 1

    levels_orig = np.sort(np.unique(difficulty_levels))
    n_levels = len(levels_orig)
    n_samples = len(stimuli)

    if method is None:
        method = 'exact' if (n_levels <= 10) and (n_samples / n_levels >= 200) else 'discretize_linear'
    stimuli_linear = linearize_stimulus_evidence(stimuli, choices, difficulty_levels, method=method,
                                                 verbosity=verbosity, **kwargs)

    if (choice_ids := tuple(sorted(np.unique(choices)))) != (0, 1):
        choices[choices == choice_ids[0]] = 0
        choices[choices == choice_ids[1]] = 1
    accuracy = (np.sign(stimuli) == np.sign(choices - 0.5)).astype(int)
    increasing_orig = np.polyfit(range(n_levels), [accuracy[difficulty_levels == level].mean()
                                                   for level in levels_orig],1)[0] > 0

    import remeta
    cfg = remeta.Configuration()
    cfg.param_type1_noise.bounds[1] = np.inf
    rem_orig = remeta.ReMeta(cfg)
    if increasing_orig:
        stimuli_orig = stimuli * difficulty_levels
        rem_orig.fit_type1(stimuli_orig / stimuli_orig.max(), choices, verbosity=verbosity, silence_warnings=True)
        result_orig = rem_orig.summary()
        title_orig = rf'Original: $\text{{AIC}} = {result_orig.summary().type1.aic:.1f}$'

    rem_linear = remeta.ReMeta(cfg)
    rem_linear.fit_type1(stimuli_linear, choices, verbosity=verbosity, silence_warnings=True)
    result_linear = rem_linear.summary()
    title_linear = rf'Linearized: $\text{{AIC}} = {result_linear.summary().type1.aic:.1f}$'

    levels_linear = [np.abs(stimuli_linear[difficulty_levels == level])[0] for level in levels_orig]

    import matplotlib.pyplot as plt
    plt.figure(figsize=(9 if increasing_orig else 6, 3))
    plt.subplot(1, 2+increasing_orig, 1)
    plt.plot([0, 1], [0, 1], 'k-', label='Perfect\nlinearity')
    plt.plot(levels_orig / np.max(levels_orig), levels_linear, label='Empirical')
    plt.xlabel('Evidence (original)', fontsize=13)
    plt.ylabel('Evidence (linearized)', fontsize=13)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend(fontsize=8, handlelength=1)

    if increasing_orig:
        ax = plt.subplot(1, 2+increasing_orig, 2)
        rem_orig.plot_psychometric(axis_mode=True)
        plt.text(0.05, 0.85, rf'$\sigma_1 = {result_orig.params["type1_noise"]:.3f}$',
                 bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=1, pad=2), transform=ax.transAxes)
        plt.title(title_orig, fontsize=11)

    ax = plt.subplot(1, 2+increasing_orig, 2+increasing_orig)
    rem_linear.plot_psychometric(axis_mode=True)
    plt.title(title_linear, fontsize=11)
    plt.text(0.05, 0.85, rf'$\sigma_1 = {result_linear.params["type1_noise"]:.3f}$',
             bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=1, pad=2), transform=ax.transAxes)

    plt.tight_layout()


def linearize_stimulus_evidence(stimuli, choices, difficulty_levels=None, method='auto',
                                rolling_size='auto', rolling_auto_nsamples=200, discretize_nlevels=10,
                                noise_model='normal', force_monotonic=True, type1_noise_bounds=(0.001, np.inf),
                                verbosity=1):
    """
    Linearize stimulus evidence. The `stimuli` variable passed to ReMeta should encode stimulus evidence *in interval
    scale*. Interval scale means that 1) identical increments of the stimulus variable anywhere along the stimulus axis
    should correspond to identical increments in evidence; and 2), the value 0 should indicate the absence of any
    evidence. This helper method transforms stimuli to interval-scale (signed) stimulus evidence.

    To check if linearization is necessary, use the helper method remeta.check_linearity().
    When *not* to use:
        If stimulus magnitude (i.e. |stimuli|) already encodes a signal-to-noise ratio, or if the stimulus "noise
        denominator" is constant and stimulus magnitude mainly encodes a continuous numerator (e.g., offset angle from a
        reference, motion coherence in percent)
    In all other cases it is sensible to linearize the input.

    It is recommended to perform linearization on the entire group data. To this aim, stimuli, choices (and optionally
    difficulty_levels) can be passed as flattened 1d arrays or n_subjects x n_trials 2d arrays. The idea is that
    non-linearity is mainly a property of the stimuli and the general architecture of the human brain, and less of
    individual participants. Moreover, only when linearizing stimuli in an identical manner for the entire group,
    type 1 parameters can be meaningfully compared between participants.

    The present linearization method estimates type 1 noise (sigma_1) along the stimulus magnitude dimension while
    setting the "signal" to 1 in each case. The original stimuli are then transformed to a signed signal-to-noise ratio
    in the form 1/sigma_1.

    Args:
        stimuli: 1d (n_samples) or 2d (n_subjects x n_samples) stimulus array
            It is recommended to use the data of the entire group!
            If stimuli is binary, difficulty_levels must be passed, otherwise the absolute value
            is assumed to encode difficulty / stimulus magnitude.
        choices: 1d (n_samples) or 2d (n_subjects x n_samples) choice array.
            It is recommended to use the data of the entire group!
            Choices must be encoded as 0/1 or -1/+1.
        difficulty_levels: 1d (n_samples) or 2d (n_subjects x n_samples) difficulty level array
            It is recommended to use the data of the entire group!
            Must be passed if the stimuli array is binary. Should encode difficulty or stimulus magnitude.
        method: 'auto', 'exact', 'rolling', 'discretize_linear' or 'discretize_quantile'
             'auto': 'exact' if at most 10 difficulty levels and >=200 samples per difficulty level; else 'rolling'.
             'exact': process each difficulty level separately (recommended if each difficulty level has >=200 samples)
             'rolling': Linearization is performed within a rolling window of size `rolling_size`.
             'discretize_linear'/'discretize_quantile': The difficulty dimension is divided in `discretize_nlevels`
             bins either in a linear (equidistant) or a quantile-based (equinumerous) manner; linearization is
             performed for each bin, although some within-bin differentiation is maintained by means of subsequent
             inter/extrapolation.
        rolling_size: Window size for discretization 'rolling'. In case of 'auto', the window is chosen such that there
                      are around `rolling_auto_nsamples` samples within a window.
        rolling_auto_nsamples: Sample size for discretization 'rolling' and rolling_size 'auto'. Windows size is
                               adaptively chosen such that the sample size for each fit is at least
                               `rolling_auto_nsamples`.
        discretize_nlevels: Number of difficulty bins for methods 'auto' / 'discretize_linear' / 'discretize_quantile'.
        noise_model: 'normal' (default) or 'logistic'
            Noise model used for linearization.
        force_monotonic:
            Enforce monotonicity via np.maximum.accumulate / np.minimum.accumulate.
            Only necessary in case of a monotonicity violation. Set False to detect such a violation.

    Returns:
        stimuli_linear: Linearized stimulus array, normalized to [-1; 1].

    """
    stim_ids = sorted(np.unique(stimuli))

    if difficulty_levels is None:
        if len(stim_ids) <= 2:
            raise ValueError('Stimulus variable seems binary and no difficulty levels passed -> cannot compute gradual '
                             'stimulus values')
        difficulty_levels = np.abs(stimuli)
        stimuli = np.sign(stimuli)
    else:
        if len(stim_ids) > 2:
            raise ValueError('Stimuli should have exactly two values')
        if (stim_ids[0] != -1) or (stim_ids[1] != 1):
            warnings.warn('Stimuli are not in a -1/+1 format. Hence, the smaller stimulus value is converted to'
                          '-1 and the larger to +1.')
            stimuli[stimuli == stim_ids[0]] = -1
            stimuli[stimuli == stim_ids[1]] = 1

    levels = np.sort(np.unique(difficulty_levels))

    n_levels = len(levels)
    n_samples = len(stimuli)

    if method == 'auto':
        method = 'exact' if (n_levels <= 10) and (n_samples / n_levels >= 200) else 'rolling'
    if method == 'exact':
        difficulty_levels_final = difficulty_levels
        levels_final = levels
        if verbosity:
            print(f'Samples per difficulty level:')
            for i, level in enumerate(levels_final):
                print(f'\tLevel {i + 1} [{level:.4g}]: {np.sum(difficulty_levels_final == level)} samples')
    else:
        if method == 'rolling':
            from scipy.stats import rankdata
            difficulty_levels_final = rankdata(difficulty_levels, method='dense') - 1
            levels_final = levels
        else:
            if method == 'discretize_linear':
                edges = np.linspace(np.min(levels), np.max(levels), discretize_nlevels + 1)
            elif method == 'discretize_quantile':
                edges = np.quantile(levels, np.linspace(0, 1, discretize_nlevels + 1))
            centers = edges[:-1] + np.diff(edges) / 2
            difficulty_levels_final = np.digitize(difficulty_levels, edges + np.hstack((-1, np.zeros(len(edges) - 2), 1)))
            levels_final = np.linspace(1, discretize_nlevels, discretize_nlevels).astype(int)
            if verbosity:
                print(f'Samples per difficulty level:')
                for i, level in enumerate(levels_final):
                    print(f'\tLevel {i + 1} [Center {centers[i]:.4g}]: {np.sum(difficulty_levels_final == level)} samples')

    stimuli_new = np.zeros_like(stimuli, dtype=float)
    params = np.full(len(levels_final), np.nan)
    import remeta
    cfg = remeta.Configuration()
    cfg.param_type1_noise.model = noise_model
    cfg.param_type1_noise.bounds = list(type1_noise_bounds)
    rem = remeta.ReMeta(cfg)
    if method == 'rolling':
        for i in range(len(levels_final)):
            if (verbosity == 2) and (np.mod(i + 1, 100) == 0):
                print(f'Computing rolling window {i + 1} / {n_levels}')
            j = 0
            if rolling_size == 'auto':
                while True:
                    if (np.sum(np.isin(difficulty_levels_final, range(max(0, i - j), min(n_levels, i + j + 1))))
                            >= rolling_auto_nsamples):
                        break
                    if j > n_levels:
                        raise ValueError('Two few samples for method `rolling`.')
                    j += 1
                cnd = np.isin(difficulty_levels_final, range(max(0, i - j), min(n_levels, i + j + 1)))
            else:
                left = (rolling_size - 1) // 2
                start = min(max(0, i - left), n_levels - rolling_size)
                stop = start + rolling_size
                cnd = np.isin(difficulty_levels_final, range(start, stop))
            rem.fit_type1(
                stimuli[cnd].flatten(),
                choices[cnd].flatten(),
                verbosity=0,
                silence_warnings=True
            )
            params[i] = rem.summary().params['type1_noise']
        params = fit_monotone_smooth(params, sigma=n_levels/10)
    else:
        for i, level in enumerate(levels_final):
            if len(stimuli[difficulty_levels_final == level]):
                rem.fit_type1(
                    stimuli[difficulty_levels_final == level].flatten(),
                    choices[difficulty_levels_final == level].flatten(),
                    verbosity=0,
                    silence_warnings=True
                )
                params[i] = rem.summary().params['type1_noise']

    valid = ~np.isnan(params)
    params_delta = np.diff(params[valid])
    is_monotonic = np.all(params_delta >= 0) or np.all(params_delta <= 0)
    # decreasing = higher stimulus magnitude ~ lower sensitivity (type1_noise)
    decreasing = np.polyfit(range(len(params[valid])), params[valid],1)[0] < 0
    if not is_monotonic:
        if not force_monotonic:
            raise ValueError('Sensitivity does not increase or decrease monotonically. Check your difficulty levels, '
                             'reduce discretize_nlevels or make sure that force_monotonic=True')
        else:
            # warnings.warn('Sensitivity does not increase or decrease monotonically. Enforcing monotonicity as per '
            #               'force_monotonic=True.')
            params[valid] = np.minimum.accumulate(params[valid]) if decreasing else np.maximum.accumulate(params[valid])
    if method == 'exact':
        for i, level in enumerate(levels_final):
            if valid[i]:
                stimuli_new[difficulty_levels_final == level] = stimuli[difficulty_levels_final == level] / params[i]
            elif len(stimuli[difficulty_levels_final == level]):
                raise ValueError('This should not happen.')
    else:
        if method == 'rolling':
            stimuli_new = np.full(stimuli.shape, fill_value=np.nan)
            for i in range(n_levels):
                cnd = difficulty_levels_final == i
                stimuli_new[cnd] = stimuli[cnd] / params[i]
        else:
            stimuli_new = stimuli * pchip_interp_monotone_extrap(centers[valid], 1 / params[valid], difficulty_levels)

    stimuli_linear = stimuli_new / np.max(np.abs(stimuli_new))

    if verbosity:
        result_old = None
        if decreasing:
            stimuli_old = stimuli * difficulty_levels
            rem.fit_type1(stimuli_old / stimuli_old.max(), choices, verbosity=0, silence_warnings=True)
            result_old = rem.summary()
            print(f'Before linearization: type1_noise = {result_old.params["type1_noise"]:.4f}, AIC = {result_old.summary().type1.aic:.1f}')

        else:
            print(f'Before linearization: <unavailable, as difficulty levels are coded inversely to stimulus magnitude>')

        rem.fit_type1(stimuli_linear, choices, verbosity=0, silence_warnings=True)
        result_new = rem.summary()
        print(f'After linearization: type1_noise = {result_new.params["type1_noise"]:.4f}, AIC = {result_new.summary().type1.aic:.1f}')

        if result_old is not None:
            if result_new.summary().type1.aic < result_old.summary().type1.aic:
                print(f'\t-> Linearization improves the model fit.')
            else:
                warnings.warn(f'\t-> Linearization impairs the model fit. This is unexpected and might indicate '
                              f'incorrect usage of the linearization method.')

    return stimuli_linear


def fit_monotone_smooth(y, sigma=2):
    """
    Return a smooth, monotonic fit to a 1D array.

    Parameters
    ----------
    y : array-like
        1D data values.
    sigma : float, default=2
        Gaussian smoothing width. Larger = smoother.

    Returns
    -------
    y_fit : ndarray
        Smooth monotonic fitted values at the original points.
    """

    from sklearn.isotonic import IsotonicRegression
    from scipy.ndimage import gaussian_filter1d

    increasing = np.polyfit(range(len(y)), y, 1)[0] > 0

    iso = IsotonicRegression(increasing=increasing, out_of_bounds="clip")
    y_mono = iso.fit_transform(np.arange(len(y)), y)

    y_fit = gaussian_filter1d(y_mono, sigma=sigma, mode="nearest")

    return y_fit


def pchip_interp_monotone_extrap(xknown, yknown, xnew):

    # PchipInterpolator for interpolation and boundary derivative for extrapolation
    # Pchip = Piecewise cubic Hermite interpolation (preserves monotony)

    from scipy.interpolate import PchipInterpolator

    order = np.argsort(xknown)
    xknown = xknown[order]
    yknown = yknown[order]

    f = PchipInterpolator(xknown, yknown, extrapolate=False)
    ynew = f(xnew)

    left_slope = f.derivative()(xknown[0])
    right_slope = f.derivative()(xknown[-1])

    mask_left = xnew < xknown[0]
    mask_right = xnew > xknown[-1]

    ynew[mask_left] = yknown[0] + left_slope * (xnew[mask_left] - xknown[0])
    ynew[mask_right] = yknown[-1] + right_slope * (xnew[mask_right] - xknown[-1])

    return ynew


def print_dataset_characteristics(sim):
    print('----------------------------------')
    if sim.cfg.skip_type2:
        print('..Generative parameters:')
        print(f'{TAB}Type 1 noise distribution: {sim.cfg.param_type1_noise.model}')
        for p, v in sim.params_type1.items():
            print(f'{TAB}{p}: {np.array2string(np.array(v), precision=3)}')
    else:
        print('..Generative model:')
        print(f'{TAB}Type 1 noise distribution: {sim.cfg.param_type1_noise.model}')
        print(f'{TAB}Type 2 noise type: {sim.cfg.type2_noise_type}')
        print(f'{TAB}Type 2 noise distribution: {sim.cfg.param_type2_noise.model}')
        print('..Generative parameters:')
        for p, v in sim.params.items():
            print(f'{TAB}{p}: {np.array2string(np.array(v), precision=3)}')
        if sim.params_extra is not None:
            if 'type2_criteria_bias' in sim.params_extra:
                print(f"{TAB}{TAB}[extra] Criterion bias: {sim.params_extra['type2_criteria_bias']:.4f}")
            if 'type2_criteria_confidence_bias' in sim.params_extra:
                print(f"{TAB}{TAB}[extra] Criterion-based confidence bias: {sim.params_extra['type2_criteria_confidence_bias']:.4f}")
            # if 'type2_criteria_absdev' in sim.params_extra:
            #     print(f"{TAB}{TAB}[extra] Criterion absolute deviation: {sim.params_extra['type2_criteria_absdev']:.4f}")
    print('..Descriptive statistics:')
    print(f'{TAB}No. subjects: {sim.nsubjects}')
    print(f"{TAB}No. samples: {np.array2string(np.array(sim.nsamples).squeeze(), separator=', ', threshold=3)}")
    if sim.type1_stats is not None:
        print(f"{TAB}Accuracy: {100 * sim.type1_stats['accuracy']:.1f}% correct")
        print(f"{TAB}d': {sim.type1_stats['dprime']:.1f}")
        # print(f"{TAB}Choice bias: {('-', '+')[int(sim.type1_stats['choice_bias'] > 0.5)]}{100*np.abs(sim.type1_stats['choice_bias'] - 0.5):.1f}%")
        print(f"{TAB}Choice bias: {100*sim.type1_stats['choice_bias']:.1f}%")
    if not sim.cfg.skip_type2 and sim.type2_stats is not None:
        print(f"{TAB}Confidence: {sim.type2_stats['confidence']:.2f}")
        print(f"{TAB}M-Ratio: {sim.type2_stats['mratio']:.2f}")
        print(f"{TAB}AUROC2: {sim.type2_stats['auroc2']:.2f}")
    print('----------------------------------')
