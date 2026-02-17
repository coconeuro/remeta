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


def create_struct_with_reprmixin(class_name):
    return type(class_name, (ReprMixin,), {})()


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


def cov_from_hessian(H, symmetrize=True):
    Hs = 0.5 * (H + H.T) + 1e-12 * np.eye(len(H)) if symmetrize else H + 1e-12 * np.eye(len(H))
    c, lower = cho_factor(Hs, lower=True, check_finite=False)  # raises LinAlgError if not SPD
    cov = cho_solve((c, lower), np.eye(Hs.shape[0]), check_finite=False)
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
