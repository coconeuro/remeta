import sys
import warnings

import numpy as np
from scipy.stats import rankdata
from copy import deepcopy
from dataclasses import fields, MISSING


TAB = '    '
SP2 = '  '
maxfloat = np.float128 if hasattr(np, 'float128') else np.longdouble
_slsqp_epsilon = np.sqrt(np.finfo(float).eps)  # scipy's default value for the SLSQP epsilon parameter


class Struct:
    pass


class ReprMixin:
    def __repr__(self):
        return f'{self.__class__.__name__}\n' + '\n'.join([f'\t{k}: {v}' for k, v in self.__dict__.items()])


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


def print_warnings(w):
    for el in set([w_.message.args[0] for w_ in w]):
        if 'delta_grad == 0.0' not in el:
            print('\tWarning: ' + el)


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
        for p, v in sim.params_type1.items():
            print(f'{TAB}{p}: {np.array2string(np.array(v), precision=3)}')
    else:
        print('..Generative model:')
        print(f'{TAB}Type 2 noise type: {sim.cfg.type2_noise_type}')
        print(f'{TAB}Type 2 noise distribution: {sim.cfg.type2_noise_dist}')
        print('..Generative parameters:')
        for p, v in sim.params.items():
            print(f'{TAB}{p}: {np.array2string(np.array(v), precision=3)}')
        if sim.params_extra is not None:
            if 'type2_criteria_absolute' in sim.params_extra:
                print(f'{TAB}Type 2 criteria (absolute): [{", ".join([f"{c:.5g}" for c in sim.params_extra['type2_criteria_absolute']])}]')
            if 'type2_criteria_bias' in sim.params_extra:
                print(f'{TAB}Criterion bias: {sim.params_extra['type2_criteria_bias']:.5g}')
    print('..Descriptive statistics:')
    print(f'{TAB}No. subjects: {sim.nsubjects}')
    print(f'{TAB}No. samples: {sim.nsamples}')
    if sim.type1_stats is not None:
        print(f'{TAB}Performance: {100 * sim.type1_stats['accuracy']:.1f}% correct')
        print(f"{TAB}Choice bias: {('-', '+')[int(sim.type1_stats['choice_bias'] > 0.5)]}{100*np.abs(sim.type1_stats['choice_bias'] - 0.5):.1f}%")
    if not sim.cfg.skip_type2 and sim.type2_stats is not None:
        print(f'{TAB}Confidence: {sim.type2_stats['confidence']:.2f}')
        print(f'{TAB}M-Ratio: {sim.type2_stats['mratio']:.2f}')
        print(f'{TAB}AUROC2: {sim.type2_stats['auroc2']:.2f}')
    print('----------------------------------')

if __name__ == '__main__':
    empty_list(2, 3, [3, 4])[0].shape