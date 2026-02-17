from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
import warnings
from matplotlib.ticker import FormatStrFormatter
from scipy.special import ndtr
from scipy.stats import sem

from remeta.simulation import simulate, Simulation
from remeta.model import Configuration
from remeta.util import _check_param, listlike, fmp

color_generative_type2 = np.array([231, 168, 116]) / 255
color_generative_type2b = np.array([47, 158, 47]) / 255
color_data = [0.6, 0.6, 0.6]

color_inc = [0.87, 0.27, 0.33]
color_cor = [0, 0.67, 0.5]

color_model = (0.55, 0.55, 0.69)
color_model_wrong = np.array([152, 75, 75]) / 255

symbols = dict(
    type1_noise=r'$\sigma_1$',
    type1_noise_heteroscedastic=r'$\sigma_\mathrm{1,1}$',
    type1_thresh=r'$\vartheta_1$',
    type1_bias=r'$\delta_1$',
    type2_noise=r'$\sigma_2$',
    type2_evidence_bias=r'$\varphi_2$',
    type2_confidence_bias=r'$\alpha_2$',
    type2_criteria=r'$\gamma^\mathrm{i}$'
)

FONTSIZE = dict(
    label=14,
    xlabel=14,
    ylabel=14,
    tick=11,
    xtick=11,
    ytick=11,
    title=14
)


class _LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(_LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):  # noqa
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self.text_props)
        # title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title


def _logistic(x, sigma, thresh, bias):
    beta = np.pi / (np.sqrt(3) * sigma)
    return \
        (np.abs(x) >= thresh) * (
                1 / (1 + np.exp(-beta * (x + bias)))) + \
        (np.abs(x) < thresh) * (1 / (1 + np.exp(-beta * bias)))


def _normal(x, sigma, thresh, bias):
    return \
        (np.abs(x) >= thresh) * ndtr((x + bias) / sigma) + \
        (np.abs(x) < thresh) * ndtr(bias / sigma)


def _linear(x, thresh, bias):
    y = (np.abs(x) > thresh) * (x - np.sign(x) * thresh) + bias
    return y



def _tanh(x, beta, thresh, offset):
    return \
        (np.abs(x) > thresh) * (
                (1 - offset) * np.tanh(beta * (x - np.sign(x) * thresh)) + np.sign(x) * offset) + \
        (np.abs(x) <= thresh) * np.sign(x) * offset


def plot_psychometric(
        stimuli_or_simulation: list[float] | np.typing.NDArray[float] | Simulation | None = None,
        choices: list[float] | np.typing.NDArray[float] | None = None,
        type1_noise: float | list[float] | None = None,
        type1_bias: float | None = None,
        type1_thresh: float | list[float] | None = None,
        stim_max: float | None = None,
        params: dict | None = None,
        type1_noise_dist: str = 'normal',
        cfg: Configuration | None = None,
        model_prediction: bool = False,
        model_only: bool = False,
        highlight_fit: bool = False,
        errorbars: bool = True,
        path_export: str | None = None
) -> None:
    """Plot psychometric function.

    Usage:
        **Psychometric curve for empirical data:**

        `plot_psychometric(stimuli, choices)`  # Data only

        `plot_psychometric(stimuli, choices, type1_noise, type1_bias, ...)`  # Data + model

        `plot_psychometric(stimuli, choices, params=params)`  # Data + model

        **Psychometric curve for a simulation instance: (`remeta.simulation.Simulation`):**

        `plot_psychometric(simulation)`  # (Simulated) Data

        `plot_psychometric(simulation, model_prediction=True)`  # (Simulated) Data + model

        **Model only:**

        `plot_psychometric(type1_noise=type1_noise, type1_bias=type1_bias, ...)`  # Model

        `plot_psychometric(params)`  # Model

    Args:
        stimuli_or_simulation: 1d stimulus array (normalized to [-1; 1]) or [Simulation][remeta.simulation.Simulation] object
        choices: 1d choice array
        type1_noise: Type 1 noise.
        type1_bias: Type 1 bias.
        type1_thresh: Type 1 threshold.
        stim_max: maximum stimulus intensity.
        params: instead of passing parameters as separate parameters, one can pass a parameter dictionary
        cfg: If a [Configuration][remeta.configuration.Configuration] instance is passed, checks are performed for
             `cfg.param_type1_bias.enable` and `cfg.param_type1_thresh.enable`
         model_prediction: whether to show model-predicted values for comparison
        model_only: Show the model prediction only (auto-set to True if no data are passed)
        highlight_fit: Emphasize the fit over the data
        errorbars: Whether to include error bars for empirical data (SD)
    """

    if type1_noise is not None or params is not None:
        model_prediction = True

    if stimuli_or_simulation is not None and choices is None:  # assume a dataset is passed
        cfg = stimuli_or_simulation.cfg
        stimuli = stimuli_or_simulation.stimuli
        choices = stimuli_or_simulation.choices
        if model_prediction:
            params = stimuli_or_simulation.params
            type1_noise = params['type1_noise']
            type1_bias = params['type1_bias'] if cfg.param_type1_bias.enable else None
            type1_thresh = params['type1_thresh'] if cfg.param_type1_thresh.enable else None
    else:
        if choices is None:
            model_only = True
        stimuli = None if model_only else stimuli_or_simulation
        if model_prediction:
            if params is None:
                if type1_noise is None:
                    raise ValueError('Model predictions requested, but type 1 noise is unspecified.')
                params = dict(type1_noise=type1_noise)
            else:
                type1_noise = params['type1_noise']
                if 'type1_bias' in params and not (cfg is not None and not cfg.param_type1_bias.enable):
                    type1_bias = params['type1_bias']
                if 'type1_thresh' in params and not (cfg is not None and not cfg.param_type1_thresh.enable):
                    type1_thresh = params['type1_thresh']

    if stim_max is None:
        stim_max = 1 if stimuli is None else np.max(np.abs(stimuli))

    xrange_neg = np.linspace(-stim_max, 0, 1000)
    xrange_pos = np.linspace(stim_max / 1000, stim_max, 1000)

    if model_prediction:
        type1_noise = _check_param(type1_noise)
        if (cfg is None and type1_thresh is not None) or (cfg is not None and cfg.param_type1_thresh.enable):
            params['type1_thresh'] = type1_thresh
            type1_thresh = _check_param(type1_thresh)
        else:
            type1_thresh = [0, 0]

        if (cfg is None and type1_bias is not None) or (cfg is not None and cfg.param_type1_bias.enable):
            params['type1_bias'] = type1_bias
            type1_bias = _check_param(type1_bias)
        else:
            type1_bias = [0, 0]

        if cfg is not None:
            type1_noise_dist = cfg.param_type1_noise.model
        cdf = _logistic if type1_noise_dist == 'logistic' else _normal
        posterior_neg = cdf(xrange_neg, type1_noise[0], type1_thresh[0], type1_bias[0])
        posterior_pos = cdf(xrange_pos, type1_noise[1], type1_thresh[1], type1_bias[1])


    fig = plt.figure(figsize=(6, 3.5))
    fig.subplots_adjust(bottom=0.2)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.5], wspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")
    plt.sca(ax)


    if not model_only:
        stimulus_ids = (stimuli > 0).astype(int)
        levels = np.unique(stimuli)
        choiceprob_neg = np.array([np.mean(choices[(stimuli == v) & (stimulus_ids == 0)] ==
                                           stimulus_ids[(stimuli == v) & (stimulus_ids == 0)])
                                   for v in levels[levels < 0]])
        choiceprob_pos = np.array([np.mean(choices[(stimuli == v) & (stimulus_ids == 1)] ==
                                           stimulus_ids[(stimuli == v) & (stimulus_ids == 1)])
                                   for v in levels[levels > 0]])
        if errorbars:
            choiceprob_neg_se = np.array([sem(choices[(stimuli == v) & (stimulus_ids == 0)] ==
                                               stimulus_ids[(stimuli == v) & (stimulus_ids == 0)])
                                       for v in levels[levels < 0]])
            choiceprob_pos_se = np.array([sem(choices[(stimuli == v) & (stimulus_ids == 1)] ==
                                               stimulus_ids[(stimuli == v) & (stimulus_ids == 1)])
                                       for v in levels[levels > 0]])
        else:
            choiceprob_neg_se, choiceprob_pos_se = np.nan, np.nan
        plt.errorbar(levels[levels < 0], 1-choiceprob_neg, yerr=choiceprob_neg_se, fmt='o', color=color_data, mec='k', ls='' if model_prediction else '-',
                     label=f'Data $S^-$ (mean{["", "±SE"][int(errorbars)]})', clip_on=False, zorder=11, alpha=(1, 0.2)[highlight_fit])
        if not model_prediction:
            plt.plot([levels[levels < 0][-1], levels[levels > 0][0]], [1-choiceprob_neg[-1], choiceprob_pos[0]], color=color_data)
        plt.errorbar(levels[levels > 0], choiceprob_pos, yerr=choiceprob_pos_se, fmt='s', color=color_data, mec='k', ls='' if model_prediction else '-',
                     label=f'Data $S^+$ (mean{["", "±SE"][int(errorbars)]})', clip_on=False, zorder=11, alpha=(1, 0.2)[highlight_fit])

    if model_prediction:
        plt.plot(xrange_neg, posterior_neg, '-', lw=(2, 5)[highlight_fit], color=color_model, clip_on=False,
                 zorder=(10, 12)[highlight_fit], label=f'Model')
        plt.plot(xrange_pos, posterior_pos, '-', lw=(2, 5)[highlight_fit], color=color_model, clip_on=False,
                 zorder=(10, 12)[highlight_fit])

    plt.plot([-stim_max, stim_max], [0.5, 0.5], 'k-', lw=0.5)
    plt.plot([0, 0], [-0.02, 1.02], 'k-', lw=0.5)

    ax.yaxis.grid('on', color=[0.9, 0.9, 0.9])
    plt.xlim((-stim_max, stim_max))
    plt.ylim((0, 1))
    plt.xlabel('Stimulus ($x$)')
    plt.ylabel('Choice probability $S^+$')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    if model_prediction:
        anot_type1 = []
        for i, (k, v) in enumerate(params.items()):
            # if (cfg is None and k in params) or (cfg is not None and getattr(cfg, f"enable_param_{k}")) and k.startswith('type1_'):
            if k.startswith('type1') and (cfg is None or (cfg is not None and getattr(cfg, f"param_{k}").enable)):
                if hasattr(v, '__len__'):
                    val = ', '.join([fmp(p) for p in v])
                    anot_type1 += [f"${symbols[k][1:-1]}=" + f"[{val}]$"]
                else:
                    anot_type1 += [f"${symbols[k][1:-1]}={fmp(v)}$"]
        plt.text(1.045, 0.1, r'Parameters:' + '\n' + '\n'.join(anot_type1), transform=ax.transAxes,
                 bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=1, pad=5), fontsize=11)
    set_fontsize(label='default', tick='default')

    # ax_leg.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=11, handlelength=1)
    ax_leg.legend(*ax.get_legend_handles_labels(), loc="upper left", fontsize=11, handlelength=1)

    if path_export is not None:
        plt.savefig(path_export, bbox_inches='tight', pad_inches=0.02)


def plot_stimulus_versus_confidence(
        stimuli_or_simulation: list[float] | np.typing.NDArray[float] | Simulation | None = None,
        confidence: list[float] | np.typing.NDArray[float] | None = None,
        choices: list[float] | np.typing.NDArray[float] | None = None,
        params: dict | None = None,
        type1_noise: float | list[float] | None = None,
        type1_bias: float | list[float] | None = None,
        type1_thresh: float | list[float] | None = None,
        type2_noise: float | None = None,
        type2_evidence_bias_mult: float | None = None,
        type2_criteria: float | list[float] | None = None,
        cfg: Configuration | None = None,
        stim_max: float | None = None,
        errorbar_type: str | None = 'SEM',
        probability_correct: bool = False,
        separate_by_accuracy: bool = False,
        model_prediction: bool = False,
        model_prediction_nsamples: int = 10000,
        model_only: bool = False,
        model_prediction_disable_type2_noise: bool = False,
        path_export: str | None = None
) -> None:
    """ Plot the relationship between stimulus levels and confidence.

    Usage:
        **Plot for empirical data:**

        `plot_stimulus_versus_confidence(stimuli, choices)`  # Data

        `plot_stimulus_versus_confidence(stimuli, choices, type1_noise, type2_noise, ...)`  # Data + Model

        `plot_stimulus_versus_confidence(stimuli, choices, params, ...)`  # Data + Model

        **Plot for simulation instance: (`remeta.simulation.Simulation`):**

        `plot_stimulus_versus_confidence(simulation)`  # (Simulated) Data

        `plot_stimulus_versus_confidence(simulation, model_prediction=True)`  # (Simulated) Data + model

        **Model only:**

        `plot_confidence_histogram(type1_noise=type1_noise, type2_noise=type2_noise, ...)`  # Model

        `plot_stimulus_versus_confidence(params)`  # Model

    Args:
        stimuli_or_simulation: 1d stimulus array (normalized to [-1; 1]) or [Simulation][remeta.simulation.Simulation] object
        confidence: 1d confidence array
        choices: 1d choice array
        params: pass parameters as a dictionary
        type1_noise: Type 1 noise.
        type1_bias: Type 1 bias.
        type1_thresh: Type 1 threshold.
        type2_noise: Metacognitive noise. Required
        type2_evidence_bias_mult: Multiplicative metacognitive bias
        type2_criteria: Confidence criteria
        cfg: Place holder - checking the configuration object is not yet implemented
        stim_max: maximum stimulus intensity
        errorbar_type: either None (disable), 'SD' (standard deviation) or 'SEM' (standard error)
        probability_correct: if True, convert confidence (range 0-1) to subjective probability correct (range 0.5-1),
        separate_by_accuracy: separate plots for correct and incorrect responses
        model_prediction: whether to show model-predicted values for comparison
        model_prediction_nsamples: number of samples used to generate model predictions
        model_only: Show the model prediction only (auto-set to True if no data are passed)
        model_prediction_disable_type2_noise: plot model prediction with ~0 metacognitive noise
    """

    if stimuli_or_simulation is not None and confidence is None:  # assume first parameter is a simulated dataset
        from copy import deepcopy
        cfg = deepcopy(stimuli_or_simulation.cfg)
        stimuli, confidence, choices = stimuli_or_simulation.stimuli, stimuli_or_simulation.confidence, stimuli_or_simulation.choices
        params = stimuli_or_simulation.params.copy()
    else:

        if separate_by_accuracy and choices is None and not model_only:
            raise ValueError('If separate_by_accuracy is True, choices must be passed.')

        if confidence is None or params is not None or (type1_noise is not None and type2_noise is not None):
            model_prediction = True
        stimuli = None if model_only else stimuli_or_simulation
        choices = None if model_only else choices
        if stimuli is None:
            model_only = True
        if model_prediction:
            if params is None:
                if type1_noise is None:
                    raise ValueError('Type 1 noise is unspecified.')
                if type2_noise is None:
                    raise ValueError('Type 2 noise is unspecified.')
                params = dict(type1_noise=type1_noise, type2_noise=type2_noise)
                for param in ('type1_bias', 'type1_thresh', 'type2_evidence_bias_mult', 'type2_criteria'):
                    if (value := eval(param)) is not None and not (cfg is not None and not getattr(cfg, f'param_{param}').enable):
                        params[param] = value
            else:
                params = params.copy()
                if 'type1_noise' not in params:
                    raise ValueError('Type 1 noise is unspecified.')
                if 'type2_noise' not in params:
                    raise ValueError('Type 2 noise is unspecified.')
                for param in ('type1_bias', 'type1_thresh', 'type2_evidence_bias_mult', 'type2_criteria'):
                    if param in params and (cfg is not None and not getattr(cfg, f'param_{param}').enable):
                        params.pop(param)

    if model_prediction and model_prediction_disable_type2_noise:
        params['type2_noise'] = 1e-5


    if probability_correct:
        confidence = (confidence + 1) / 2

    fig = plt.figure(figsize=(6, 3.5))
    fig.subplots_adjust(bottom=0.2)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.5], wspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")
    plt.sca(ax)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    if stim_max is None:
        stim_max = 1 if stimuli is None else np.max(np.abs(stimuli))


    if errorbar_type is not None:
        if errorbar_type == 'SD':
            errorfun = np.std
        elif errorbar_type == 'SEM':
            errorfun = sem
        else:
            raise ValueError(f"Unknown errorbar type '{errorbar_type}' (valid: 'SD', 'SEM')")

    chance_level_ref = 0.75 if probability_correct else 0.5
    if not separate_by_accuracy:
        plt.plot([-1.05*stim_max, 1.05*stim_max], [chance_level_ref, chance_level_ref], lw=1, color=[0.2, 0.2, 0.2],
                 ls=':', label='Theoretical value\nat chance level')
    plt.plot([0, 0], [0, 1], 'k-', lw=0.5)

    if not model_only:
        stim_levels = sorted(np.unique(stimuli))
        if separate_by_accuracy:
            accuracy = np.sign(stimuli) == np.sign(choices - 0.5)
            conf_av_inc = np.array([np.nan if (cnd :=(~accuracy & (stimuli == stim_level))).sum() == 0 else np.mean(confidence[cnd]) for stim_level in stim_levels])
            conf_av_cor = np.array([np.nan if (cnd :=(accuracy & (stimuli == stim_level))).sum() == 0 else np.mean(confidence[cnd]) for stim_level in stim_levels])
            if errorbar_type is not None:
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=RuntimeWarning)
                    conf_err_inc = np.array([np.nan if (cnd :=(~accuracy & (stimuli == stim_level))).sum() == 0 else errorfun(confidence[cnd]) for stim_level in stim_levels])
                    conf_err_cor = np.array([np.nan if (cnd :=(accuracy & (stimuli == stim_level))).sum() == 0 else errorfun(confidence[cnd]) for stim_level in stim_levels])
            else:
                conf_err_inc = np.zeros(len(stim_levels))
                conf_err_cor = np.zeros(len(stim_levels))
            plt.errorbar(
                stim_levels,
                conf_av_cor,
                yerr=None if errorbar_type is None else conf_err_cor,
                marker='o', markersize=5, mew=1, mec='k', color='None', ecolor=color_cor, mfc=color_cor, clip_on=False,
                elinewidth=1.5, capsize=5, label=f'Data (correct,\nmean{"" if errorbar_type is None else f"±{errorbar_type}"})'
            )
            plt.errorbar(
                stim_levels,
                conf_av_inc,
                yerr=None if errorbar_type is None else conf_err_inc,
                marker='o', markersize=5, mew=1, mec='k', color='None', ecolor=color_inc, mfc=color_inc, clip_on=False,
                elinewidth=1.5, capsize=5, label=f'Data (incorrect,\nmean{"" if errorbar_type is None else f"±{errorbar_type}"})'
            )
            conf_min_data = min(np.nanmin(conf_av_inc - conf_err_inc), np.nanmin(conf_av_cor - conf_err_cor))
        else:
            conf_av = np.array([np.mean(confidence[stimuli == stim_level]) for stim_level in stim_levels])
            if errorbar_type is not None:
                conf_err = np.array([errorfun(confidence[stimuli == stim_level]) for stim_level in stim_levels])
            else:
                conf_err = np.zeros(len(stim_levels))

            plt.errorbar(
                stim_levels,
                conf_av,
                ls='' if model_prediction else '-',
                yerr=None if errorbar_type is None else conf_err,
                marker='o', markersize=5, mew=1, mec='k', color=color_data, ecolor='k', mfc=color_data, clip_on=False,
                elinewidth=1.5, capsize=5, label=f'Data (mean{"" if errorbar_type is None else f"±{errorbar_type}"})'
            )
            conf_min_data = np.min(conf_av - conf_err)

    if model_prediction:

        # nlevels = 100
        # nrows_per_level = int(np.ceil(model_prediction_nsamples / nlevels / 2))
        # stim_min = stim_max / nlevels
        # levels = np.hstack((np.linspace(-stim_max, -stim_min, nlevels), np.linspace(stim_min, stim_max, nlevels)))
        # x_stim = np.tile(levels, (nrows_per_level, 1))
        #
        # y_decval = stimulus_to_decision_value(x_stim, params, return_only_decval=True)
        # c_conf = type1_evidence_to_confidence(
        #     z1_type1_evidence=np.abs(y_decval), y_decval=y_decval,
        #     **params
        # ).mean(axis=0)

        _nsamples = int(np.ceil(model_prediction_nsamples / 2))
        stim_min = stim_max / _nsamples
        levels = np.hstack((np.linspace(-stim_max, -stim_min, _nsamples), np.linspace(stim_min, stim_max, _nsamples)))
        # y_decval = stimulus_to_decision_value(levels, params, return_only_decval=True)
        # c_conf = type1_evidence_to_confidence(
        #     z1_type1_evidence=np.abs(y_decval), x_stim=levels,
        #     **params
        # )
        ds = simulate(
            nsubjects=100,
            params=params, cfg=cfg, custom_stimuli=levels, verbosity=False,
            stim_max=stim_max, squeeze=True, compute_stats=False,
            silence_warnings=True
        )
        c_conf = (ds.confidence + 1) / 2 if probability_correct else ds.confidence

        # from scipy.interpolate import UnivariateSpline

        if 'type1_bias' in params:
            if listlike(params['type1_bias']):
                warnings.warn(f'Stimulus-dependent bias is currently not supported for this plot.')
                bias = params['type1_bias'][0]
            else:
                bias = params['type1_bias']
        else:
            bias = 0

        # Constrain the spline in a way that it is exactly chance_level_ref for x_stim = -bias
        # We do this by weighting an additional datapoint at x_stim = -bias very highly
        # ind_levels_min = np.argmin(np.abs(levels + bias))
        # # insert data point at the position that is closest to already present values, while not
        # # preserving ordering
        # insert_idx = ind_levels_min + 1 if (levels[ind_levels_min] + bias < 0) else ind_levels_min
        # weights = np.insert(np.ones_like(c_conf), insert_idx, 1e8)
        # c_conf_aug = np.insert(c_conf, insert_idx, chance_level_ref)
        # levels_aug = np.insert(levels, insert_idx, -bias)
        # c_conf_final = UnivariateSpline(levels_aug, c_conf_aug, k=3, w=weights)(levels)

        from scipy.ndimage import gaussian_filter1d
        if separate_by_accuracy:
            accuracy = np.sign(ds.choices - 0.5) == np.sign(ds.stimuli)
            c_conf_inc_, c_conf_cor_ = c_conf.copy(), c_conf.copy()
            c_conf_inc_[accuracy] = np.nan
            c_conf_cor_[~accuracy] = np.nan
            with warnings.catch_warnings():
                warnings.simplefilter('ignore', category=RuntimeWarning)
                c_conf_inc_mean, c_conf_cor_mean = np.nanmean(c_conf_inc_, axis=0), np.nanmean(c_conf_cor_, axis=0)
            # Interpolate nans
            isnan_inc, isnan_cor = np.isnan(c_conf_inc_mean), np.isnan(c_conf_cor_mean)
            c_conf_inc_mean[isnan_inc] = np.interp(np.arange(2*_nsamples)[isnan_inc], np.arange(2*_nsamples)[~isnan_inc], c_conf_inc_mean[~isnan_inc])
            c_conf_cor_mean[isnan_cor] = np.interp(np.arange(2*_nsamples)[isnan_cor], np.arange(2*_nsamples)[~isnan_cor], c_conf_cor_mean[~isnan_cor])
            c_conf_inc = gaussian_filter1d(c_conf_inc_mean, sigma=model_prediction_nsamples/20)
            c_conf_cor = gaussian_filter1d(c_conf_cor_mean, sigma=model_prediction_nsamples/20)
            plt.plot(levels, c_conf_cor, '-', lw=2, color=color_cor, clip_on=False,
                     label='Model prediction\n(correct)', alpha=0.5)
                     # label='Model prediction' + (r"($\sigma_2=0$)" if model_prediction_disable_type2_noise else "")+'\n(correct)', alpha=0.5)
            plt.plot(levels, c_conf_inc, '-', lw=2, color=color_inc, clip_on=False,
                     label='Model prediction\n(incorrect)', alpha=0.5)
                     # label='Model prediction' + (r"($\sigma_2=0$)" if model_prediction_disable_type2_noise else "")+'\n(incorrect)', alpha=0.5)
            conf_min_model = min(min(c_conf_inc), min(c_conf_cor))
        else:
            c_conf_final = gaussian_filter1d(c_conf.mean(axis=0), sigma=model_prediction_nsamples/20)
            plt.plot(levels, c_conf_final, '-', lw=2, color=color_model, clip_on=False,
                     label=f'Model prediction')
                     # label=f'Model prediction' + (r"($\sigma_2=0$)" if model_prediction_disable_type2_noise else ""))
            conf_min_model = min(c_conf_final)

    plt.xlim(-1.05*stim_max, 1.05*stim_max)
    conf_min = conf_min_model if model_only else (min(conf_min_data, conf_min_model) if model_prediction else conf_min_data)
    step = 0.05 if probability_correct else 0.1
    ymin = (np.floor(conf_min / step) * step) if conf_min < chance_level_ref else chance_level_ref - step / 2
    plt.ylim(ymin, 1)
    plt.xlabel('Stimulus ($x$)')
    plt.ylabel('Subjective prob. correct' if probability_correct else 'Confidence ($C$)')

    if model_prediction:
        anot_type2 = []
        for i, (k, v) in enumerate(params.items()):
            if k.startswith('type2_'):
                if listlike(v):
                    val = ', '.join([fmp(p) for p in v])
                    anot_type2 += [f"${symbols[k][1:-1]}=" + f"[{val}]$"]
                else:
                    anot_type2 += [f"${symbols[k][1:-1]}={fmp(v)}$"]
        plt.text(1.045, 0.1-0.2*separate_by_accuracy, r'Type 2 parameters:' + '\n' + '\n'.join(anot_type2), transform=ax.transAxes,
                 bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=1, pad=5), fontsize=11)

    # plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=11, handlelength=1)
    ax_leg.legend(*ax.get_legend_handles_labels(), loc="upper left", fontsize=11, handlelength=1)

    set_fontsize(label='default', tick='default')
    if path_export is not None:
        plt.savefig(path_export, bbox_inches='tight', pad_inches=0.02)




def plot_confidence_histogram(
        confidence_or_simulation: list[float] | np.typing.NDArray[float] | Simulation | None = None,
        stimuli: list[float] | np.typing.NDArray[float] | None = None,
        choices: list[float] | np.typing.NDArray[float] | None = None,
        params: dict[str, float] | None = None,
        type1_noise: float | list[float] | None = None,
        type1_bias: float | list[float] | None = None,
        type1_thresh: float | list[float] | None = None,
        type2_noise: float | None = None,
        type2_evidence_bias_mult: float | None = None,
        type2_criteria: list[float] | None = None,
        cfg: Configuration | None = None,
        stim_max: float | None = None,
        probability_correct: bool = False,
        model_prediction: bool = False,
        model_only: bool = False,
        separate_by_category: bool = False,
        separate_by_accuracy: bool = False,
        model_prediction_nsamples: int = 10000,
        path_export: str | None = None
) -> None:
    """ Plot the relationship between stimulus levels and confidence.

    Usage:
        **Plot for empirical data:**

        `plot_confidence_histogram(confidence)`  # Data

        `plot_confidence_histogram(confidence, stimuli, choices, separate_by_accuracy=True)`  # Data

        `plot_confidence_histogram(confidence, type1_noise, type2_noise, ...)`  # Data + Model

        `plot_confidence_histogram(confidence, params, ...)`  # Data + Model

        **Plot for simulation instance: (`remeta.simulation.Simulation`):**

        `plot_confidence_histogram(simulation)`  # (Simulated) Data

        `plot_confidence_histogram(simulation, model_prediction=True)`  # (Simulated) Data + model

        **Model only:**

        `plot_confidence_histogram(type1_noise=type1_noise, type2_noise=type2_noise, ...)`  # Model

        `plot_confidence_histogram(params)`  # Model

    Args:
        confidence_or_simulation: 1d stimulus array (normalized to [-1; 1]) or [Simulation][remeta.simulation.Simulation] object
        stimuli: 1d stimulus array
        choices: 1d choice array
        params: pass parameters as a dictionary
        type1_noise: Type 1 noise.
        type1_bias: Type 1 bias.
        type1_thresh: Type 1 threshold.
        type2_noise: Metacognitive noise. Required
        type2_evidence_bias_mult: Multiplicative metacognitive bias
        type2_criteria: Confidence criteria
        cfg: Place holder - checking the configuration object is not yet implemented
        stim_max: float | None = None,
        probability_correct: if True, convert confidence (range 0-1) to subjective probability correct (range 0.5-1)
        model_prediction: whether to show model-predicted values for comparison
        model_only: Show the model prediction only (auto-set to True if no data are passed)
        separate_by_category: separate histograms for the two stimulus categories
        separate_by_accuracy: separate histograms for correct and incorrect responses
        model_prediction_nsamples: number of samples used to generate model predictions
    """

    if separate_by_accuracy:
        separate_by_category = False

    if model_only:
        model_prediction = True

    if isinstance(confidence_or_simulation, Simulation):
        from copy import deepcopy
        cfg = deepcopy(confidence_or_simulation.cfg)
        stimuli, confidence, choices = confidence_or_simulation.stimuli, confidence_or_simulation.confidence, confidence_or_simulation.choices
        params = confidence_or_simulation.params.copy()
    else:

        if params is not None or (type1_noise is not None and type2_noise is not None):
            model_prediction = True
        confidence = None if model_only else confidence_or_simulation
        stimuli = None if model_only else stimuli
        choices = None if model_only else choices
        if confidence is None:
            model_only = True

        if separate_by_accuracy and choices is None and not model_only:
            raise ValueError('If separate_by_accuracy is True, choices must be passed.')

        if model_prediction:
            if params is None:
                if type1_noise is None:
                    raise ValueError('Type 1 noise is unspecified.')
                if type2_noise is None:
                    raise ValueError('Type 2 noise is unspecified.')
                params = dict(type1_noise=type1_noise, type2_noise=type2_noise)
                for param in ('type1_bias', 'type1_thresh', 'type2_evidence_bias_mult', 'type2_criteria'):
                    if (value := eval(param)) is not None and not (cfg is not None and not getattr(cfg, f'param_{param}').enable):
                        params[param] = value
            else:
                params = params.copy()
                if 'type1_noise' not in params:
                    raise ValueError('Type 1 noise is unspecified.')
                if 'type2_noise' not in params:
                    raise ValueError('Type 2 noise is unspecified.')
                for param in ('type1_bias', 'type1_thresh', 'type2_evidence_bias_mult', 'type2_criteria'):
                    if param in params and (cfg is not None and not getattr(cfg, f'param_{param}').enable):
                        params.pop(param)


    if probability_correct:
        confidence = (confidence + 1) / 2

    if stim_max is None:
        stim_max = 1 if stimuli is None else np.max(np.abs(stimuli))

    fig = plt.figure(figsize=(6, 3.5))
    fig.subplots_adjust(bottom=0.2)
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 0.5], wspace=0)
    ax = fig.add_subplot(gs[0, 0])
    ax_leg = fig.add_subplot(gs[0, 1])
    ax_leg.axis("off")
    plt.sca(ax)
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.3g'))

    if not model_only:
        conf_levels = np.sort(np.unique(confidence))
        n_conf_levels = len(conf_levels)
        bins = np.linspace(0, 1, n_conf_levels+1) if n_conf_levels >= 8 else np.hstack((0, conf_levels[1:] - np.diff(conf_levels) / 2, 1))

        if separate_by_category:
            counts0 = np.histogram(confidence[np.sign(stimuli) == -1], bins=bins)[0]
            counts1 = np.histogram(confidence[np.sign(stimuli) == 1], bins=bins)[0]
            centers = 0.5 * (bins[:-1] + bins[1:])
            widths  = bins[1:] - bins[:-1]
            for i, (x, w, c0, c1) in enumerate(zip(centers, widths, counts0, counts1)):
                if c0 < c1:
                    plt.bar(x, c1, width=w, color=(0.8, 0.8, 0.8), ec='grey', align='center', zorder=1, label='Data ($S^+$)' if i == 0 else None)
                    plt.bar(x, c0, width=w, color=(0.4, 0.4, 0.4), ec='grey', align='center', zorder=2, label='Data ($S^-$)' if i == 0 else None)
                else:
                    plt.bar(x, c0, width=w, color=(0.4, 0.4, 0.4), ec='grey', align='center', zorder=1, label='Data ($S^-$)' if i == 0 else None)
                    plt.bar(x, c1, width=w, color=(0.8, 0.8, 0.8), ec='grey', align='center', zorder=2, label='Data ($S^+$)' if i == 0 else None)
        elif separate_by_accuracy:
            accuracy = np.sign(stimuli) == np.sign(choices - 0.5)
            counts_inc = np.histogram(confidence[~accuracy], bins=bins)[0]
            counts_cor = np.histogram(confidence[accuracy], bins=bins)[0]
            centers = 0.5 * (bins[:-1] + bins[1:])
            widths  = bins[1:] - bins[:-1]
            for i, (x, w, c0, c1) in enumerate(zip(centers, widths, counts_inc, counts_cor)):
                if c0 < c1:
                    plt.bar(x, c1, width=w, color=color_cor, ec='grey', align='center', zorder=1, label='Data (correct)' if i == 0 else None)
                    plt.bar(x, c0, width=w, color=color_inc, ec='grey', align='center', zorder=2, label='Data (incorrect)' if i == 0 else None)
                else:
                    plt.bar(x, c0, width=w, color=color_inc, ec='grey', align='center', zorder=1, label='Data (incorrect)' if i == 0 else None)
                    plt.bar(x, c1, width=w, color=color_cor, ec='grey', align='center', zorder=2, label='Data (correct)' if i == 0 else None)
        else:
            plt.hist(confidence, bins=bins, color=(0.8, 0.8, 0.8), edgecolor='grey', clip_on=False, label='Data')

    if model_prediction:
        if model_only:
            _nsamples = int(np.ceil(model_prediction_nsamples / 2))
            bins = np.linspace(0, 1, 5)
        else:
            _nsamples = len(confidence)
        stim_min = stim_max / _nsamples
        levels = np.hstack((np.linspace(-stim_max, -stim_min, _nsamples), np.linspace(stim_min, stim_max, _nsamples)))
        # y_decval = stimulus_to_decision_value(levels, params, return_only_decval=True)
        # c_conf = type1_evidence_to_confidence(
        #     z1_type1_evidence=np.abs(y_decval), x_stim=levels,
        #     **params
        # )
        nsubjects = 100 if model_only else 500
        ds = simulate(
            nsubjects=nsubjects,
            params=params, cfg=cfg, custom_stimuli=levels, verbosity=False,
            stim_max=stim_max, squeeze=True, compute_stats=False,
            silence_warnings=True
        )
        c_conf = (ds.confidence + 1) / 2 if probability_correct else ds.confidence

        if separate_by_category:
            counts0_ = [np.histogram(c_conf[s][np.sign(ds.stimuli[s]) == -1], bins=bins)[0] for s in range(nsubjects)]
            counts1_ = [np.histogram(c_conf[s][np.sign(ds.stimuli[s]) == 1], bins=bins)[0] for s in range(nsubjects)]
            counts0 = np.array([(counts0_[s] / (2*_nsamples if model_only else counts0_[s].sum())) * (1 if model_only else (stimuli < 0).sum()) for s in range(nsubjects)])
            counts1 = np.array([(counts1_[s] / (2*_nsamples if model_only else counts1_[s].sum())) * (1 if model_only else (stimuli > 0).sum()) for s in range(nsubjects)])
            plt.errorbar(
                bins[:-1] + np.diff(bins) / 2 - 0.04, counts0.mean(axis=0),
                yerr=np.percentile(counts0, 97.5, axis=0) - np.percentile(counts0, 2.5, axis=0),
                mew=2, elinewidth=2, capsize=7, capthick=2, fmt='o', markersize=8, mec=color_model, mfc=(0.4, 0.4, 0.4),
                color=color_model, lw=2, label='Model prediction ($S^-$)'
            )
            plt.errorbar(
                bins[:-1] + np.diff(bins) / 2 + 0.04, counts1.mean(axis=0),
                yerr=np.percentile(counts1, 97.5, axis=0) - np.percentile(counts1, 2.5, axis=0),
                mew=2, elinewidth=2, capsize=7, capthick=2, fmt='o', markersize=8, mec=color_model, mfc=(0.8, 0.8, 0.8),
                color=color_model, lw=2, label='Model prediction ($S^+$)'
            )
        elif separate_by_accuracy:
            acc = [np.sign(ds.stimuli[s]) == np.sign(ds.choices[s] - 0.5) for s in range(nsubjects)]
            counts_inc_ = [np.histogram(c_conf[s][~acc[s]], bins=bins)[0] for s in range(nsubjects)]
            counts_cor_ = [np.histogram(c_conf[s][acc[s]], bins=bins)[0] for s in range(nsubjects)]
            print(_nsamples, counts_inc_[0].sum()+counts_cor_[0].sum())
            counts_inc = np.array([(counts_inc_[s] / (2*_nsamples if model_only else counts_inc_[s].sum())) * (1 if model_only else (~accuracy).sum())for s in range(nsubjects)])
            counts_cor = np.array([(counts_cor_[s] / (2*_nsamples if model_only else counts_cor_[s].sum())) * (1 if model_only else accuracy.sum()) for s in range(nsubjects)])
            plt.errorbar(
                bins[:-1] + np.diff(bins) / 2 - 0.04, counts_inc.mean(axis=0),
                yerr=np.percentile(counts_inc, 97.5, axis=0) - np.percentile(counts_inc, 2.5, axis=0),
                mew=2, elinewidth=2, capsize=7, capthick=2, fmt='o', markersize=8, mec=color_model, mfc=color_inc,
                color=color_model, lw=2, label='Model prediction\n(incorrect)'
            )
            plt.errorbar(
                bins[:-1] + np.diff(bins) / 2 + 0.04, counts_cor.mean(axis=0),
                yerr=np.percentile(counts_cor, 97.5, axis=0) - np.percentile(counts_cor, 2.5, axis=0),
                mew=2, elinewidth=2, capsize=7, capthick=2, fmt='o', markersize=8, mec=color_model, mfc=color_cor,
                color=color_model, lw=2, label='Model prediction\n(correct)'
            )
        else:
            counts_ = [np.histogram(c_conf[s], bins=bins)[0] for s in range(nsubjects)]
            counts = np.array([(counts_[s] / (2*_nsamples)) * (1 if model_only else len(confidence)) for s in range(nsubjects)])
            plt.errorbar(
                bins[:-1] + np.diff(bins) / 2, counts.mean(axis=0),
                yerr=np.percentile(counts, 97.5, axis=0) - np.percentile(counts, 2.5, axis=0),
                mew=2, elinewidth=2, capsize=7, capthick=2, fmt='o', markersize=8, mec=color_model, mfc=(0.8, 0.8, 0.8),
                color=color_model, lw=2, label='Model prediction'
            )


    plt.xticks(bins)
    plt.xlabel('Confidence')
    plt.ylabel('Probability' if model_only else 'Count')
    plt.xlim(0, 1)

    if model_prediction:
        anot_type2 = []
        cfg = None  # keep in case cfg is implemented in the future
        for i, (k, v) in enumerate(params.items()):
            if k.startswith('type2_'):
                if listlike(v):
                    val = ', '.join([fmp(p) for p in v])
                    anot_type2 += [f"${symbols[k][1:-1]}=" + f"[{val}]$"]
                else:
                    anot_type2 += [f"${symbols[k][1:-1]}={fmp(v)}$"]
        plt.text(1.055, 0.1-0.2*(separate_by_accuracy | separate_by_category), r'Type 2 parameters:' + '\n' + '\n'.join(anot_type2), transform=ax.transAxes,
                 bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=1, pad=5), fontsize=11)

    # plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=11, handlelength=1)
    ax_leg.legend(*ax.get_legend_handles_labels(), loc="upper left", fontsize=11, handlelength=1)

    set_fontsize(label='default', tick='default')

    if path_export is not None:
        plt.savefig(path_export, bbox_inches='tight', pad_inches=0.02)


def set_fontsize(label=None, xlabel=None, ylabel=None, tick=None, xtick=None, ytick=None, title=None):

    fig = plt.gcf()

    for ax in fig.axes:
        if xlabel is not None:
            ax.xaxis.label.set_size(FONTSIZE['xlabel'] if xlabel == 'default' else xlabel)
        elif label is not None:
            ax.xaxis.label.set_size(FONTSIZE['label'] if label == 'default' else label)
        if ylabel is not None:
            ax.yaxis.label.set_size(FONTSIZE['ylabel'] if ylabel == 'default' else ylabel)
        elif label is not None:
            ax.yaxis.label.set_size(FONTSIZE['label'] if label == 'default' else label)

        if xtick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(FONTSIZE['xtick'] if xtick == 'default' else xtick)
        elif tick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(FONTSIZE['tick'] if tick == 'default' else tick)
        if ytick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(FONTSIZE['ytick'] if ytick == 'default' else ytick)
        elif tick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(FONTSIZE['tick'] if tick == 'default' else tick)

        if title is not None:
            ax.title.set_fontsize(FONTSIZE['title'] if title == 'default' else title)


if __name__ == '__main__':

    import remeta
    np.random.seed(42)  # make notebook reproducible
    import warnings
    warnings.filterwarnings('error')


    # x_stim, d_dec, c_conf, params, y_decval = remeta.load_dataset(
    # 'type2_simple', return_params=True, return_y_decval=True
    # )
    # remeta.plot_evidence_versus_confidence(x_stim, c_conf, y_decval, params, plot_bias_free=True)

    remeta.plot_psychometric(type1_noise=[0.5, 0.2])