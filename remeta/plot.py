import matplotlib.pyplot as plt
import matplotlib.text as mtext
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FormatStrFormatter
from scipy.stats import gaussian_kde, sem

from .model import Configuration
from .gendata import simu_data
from .type2_dist import get_type2_dist, get_likelihood
from .transform import type1_evidence_to_confidence
from .util import _check_param


color_logistic = (0.55, 0.55, 0.69)
color_generative_type2 = np.array([231, 168, 116]) / 255
color_generative_type2b = np.array([47, 158, 47]) / 255
color_data = [0.6, 0.6, 0.6]

color_model = np.array([57, 127, 95]) / 255
color_model_wrong = np.array([152, 75, 75]) / 255

symbols = dict(
    type1_noise=r'$\sigma_\mathrm{s}$',
    type1_noise_heteroscedastic=r'$\sigma_\mathrm{s,1}$',
    type1_thresh=r'$\vartheta_\mathrm{s}$',
    type1_bias=r'$\delta_\mathrm{s}$',
    type2_noise=r'$\sigma_\mathrm{m}$',
    type2_evidence_bias_mult=r'$\varphi_\mathrm{m}$',
    type2_criteria=r'$c_\mathrm{i}$'
)


class LegendTitle(object):
    def __init__(self, text_props=None):
        self.text_props = text_props or {}
        super(LegendTitle, self).__init__()

    def legend_artist(self, legend, orig_handle, fontsize, handlebox):  # noqa
        x0, y0 = handlebox.xdescent, handlebox.ydescent
        title = mtext.Text(x0, y0, orig_handle, **self.text_props)
        # title = mtext.Text(x0, y0, r'\underline{' + orig_handle + '}', usetex=True, **self.text_props)
        handlebox.add_artist(title)
        return title


def logistic(x, sigma, thresh, bias):
    beta = np.pi / (np.sqrt(3) * sigma)
    return \
        (np.abs(x) >= thresh) * (
                1 / (1 + np.exp(-beta * (x + bias)))) + \
        (np.abs(x) < thresh) * (1 / (1 + np.exp(-beta * bias)))


def logistic_old(x, sigma, thresh, bias):
    beta = np.pi / (np.sqrt(3) * sigma)
    return \
        (np.abs(x) >= thresh) * (
                1 / (1 + np.exp(-beta * (x + bias - np.sign(x) * thresh)))) + \
        (np.abs(x) < thresh) * (1 / (1 + np.exp(beta * bias)))


def linear(x, thresh, bias):
    y = (np.abs(x) > thresh) * (x - np.sign(x) * thresh) + bias
    return y


def tanh(x, beta, thresh, offset):
    return \
        (np.abs(x) > thresh) * (
                (1 - offset) * np.tanh(beta * (x - np.sign(x) * thresh)) + np.sign(x) * offset) + \
        (np.abs(x) <= thresh) * np.sign(x) * offset


def plot_type2_condensed(ax, s, m, m2=None, nsamples_gen=1000):
    cfg = m.cfg

    if hasattr(m, 'model'):
        type1_params = m.model.params_type1
        type2_params = m.model.params_type2
        if '_criteria' in cfg.type1_evidence_to_confidence:
            type2_params['type2_evidence_bias_mult'] = [v for k, v in m.model.params_type2.items() if
                                                        'type2_criterion' in k]
        data = m.data.data
        stimuli_norm = data.stimuli_norm
        c_conf = data.c_conf
    else:
        type1_params = m.params_type1
        type2_params = m.params_type2
        stimuli_norm = m.x_stim
        c_conf = m.c_conf

    simu = simu_data({**type1_params, **type2_params}, nsamples=len(stimuli_norm), nsubjects=nsamples_gen,
                     cfg=cfg, stimuli_external=stimuli_norm, verbose=False)

    if 'type1_thresh' not in type1_params:
        type1_params['type1_thresh'] = 0
    if 'type1_bias' not in type1_params:
        type1_params['type1_bias'] = 0

    levels = np.unique(stimuli_norm)
    nbins = 20

    if m2 is not None:
        cfg2 = m2.cfg
        if hasattr(m2, 'model'):
            type1_params2 = m2.model.params_type1
            type2_params2 = m2.model.params_type2
            if '_criteria' in cfg2.type1_evidence_to_confidence:
                type2_params2['type2_evidence_bias_mult'] = [v for k, v in m2.model.params_type2.items() if
                                                                     'type2_criterion' in k]
        else:
            type1_params2 = m2.params_type1
            type2_params2 = m2.params_type2
        simu2 = simu_data({**type1_params2, **type2_params2}, nsamples=len(stimuli_norm), nsubjects=nsamples_gen,
                          cfg=cfg2, stimuli_external=stimuli_norm, verbose=False)

        if 'type1_thresh' not in type1_params2:
            type1_params2['type1_thresh'] = 0
        if 'type1_bias' not in type1_params2:
            type1_params2['type1_bias'] = 0
        counts_gen2 = [[] for _ in range(2)]

    counts, counts_gen, bins = [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)]
    for k in range(2):
        levels_ = (levels[levels < 0], levels[levels > 0])[k]
        for i, v in enumerate(levels_):
            hist = np.histogram(c_conf[stimuli_norm == v], density=True, bins=nbins)
            counts[k] += [hist[0]]
            bins[k] += [hist[1]]
            counts_gen[k] += [np.histogram(simu.c_conf[np.tile(stimuli_norm, (nsamples_gen, 1)) == v], density=True,
                                           bins=bins[k][i])[0] / (len(bins[k][i]) - 1)]
            if m2 is not None:
                counts_gen2[k] += [np.histogram(simu2.c_conf[np.tile(stimuli_norm, (nsamples_gen, 1)) == v],  # noqa
                                                density=True, bins=bins[k][i])[0] / (len(bins[k][i]) - 1)]
    counts = np.array(counts) / np.max(counts)
    counts_gen = np.array(counts_gen) / np.max(counts_gen)
    if m2 is not None:
        counts_gen2 = np.array(counts_gen2) / np.max(counts_gen2)
    bins = np.array(bins)

    for k in range(2):
        levels_ = (levels[levels < 0], levels[levels > 0])[k]
        for i, v in enumerate(levels_):
            plt.barh(y=bins[k, i][:-1] + np.diff(bins[k, i]) / 2, width=((1, -1)[k]) * 0.3 * counts[k, i],
                     height=1 / nbins, left=0.005 + v, color=color_data, linewidth=0, alpha=1, zorder=10,
                     label='Data: histogram' if ((k == 0) & (i == 0)) else None)
            plt.plot(v + (1, -1)[k] * 0.3 * counts_gen[k, i], bins[k, i][:-1] + (bins[k, i][1] - bins[k, i][0]) / 2,
                     color=0.85 * color_generative_type2, zorder=11, lw=1.5,
                     label='Model: density' if ((k == 0) & (i == 0)) else None)
            if m2 is not None:
                plt.plot(v + (1, -1)[k] * 0.3 * counts_gen2[k, i],  # noqa
                         bins[k, i][:-1] + (bins[k, i][1] - bins[k, i][0]) / 2, '--', dashes=(3, 2.4),
                         color=0.85 * color_generative_type2b,
                         zorder=11, lw=1.5, label='Model: density' if ((k == 0) & (i == 0)) else None)

    plt.xlim((-1, 1))
    ylim = (-0.01, 1.01)
    plt.plot([0, 0], ylim, 'k-', lw=0.5)
    plt.ylim(ylim)

    if s == 17:
        plt.xlabel('Stimulus ($x$)', fontsize=11)
        ax.xaxis.set_label_coords(1.1, -0.18)
    if s == 8:
        plt.ylabel('c_conf', fontsize=11)
    if s < 16:
        plt.xticks([])
    if np.mod(s, 4) != 0:
        plt.yticks([])
    title = r"$\varphi_\mathrm{m}$=" + f"${type2_params['type2_evidence_bias_mult']:.2f}$ " + \
            r"$\sigma_\mathrm{m}$=" + f"${type2_params['type2_noise']:.2f}$"
    if m2 is not None:
        params_type2_ = type2_params2  # noqa
        title2 = r"$\varphi_\mathrm{m}$=" + f"${params_type2_['type2_evidence_bias_mult']:.2f}$ " + \
                 r"$\sigma_\mathrm{m}$=" + f"${params_type2_['type2_noise']:.2f}$"
        plt.text(0, 1.23, title, fontsize=8.5, color=np.array([165, 110, 0])/255, ha='center')
        plt.text(0, 1.13, title2, fontsize=8.5, color=np.array([30, 98, 38])/255, ha='center')
    else:
        plt.title(title, fontsize=9, y=0.97)
    plt.text(0, 0.8, f'{s + 1}', bbox=dict(fc=[0.8, 0.8, 0.8], ec=[0.5, 0.5, 0.5], lw=0.5, pad=2, alpha=0.8),
             fontsize=10, ha='center')


def plot_psychometric_sim(data, figure_paper=False):
    plot_psychometric(data.choices, data.stimuli, data.params_type1, cfg=data.cfg, figure_paper=figure_paper)


def plot_psychometric(stimuli, choices, params, cfg=None, figure_paper=False,
                      fit_only=False, highlight_fit=False):

    params_type1 = {k: v for k, v in params.items() if k.startswith('type1')}

    type1_noise = _check_param(params_type1['type1_noise'])
    if (cfg is None and 'type1_thresh' in params_type1) or (cfg is not None and cfg.enable_type1_param_thresh):
        type1_thresh = _check_param(params_type1['type1_thresh'])
    else:
        type1_thresh = [0, 0]
    if (cfg is None and 'type1_bias' in params_type1) or (cfg is not None and cfg.enable_type1_param_bias):
        type1_bias = _check_param(params_type1['type1_bias'])
    else:
        type1_bias = [0, 0]

    xrange_neg = np.arange(-1, 0.001, 0.001)
    xrange_pos = np.arange(0.001, 1.001, 0.001)

    posterior_neg = logistic(xrange_neg, type1_noise[0], type1_thresh[0], type1_bias[0])
    posterior_pos = logistic(xrange_pos, type1_noise[1], type1_thresh[1], type1_bias[1])

    ax = plt.gca()

    if not fit_only:
        stimulus_ids = (stimuli > 0).astype(int)
        levels = np.unique(stimuli)
        choiceprob_neg = np.array([np.mean(choices[(stimuli == v) & (stimulus_ids == 0)] ==
                                           stimulus_ids[(stimuli == v) & (stimulus_ids == 0)])
                                   for v in levels[levels < 0]])
        choiceprob_pos = np.array([np.mean(choices[(stimuli == v) & (stimulus_ids == 1)] ==
                                           stimulus_ids[(stimuli == v) & (stimulus_ids == 1)])
                                   for v in levels[levels > 0]])
        plt.plot(levels[levels < 0], 1 - choiceprob_neg, 'o', markersize=6.5, mew=1, mec='k', label='Data: $S^-$ Mean',
                 color=color_data, clip_on=False, zorder=11, alpha=(1, 0.2)[highlight_fit])
        plt.plot(levels[levels > 0], choiceprob_pos, 's', markersize=5.5, mew=1, mec='k', label='Data: $S^+$ Mean',
                 color=color_data, clip_on=False, zorder=11, alpha=(1, 0.2)[highlight_fit])

    plt.plot(xrange_neg, posterior_neg, '-', lw=(2, 5)[highlight_fit], color=color_logistic, clip_on=False,
             zorder=(10, 12)[highlight_fit], label=f'Model fit')
    plt.plot(xrange_pos, posterior_pos, '-', lw=(2, 5)[highlight_fit], color=color_logistic, clip_on=False,
             zorder=(10, 12)[highlight_fit])

    plt.plot([-1, 1], [0.5, 0.5], 'k-', lw=0.5)
    plt.plot([0, 0], [-0.02, 1.02], 'k-', lw=0.5)

    ax.yaxis.grid('on', color=[0.9, 0.9, 0.9])
    plt.xlim((-1, 1))
    plt.ylim((0, 1))
    plt.xlabel('Stimulus ($x$)')
    plt.ylabel('Choice probability $S^+$')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    leg = plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9, handlelength=0.5)
    for lh in (leg.legendHandles if hasattr(leg, 'legendHandles') else leg.legend_handles):
        if hasattr(lh, '_legmarker'):
            lh._legmarker.set_alpha(1)  # noqa
        elif hasattr(lh, 'legmarker'):
            lh.legmarker.set_alpha(1)  # noqa
    anot_type1 = []
    for i, (k, v) in enumerate(params_type1.items()):
        if (cfg is None and k in params_type1) or (cfg is not None and getattr(cfg, f"enable_{k.replace('type1', 'type1_param')}")):
            if hasattr(v, '__len__'):
                val = ', '.join([f"{p:{'.0f' if p == 0 else '.3g'}}" for p in v])
                anot_type1 += [f"${symbols[k][1:-1]}=" + f"[{val}]$"]
            else:
                anot_type1 += [f"${symbols[k][1:-1]}={v:{'.0f' if v == 0 else '.3g'}}$"]
    plt.text(1.045, -0.1, r'Estimated parameters:' + '\n' + '\n'.join(anot_type1), transform=plt.gca().transAxes,
             bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=1, pad=5), fontsize=9)
    set_fontsize(label=13, tick=11)


def plot_confidence(stimuli_or_data_object, confidence=None):
    if confidence is None:
        stimuli, confidence = stimuli_or_data_object.stimuli, stimuli_or_data_object.confidence
    else:
        stimuli = stimuli_or_data_object
    ax = plt.gca()

    for v in sorted(np.unique(stimuli)):
        plt.errorbar(v, np.mean(confidence[stimuli == v]), yerr=sem(confidence[stimuli == v]), marker='o', markersize=5,
                     mew=1, mec='k', color='None', ecolor='k', mfc=color_data, clip_on=False, elinewidth=1.5,
                     capsize=5)
    plt.plot([0, 0], [0, 1], 'k-', lw=0.5)
    plt.ylim(0, 1)
    plt.xlabel('Stimulus ($x$)')
    plt.ylabel('Confidence ($c$)')
    set_fontsize(label=13, tick=11)


def plot_evidence_versus_confidence(stimuli, confidence, y_decval, params, cfg=None,
                                    type2_noise_type=None, type2_noise_dist=None,
                                    type1_noise_signal_dependency='none',
                                    plot_data=True, plot_generative_data=True, plot_likelihood=False,
                                    plot_bias_free=False, display_parameters=True,
                                    var_likelihood=None, y_decval_range=(45, 50, 55),
                                    nsamples_gen=1000, nsamples_dist=100000, bw=0.03, color_linkfunction=(0.55, 0.55, 0.69),
                                    label_linkfunction='Link function',
                                    figure_paper=False):

    params = params.copy()

    if cfg is not None:
        type2_noise_dist = cfg.type2_noise_dist
        type2_noise_type = cfg.type2_noise_type
        type1_noise_signal_dependency = cfg.type1_noise_signal_dependency
    else:
        cfg = Configuration()
        # We disable parameters that are not contained in params
        for k, v in cfg.__dict__.items():
            if k.startswith('enable_') and (v > 0) and not ((k.replace('enable_type1_param', 'type1') in params) or
                    (k.replace('enable_type2_param', 'type2') in params)):
                setattr(cfg, k, 0)


    generative = simu_data(params, nsamples=len(stimuli), nsubjects=nsamples_gen, cfg=cfg, stimuli_external=stimuli,
                           verbose=False, squeeze=True)

    ax = plt.gca()
    vals_decval = np.unique(y_decval)
    vals_dv_gen = np.unique(generative.y_decval_latent)
    for k in range(2):

        vals_dv_ = vals_decval[vals_decval < 0] if k == 0 else vals_decval[vals_decval > 0]
        vals_dv_gen_ = vals_dv_gen[vals_dv_gen < 0] if k == 0 else vals_dv_gen[vals_dv_gen > 0]

        conf_data_means = [np.mean(confidence[y_decval == v]) for v in vals_dv_]
        conf_data_std_neg = [np.std(confidence[(y_decval == v) & (confidence < conf_data_means[i])])
                             for i, v in enumerate(vals_dv_)]
        conf_data_std_pos = [np.std(confidence[(y_decval == v) & (confidence >= conf_data_means[i])])
                             for i, v in enumerate(vals_dv_)]

        conf_gen_means = [np.mean(generative.confidence[generative.y_decval_latent == v]) for v in vals_dv_gen_]
        conf_gen_std_neg = [np.std(
            generative.confidence[(generative.y_decval_latent == v) & (generative.confidence < conf_gen_means[i])])
            for i, v in enumerate(vals_dv_gen_)]
        conf_gen_std_pos = [np.std(
            generative.confidence[(generative.y_decval_latent == v) & (generative.confidence > conf_gen_means[i])])
            for i, v in enumerate(vals_dv_gen_)]

        if plot_data:
            _, cap, barlinecols = plt.errorbar(
                vals_dv_-0.015, conf_data_means, yerr=[conf_data_std_neg, conf_data_std_pos],
                label='Data: Mean (SD)' if k == 0 else None, marker='o', markersize=7, mew=1, mec='k', color='None',
                ecolor='k', mfc=color_data, clip_on=False, zorder=35, elinewidth=1.5, capsize=5
            )
            [cap[i].set_markeredgewidth(1.5) for i in range(len(cap))]
            [cap[i].set_clip_on(False) for i in range(len(cap))]
            barlinecols[0].set_clip_on(False)

        if plot_generative_data:
            _, cap, barlinecols = plt.errorbar(
                vals_dv_gen_+0.015, conf_gen_means, yerr=[conf_gen_std_neg, conf_gen_std_pos],
                label='Generative model' if k == 0 else None, marker='o', markersize=7, mew=1, mec='k',
                color='None', ecolor=color_generative_type2, mfc=color_generative_type2, clip_on=False, zorder=35,
                elinewidth=1.5, capsize=5
            )
            [cap[i].set_markeredgewidth(1.5) for i in range(len(cap))]
            [cap[i].set_clip_on(False) for i in range(len(cap))]
            barlinecols[0].set_clip_on(False)

        if plot_likelihood:
            var_likelihood_means = [np.nanmean(var_likelihood[y_decval == v, y_decval_range[1]]) for v in
                                    vals_dv_]

            for i, v in enumerate(vals_dv_):

                x = np.linspace(0, 1, 1000)

                if type2_noise_type == 'noisy_report':
                    likelihood = get_likelihood(x, type2_noise_dist, np.maximum(1e-3, var_likelihood_means[i]),
                                                params['type2_noise'], logarithm=False)
                else:
                    dist = get_type2_dist(type2_noise_dist, np.maximum(1e-3, var_likelihood_means[i]), params['type2_noise'])
                    z1_type1_evidence_generative = dist.rvs(nsamples_dist)
                    c_conf_generative = type1_evidence_to_confidence(
                        z1_type1_evidence_generative, x_stim=stimuli, y_decval=z1_type1_evidence_generative,
                        type1_noise_signal_dependency=type1_noise_signal_dependency, **params
                    )
                    likelihood = gaussian_kde(c_conf_generative, bw_method=bw).evaluate(x)
                likelihood -= likelihood.min()
                like_max = likelihood.max()
                likelihood_norm = likelihood / like_max if like_max > 0 else np.zeros(likelihood.shape)
                plt.plot(v + (0.26 * likelihood_norm + 0.005) * ((1, -1)[k]), x, color=color_model, zorder=25, lw=2.5,
                         label=(None, r'Likelihood for $y_i^*$')[int((k == 0) & (i == 0))])

                ax.annotate(rf"$\mathbf{{y}}_{{{('', '+')[k]}{(i - len(vals_dv_), i + 1)[k]}}}^*$",
                            xy=(v, 1.008), xycoords='data', xytext=(v, 1.09), color=color_model, weight='bold',
                            fontsize=9, ha='center', bbox=dict(pad=0, facecolor='w', lw=0),
                            arrowprops=dict(facecolor=color_model, headwidth=7, lw=0, headlength=3, width=2))

        xrange = np.arange(-5, 0.001, 0.001) if k == 0 else np.arange(0, 5.001, 0.001)

        type2_evidence_bias_mult = params['type2_evidence_bias_mult'] if 'type2_evidence_bias_mult' in params else 1
        type2_evidence_bias_add = params['type2_evidence_bias_add'] if 'type2_evidence_bias_add' in params else 0
        conf_model = type1_evidence_to_confidence(
            type2_evidence_bias_mult * np.abs(xrange) + type2_evidence_bias_add,
            # np.abs(xrange),
            x_stim=xrange, y_decval=xrange,
            type1_noise_signal_dependency=type1_noise_signal_dependency, **params
        )
        if 'type2_criteria' in params:
            criteria = [np.sum(params['type2_criteria'][:i+1]) for i in range(len(params['type2_criteria']))]
            plt.plot(xrange, (np.digitize(conf_model, criteria) + 0.5) / (len(criteria) + 1),
                     color=color_linkfunction, lw=3.5, zorder=5, alpha=0.9,
                     label=label_linkfunction if k == 0 else None)
        else:
            plt.plot(xrange, conf_model, color=color_linkfunction, lw=3.5, zorder=5, alpha=0.9,
                     label=label_linkfunction if k == 0 else None)
        if plot_bias_free:
            conf_model_bf = type1_evidence_to_confidence(
                np.abs(xrange), x_stim=xrange, y_decval=xrange,
                type1_noise_signal_dependency=type1_noise_signal_dependency, **params
            )
            if 'type2_criteria' in params:
                criteria_bf = np.arange(1/(len(params['type2_criteria'])+1), 1, 1/(len(params['type2_criteria'])+1))
                conf_model_bf = (np.digitize(conf_model_bf, criteria_bf) + 0.5) / (len(criteria) + 1)
            plt.plot(xrange, conf_model_bf, color='green', lw=3.5, zorder=5, alpha=0.9,
                     label='Link function (bias-free)' if k == 0 else None)

    ylim = (-0.01, 1.025)
    plt.plot([0, 0], ylim, 'k-', lw=0.5)
    plt.ylim(ylim)
    plt.xlim((-1.1 * np.abs(vals_decval).max(), 1.1 * np.abs(vals_decval).max()))
    plt.xlabel(r'Type 1 decision value ($y$)')
    plt.ylabel('Confidence ($c$)')
    handles, labels = plt.gca().get_legend_handles_labels()
    if plot_likelihood:
        order = [2, 1, 0, 3]
        plt.legend([handles[i] for i in order], [labels[i] for i in order], bbox_to_anchor=(1.02, 1), loc="upper left",
                   fontsize=9)
    else:
        plt.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.yaxis.grid('on', color=[0.9, 0.9, 0.9])

    if display_parameters:
        an_params = [p for p in params if 'type2_' in p]
        an_type2 = []
        for p in an_params:
            if hasattr(params[p], '__len__'):
                an_type2 += [f"${symbols[p][1:-1]}=${[float(f'{v:.3f}') for v in params[p]]}"]
            else:
                an_type2 += [f"${symbols[p][1:-1]}={params[p]:{'.0f' if params[p] == 0 else ('.3f', '.2f')[figure_paper]}}$"]  # noqa
        plt.text(1.045, -0.2, r'Estimated parameters:' + '\n' + '\n'.join(an_type2), transform=plt.gca().transAxes,
                 bbox=dict(fc=[1, 1, 1], ec=[0.5, 0.5, 0.5], lw=1, pad=5), fontsize=9)

    set_fontsize(label=13, tick=11)


def plot_confidence_dist(cfg, stimuli, confidence, params, nsamples_gen=1000,
                         plot_likelihood=True, var_likelihood_grid=None, y_decval_grid=None,
                         likelihood_weighting=None, dv_range=(45, 50, 55), nsamples_dist=10000, bw=0.03,
                         figure_paper=False):
    generative = simu_data(params, nsamples=len(stimuli), nsubjects=nsamples_gen, cfg=cfg, stimuli_external=stimuli,
                           verbose=False)

    nbins = 20
    levels = np.unique(stimuli)
    counts, counts_gen, bins = [[] for _ in range(2)], [[] for _ in range(2)], [[] for _ in range(2)]
    for k in range(2):
        levels_ = (levels[levels < 0], levels[levels > 0])[k]
        for i, v in enumerate(levels_):
            hist = np.histogram(confidence[stimuli == v], density=True, bins=nbins)
            counts[k] += [hist[0]]
            bins[k] += [hist[1]]
            counts_gen[k] += [np.histogram(generative.c_conf[np.tile(stimuli, (nsamples_gen, 1)) == v],
                                           density=True, bins=bins[k][i])[0] / (len(bins[k][i]) - 1)]
    counts = [np.array(count) / np.max([np.max(c) for c in counts]) for count in counts]
    counts_gen = [np.array(count) / np.max([np.max(c) for c in counts_gen]) for count in counts_gen]

    ax = plt.gca()

    dist_labels = [r'for $y_i^*$ $−$ 0.5 SD', r'for $y_i^*$', r'for $y_i^*$ $+$ 0.5 SD']
    for k in range(2):

        levels_ = (levels[levels < 0], levels[levels > 0])[k]

        if plot_likelihood:
            confp_means = [[np.nanmean(var_likelihood_grid[stimuli == v, z]) for z in dv_range] for v in levels_]
            weighting_p = np.array([[np.nanmean(likelihood_weighting[stimuli == v, z]) for z in dv_range] for v in
                                    levels_])
            weighting_p /= np.max(weighting_p)

        for i, v in enumerate(levels_):

            plt.barh(y=bins[k][i][:-1] + np.diff(bins[k][i]) / 2, width=((1, -1)[k]) * 0.26 * counts[k][i],
                     height=1 / nbins, left=0.005 + v, color=color_data, linewidth=0, alpha=1, zorder=10,
                     label='Data: histogram' if ((k == 0) & (i == 0)) else None)

            plt.plot(v + (1, -1)[k] * 0.26 * counts_gen[k][i], bins[k][i][:-1] + (bins[k][i][1] - bins[k][i][0]) / 2,
                     color=color_generative_type2, zorder=11, lw=2,
                     label='Generative model' if ((k == 0) & (i == 0)) else None)

            if plot_likelihood:
                for j, dv in enumerate(dv_range):
                    x = np.linspace(0, 1, 1000)
                    if cfg.type2_noise_type == 'noisy_report':
                        likelihood = get_likelihood(x, cfg.type2_noise_dist,
                                                    np.maximum(1e-3, confp_means[i][j]),  # noqa
                                                    params['type2_noise'], logarithm=False)
                    else:
                        dist = get_type2_dist(cfg.type2_noise_dist, np.maximum(1e-3, confp_means[i][j]), params['type2_noise'])
                        z1_type1_evidence = dist.rvs(nsamples_dist)
                        if 'censored_' in cfg.type2_noise_dist:
                            z1_type1_evidence[z1_type1_evidence < 0] = 0
                        c_conf = type1_evidence_to_confidence(
                            z1_type1_evidence, cfg.type1_evidence_to_confidence, x_stim=z1_type1_evidence,
                            type1_noise_signal_dependency=cfg.type1_noise_transform,
                            **params
                        )
                        likelihood = gaussian_kde(c_conf, bw_method=bw).evaluate(x)
                    likelihood -= likelihood.min()
                    likelihood_max = likelihood.max()
                    likelihood_norm = likelihood / likelihood_max if likelihood_max > 0 else np.zeros(likelihood.shape)
                    likelihood_norm[likelihood_norm < 0.05] = np.nan
                    correct = np.sign(y_decval_grid[stimuli == v, dv][0]) == (-1, 1)[k]
                    color_shade = [[0.175], [0], [0.175]][j]
                    plt.plot(v + (weighting_p[i][j] * 0.26 * likelihood_norm + 0.005) * ((1, -1)[k]),  # noqa
                             x, color=(color_model_wrong, color_model)[int(correct)] + color_shade,
                             zorder=25, lw=2.5, dashes=[(2, 1), (None, None), (None, None)][j],
                             label=(None, dist_labels[j])[int((k == 1) & (i == 1))])

    plt.xlim((-1.05 * np.abs(levels).max(), 1.05 * np.abs(levels).max()))
    ylim = (-0.01, 1.01)
    plt.plot([0, 0], ylim, 'k-', lw=0.5)
    plt.ylim(ylim)

    plt.xlabel('Stimulus ($x$)')
    plt.ylabel('c_conf')
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2g'))

    if plot_likelihood:
        handles, labels = plt.gca().get_legend_handles_labels()
        handles += ['', 'Likelihood']
        labels += ['', '']
        if figure_paper:
            labels += [r'for $y_i^*$ $+$ ' + '0.5 SD\n(incorrect choice)']
            handles += [Line2D([0], [0], color=color_model_wrong + 0.175,
                               **{k: getattr(handles[3], f'_{k}') for k in ('linestyle', 'linewidth')})]
            order = [4, 0, 5, 6, 2, 1, 3, 7]
        else:
            order = [4, 0, 5, 6, 2, 1, 3]
        plt.legend([handles[i] for i in order], [labels[i] for i in order],
                   bbox_to_anchor=(1.02, 1.1 if figure_paper else 1), loc="upper left", fontsize=9,
                   handler_map={str: LegendTitle()})
    else:
        plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    ax.yaxis.grid('on', color=[0.9, 0.9, 0.9], zorder=-10)

    set_fontsize(label=13, tick=11)

def plot_type1_type2(m, plot_subject_id=False, nsamples_gen=1000, figure_paper=False):

    if hasattr(m, 'model'):
        # In case m is model fit to data
        simulation = False
        type1_params = m.model.params_type1
        type2_params = m.model.params_type2
        data = m.data.data
        stimuli_norm = data.stimuli_norm
        choices = data.d_dec
        c_conf = data.c_conf
        var_likelihood = dict(noisy_report=m.model.extended.c_conf,
                              noisy_readout=m.model.extended.z1_type1_evidence_grid)[m.cfg.type2_noise_type]
        likelihood_weighting = m.model.extended.y_decval_pmf_grid
        y_decval = m.model.extended.y_decval_grid
        y_decval_mode = m.model.extended.y_decval_grid
    else:
        # In case m is a simulation
        simulation = True
        type1_params = m.params_type1
        type2_params = m.params_type2
        choices = m.d_dec
        stimuli_norm = m.x_stim
        c_conf = m.c_conf
        y_decval_mode = m.y_decval.squeeze()
        y_decval = None
        likelihood_weighting = None
        var_likelihood = None

    if 'type1_evidence_bias_mult' not in type2_params:
        type2_params['type1_evidence_bias_mult'] = 1
    if 'type1_thresh' not in type1_params:
        type1_params['type1_thresh'] = 0
    if 'type1_bias' not in type1_params:
        type1_params['type1_bias'] = 0

    params = {**type1_params, **type2_params}

    fig = plt.figure(figsize=(8, 7))
    if plot_subject_id and hasattr(m, 'subject_id') and (m.subject_id is not None):
        fig.suptitle(f'Subject {m.subject_id}', fontsize=16)

    plt.subplot(3, 1, 1)
    ax1 = plot_psychometric(choices, stimuli_norm, type1_params, cfg=m.cfg, figure_paper=figure_paper)
    ax1.yaxis.set_label_coords(-0.1, 0.43)
    plt.text(-0.15, 1.01, 'A', transform=ax1.transAxes, fontsize=19)

    plt.subplot(3, 1, 2)
    ax2 = plot_evidence_versus_confidence(
        stimuli_norm, c_conf, y_decval_mode, params, cfg=m.cfg,
        plot_likelihood=not simulation, var_likelihood=var_likelihood,
        y_decval_range=(0,) if simulation else (45, 50, 55),
        figure_paper=figure_paper
    )
    plt.text(-0.15, 1.01, 'B', transform=ax2.transAxes, fontsize=19)

    plt.subplot(3, 1, 3)
    ax3 = plot_confidence_dist(
        m.cfg, stimuli_norm, c_conf, params, nsamples_gen,
        plot_likelihood=not simulation, var_likelihood_grid=var_likelihood,
        y_decval_grid=y_decval, likelihood_weighting=likelihood_weighting, dv_range=(0,) if simulation else (45, 50, 55),
        figure_paper=figure_paper
    )
    plt.text(-0.15, 1.01, 'C', transform=ax3.transAxes, fontsize=19)

    # hack to not cut the right edges in saved images
    # if figure_paper:
    #     plt.text(1.29, 1.01, 'C', transform=plt.gca().transAxes, color='r', fontsize=9)

    set_fontsize(label=11, tick=10)
    plt.subplots_adjust(hspace=0.5, top=0.96, right=0.7, left=0.1)
    ax2.set_position([*(np.array(ax2.get_position())[0] + (0, -0.02)),
                      ax2.get_position().width, ax2.get_position().height])
    ax3.set_position([*(np.array(ax3.get_position())[0] + (0, -0.02)),
                      ax3.get_position().width, ax3.get_position().height])


def set_fontsize(label=None, xlabel=None, ylabel=None, tick=None, xtick=None, ytick=None, title=None):

    fig = plt.gcf()

    for ax in fig.axes:
        if xlabel is not None:
            ax.xaxis.label.set_size(xlabel)
        elif label is not None:
            ax.xaxis.label.set_size(label)
        if ylabel is not None:
            ax.yaxis.label.set_size(ylabel)
        elif label is not None:
            ax.yaxis.label.set_size(label)

        if xtick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(xtick)
        elif tick is not None:
            for ticklabel in (ax.get_xticklabels()):
                ticklabel.set_fontsize(tick)
        if ytick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(ytick)
        elif tick is not None:
            for ticklabel in (ax.get_yticklabels()):
                ticklabel.set_fontsize(tick)

        if title is not None:
            ax.title.set_fontsize(title)


if __name__ == '__main__':

    import remeta
    np.random.seed(42)  # make notebook reproducible
    import warnings
    warnings.filterwarnings('error')

    # cfg = remeta.Configuration()
    #
    # cfg.enable_type1_param_noise = 2
    # cfg.enable_type2_param_evidence_bias_add = 1
    # params_true = dict(
    #     type1_noise=[0.5, 0.7],
    #     type1_bias=-0.1,
    #     type2_noise=0.1,
    #     type2_evidence_bias_mult=1.3,
    #     type2_evidence_bias_add=-0.1
    # )
    # data = remeta.simu_data(params_true, nsamples=1000, nsubjects=1, squeeze=True, x_stim_stepsize=0.25, cfg=cfg)
    # # remeta.plot_psychometric_sim(data)
    # # remeta.plot_c_conf_sim(data);
    # cfg.true_params = params_true
    # cfg.gridsearch = False
    # rem = remeta.ReMeta(cfg=cfg)
    # rem.fit(data.x_stim, data.d_dec, data.c_conf)
    # # rem.plot_link_function()
    # rem.plot_c_conf_dist()


    x_stim, d_dec, c_conf, params, y_decval = remeta.load_dataset(
    'type2_simple', return_params=True, return_y_decval=True
    )
    remeta.plot_evidence_versus_confidence(x_stim, c_conf, y_decval, params, plot_bias_free=True)