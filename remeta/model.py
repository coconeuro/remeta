import os
import pathlib
import pickle
import timeit
import warnings
from dataclasses import make_dataclass

import numpy as np
from scipy.optimize import minimize, OptimizeResult
from scipy.stats import logistic as logistic_dist
from scipy.special import expit

from .configuration import Configuration
from .type2_dist import get_type2_dist
from .fit import fmincon
from .gendata import simu_data
from .modelspec import Model, Data
from .plot import plot_evidence_versus_confidence, plot_confidence_dist
from .transform import (compute_signal_dependent_type1_noise, logistic, type1_evidence_to_confidence,
                        confidence_to_type1_evidence, confidence_to_type1_noise, type1_noise_to_confidence,
                        check_criteria_sum)
from .util import _check_param, TAB, maxfloat, print_dataset_characteristics

class ReMeta:

    def __init__(self, cfg=None, **kwargs):
        """
        Main class of the ReMeta toolbox

        Parameters
        ----------
        cfg : util.Configuration
            Configuration object. If None is passed, the default configuration is used (but see kwargs).
        kwargs : dict
            The kwargs dictionary is parsed for keywords that match keywords of util.Configuration; in case of a match,
            the configuration is set.
        """

        if cfg is None:
            # Set configuration attributes that match keyword arguments
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in Configuration.__dict__}
            self.cfg = Configuration(**cfg_kwargs)
        else:
            self.cfg = cfg
        self.cfg.setup()

        self.model = Model(cfg=self.cfg)
        self.data = None

        self.type1_is_fitted = False
        self.type2_is_fitted = False


    def fit(self, x_stim, d_dec, c_conf, precomputed_parameters=None, guess_type2=None, verbose=True,
            ignore_warnings=False):
        """
        Fit type 1 and type 2 parameters

        Parameters
        ----------
        x_stim : array-like of shape (n_samples)
            Array of signed stimulus intensity values, where the sign codes the stimulus category (cat 1: -, cat2: +)
            and the absolut value codes the intensity. Must be normalized to [-1; 1], or set
            `normalize_stimuli_by_max=True`.
        d_dec : array-like of shape (n_samples)
            Array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive
            stimulus category.
        c_conf : array-like of shape (n_samples)
            Confidence ratings; must be normalized to the range [0;1].
        precomputed_parameters : dict
            Provide pre-computed parameters. A dictionary with all parameters defined by the model must be passed. This
            can sometimes be useful to obtain information from the model without having to fit the model.
            [ToDO: which information?]
        guess_type2 : array-like of shape (n_params_type1)
            For testing: provide an initial guess for the optimization of the type 2 level
        verbose : bool
            If True, information of the model fitting procedure is printed.
        ignore_warnings : bool
            If True, warnings during model fitting are supressed.
        """

        # Instantiate util.Data object
        self.data = Data(self.cfg, x_stim, d_dec, c_conf)

        self.fit_type1(precomputed_parameters=precomputed_parameters, verbose=verbose, ignore_warnings=ignore_warnings)

        if not self.cfg.skip_type2:
            self.fit_type2(precomputed_parameters=precomputed_parameters, guess_type2=guess_type2, verbose=verbose,
                           ignore_warnings=ignore_warnings)


    def fit_type1(self, x_stim=None, d_dec=None, c_conf=None, precomputed_parameters=None, verbose=True,
                  ignore_warnings=False):

        if self.data is None:
            if x_stim is None or d_dec is None:
                raise ValueError('If the data attribute of the ReMeta instance is None, at least x_stim (stimuli) '
                                 'and d_dec (choices) have to be passed to fit_type1()')
            else:
                self.data = Data(self.cfg, x_stim, d_dec, c_conf)

        if verbose:
            print('\n+++ Type 1 level +++')
        # with warnings.catch_warnings(record=True) as w:
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize',
                                    message='delta_grad == 0.0. Check if the approximated function is linear. If the '
                                            'function is linear better results can be obtained by defining the Hessian '
                                            'as zero instead of using quasi-Newton approximations.')
            if ignore_warnings:
                warnings.filterwarnings('ignore')
            if isinstance(precomputed_parameters, dict):
                if not np.all([p in precomputed_parameters for p in self.cfg.paramset_type1.param_names_flat]):
                    raise ValueError('Set of precomputed type 1 parameters is incomplete.')
                self.model.fit.fit_type1 = OptimizeResult(
                    x=[precomputed_parameters[p] for p in self.cfg.paramset_type1.param_names_flat],
                    fun=self._negll_type1([precomputed_parameters[p] for p in self.cfg.paramset_type1.param_names_flat])
                )
            else:
                if self.cfg.paramset_type1.nparams > 0:
                    if verbose:
                        negll_initial_guess = self._negll_type1(self.cfg.paramset_type1.guess)
                        print(f'Initial guess (neg. LL: {negll_initial_guess:.2f})')
                        for i, p in enumerate(self.cfg.paramset_type1.param_names_flat):
                            print(f'{TAB}[guess] {p}: {self.cfg.paramset_type1.guess[i]:.4g}')
                        print('Performing local optimization')
                    t0 = timeit.default_timer()
                    self.model.fit.fit_type1 = minimize(
                        self._negll_type1, self.cfg.paramset_type1.guess, bounds=self.cfg.paramset_type1.bounds,
                        method='trust-constr'
                    )
                    if self.cfg.enable_type1_param_thresh:
                        fit_powell = minimize(
                            self._negll_type1, self.cfg.paramset_type1.guess, bounds=self.cfg.paramset_type1.bounds,
                            method='Powell'
                        )
                        if fit_powell.fun < self.model.fit.fit_type1.fun:
                            self.model.fit.fit_type1 = fit_powell
                    self.model.fit.fit_type1.execution_time = timeit.default_timer() - t0

                else:
                    self.model.fit.fit_type1 = OptimizeResult(x=None)
            if isinstance(self.cfg.true_params, dict):
                if not np.all([p in self.cfg.true_params for p in self.cfg.paramset_type1.param_names]):
                    raise ValueError('Set of provided true type 1 parameters is incomplete.')
                params_true = sum([[self.cfg.true_params[p]] if n == 1 else self.cfg.true_params[p] for n, p in
                                  zip(self.cfg.paramset_type1.param_len, self.cfg.paramset_type1.param_names)], [])
                self.model.fit.fit_type1.negll_true = self._negll_type1(params_true)

            # call once again with final=True to save the model fit
            negll_type1, params_type1, type1_likelihood, type1_posterior = \
                self._negll_type1(self.model.fit.fit_type1.x, final=True)
            if 'type1_thresh' in params_type1 and params_type1['type1_thresh'] < self.data.stimuli_min:
                warnings.warn('Fitted threshold is below the minimal stimulus intensity; consider disabling '
                              'the type 1 threshold by setting enable_type1_param_thresh to 0', category=UserWarning)
            self.model.store_type1(negll_type1=negll_type1, params_type1=params_type1, type1_likelihood=type1_likelihood,
                                   type1_posterior=type1_posterior, stimuli_max=self.data.stimuli_max)
            self.model.report_fit_type1(verbose)
            self.type1_is_fitted = True

        # if not ignore_warnings and verbose:
        #     print_warnings(w)

        self.model.params = self.model.params_type1


    def fit_type2(self, precomputed_parameters=None, guess_type2=None, verbose=True, ignore_warnings=False):

        # compute decision values
        self._compute_decision_values()

        if self.cfg.type2_noise_type == 'noisy_temperature':
            # We precompute a few things for noisy_temperature model
            self.data.jacobian_noisy_temperature = ((2 * np.sqrt(3)) / np.pi) / (1 - np.minimum(1-1e-8, self.data.c_conf)**2)
            self.model.quintiles_noisy_temperature = np.arange(self.cfg.resolution_noisy_temperature, 1, self.cfg.resolution_noisy_temperature)
            # self.model.type1_dist = logistic_dist(loc=self.model.y_decval,
            #                                       scale=self.model.params_type1['type1_noise'] * np.sqrt(3) / np.pi)

        if verbose:
            print('\n+++ Type 2 level +++')

        args_type2 = [self.cfg.type2_noise_type, ignore_warnings]
        if precomputed_parameters is not None:
            if not np.all([p in precomputed_parameters for p in self.cfg.paramset_type2.param_names_flat]):
                raise ValueError('Set of precomputed type 2 parameters is incomplete.')
            self.model.params_type2 = {p: precomputed_parameters[p] for p in self.cfg.paramset_type2.param_names_flat}
            self.model.fit.fit_type2 = OptimizeResult(
                x=[precomputed_parameters[p] for p in self.cfg.paramset_type2.param_names_flat],
                fun=self.fun_negll_type2([precomputed_parameters[p] for p in self.cfg.paramset_type2.param_names_flat])
            )
            fitinfo_type2 = self.compute_type2_negll(list(self.model.params_type2.values()), *args_type2, final=True)  # noqa
            self.model.store_type2(**fitinfo_type2)
        else:
            with warnings.catch_warnings(record=True) as w:  # noqa
                warnings.filterwarnings('ignore', module='scipy.optimize')
                if self.cfg.paramset_type2.nparams > 0:
                    self.model.fit.fit_type2 = fmincon(
                        self.compute_type2_negll, self.cfg.paramset_type2, args_type2,
                        gridsearch=self.cfg.gridsearch, grid_multiproc=self.cfg.grid_multiproc,
                        minimize_along_grid=self.cfg.minimize_along_grid,
                        global_minimization=self.cfg.global_minimization,
                        fine_gridsearch=self.cfg.fine_gridsearch,
                        minimize_solver=self.cfg.minimize_solver, slsqp_epsilon=self.cfg.slsqp_epsilon,
                        guess=guess_type2,
                        verbose=verbose
                    )
                else:
                    self.model.fit.fit_type2 = OptimizeResult(x=None)

            # call once again with final=True to save the model fit
            fitinfo_type2 = self.compute_type2_negll(self.model.fit.fit_type2.x, *args_type2, final=True)
            self.model.store_type2(**fitinfo_type2)

        if self.cfg.true_params is not None:
            type2_params_true = self.cfg.true_params.copy()
            if self.cfg.enable_type2_param_criteria and not 'type2_criteria' in type2_params_true:
                type2_params_true[f'type2_criteria'] = [1/self.cfg.n_discrete_confidence_levels for _ in range(self.cfg.n_discrete_confidence_levels-1)]
            type2_params_true_values = sum([([type2_params_true[p]] if n == 1 else type2_params_true[p])
                                                        if p in type2_params_true else [None]*n for n, p in
                                            zip(self.cfg.paramset_type2.param_len, self.cfg.paramset_type2.param_names)],
                                           [])
            self.model.fit.fit_type2.negll_true = self.compute_type2_negll(type2_params_true_values, self.cfg.type2_noise_type)

        self.model.report_fit_type2(verbose)

        self.model.params = ({} if self.model.params is None else self.model.params) | self.model.params_type2

        self.type2_is_fitted = True

        # if not ignore_warnings:
        #     print_warnings(w)

    def summary(self, extended=False, generative=True, generative_nsamples=1000):
        """
        Provides information about the model fit.

        Parameters
        ----------
        extended : bool
            If True, store various model variables in the summary object.
        generative : bool
            If True, compare model predictions of confidence with empirical confidence by repeatedly sampling from
            the generative model.
        generative_nsamples : int
            Number of samples used for the generative model (higher = more accurate).

        Returns
        ----------
        summary : dataclass
            Information about model fit.
        """

        if self.type2_is_fitted and generative:
            c_conf_generative = simu_data(self.model.params, nsamples=self.data.nsamples, nsubjects=generative_nsamples,
                                          cfg=self.cfg, x_stim_external=self.data.x_stim, verbose=False).c_conf
        else:
            c_conf_generative = None
        summary_model = self.model.summary(
            extended=extended, c_conf_empirical=self.data.c_conf, c_conf_generative=c_conf_generative
        )
        desc = dict(data=self.data.summary(extended), model=summary_model, cfg=self.cfg)

        summary_ = make_dataclass('Summary', desc.keys())

        def repr_(self_):
            txt = f'***{self_.__class__.__name__}***\n'
            for k, v in self_.__dict__.items():
                if k == 'cfg':
                    txt += f"\n{k}: {type(desc['cfg'])} <not displayed>"
                else:
                    txt += f"\n{k}: {v}"
            return txt

        summary_.__repr__ = repr_
        summary_.__module__ = '__main__'
        summary = summary_(**desc)
        return summary

    def _negll_type1(self, params, final=False):
        """
        Likelihood function for the type 1 level

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 1 level.
        final : bool
            If True, store latent variables and parameters.

        Returns:
        --------
        negll: float
            Negative (summed) log likelihood.
        """

        bl = self.cfg.paramset_type1.param_len
        params_type1 = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                        for i, (p, n) in enumerate(zip(self.cfg.paramset_type1.param_names, bl))}
        type1_thresh = _check_param(params_type1['type1_thresh'] if self.cfg.enable_type1_param_thresh else 0)
        type1_bias = _check_param(params_type1['type1_bias'] if self.cfg.enable_type1_param_bias else 0)

        cond_neg, cond_pos = self.data.x_stim < 0, self.data.x_stim >= 0
        y_decval = np.full(self.data.x_stim.shape, np.nan)
        y_decval[cond_neg] = (np.abs(self.data.x_stim[cond_neg]) > type1_thresh[0]) * self.data.x_stim[cond_neg] + type1_bias[0]
        y_decval[cond_pos] = (np.abs(self.data.x_stim[cond_pos]) > type1_thresh[1]) * self.data.x_stim[cond_pos] + type1_bias[1]

        if (self.cfg.type1_noise_signal_dependency != 'none') or (self.cfg.enable_type1_param_noise == 2):
            type1_noise = compute_signal_dependent_type1_noise(
                x_stim=self.data.x_stim, type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency, **params_type1
            )
        else:
            type1_noise = params_type1['type1_noise']

        posterior = logistic(y_decval, type1_noise)
        likelihood_type1 = (self.data.d_dec == 1) * posterior + (self.data.d_dec == 0) * (1 - posterior)
        negll = np.sum(-np.log(np.maximum(likelihood_type1, self.cfg.min_type1_likelihood)))

        return (negll, params_type1, likelihood_type1, posterior) if final else negll

    def compute_type2_negll(self, params, type2_noise_type, ignore_warnings=False, final=False):
        """
        Negative log likelihood minimization

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 2 level.
        type2_noise_type : str
            Type 2 noise type: 'noisy_report', 'noisy_readout' or 'noisy_temperature'
        ignore_warnings : bool
            If True, suppress warnings during minimization.
        final : bool
            If True, return latent variables and parameters.
            Note: this has to be the final parameter in the method definition!

        Returns:
        --------
        By default, the method returns the negative log likelihood. However, depending on the arguments various
        combinations of variables are returned (see Parameters).
        negll_type2: float
            Negative (summed) log likelihood.
        """

        bl = self.cfg.paramset_type2.param_len
        params_type2 = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                        for i, (p, n) in enumerate(zip(self.cfg.paramset_type2.param_names, bl))}

        # if hasattr(self.data, 'z1_type1_evidence'):
        #     z1_type1_evidence = self.data.z1_type1_evidence_grid
        # else:
        # z1_type1_evidence = self.model.z1_type1_evidence_grid


        if self.cfg.type2_fitting_type == 'criteria':
            type2_likelihood_grid = self._compute_type2_likelihood_criteria(params_type2, type2_noise_type=type2_noise_type)
        else:
            type2_likelihood_grid = self._compute_type2_likelihood_continuous(params_type2, type2_noise_type=type2_noise_type)

        if final and ('type2_criteria' in params_type2) and (np.sum(params_type2['type2_criteria']) > 1.001):
            params_type2['type2_criteria'] = check_criteria_sum(params_type2['type2_criteria'])

        if type2_noise_type == 'noisy_temperature':
            type2_likelihood = type2_likelihood_grid  # in case of the noisy-temp model there is no grid
            type2_likelihood_grid = None
        else:
            if not self.cfg.experimental_include_incongruent_y_decval:
                type2_likelihood_grid[self.model.y_decval_grid_invalid] = np.nan
            # compute log likelihood
            type2_likelihood = np.nansum(self.model.y_decval_pmf_grid * type2_likelihood_grid, axis=1)
            # This is equivalent:
            # type2_cum_likelihood2 = np.trapezoid(self.model.y_decval_grid_pdf * np.nan_to_num(type2_likelihood_grid, 0),
            #                                      self.model.y_decval_grid)

        if self.cfg.experimental_min_uniform_type2_likelihood:
            # use an upper bound for the negative log likelihood based on a uniform 'guessing' model
            negll_type2 = min(self.model.uniform_type2_negll, -np.sum(np.log(np.maximum(type2_likelihood, 1e-200))))
        else:
            negll_type2 = -np.sum(np.log(np.maximum(type2_likelihood, self.cfg.min_type2_likelihood)))

        if final:
            if type2_noise_type != 'noisy_temperature':
                self.model.c_conf_grid = self._type1_evidence_to_confidence(self.model.z1_type1_evidence_grid, params_type2)
                if not self.cfg.experimental_include_incongruent_y_decval:
                    self.model.c_conf_grid[self.model.y_decval_grid_invalid] = np.nan
            self.model.c_conf = self._type1_evidence_to_confidence(self.model.z1_type1_evidence, params_type2)
            # self._compute_type2_likelihood_continuous(params_type2, type2_noise_type=type2_noise_type)
            # self._compute_type2_likelihood_criteria(params_type2, type2_noise_type=type2_noise_type)
            return dict(negll_type2=negll_type2, params_type2=params_type2, type2_likelihood_grid=type2_likelihood_grid,
                        type2_likelihood=type2_likelihood)
        else:
            return negll_type2


    def _compute_type2_likelihood_continuous(self, params_type2, type2_noise_type='noisy_report'):
        if self.cfg.experimental_wrap_type2_integration_window:
            wrap_neg = (self.cfg.type2_binsize -
                        np.abs(np.minimum(1, self.data.c_conf_2d + self.cfg.type2_binsize) - self.data.c_conf_2d))
            wrap_pos = (self.cfg.type2_binsize -
                        np.abs(np.maximum(0, self.data.c_conf_2d - self.cfg.type2_binsize) - self.data.c_conf_2d))
            binsize_neg, binsize_pos = self.cfg.type2_binsize + wrap_neg, self.cfg.type2_binsize + wrap_pos
        else:
            binsize_neg, binsize_pos = self.cfg.type2_binsize, self.cfg.type2_binsize

        if type2_noise_type == 'noisy_temperature':
            if self.cfg.experimental_disable_type2_binsize:
                data = self.data.c_conf
            else:
                data_lower = np.maximum(0, self.data.c_conf - binsize_neg)
                data_upper = np.minimum(1, self.data.c_conf + binsize_pos)
            dist = self.get_noisy_temperature_dist(params_type2)
        elif type2_noise_type == 'noisy_report':
            c_conf_grid = self._type1_evidence_to_confidence(self.model.z1_type1_evidence_grid, params_type2)
            if self.cfg.experimental_disable_type2_binsize:
                data = self.data.c_conf_2d
            else:
                data_lower = np.maximum(0, self.data.c_conf_2d - binsize_neg)
                data_upper = np.minimum(1, self.data.c_conf_2d + binsize_pos)
            dist = get_type2_dist(self.cfg.type2_noise_dist, type2_center=c_conf_grid, type2_noise=params_type2['type2_noise'],
                                  type2_noise_type='noisy_report')
        elif type2_noise_type == 'noisy_readout':
            if self.cfg.experimental_disable_type2_binsize:
                data = self._confidence_to_type1_evidence(self.data.c_conf_2d, params_type2)
            else:
                data_lower = self._confidence_to_type1_evidence(
                    np.maximum(0, self.data.c_conf_2d - binsize_neg), params_type2)
                data_upper = self._confidence_to_type1_evidence(
                    np.minimum(1, self.data.c_conf_2d + binsize_pos), params_type2)
            dist = get_type2_dist(self.cfg.type2_noise_dist, type2_center=self.model.z1_type1_evidence_grid, type2_noise=params_type2['type2_noise'],
                                  type2_noise_type='noisy_readout')

        if self.cfg.experimental_disable_type2_binsize:
            type2_likelihood = dist.pdf(data)
        else:
            type2_likelihood = dist.cdf(data_upper) - dist.cdf(data_lower)

        return type2_likelihood

    def get_noisy_temperature_dist(self, params_type2, mask=None):

        def __init__(self_):

            self_.type2_dist = get_type2_dist(self.cfg.type2_noise_dist, type2_center=self.model.params_type1['type1_noise'], type2_noise=params_type2['type2_noise'],
                                              type2_noise_type='noisy_temperature')
            self_.params_type2 = params_type2
            # self_.type1_noise_range = self_.type2_dist.ppf(self.model.quintiles_noisy_temperature)[:, None]
            # self_.type1_noise_range = np.maximum.accumulate(self_.type2_dist.ppf(self.model.quintiles_noisy_temperature))[:, None]
            self_.type1_noise_range = self_.type2_dist.ppf(self.model.quintiles_noisy_temperature)
            if np.any(np.diff(self_.type1_noise_range) <= 0):
                raise ValueError('Numerical instability in the type 2 noise distribution. The lower bound'
                                 'of the type 2 noise parameter might be too small.')
            self_.type1_noise_range = self_.type1_noise_range[:, None]


        def conf_to_decval(self_, confidence):
            z1 = self._confidence_to_type1_evidence(confidence, self_.params_type2, type1_noise=self_.type1_noise_range,
                                                    tile_on_type1_uncertainty=False)
            y = z1 * (self.data.d_dec_sign if mask is None else self.data.d_dec_sign[mask])
            return y

        def pdf(self_, data):
            y = self_.conf_to_decval(data)
            jac = (self.data.jacobian_noisy_temperature /
                   (self_.params_type2['type2_evidence_bias_mult'] if self.cfg.enable_type2_param_evidence_bias_mult else 1))
            # pdf_y = self.model.type1_dist.pdf(y)
            # slightly faster:
            ez = expit((y - self.model.y_decval) / (self.model.params_type1['type1_noise'] * np.sqrt(3) / np.pi))
            pdf_y = (ez * (1 - ez)) / (self.model.params_type1['type1_noise'] * np.sqrt(3) / np.pi)
            integrand = self_.type1_noise_range * self_.type2_dist.pdf(self_.type1_noise_range) * pdf_y
            pdf_ = jac * np.trapezoid(integrand, x=self_.type1_noise_range.squeeze(), axis=0)
            return pdf_

        def cdf(self_, data):
            y = self_.conf_to_decval(data)
            # cdf_y = self.model.type1_dist.cdf(y) - self.model.type1_dist.cdf(0) # slower!
            y_decval = self.model.y_decval if mask is None else self.model.y_decval[mask]
            cdf_y = np.abs(logistic(y_decval, self.model.params_type1['type1_noise']) -
                           logistic(y_decval - y, self.model.params_type1['type1_noise']))
            # (999, 13)
            # neg = y < 0
            # if neg.sum():
            #     cdf_y[neg] = -cdf_y[neg]
            # Marginalize type 1 noise
            integrand = cdf_y * self_.type2_dist.pdf(self_.type1_noise_range)
            cdf_ = np.trapezoid(integrand, x=self_.type1_noise_range.squeeze(), axis=0)
            return cdf_

        Dist = type("Dist", (), {"__init__": __init__, "pdf": pdf, "cdf": cdf, "conf_to_decval": conf_to_decval})

        return Dist()


    def _compute_type2_likelihood_criteria(self, params_type2, type2_noise_type='noisy_report'):

        type2_likelihood = np.empty_like(self.model.y_decval, float) if type2_noise_type == 'noisy_temperature' \
                      else np.empty_like(self.model.z1_type1_evidence_grid, float)

        if self.cfg.enable_type2_param_criteria:
            criteria_ = np.hstack((params_type2['type2_criteria'], 1))
        else:
            criteria_ = np.hstack((np.ones(self.cfg.n_discrete_confidence_levels - 1) / self.cfg.n_discrete_confidence_levels, 1))

        for i, crit in enumerate(criteria_):
            cnd = self.data.c_conf_discrete == i
            if cnd.sum():
                if i == 0:
                    lower_bin_edge = 0
                    upper_bin_edge = crit
                else:
                    lower_bin_edge = np.sum(criteria_[:i])
                    upper_bin_edge = np.sum(criteria_[:i+1])
                if type2_noise_type == 'noisy_report':
                    c_conf_grid = self._type1_evidence_to_confidence(self.model.z1_type1_evidence_grid, params_type2)
                    data_lower = lower_bin_edge
                    data_upper = min(1, upper_bin_edge)
                    type2_noise_dist = get_type2_dist(
                        self.cfg.type2_noise_dist, type2_center=c_conf_grid[cnd], type2_noise=params_type2['type2_noise'],
                        type2_noise_type='noisy_report'
                    )
                elif type2_noise_type == 'noisy_readout':
                    data_lower = self._confidence_to_type1_evidence(lower_bin_edge, params_type2, mask=cnd)
                    data_upper = self._confidence_to_type1_evidence(upper_bin_edge, params_type2, mask=cnd)
                    type2_noise_dist = get_type2_dist(
                        self.cfg.type2_noise_dist, type2_center=self.model.z1_type1_evidence_grid[cnd], type2_noise=params_type2['type2_noise'],
                        type2_noise_type='noisy_readout'
                    )
                elif type2_noise_type == 'noisy_temperature':
                    data_lower = lower_bin_edge
                    data_upper = min(1, upper_bin_edge)
                    type2_noise_dist = self.get_noisy_temperature_dist(params_type2, mask=cnd)

                type2_likelihood[cnd] = type2_noise_dist.cdf(data_upper) - type2_noise_dist.cdf(data_lower)  # noqa

        return type2_likelihood


    def _type1_evidence_to_confidence(self, z1_type1_evidence, params_type2):
        """
        Helper function to convert type 1 evidence to confidence
        """
        return type1_evidence_to_confidence(
            z1_type1_evidence=z1_type1_evidence, y_decval=self.model.y_decval_grid,
            x_stim=self.data.x_stim, type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
            **self.model.params_type1, **params_type2
        )

    def _type1_noise_to_confidence(self, type1_noise, params_type2):
        """
        Helper function to convert type 1 noise to confidence
        """
        return type1_noise_to_confidence(
            y_decval=self.model.y_decval_grid,
            x_stim=self.data.x_stim, type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
            **{**self.model.params_type1, **dict(type1_noise=type1_noise)}, **params_type2
        )

    def _confidence_to_type1_evidence(self, c_conf, params_type2, type1_noise=None, tile_on_type1_uncertainty=True,
                                      mask=None):
        """
        Helper function to convert confidence to type 1 evidence
        """
        return confidence_to_type1_evidence(
            c_conf=c_conf,
            x_stim=self.data.x_stim if mask is None else self.data.x_stim[mask],
            y_decval=self.model.y_decval_grid if mask is None else self.model.y_decval_grid[mask],
            type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
            tile_on_type1_uncertainty=tile_on_type1_uncertainty,
            **(self.model.params_type1 if type1_noise is None else
               {**self.model.params_type1, **dict(type1_noise=type1_noise)}),
            **params_type2
        )

    def _confidence_to_type1_noise(self, c_conf, params_type2, mask=None):
        """
        Helper function to convert confidence to type 1 noise
        """
        return confidence_to_type1_noise(
            c_conf=c_conf,
            x_stim=self.data.x_stim if mask is None else self.data.x_stim[mask],
            y_decval=self.model.y_decval_grid if mask is None else self.model.y_decval_grid[mask],
            type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
            **self.model.params_type1, **params_type2
        )

    def _compute_decision_values(self):
        """
        Compute type 1 decision values
        """

        type1_noise_trialwise = compute_signal_dependent_type1_noise(
            x_stim=self.data.x_stim_2d, type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency, **self.model.params_type1
        )
        type1_thresh = _check_param(self.model.params_type1['type1_thresh'] if self.cfg.enable_type1_param_thresh else 0)
        type1_bias = _check_param(self.model.params_type1['type1_bias'] if self.cfg.enable_type1_param_bias else 0)

        cond_neg, cond_pos = (self.data.x_stim_2d < 0).squeeze(), (self.data.x_stim_2d >= 0).squeeze()
        y_decval = np.full(self.data.x_stim_2d.shape, np.nan)
        y_decval[cond_neg] = (np.abs(self.data.x_stim_2d[cond_neg]) >= type1_thresh[0]) * \
            self.data.x_stim_2d[cond_neg] + type1_bias[0]
        y_decval[cond_pos] = (np.abs(self.data.x_stim_2d[cond_pos]) >= type1_thresh[1]) * \
            self.data.x_stim_2d[cond_pos] + type1_bias[1]

        range_ = np.linspace(0, self.cfg.y_decval_range_nsds, int((self.cfg.y_decval_range_nbins + 1) / 2))[1:]
        yrange = np.hstack((-range_[::-1], 0, range_))
        self.model.y_decval_grid = np.full((y_decval.shape[0], yrange.shape[0]), np.nan)
        self.model.y_decval_grid[cond_neg] = y_decval[cond_neg] + yrange * np.mean(type1_noise_trialwise[cond_neg])
        self.model.y_decval_grid[cond_pos] = y_decval[cond_pos] + yrange * np.mean(type1_noise_trialwise[cond_pos])

        logistic_neg = logistic_dist(loc=y_decval[cond_neg], scale=type1_noise_trialwise[cond_neg] * np.sqrt(3) / np.pi)
        logistic_pos = logistic_dist(loc=y_decval[cond_pos], scale=type1_noise_trialwise[cond_pos] * np.sqrt(3) / np.pi)
        margin_neg = type1_noise_trialwise[cond_neg] * self.cfg.y_decval_range_nsds / self.cfg.y_decval_range_nbins
        margin_pos = type1_noise_trialwise[cond_pos] * self.cfg.y_decval_range_nsds / self.cfg.y_decval_range_nbins
        if self.cfg.type2_noise_type != 'noisy_temperature':
            self.model.y_decval_pmf_grid = np.full(self.model.y_decval_grid.shape, np.nan)
            self.model.y_decval_pmf_grid[cond_neg] = (logistic_neg.cdf(self.model.y_decval_grid[cond_neg] + margin_neg) -
                                                      logistic_neg.cdf(self.model.y_decval_grid[cond_neg] - margin_neg))
            self.model.y_decval_pmf_grid[cond_pos] = (logistic_pos.cdf(self.model.y_decval_grid[cond_pos] + margin_pos) -
                                                      logistic_pos.cdf(self.model.y_decval_grid[cond_pos] - margin_pos))
            # pdf is only necessary if we use trapezoid integration at the type 2 level
            # self.model.y_decval_pdf_grid = np.full(self.model.y_decval_grid.shape, np.nan)
            # self.model.y_decval_pdf_grid[cond_neg] = logistic_neg.pdf(self.model.y_decval_grid[cond_neg])
            # self.model.y_decval_pdf_grid[cond_pos] = logistic_pos.pdf(self.model.y_decval_grid[cond_pos])
            # normalize PMF
            self.model.y_decval_pmf_grid = self.model.y_decval_pmf_grid / self.model.y_decval_pmf_grid.sum(axis=1).reshape(-1, 1)
            # invalidate invalid decision values
            if not self.cfg.experimental_include_incongruent_y_decval:
                self.model.y_decval_grid_invalid = np.sign(self.model.y_decval_grid) != \
                                                   np.sign(self.data.d_dec_2d - 0.5)
                self.model.y_decval_pmf_grid[self.model.y_decval_grid_invalid] = np.nan

            if self.cfg.experimental_min_uniform_type2_likelihood:
                # self.cfg.type2_binsize*2 is the probability for a given confidence rating assuming a uniform
                # distribution for confidence. This 'confidence guessing model' serves as a upper bound for the
                # negative log likelihood.
                min_type2_likelihood_grid = self.cfg.type2_binsize * 2 * np.ones(self.model.y_decval_pmf_grid.shape)
                min_type2_likelihood = np.nansum(min_type2_likelihood_grid * self.model.y_decval_pmf_grid, axis=1)
                self.model.uniform_type2_negll = -np.log(min_type2_likelihood).sum()

            self.model.z1_type1_evidence_grid = np.abs(self.model.y_decval_grid)
        self.model.y_decval = y_decval.flatten()
        self.model.z1_type1_evidence = np.abs(self.model.y_decval)


    def _check_fit(self):
        if not self.type1_is_fitted and not self.type2_is_fitted:
            raise RuntimeError('Please fit the model before plotting.')
        elif self.type1_is_fitted and not self.type2_is_fitted:
            raise RuntimeError('Only the type 1 level was fitted. Please also fit the type 2 level to plot'
                               'a link function.')

    def plot_evidence_versus_confidence(self, **kwargs):
        self._check_fit()
        plot_evidence_versus_confidence(
            self.data.x_stim, self.data.c_conf, self.model.y_decval, self.model.params, cfg=self.cfg, **kwargs
        )

    def plot_confidence_dist(self, **kwargs):
        self._check_fit()
        varlik = self.model.z1_type1_evidence_grid if self.cfg.type2_noise_type == 'noisy_readout' else self.model.c_conf_grid
        plot_confidence_dist(
            self.cfg, self.data.x_stim, self.data.c_conf, self.model.params, var_likelihood_grid=varlik,
            y_decval_grid=self.model.y_decval_grid,
            likelihood_weighting=self.model.y_decval_pmf_grid, **kwargs
        )


def load_dataset(name, verbose=True, return_data_only=False):
    import gzip
    path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'demo_data', f'example_data_{name}.pkl.gz')
    if os.path.exists(path):
        with gzip.open(path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise FileNotFoundError(f'[Dataset does not exist!] No such file: {path}')

    if verbose:
        print_dataset_characteristics(dataset)

    if return_data_only:
        return (dataset.x_stim, dataset.d_dec, dataset.params)
    else:
        return dataset