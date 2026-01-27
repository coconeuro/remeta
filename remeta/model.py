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
from .fit import subject_estimation, group_estimation
from .gendata import simu_data
from .modelspec import ModelData, Data, Summary
from .plot import plot_evidence_versus_confidence, plot_confidence_dist
from .transform import (compute_signal_dependent_type1_noise, logistic, type1_evidence_to_confidence,
                        confidence_to_type1_evidence, confidence_to_type1_noise, type1_noise_to_confidence,
                        check_criteria_sum, compute_nonlinear_encoding)
from .util import _check_param, TAB, SP2, maxfloat, print_dataset_characteristics, print_warnings, empty_list

class ReMeta:

    def __init__(self, cfg=None, **kwargs):
        """
        Main class of the ReMeta toolbox

        Parameters
        ----------
        cfg : remeta.Configuration
            Configuration object. If None is passed, the default configuration is used (but see kwargs).
        kwargs : dict
            The kwargs dictionary is parsed for keywords that match keywords of Configuration; in case of a match,
            the configuration is set.
        """

        if cfg is None:
            # Set configuration attributes that match keyword arguments
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in Configuration.__dict__}
            self.cfg = Configuration(**cfg_kwargs)
        else:
            self.cfg = cfg
        self.cfg.setup()

        self.modeldata = ModelData(cfg=self.cfg)
        self.data = None
        self.result = None

        self.type1_is_fitted = False
        self.type2_is_fitted = False


    def fit(self, stimuli, choices, confidence, precomputed_parameters=None, initial_guess=None, verbose=True,
            ignore_warnings=False):
        """
        Fit type 1 and type 2 parameters

        Parameters
        ----------
        stimuli : array-like of shape (n_samples,) or array-like of shape (n_subjects, n_samples)
                    or list of array-like (variable n_samples per subject)
            Array of signed stimulus intensity values, where the sign codes the stimulus category (cat 1: -, cat2: +)
            and the absolut value codes the intensity. Must be normalized to [-1; 1], or set
            `normalize_stimuli_by_max=True`.
        choices : array-like of shape (n_samples,) or array-like of shape (n_subjects, n_samples)
                    or list of array-like (variable n_samples per subject)
            Array of choices coded as 0 (or alternatively -1) for the negative stimuli category and 1 for the positive
            stimulus category.
        confidence : array-like of shape (n_samples,) or array-like of shape (n_subjects, n_samples)
                    or list of array-like (variable n_samples per subject)
            Confidence ratings; must be normalized to the range [0;1].
        precomputed_parameters : dict
            Provide pre-computed parameters. A dictionary with all parameters defined by the model must be passed. This
            can sometimes be useful to obtain information from the model without having to fit the model.
            [ToDO: which information?]
        initial_guess : array-like of length n_params
            For testing: provide an initial guess for parameters
        verbose : bool
            If True, information of the model fitting procedure is printed.
        ignore_warnings : bool
            If True, warnings during model fitting are supressed.
        """

        # Instantiate util.Data object
        self.data = Data(self.cfg, stimuli, choices, confidence)

        self.result = Summary(self.data, self.cfg)

        self.fit_type1(precomputed_parameters=precomputed_parameters, initial_guess=initial_guess, verbose=verbose,
                       ignore_warnings=ignore_warnings)

        if not self.cfg.skip_type2:
            self.fit_type2(precomputed_parameters=precomputed_parameters, initial_guess=initial_guess, verbose=verbose,
                           ignore_warnings=ignore_warnings)


    def fit_type1(self, stimuli=None, choices=None, confidence=None, precomputed_parameters=None, initial_guess=None,
                  verbose=True, ignore_warnings=False):

        if self.data is None:
            if stimuli is None or choices is None:
                raise ValueError('If the data attribute of the ReMeta instance is None, at least stimuli '
                                 'and choices have to be passed to fits_type1_subject()')
            else:
                self.data = Data(self.cfg, stimuli, choices, confidence)

        self.result = Summary(self.data, self.cfg)

        if verbose:
            print('\n+++ Type 1 level +++')
        with warnings.catch_warnings(record=True) as w:
        # with (warnings.catch_warnings()):
            warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize',
                                    message='delta_grad == 0.0. Check if the approximated function is linear. If the '
                                            'function is linear better results can be obtained by defining the Hessian '
                                            'as zero instead of using quasi-Newton approximations.')
            # if ignore_warnings:
            #     warnings.filterwarnings('ignore')
            if isinstance(precomputed_parameters, dict):
                if not np.all([p in precomputed_parameters for p in self.cfg.paramset_type1.param_names_flat]):
                    raise ValueError('Set of precomputed type 1 parameters is incomplete.')
                fits_type1_subject = [OptimizeResult(
                    x=[precomputed_parameters[p] for p in self.cfg.paramset_type1.param_names_flat],
                    fun=self.compute_type1_negll([precomputed_parameters[p] for p in self.cfg.paramset_type1.param_names_flat], s)
                ) for s in range(self.data.nsubjects)]
            else:
                fits_type1_subject, fit_type1_group = None, None
                if self.cfg.paramset_type1.nparams > 0:

                    if verbose:
                        print(f'{SP2}Subject-level estimation (MLE)')
                        tind = timeit.default_timer()

                    # Single-subject fits via MLE
                    fits_type1_subject = [None for _ in range(self.data.nsubjects)]
                    for s in range(self.data.nsubjects):
                        fits_type1_subject[s] = subject_estimation(
                            self.compute_type1_negll, self.cfg.paramset_type1, args=[s],
                            gridsearch=self.cfg.optim_type1_gridsearch,
                            scipy_solvers=self.cfg.optim_type1_scipy_solvers,
                            grid_multiproc=self.cfg.optim_grid_multiproc,
                            minimize_along_grid=self.cfg.optim_type1_minimize_along_grid,
                            global_minimization=self.cfg.optim_type1_global_minimization,
                            fine_gridsearch=self.cfg.optim_type1_fine_gridsearch,
                            guess=initial_guess,
                            verbose=verbose
                        )
                    # Store single-subject results
                    params_subject = [fits_type1_subject[s].x for s in range(self.data.nsubjects)]
                    self.result.type1.subject.store(
                        self.cfg, self.data,
                        params=params_subject, fun=self.compute_type1_negll, args=[],
                        stage='type1', fit=fits_type1_subject
                    )
                    if verbose:
                        print(f'{TAB}.. finished ({timeit.default_timer() - tind:.1f} secs).')

                    if self.data.nsubjects > 1:
                        idx_fe = np.array([i for i, p in enumerate(self.cfg.paramset_type1.parameters_flat.values()) if p.group == 'fixed'])
                        idx_re = np.array([i for i, p in enumerate(self.cfg.paramset_type1.parameters_flat.values()) if p.group == 'random'])
                        if (len(idx_fe) > 0) or (len(idx_re) > 0):
                            fit_type1_group = group_estimation(
                                fun=self.compute_type1_negll,
                                nsubjects=self.data.nsubjects,
                                params_init=params_subject,
                                bounds=self.cfg.paramset_type1.bounds,
                                idx_fe=idx_fe,
                                idx_re=idx_re,
                                max_iter=30, sigma_floor=1e-3, damping=0.5,
                                verbose=verbose
                            )
                            self.result.type1.init_group()
                            self.result.type1.group.store(
                                self.cfg, self.data,
                                params=[fit_type1_group.x[s] for s in range(self.data.nsubjects)],
                                fun=self.compute_type1_negll, args=[],
                                stage='type1',
                                pop_mean_sd=fit_type1_group.x_re_pop_mean_sd
                            )
            self.result.type1.store(cfg=self.cfg, data=self.data, fun=self.compute_type1_negll, args=[])
            if verbose:
                self.result.type1.report_fit(self.cfg)
            self.type1_is_fitted = True

        if not ignore_warnings and verbose:
            print_warnings(w)

        if self.cfg.skip_type2:
            self.result.store(self.cfg)

        if verbose:
            print('Type 1 level finished')


    def fit_type2(self, precomputed_parameters=None, initial_guess=None, verbose=True, ignore_warnings=False):

        # compute decision values
        self._compute_decision_values()

        if self.cfg.type2_noise_type == 'noisy_temperature':
            # We precompute a few things for noisy_temperature model
            self.modeldata.precomputed.jacobian_noisy_temperature = \
                [((2 * np.sqrt(3)) / np.pi) / (1 - np.minimum(1-1e-8, self.data.c_conf[s])**2) for s in range(self.data.nsubjects)]
            self.modeldata.precomputed.quintiles_noisy_temperature = np.arange(self.cfg.resolution_noisy_temperature, 1, self.cfg.resolution_noisy_temperature)

        if verbose:
            print('\n+++ Type 2 level +++')

        args_type2 = [self.cfg.type2_noise_type]
        if precomputed_parameters is not None:
            if not np.all([p in precomputed_parameters for p in self.cfg.paramset_type2.param_names_flat]):
                raise ValueError('Set of precomputed type 2 parameters is incomplete.')
            self.result.type2.params = [{p: precomputed_parameters[p] for p in self.cfg.paramset_type2.param_names_flat}] * self.data.nsubjects
            fits_type2_subject = [OptimizeResult(
                x=[precomputed_parameters[p] for p in self.cfg.paramset_type2.param_names_flat],
                fun=self.fun_negll_type2([precomputed_parameters[p] for p in self.cfg.paramset_type2.param_names_flat])
            )] * self.data.nsubjects
            negll_type2 = np.empty(self.data.nsubjects)
            params_type2 = [np.empty(self.cfg.paramset_type2.nparams) for _ in range(self.data.nsubjects)]
            type2_likelihood_grid = [np.empty((self.data.nsamples[s], self.cfg.y_decval_range_nbins)) for s in range(self.data.nsubjects)]
            type2_likelihood = [np.empty(self.data.nsamples[s]) for s in range(self.data.nsubjects)]
            for s in range(self.data.nsubjects):
                negll_type2[s], params_type2[s], type2_likelihood_grid[s], type2_likelihood[s] = \
                    self.compute_type2_negll(self.result.type2.params[s], s, *args_type2, save_target='subject')
            self.modeldata.store_type2(fits_type2_subject, None, negll_type2=negll_type2, negll_type2_true=None, params_type2=params_type2,
                                       type2_likelihood=type2_likelihood, type2_likelihood_grid=type2_likelihood_grid)
        else:
            with warnings.catch_warnings(record=True) as w:  # noqa
                warnings.filterwarnings('ignore', module='scipy.optimize')
                if self.cfg.paramset_type2.nparams > 0:

                    if verbose:
                        print(f'{SP2}Subject-level estimation (MLE)')
                        tind = timeit.default_timer()
                        # print(f'{SP2}Scipy solvers: {self.cfg.optim_type1_scipy_solvers}')

                    fits_type2_subject = [None for _ in range(self.data.nsubjects)]
                    for s in range(self.data.nsubjects):
                        # if verbose and (self.data.nsubjects > 1):
                        #     print(f'{TAB}Subject {s + 1} / {self.data.nsubjects}')
                        fits_type2_subject[s] = subject_estimation(
                            self.compute_type2_negll, self.cfg.paramset_type2, args=[s] + args_type2,
                            gridsearch=self.cfg.optim_type2_gridsearch, grid_multiproc=self.cfg.optim_grid_multiproc,
                            minimize_along_grid=self.cfg.optim_type2_minimize_along_grid,
                            global_minimization=self.cfg.optim_type2_global_minimization,
                            fine_gridsearch=self.cfg.optim_type2_fine_gridsearch,
                            scipy_solvers=self.cfg.optim_type2_scipy_solvers, slsqp_epsilon=self.cfg.optim_type2_slsqp_epsilon,
                            guess=initial_guess,
                            verbose=verbose
                        )

                    # Store single-subject results
                    params_subject = [fits_type2_subject[s].x for s in range(self.data.nsubjects)]
                    self.result.type2.subject.store(
                        self.cfg, self.data,
                        params=params_subject, fun=self.compute_type2_negll, args=args_type2,
                        stage='type2', fit=fits_type2_subject
                    )
                    if verbose:
                        print(f'{TAB}.. finished ({timeit.default_timer() - tind:.1f} secs).')

                    # Group fit
                    if self.data.nsubjects > 1:
                        idx_fe = np.array([i for i, p in enumerate(self.cfg.paramset_type2.parameters_flat.values()) if p.group == 'fixed'])
                        idx_re = np.array([i for i, p in enumerate(self.cfg.paramset_type2.parameters_flat.values()) if p.group == 'random'])
                        if (len(idx_fe) > 0) or (len(idx_re) > 0):
                            fit_type2_group = group_estimation(
                                fun=self.compute_type2_negll,
                                nsubjects=self.data.nsubjects,
                                params_init=params_subject,
                                bounds=self.cfg.paramset_type2.bounds,
                                idx_fe=idx_fe,
                                idx_re=idx_re,
                                max_iter=30, sigma_floor=1e-3, damping=0.5,
                                verbose=verbose
                            )
                            self.result.type2.init_group()
                            self.result.type2.group.store(
                                self.cfg, self.data,
                                params=[fit_type2_group.x[s] for s in range(self.data.nsubjects)],
                                fun=self.compute_type2_negll, args=args_type2,
                                stage='type2',
                                pop_mean_sd=fit_type2_group.x_re_pop_mean_sd
                            )
            self.result.type2.store(cfg=self.cfg, data=self.data, fun=self.compute_type2_negll, args=args_type2)
            if verbose:
                self.result.type2.report_fit(self.cfg)

        self.result.store(self.cfg)
        self.type2_is_fitted = True

        if not ignore_warnings:
            print_warnings(w)
        if verbose:
            print('Type 2 level finished')

    def summary(self, generative=True, generative_nsamples=1000, squeeze=True):
        """
        Provides information about the model fit.

        Parameters
        ----------
        generative : bool
            If True, compare model predictions of confidence with empirical confidence by repeatedly sampling from
            the generative model.
        generative_nsamples : int
            Number of samples used for the generative model (higher = more accurate).
        squeeze : bool (default: True)
            If True, return flattened results in case of only a single participant.

        Returns
        ----------
        summary : dataclass
            Information about model fit.
        """

        if self.type2_is_fitted and generative:
            c_conf_generative = [np.empty((generative_nsamples, self.data.nsamples[s])) for s in range(self.data.nsubjects)]
            for s in range(self.data.nsubjects):
                c_conf_generative[s] = simu_data(self.result.params[s], nsubjects=generative_nsamples, nsamples=self.data.nsamples[s],
                                                 cfg=self.cfg, stimuli_external=self.data.x_stim[s], verbose=False).confidence
        else:
            c_conf_generative = None
        model_summary = self.result.summary(
            c_conf_empirical=self.data.c_conf, c_conf_generative=c_conf_generative, squeeze=squeeze
        )
        return model_summary


    def compute_type1_negll(self, params, sub_ind=0, save_target=None):
        """
        Likelihood function for the type 1 level

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 1 level.
        sub_ind : int
            Subject index (only valid for 2d multi-subject datasets)
        save_target : None | str
            If 'subject' or 'group', store latent variables and parameters.

        Returns:
        --------
        negll: float
            Negative (summed) log likelihood.
        """

        bl = self.cfg.paramset_type1.param_len_list
        params_type1 = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                        for i, (p, n) in enumerate(zip(self.cfg.paramset_type1.param_names, bl))}
        type1_thresh = _check_param(params_type1['type1_thresh'] if self.cfg.enable_type1_param_thresh else 0)
        type1_bias = _check_param(params_type1['type1_bias'] if self.cfg.enable_type1_param_bias else 0)

        if self.cfg.enable_type1_param_nonlinear_encoding_gain:
            x_stim_transform = compute_nonlinear_encoding(
                self.data.x_stim[sub_ind], params_type1['type1_nonlinear_encoding_gain'],
                params_type1['type1_nonlinear_encoding_transition'] if self.cfg.enable_type1_param_nonlinear_encoding_transition else 1)
        else:
            x_stim_transform = self.data.x_stim[sub_ind]

        cond_neg, cond_pos = self.data.x_stim[sub_ind] < 0, self.data.x_stim[sub_ind] >= 0
        y_decval = np.full(self.data.x_stim[sub_ind].shape, np.nan)
        y_decval[cond_neg] = (np.abs(x_stim_transform[cond_neg]) > type1_thresh[0]) * x_stim_transform[cond_neg] + type1_bias[0]
        y_decval[cond_pos] = (np.abs(x_stim_transform[cond_pos]) > type1_thresh[1]) * x_stim_transform[cond_pos] + type1_bias[1]

        if (self.cfg.type1_noise_signal_dependency != 'none') or (self.cfg.enable_type1_param_noise == 2):
            type1_noise = compute_signal_dependent_type1_noise(
                x_stim=x_stim_transform, type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency, **params_type1
            )
        else:
            type1_noise = params_type1['type1_noise']

        posterior = logistic(y_decval, type1_noise)
        likelihood = (self.data.d_dec[sub_ind] == 1) * posterior + (self.data.d_dec[sub_ind] == 0) * (1 - posterior)
        negll = np.sum(-np.log(np.maximum(likelihood, self.cfg.min_type1_likelihood)))

        # Add negative log likelihood of (fixed) Normal priors
        priors = [((params_type1[k] - p.prior[0])**2) / (2 * p.prior[1]**2) for k, p in
                  self.cfg.paramset_type1.parameters_flat.items() if isinstance(p.prior, tuple)]
        if len(priors) > 0:
            negll += np.sum(priors)

        if save_target:
            getattr(self.result.type1, save_target).params[sub_ind] = params_type1
            getattr(self.result.type1, save_target).negll[sub_ind] = negll
            if self.modeldata.type1_posterior is None:
                self.modeldata.type1_posterior = empty_list(self.data.nsubjects, self.data.nsamples)
                self.modeldata.type1_likelihood = empty_list(self.data.nsubjects, self.data.nsamples)
            self.modeldata.type1_posterior[sub_ind] = posterior
            self.modeldata.type1_likelihood[sub_ind] = likelihood
        return negll

    def compute_type2_negll(self, params, sub_ind=0, type2_noise_type='noisy_report', save_target=None):
        """
        Negative log likelihood minimization

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 2 level.
        sub_ind : int
            Subject index (only valid for 2d multi-subject datasets)
        type2_noise_type : str
            Type 2 noise type: 'noisy_report', 'noisy_readout' or 'noisy_temperature'
        save_target : None | str
            If 'subject' or 'group', store latent variables and parameters.

        Returns:
        --------
        By default, the method returns the negative log likelihood. However, depending on the arguments various
        combinations of variables are returned (see Parameters).
        negll: float
            Negative (summed) log likelihood.
        """

        bl = self.cfg.paramset_type2.param_len_list
        params_type2 = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
                        for i, (p, n) in enumerate(zip(self.cfg.paramset_type2.param_names, bl))}

        if self.cfg.type2_fitting_type == 'criteria':
            likelihood_grid = self._compute_type2_likelihood_criteria(params_type2, sub_ind, type2_noise_type=type2_noise_type)
        else:
            likelihood_grid = self._compute_type2_likelihood_continuous(params_type2, sub_ind, type2_noise_type=type2_noise_type)

        if save_target is not None and ('type2_criteria' in params_type2) and (np.sum(params_type2['type2_criteria']) > 1.001):
            params_type2['type2_criteria'] = check_criteria_sum(params_type2['type2_criteria'])

        if type2_noise_type == 'noisy_temperature':
            likelihood = likelihood_grid  # in case of the noisy-temp model there is no grid
            likelihood_grid = None
        else:
            if not self.cfg.experimental_include_incongruent_y_decval:
                likelihood_grid[self.modeldata.y_decval_grid_invalid[sub_ind]] = np.nan
            # compute log likelihood
            likelihood = np.nansum(self.modeldata.y_decval_pmf_grid[sub_ind] * likelihood_grid, axis=1)
            # This is equivalent:
            # type2_cum_likelihood2 = np.trapezoid(self.model.y_decval_grid_pdf * np.nan_to_num(likelihood_grid, 0),
            #                                      self.model.y_decval_grid)

        if self.cfg.experimental_min_uniform_type2_likelihood:
            # use an upper bound for the negative log likelihood based on a uniform 'guessing' model
            negll = min(self.modeldata.precomputed.uniform_type2_negll[sub_ind], -np.sum(np.log(np.maximum(likelihood, 1e-200))))
        else:
            negll = -np.sum(np.log(np.maximum(likelihood, self.cfg.min_type2_likelihood)))

        # Add negative log likelihood of (fixed) Normal priors
        priors = [((params_type2[k] - p.prior[0])**2) / (2 * p.prior[1]**2) for k, p in
                  self.cfg.paramset_type2.parameters_flat.items() if isinstance(p.prior, tuple)]
        if len(priors) > 0:
            negll += np.sum(priors)

        if save_target is not None:
            if type2_noise_type != 'noisy_temperature':
                if self.modeldata.c_conf_grid is None:
                    self.modeldata.c_conf_grid = empty_list(self.data.nsubjects, self.data.nsamples, self.cfg.y_decval_range_nbins)
                self.modeldata.c_conf_grid[sub_ind] = self._type1_evidence_to_confidence(self.modeldata.z1_type1_evidence_grid[sub_ind], params_type2, sub_ind)
                if not self.cfg.experimental_include_incongruent_y_decval:
                    self.modeldata.c_conf_grid[sub_ind][self.modeldata.y_decval_grid_invalid[sub_ind]] = np.nan
            if self.modeldata.c_conf is None:
                self.modeldata.c_conf = empty_list(self.data.nsubjects, self.data.nsamples)
            self.modeldata.c_conf[sub_ind] = self._type1_evidence_to_confidence(self.modeldata.z1_type1_evidence[sub_ind], params_type2, sub_ind)
            if self.modeldata.type2_likelihood is None:
                self.modeldata.type2_likelihood = empty_list(self.data.nsubjects, self.data.nsamples)
                self.modeldata.type2_likelihood_grid = empty_list(self.data.nsubjects, self.data.nsamples, self.cfg.y_decval_range_nbins)
            self.modeldata.type2_likelihood[sub_ind] = likelihood
            self.modeldata.type2_likelihood_grid[sub_ind] = likelihood_grid

            getattr(self.result.type2, save_target).params[sub_ind] = params_type2
            getattr(self.result.type2, save_target).negll[sub_ind] = negll

        return negll


    def _compute_type2_likelihood_continuous(self, params_type2, sub_ind, type2_noise_type='noisy_report'):
        if self.cfg.experimental_wrap_type2_integration_window:
            wrap_neg = (self.cfg.type2_binsize -
                        np.abs(np.minimum(1, self.data.c_conf_3d[sub_ind] + self.cfg.type2_binsize) - self.data.c_conf_3d[sub_ind]))
            wrap_pos = (self.cfg.type2_binsize -
                        np.abs(np.maximum(0, self.data.c_conf_3d[sub_ind] - self.cfg.type2_binsize) - self.data.c_conf_3d[sub_ind]))
            binsize_neg, binsize_pos = self.cfg.type2_binsize + wrap_neg, self.cfg.type2_binsize + wrap_pos
        else:
            binsize_neg, binsize_pos = self.cfg.type2_binsize, self.cfg.type2_binsize

        if type2_noise_type == 'noisy_temperature':
            if self.cfg.experimental_disable_type2_binsize:
                data = self.data.c_conf[sub_ind]
            else:
                data_lower = np.maximum(0, self.data.c_conf[sub_ind] - binsize_neg)
                data_upper = np.minimum(1, self.data.c_conf[sub_ind] + binsize_pos)
            dist = self.get_noisy_temperature_dist(params_type2)
        elif type2_noise_type == 'noisy_report':
            c_conf_grid = self._type1_evidence_to_confidence(self.modeldata.z1_type1_evidence_grid[sub_ind], params_type2, sub_ind)
            if self.cfg.experimental_disable_type2_binsize:
                data = self.data.c_conf_3d[sub_ind]
            else:
                data_lower = np.maximum(0, self.data.c_conf_3d[sub_ind] - binsize_neg)
                data_upper = np.minimum(1, self.data.c_conf_3d[sub_ind] + binsize_pos)
            dist = get_type2_dist(self.cfg.type2_noise_dist, type2_center=c_conf_grid, type2_noise=params_type2['type2_noise'],
                                  type2_noise_type='noisy_report')
        elif type2_noise_type == 'noisy_readout':
            if self.cfg.experimental_disable_type2_binsize:
                data = self._confidence_to_type1_evidence(self.data.c_conf_3d[sub_ind], params_type2, sub_ind)
            else:
                data_lower = self._confidence_to_type1_evidence(
                    np.maximum(0, self.data.c_conf_3d[sub_ind] - binsize_neg), params_type2, sub_ind)
                data_upper = self._confidence_to_type1_evidence(
                    np.minimum(1, self.data.c_conf_3d[sub_ind] + binsize_pos), params_type2, sub_ind)
            dist = get_type2_dist(self.cfg.type2_noise_dist, type2_center=self.modeldata.z1_type1_evidence_grid[sub_ind], type2_noise=params_type2['type2_noise'],
                                  type2_noise_type='noisy_readout')

        if self.cfg.experimental_disable_type2_binsize:
            type2_likelihood = dist.pdf(data)
        else:
            type2_likelihood = dist.cdf(data_upper) - dist.cdf(data_lower)

        return type2_likelihood

    def get_noisy_temperature_dist(self, params_type2, sub_ind=0, mask=None):

        def __init__(self_):

            self_.type2_dist = get_type2_dist(self.cfg.type2_noise_dist, type2_center=self.result.type1.params[sub_ind]['type1_noise'], type2_noise=params_type2['type2_noise'],
                                              type2_noise_type='noisy_temperature')
            self_.params_type2 = params_type2
            # self_.type1_noise_range = self_.type2_dist.ppf(self.modeldata.precomputed_variables.quintiles_noisy_temperature)[:, None]
            # self_.type1_noise_range = np.maximum.accumulate(self_.type2_dist.ppf(self.modeldata.precomputed_variables.quintiles_noisy_temperature))[:, None]
            self_.type1_noise_range = self_.type2_dist.ppf(self.modeldata.precomputed.quintiles_noisy_temperature)
            if np.any(np.diff(self_.type1_noise_range) <= 0):
                raise ValueError('Numerical instability in the type 2 noise distribution. The lower bound'
                                 'of the type 2 noise parameter might be too small.')
            self_.type1_noise_range = self_.type1_noise_range[:, None]


        def conf_to_decval(self_, confidence):
            z1 = self._confidence_to_type1_evidence(confidence, self_.params_type2, type1_noise=self_.type1_noise_range,
                                                    tile_on_type1_uncertainty=False)
            y = z1 * (self.data.d_dec_sign[sub_ind] if mask is None else self.data.d_dec_sign[sub_ind][mask])
            return y

        def pdf(self_, data):
            y = self_.conf_to_decval(data)
            jac = (self.modeldata.self.precomputed.jacobian_noisy_temperature[sub_ind][sub_ind] /
                   (self_.params_type2['type2_evidence_bias_mult'] if self.cfg.enable_type2_param_evidence_bias_mult else 1))
            # pdf_y = self.model.type1_dist.pdf(y)
            # slightly faster:
            ez = expit((y - self.modeldata.y_decval[sub_ind]) / (self.result.type1.params[sub_ind]['type1_noise'] * np.sqrt(3) / np.pi))
            pdf_y = (ez * (1 - ez)) / (self.result.type1.params[sub_ind]['type1_noise'] * np.sqrt(3) / np.pi)
            integrand = self_.type1_noise_range * self_.type2_dist.pdf(self_.type1_noise_range) * pdf_y
            pdf_ = jac * np.trapezoid(integrand, x=self_.type1_noise_range.squeeze(), axis=0)
            return pdf_

        def cdf(self_, data):
            y = self_.conf_to_decval(data)
            # cdf_y = self.model.type1_dist.cdf(y) - self.model.type1_dist.cdf(0) # slower!
            y_decval = self.modeldata.y_decval[sub_ind] if mask is None else self.modeldata.y_decval[sub_ind][mask]
            cdf_y = np.abs(logistic(y_decval, self.result.type1.params[sub_ind]['type1_noise']) -
                           logistic(y_decval - y, self.result.type1.params[sub_ind]['type1_noise']))
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


    def _compute_type2_likelihood_criteria(self, params_type2, sub_ind, type2_noise_type='noisy_report'):

        type2_likelihood = np.empty_like(self.modeldata.y_decval[sub_ind], float) if type2_noise_type == 'noisy_temperature' \
                      else np.empty_like(self.modeldata.z1_type1_evidence_grid[sub_ind], float)

        if self.cfg.enable_type2_param_criteria:
            criteria_ = np.hstack((params_type2['type2_criteria'], 1))
        else:
            criteria_ = np.hstack((np.ones(self.cfg.n_discrete_confidence_levels - 1) / self.cfg.n_discrete_confidence_levels, 1))

        for i, crit in enumerate(criteria_):
            cnd = self.data.c_conf_discrete[sub_ind] == i
            if cnd.sum():
                if i == 0:
                    lower_bin_edge = 0
                    upper_bin_edge = crit
                else:
                    lower_bin_edge = np.sum(criteria_[:i])
                    upper_bin_edge = np.sum(criteria_[:i+1])
                if type2_noise_type == 'noisy_report':
                    c_conf_grid = self._type1_evidence_to_confidence(self.modeldata.z1_type1_evidence_grid[sub_ind], params_type2, sub_ind)
                    data_lower = lower_bin_edge
                    data_upper = min(1, upper_bin_edge)
                    type2_noise_dist = get_type2_dist(
                        self.cfg.type2_noise_dist, type2_center=c_conf_grid[cnd], type2_noise=params_type2['type2_noise'],
                        type2_noise_type='noisy_report'
                    )
                elif type2_noise_type == 'noisy_readout':
                    data_lower = self._confidence_to_type1_evidence(lower_bin_edge, params_type2, sub_ind, mask=cnd)
                    data_upper = self._confidence_to_type1_evidence(upper_bin_edge, params_type2, sub_ind, mask=cnd)
                    type2_noise_dist = get_type2_dist(
                        self.cfg.type2_noise_dist, type2_center=self.modeldata.z1_type1_evidence_grid[sub_ind][cnd], type2_noise=params_type2['type2_noise'],
                        type2_noise_type='noisy_readout'
                    )
                elif type2_noise_type == 'noisy_temperature':
                    data_lower = lower_bin_edge
                    data_upper = min(1, upper_bin_edge)
                    type2_noise_dist = self.get_noisy_temperature_dist(params_type2, sub_ind, mask=cnd)

                type2_likelihood[cnd] = type2_noise_dist.cdf(data_upper) - type2_noise_dist.cdf(data_lower)  # noqa

        return type2_likelihood


    def _type1_evidence_to_confidence(self, z1_type1_evidence, params_type2, sub_ind=0):
        """
        Helper function to convert type 1 evidence to confidence
        """
        return type1_evidence_to_confidence(
            z1_type1_evidence=z1_type1_evidence, y_decval=self.modeldata.y_decval_grid[sub_ind],
            x_stim=self.data.x_stim[sub_ind], type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
            **self.result.type1.params[sub_ind], **params_type2
        )

    def _type1_noise_to_confidence(self, type1_noise, params_type2, sub_ind=0):
        """
        Helper function to convert type 1 noise to confidence
        """
        return type1_noise_to_confidence(
            y_decval=self.modeldata.y_decval_grid[sub_ind],
            x_stim=self.data.x_stim[sub_ind], type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
            **{**self.result.type1.params[sub_ind], **dict(type1_noise=type1_noise)}, **params_type2
        )

    def _confidence_to_type1_evidence(self, c_conf, params_type2, sub_ind=0, type1_noise=None,
                                      tile_on_type1_uncertainty=True, mask=None):
        """
        Helper function to convert confidence to type 1 evidence
        """
        return confidence_to_type1_evidence(
            c_conf=c_conf,
            x_stim=self.data.x_stim[sub_ind] if mask is None else self.data.x_stim[sub_ind][mask],
            y_decval=self.modeldata.y_decval_grid[sub_ind] if mask is None else self.modeldata.y_decval_grid[sub_ind][mask],
            type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
            tile_on_type1_uncertainty=tile_on_type1_uncertainty,
            **(self.result.type1.params[sub_ind] if type1_noise is None else
               {**self.result.type1.params[sub_ind], **dict(type1_noise=type1_noise)}),
            **params_type2
        )

    def _confidence_to_type1_noise(self, c_conf, params_type2, sub_ind=0, mask=None):
        """
        Helper function to convert confidence to type 1 noise
        """
        return confidence_to_type1_noise(
            c_conf=c_conf,
            x_stim=self.data.x_stim[sub_ind] if mask is None else self.data.x_stim[sub_ind][mask],
            y_decval=self.modeldata.y_decval_grid[sub_ind] if mask is None else self.modeldata.y_decval_grid[sub_ind][mask],
            type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency,
            **self.result.type1.params[sub_ind], **params_type2
        )

    def _compute_decision_values(self):
        """
        Compute type 1 decision values
        """

        range_ = np.linspace(0, self.cfg.y_decval_range_nsds, int((self.cfg.y_decval_range_nbins + 1) / 2))[1:]
        yrange = np.hstack((-range_[::-1], 0, range_))
        self.modeldata.y_decval_grid = [np.empty((self.data.nsamples[s], yrange.shape[0])) for s in range(self.data.nsubjects)]
        self.modeldata.y_decval = [np.empty(self.data.nsamples[s]) for s in range(self.data.nsubjects)]
        if self.cfg.type2_noise_type != 'noisy_temperature':
            self.modeldata.y_decval_pmf_grid = [np.empty(self.modeldata.y_decval_grid[s].shape) for s in range(self.data.nsubjects)]
            if not self.cfg.experimental_include_incongruent_y_decval:
                self.modeldata.y_decval_grid_invalid = [np.empty(self.modeldata.y_decval_grid[s].shape, dtype=bool) for s in range(self.data.nsubjects)]
        if self.cfg.experimental_min_uniform_type2_likelihood:
            self.modeldata.precomputed.uniform_type2_negll = np.empty(self.data.nsubjects)
        for s in range(self.data.nsubjects):

            type1_noise_trialwise = compute_signal_dependent_type1_noise(
                x_stim=self.data.x_stim_3d[s], type1_noise_signal_dependency=self.cfg.type1_noise_signal_dependency, **self.result.type1.params[s]
            )
            type1_thresh = _check_param(self.result.type1.params[s]['type1_thresh'] if self.cfg.enable_type1_param_thresh else 0)
            type1_bias = _check_param(self.result.type1.params[s]['type1_bias'] if self.cfg.enable_type1_param_bias else 0)

            cond_neg, cond_pos = (self.data.x_stim_3d[s] < 0).squeeze(), (self.data.x_stim_3d[s] >= 0).squeeze()
            y_decval = np.empty(self.data.x_stim_3d[s].shape)
            y_decval[cond_neg] = (np.abs(self.data.x_stim_3d[s][cond_neg]) >= type1_thresh[0]) * \
                self.data.x_stim_3d[s][cond_neg] + type1_bias[0]
            y_decval[cond_pos] = (np.abs(self.data.x_stim_3d[s][cond_pos]) >= type1_thresh[1]) * \
                self.data.x_stim_3d[s][cond_pos] + type1_bias[1]
            self.modeldata.y_decval[s] = y_decval.flatten()

            self.modeldata.y_decval_grid[s][cond_neg] = y_decval[cond_neg] + yrange * np.mean(type1_noise_trialwise[cond_neg])
            self.modeldata.y_decval_grid[s][cond_pos] = y_decval[cond_pos] + yrange * np.mean(type1_noise_trialwise[cond_pos])

            logistic_neg = logistic_dist(loc=y_decval[cond_neg], scale=type1_noise_trialwise[cond_neg] * np.sqrt(3) / np.pi)
            logistic_pos = logistic_dist(loc=y_decval[cond_pos], scale=type1_noise_trialwise[cond_pos] * np.sqrt(3) / np.pi)
            margin_neg = type1_noise_trialwise[cond_neg] * self.cfg.y_decval_range_nsds / self.cfg.y_decval_range_nbins
            margin_pos = type1_noise_trialwise[cond_pos] * self.cfg.y_decval_range_nsds / self.cfg.y_decval_range_nbins
            if self.cfg.type2_noise_type != 'noisy_temperature':
                self.modeldata.y_decval_pmf_grid[s][cond_neg] = (logistic_neg.cdf(self.modeldata.y_decval_grid[s][cond_neg] + margin_neg) -
                                                                 logistic_neg.cdf(self.modeldata.y_decval_grid[s][cond_neg] - margin_neg))
                self.modeldata.y_decval_pmf_grid[s][cond_pos] = (logistic_pos.cdf(self.modeldata.y_decval_grid[s][cond_pos] + margin_pos) -
                                                                 logistic_pos.cdf(self.modeldata.y_decval_grid[s][cond_pos] - margin_pos))
                # pdf is only necessary if we use trapezoid integration at the type 2 level
                # self.model.y_decval_pdf_grid = np.full(self.model.y_decval_grid.shape, np.nan)
                # self.model.y_decval_pdf_grid[cond_neg] = logistic_neg.pdf(self.model.y_decval_grid[cond_neg])
                # self.model.y_decval_pdf_grid[cond_pos] = logistic_pos.pdf(self.model.y_decval_grid[cond_pos])
                # normalize PMF
                self.modeldata.y_decval_pmf_grid[s] = self.modeldata.y_decval_pmf_grid[s] / self.modeldata.y_decval_pmf_grid[s].sum(axis=-1).reshape(-1, 1)
                # invalidate invalid decision values
                if not self.cfg.experimental_include_incongruent_y_decval:
                    self.modeldata.y_decval_grid_invalid[s] = np.sign(self.modeldata.y_decval_grid[s]) != \
                                                              np.sign(self.data.d_dec_3d[s] - 0.5)
                    self.modeldata.y_decval_pmf_grid[s][self.modeldata.y_decval_grid_invalid[s]] = np.nan

                if self.cfg.experimental_min_uniform_type2_likelihood:
                    # self.cfg.type2_binsize*2 is the probability for a given confidence rating assuming a uniform
                    # distribution for confidence. This 'confidence guessing model' serves as a upper bound for the
                    # negative log likelihood.
                    min_type2_likelihood_grid = self.cfg.type2_binsize * 2 * np.ones(self.modeldata.y_decval_pmf_grid[s].shape)
                    min_type2_likelihood = np.nansum(min_type2_likelihood_grid * self.modeldata.y_decval_pmf_grid[s], axis=1)
                    self.modeldata.precomputed.uniform_type2_negll[s] = -np.log(min_type2_likelihood).sum()

        if self.cfg.type2_noise_type != 'noisy_temperature':
            self.modeldata.z1_type1_evidence_grid = [np.abs(self.modeldata.y_decval_grid[s]) for s in range(self.data.nsubjects)]

        self.modeldata.z1_type1_evidence = [np.abs(self.modeldata.y_decval[s]) for s in range(self.data.nsubjects)]


    def _check_fit(self):
        if not self.type1_is_fitted and not self.type2_is_fitted:
            raise RuntimeError('Please fit the model before plotting.')
        elif self.type1_is_fitted and not self.type2_is_fitted:
            raise RuntimeError('Only the type 1 level was fitted. Please also fit the type 2 level to plot'
                               'a link function.')

    def plot_evidence_versus_confidence(self, sub_ind=0, **kwargs):
        self._check_fit()
        plot_evidence_versus_confidence(
            self.data.x_stim[sub_ind], self.data.c_conf[sub_ind], self.modeldata.y_decval[sub_ind], self.result.params[sub_ind], cfg=self.cfg, **kwargs
        )

    def plot_confidence_dist(self, sub_ind=0, **kwargs):
        self._check_fit()
        varlik = self.modeldata.z1_type1_evidence_grid[sub_ind] if self.cfg.type2_noise_type == 'noisy_readout' else self.modeldata.c_conf_grid[sub_ind]
        plot_confidence_dist(
            self.cfg, self.data.x_stim[sub_ind], self.data.c_conf[sub_ind], self.result.params[sub_ind], var_likelihood_grid=varlik,
            y_decval_grid=self.modeldata.y_decval_grid[sub_ind],
            likelihood_weighting=self.modeldata.y_decval_pmf_grid[sub_ind], **kwargs
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