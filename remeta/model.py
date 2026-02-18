import os
import pathlib
import pickle
import timeit
import warnings

try:  # only necessary if multiple cores should be used
    from multiprocessing_on_dill.pool import Pool as DillPool
except ModuleNotFoundError:
    pass

import numpy as np
from scipy.optimize import OptimizeResult
from scipy.special import expit, ndtr, erfinv

from .configuration import Configuration
from .type2_dist import get_type2_dist
from .fit import subject_estimation, group_estimation
from .simulation import simulate
from .modelspec import ModelData, Data, Summary
from .plotting import plot_psychometric, plot_stimulus_versus_confidence, plot_confidence_histogram
from .transform import (compute_signal_dependent_type1_noise, logistic, type1_evidence_to_confidence,
                        confidence_to_type1_evidence, confidence_to_type1_noise, type1_noise_to_confidence,
                        compute_nonlinear_encoding)
from .util import _check_param, TAB, SP2, print_dataset_characteristics, print_warnings, empty_list

class ReMeta:

    def __init__(
        self,
        cfg: Configuration = None,
        **kwargs
    ):
        """

        Usage:
            ```
            rem = remeta.ReMeta()
            ```

            for a default model, or otherwise:

            ```
            cfg = remeta.Configuration()
            cfg.<some_setting> = <some_value>
            rem = remeta.ReMeta(cfg)
            ```

        Args:
            cfg: Configuration object. If None is passed, the default configuration (but see kwargs).
            **kwargs: kwargs entries are passed to the Configuration

        """

        if cfg is None:
            # Set configuration attributes that match keyword arguments
            cfg_kwargs = {k: v for k, v in kwargs.items() if k in Configuration.__dict__}
            self.cfg = Configuration(**cfg_kwargs)
        else:
            self.cfg = cfg
            for k, v in kwargs.items():
                if k in Configuration.__dict__:
                    setattr(self.cfg, k, v)
        self.cfg.setup()

        self.modeldata = ModelData(cfg=self.cfg)
        self.data = None
        self.result = None

        self.type1_is_fitted = False
        self.type2_is_fitted = False


    def fit(
        self,
        stimuli: list[float] | list[list[float]]  | np.typing.NDArray[float],
        choices: list[float] | list[list[float]]  | np.typing.NDArray[float],
        confidence: list[float] | list[list[float]]  | np.typing.NDArray[float] = None,
        verbosity: int = 1,
        silence_warnings: bool = False
    ):
        """

        Usage:
            ```
            rem = ReMeta()
            rem.fit(stimuli, choices, confidence)
            ```

        Args:
            stimuli: 1d or 2d array or list of signed stimulus intensities
            choices: 1d or 2d array or list of choices (coded as -1/1 or 0/1)
            confidence: 1d or 2d array or list of confidence ratings (normalized to the range 0-1)
            verbosity: verbosity level (possible values: 0, 1, 2)
            silence_warnings: if `True`, warnings during model fitting are supressed.
        """

        self.data = Data(self.cfg, stimuli, choices, confidence)

        self.result = Summary(self.data, self.cfg)

        self.fit_type1(verbosity=verbosity, store_final_results=self.cfg.skip_type2, silence_warnings=silence_warnings)

        if not self.cfg.skip_type2:
            self.fit_type2(verbosity=verbosity, silence_warnings=silence_warnings)


    def fit_type1(
        self,
        stimuli: None | list[float] | list[list[float]]  | np.typing.NDArray[float] = None,
        choices: None | list[float] | list[list[float]]  | np.typing.NDArray[float] = None,
        confidence: None | list[float] | list[list[float]]  | np.typing.NDArray[float] = None,
        store_final_results: bool = True,
        verbosity: int = 1,
        silence_warnings: bool = False
    ):
        """

        Usage:
            ```
            rem = ReMeta()
            rem.fit_type1(stimuli, choices)
            ```

        Args:
            stimuli: 1d or 2d array or list of signed stimulus intensities
            choices: 1d or 2d array or list of choices (coded as -1/1 or 0/1)
            confidence: 1d or 2d array or list of confidence ratings (normalized to the range 0-1)
            store_final_results: if `True`, save final results. Mostly used internally - will be set to `False`,
                if followed by `fit_type2`.
            verbosity: verbosity level (possible values: 0, 1, 2)
            silence_warnings: if `True`, warnings during model fitting are supressed.
        """

        if self.data is None:
            if stimuli is None or choices is None:
                raise ValueError('If the data attribute of the ReMeta instance is None, at least stimuli '
                                 'and choices have to be passed to fits_type1_subject()')
            else:
                self.data = Data(self.cfg, stimuli, choices, confidence)

        if verbosity >= 1:
            print(f'Dataset characteristics:')
            print(f'{TAB}No. subjects: {self.data.nsubjects}')
            print(f"{TAB}No. samples: {np.array2string(np.array(self.data.nsamples).squeeze(), separator=', ', threshold=3)}")
            print(f'{TAB}Accuracy: {100*self.data.stats.accuracy:.1f}% correct')
            print(f"{TAB}d': {self.data.stats.dprime:.3f}")
            print(f"{TAB}Choice bias: {100*self.data.stats.choice_bias:.1f}%")
            if not self.cfg.skip_type2 and not store_final_results:
                print(f"{TAB}Mean confidence: {self.data.stats.mean_confidence:.3f} "
                      f"(min: {np.min(self.data.c_conf):.3f}, max: {np.max(self.data.c_conf):.3f})")

        self.result = Summary(self.data, self.cfg)

        if verbosity:
            print('\n+++ Type 1 level +++')
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning, module='scipy.optimize',
                                    message='delta_grad == 0.0. Check if the approximated function is linear. If the '
                                            'function is linear better results can be obtained by defining the Hessian '
                                            'as zero instead of using quasi-Newton approximations.')

            fits_type1_subject, fit_type1_group = None, None
            if self.cfg._paramset_type1.nparams > 0:

                if verbosity:
                    print(f'{SP2}Subject-level estimation (MLE)')
                    tind = timeit.default_timer()

                # Single-subject fits via MLE
                use_multiproc_for_subject_loop = (self.cfg._optim_num_cores >= 8) and (self.data.nsubjects >= 8)
                def subject_loop(s):
                    return subject_estimation(
                        self.compute_type1_negll, self.cfg._paramset_type1, args=[s],
                        gridsearch=self.cfg.optim_type1_gridsearch,
                        scipy_solvers=self.cfg.optim_type1_scipy_solvers,
                        num_cores=1 if use_multiproc_for_subject_loop else self.cfg._optim_num_cores,
                        minimize_along_grid=self.cfg.optim_type1_minimize_along_grid,
                        global_minimization=self.cfg.optim_type1_global_minimization,
                        # fine_gridsearch=self.cfg.optim_type1_fine_gridsearch,
                        verbosity=verbosity
                    )
                if use_multiproc_for_subject_loop:
                    with DillPool(self.cfg._optim_num_cores) as pool:
                        fits_type1_subject = pool.map(subject_loop, range(self.data.nsubjects))
                else:
                    fits_type1_subject = [None for _ in range(self.data.nsubjects)]
                    for s in range(self.data.nsubjects):
                        if (verbosity > 0) and (self.data.nsubjects > 1):
                            print(f'{TAB} Subject {s + 1} / {self.data.nsubjects}')
                        fits_type1_subject[s] = subject_loop(s)
                # Store single-subject results
                params_subject = [fits_type1_subject[s].x for s in range(self.data.nsubjects)]
                params_hessian_subject = [fits_type1_subject[s].hessian for s in range(self.data.nsubjects)]
                self.result.type1.subject.store(
                    'type1', self.cfg, self.data, self.compute_type1_negll, params_subject,
                    hessian=params_hessian_subject, fit=fits_type1_subject,
                    execution_time=np.sum([fits_type1_subject[s].execution_time for s in range(self.data.nsubjects)])
                )

                if verbosity:
                    print(f'{TAB}.. finished ({timeit.default_timer() - tind:.1f} secs).')

                if self.data.nsubjects > 1:
                    idx_fe = np.array([i for i, p in enumerate(self.cfg._paramset_type1.parameters_flat.values()) if p.group == 'fixed'])
                    idx_re = np.array([i for i, p in enumerate(self.cfg._paramset_type1.parameters_flat.values()) if p.group == 'random'])
                    if (len(idx_fe) > 0) or (len(idx_re) > 0):

                        fit_type1_group = group_estimation(
                            fun=self.compute_type1_negll,
                            nsubjects=self.data.nsubjects,
                            params_init=params_subject,
                            bounds=self.cfg._paramset_type1.bounds,
                            idx_fe=idx_fe,
                            idx_re=idx_re,
                            num_cores=self.cfg._optim_num_cores,
                            max_iter=30, sigma_floor=1e-3,
                            verbosity=verbosity,
                            # tau=0.05
                        )

                        self.result.type1.init_group()
                        self.result.type1.group.store(
                            'type1', self.cfg, self.data, self.compute_type1_negll,
                            params=[fit_type1_group.x[s] for s in range(self.data.nsubjects)],
                            params_se=[fit_type1_group.x_se[s] for s in range(self.data.nsubjects)],
                            params_cov=[fit_type1_group.x_cov[s] for s in range(self.data.nsubjects)],
                            pop_mean_sd=fit_type1_group.x_re_pop_mean_sd,
                            execution_time=fit_type1_group.execution_time
                        )

            self.result.type1.store(cfg=self.cfg, data=self.data, fun=self.compute_type1_negll)
            if verbosity:
                self.result.type1.report_fit(self.cfg)
            self.type1_is_fitted = True

        if store_final_results:
            self.result.store(store_type1_only=True)

        if verbosity:
            print('Type 1 level finished')


    def fit_type2(self, verbosity=1, silence_warnings=False):

        # compute decision values
        self._compute_decision_values()

        if self.cfg.type2_noise_type == 'temperature':
            # We precompute a few things for noisy temperature model
            if self.cfg.param_type1_noise.model == 'normal':
                # normal: c = 2*ɸ(z) - 1 -> z = ɸ^-1(0.5*(c+1)) -> Jacobian = dz/dc = sqrt(π/2)*exp((erf^-1(c))**2)
                self.modeldata.precomputed.jacobian_temperature = \
                [np.sqrt(np.pi / 2.0) * np.exp(erfinv(np.minimum(1-1e-8, self.data.c_conf[s]))**2) for s in range(self.data.nsubjects)]
            elif self.cfg.param_type1_noise.model == 'logistic':
                # logistic: c = tanh(z * (π/2√3)) -> z = (2√3/π) * atanh(c) -> Jacobian = dz/dc = (2√3/π) / (1 - c**2)
                self.modeldata.precomputed.jacobian_temperature = \
                    [((2 * np.sqrt(3)) / np.pi) / (1 - np.minimum(1-1e-8, self.data.c_conf[s])**2) for s in range(self.data.nsubjects)]
            self.modeldata.precomputed.quintiles_temperature = np.arange(self.cfg.temperature_marg_res, 1, self.cfg.temperature_marg_res)

        if verbosity:
            print('\n+++ Type 2 level +++')

        with warnings.catch_warnings():  # noqa
            warnings.filterwarnings('ignore', module='scipy.optimize')
            if self.cfg._paramset_type2.nparams > 0:

                if verbosity:
                    print(f'{SP2}Subject-level estimation (MLE)')
                    tind = timeit.default_timer()
                    # print(f'{SP2}Scipy solvers: {self.cfg.optim_type1_scipy_solvers}')

                # Single-subject fits via MLE
                use_multiproc_for_subject_loop = (self.cfg._optim_num_cores >= 8) and (self.data.nsubjects >= 8)
                def subject_loop(s):
                    return subject_estimation(
                        self.compute_type2_negll, self.cfg._paramset_type2, args=[s],
                        gridsearch=self.cfg.optim_type2_gridsearch,
                        num_cores=1 if use_multiproc_for_subject_loop else self.cfg._optim_num_cores,
                        minimize_along_grid=self.cfg.optim_type2_minimize_along_grid,
                        global_minimization=self.cfg.optim_type2_global_minimization,
                        # fine_gridsearch=self.cfg.optim_type2_fine_gridsearch,
                        scipy_solvers=self.cfg.optim_type2_scipy_solvers, slsqp_epsilon=self.cfg.optim_type2_slsqp_epsilon,
                        verbosity=verbosity
                    )
                if use_multiproc_for_subject_loop:
                    with DillPool(self.cfg._optim_multiproc_cores_effective) as pool:
                        fits_type2_subject = pool.map(subject_loop, range(self.data.nsubjects))
                else:
                    fits_type2_subject = [None for _ in range(self.data.nsubjects)]
                    for s in range(self.data.nsubjects):
                        if (verbosity > 0) and (self.data.nsubjects > 1):
                            print(f'{TAB} Subject {s + 1} / {self.data.nsubjects}')
                        fits_type2_subject[s] = subject_loop(s)

                # Store single-subject results
                params_subject = [fits_type2_subject[s].x for s in range(self.data.nsubjects)]
                params_hessian_subject = [fits_type2_subject[s].hessian for s in range(self.data.nsubjects)]
                self.result.type2.subject.store(
                    'type2', self.cfg, self.data, self.compute_type2_negll, params_subject,
                    hessian=params_hessian_subject,
                    fit=fits_type2_subject,
                    execution_time=np.sum([fits_type2_subject[s].execution_time for s in range(self.data.nsubjects)])
                )

                if verbosity:
                    print(f'{TAB}.. finished ({timeit.default_timer() - tind:.1f} secs).')

                # Group fit
                if self.data.nsubjects > 1:
                    idx_fe = np.array([i for i, p in enumerate(self.cfg._paramset_type2.parameters_flat.values()) if p.group == 'fixed'])
                    idx_re = np.array([i for i, p in enumerate(self.cfg._paramset_type2.parameters_flat.values()) if p.group == 'random'])
                    if (len(idx_fe) > 0) or (len(idx_re) > 0):

                        fit_type2_group = group_estimation(
                            fun=self.compute_type2_negll,
                            nsubjects=self.data.nsubjects,
                            params_init=params_subject,
                            bounds=self.cfg._paramset_type2.bounds,
                            idx_fe=idx_fe,
                            idx_re=idx_re,
                            num_cores=self.cfg._optim_num_cores,
                            max_iter=30, sigma_floor=1e-3,
                            verbosity=verbosity
                        )
                        self.result.type2.init_group()
                        self.result.type2.group.store(
                            'type2', self.cfg, self.data, self.compute_type2_negll,
                            params=[fit_type2_group.x[s] for s in range(self.data.nsubjects)],
                            params_se=[fit_type2_group.x_se[s] for s in range(self.data.nsubjects)],
                            params_cov=None if fit_type2_group.x_cov is None else [fit_type2_group.x_cov[s] for s in range(self.data.nsubjects)],
                            pop_mean_sd=fit_type2_group.x_re_pop_mean_sd,
                            execution_time=fit_type2_group.execution_time
                        )

            self.result.type2.store(cfg=self.cfg, data=self.data, fun=self.compute_type2_negll)
            if verbosity:
                self.result.type2.report_fit(self.cfg)

        self.result.store(store_type1_only=False)
        self.type2_is_fitted = True

        # if not silence_warnings:
        #     print_warnings(w)
        if verbosity:
            print('Type 2 level finished')

    def summary(self, generative=False, generative_nsamples=1000, squeeze=True):
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
                c_conf_generative[s] = simulate(self.result.params[s], nsubjects=generative_nsamples, nsamples=self.data.nsamples[s],
                                                cfg=self.cfg, custom_stimuli=self.data.x_stim[s], verbosity=0).confidence
        else:
            c_conf_generative = None
        model_summary = self.result.summary(
            c_conf_empirical=self.data.c_conf, c_conf_generative=c_conf_generative, squeeze=squeeze
        )
        return model_summary


    def compute_type1_negll(self, params, sub_ind=0, save_type=None):
        """
        Likelihood function for the type 1 level

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 1 level.
        sub_ind : int
            Subject index (only valid for 2d multi-subject datasets)
        save_type : None | str
            If 'subject' or 'group', store latent variables and parameters.

        Returns:
        --------
        negll: float
            Negative (summed) log likelihood.
        """

        # bl = self.cfg._paramset_type1.param_len_list
        # params_type1 = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
        #                 for i, (p, n) in enumerate(zip(self.cfg._paramset_type1.param_names, bl))}
        params_type1 = {p: params[ind] for p, ind in self.cfg._paramset_type1.param_ind.items()}

        type1_thresh = _check_param(params_type1['type1_thresh'] if self.cfg.param_type1_thresh.enable else self.cfg.param_type1_thresh.default)
        type1_bias = _check_param(params_type1['type1_bias'] if self.cfg.param_type1_bias.enable else self.cfg.param_type1_bias.default)

        if self.cfg.param_type1_nonlinear_gain.enable:
            x_stim_transform = compute_nonlinear_encoding(
                self.data.x_stim[sub_ind], params_type1['type1_nonlinear_gain'],
                params_type1['type1_nonlinear_scale'] if self.cfg.param_type1_nonlinear_scale.enable else
                (self.data.x_stim_max if self.cfg.param_type1_nonlinear_scale.default is None else self.cfg.param_type1_nonlinear_scale.default))
        else:
            x_stim_transform = self.data.x_stim[sub_ind]

        cond_neg, cond_pos = self.data.x_stim[sub_ind] < 0, self.data.x_stim[sub_ind] >= 0
        y_decval = np.full(self.data.x_stim[sub_ind].shape, np.nan)
        y_decval[cond_neg] = (np.abs(x_stim_transform[cond_neg]) > type1_thresh[0]) * x_stim_transform[cond_neg] + type1_bias[0]
        y_decval[cond_pos] = (np.abs(x_stim_transform[cond_pos]) > type1_thresh[1]) * x_stim_transform[cond_pos] + type1_bias[1]

        if self.cfg.param_type1_noise_heteroscedastic.enable or (self.cfg.param_type1_noise.enable == 2):
            type1_noise = compute_signal_dependent_type1_noise(
                x_stim=x_stim_transform,
                type1_noise_signal_dependency=self.cfg.param_type1_noise_heteroscedastic.model if self.cfg.param_type1_noise_heteroscedastic.enable else None,
                **params_type1
            )
        else:
            type1_noise = params_type1['type1_noise']

        if self.cfg.param_type1_noise.model == 'normal':
            posterior = ndtr(y_decval / type1_noise)
        elif self.cfg.param_type1_noise.model == 'logistic':
            posterior = logistic(y_decval, type1_noise)
        likelihood = (self.data.d_dec[sub_ind] == 1) * posterior + (self.data.d_dec[sub_ind] == 0) * (1 - posterior)
        negll = np.sum(-np.log(np.maximum(likelihood, self.cfg.min_type1_like)))

        # Add negative log likelihood of (fixed) Normal priors
        priors = [((params_type1[k] - p.prior[0])**2) / (2 * p.prior[1]**2) for k, p in
                  self.cfg._paramset_type1.parameters_flat.items() if isinstance(p.prior, tuple)]
        if len(priors) > 0:
            negll += np.sum(priors)

        if save_type is not None and (save_type != 'mock'):
            getattr(self.result.type1, save_type).params[sub_ind] = params_type1
            getattr(self.result.type1, save_type).loglik[sub_ind] = -negll
            if self.modeldata.type1_posterior is None:
                self.modeldata.type1_posterior = empty_list(self.data.nsubjects, self.data.nsamples)
                self.modeldata.type1_likelihood = empty_list(self.data.nsubjects, self.data.nsamples)
            self.modeldata.type1_posterior[sub_ind] = posterior
            self.modeldata.type1_likelihood[sub_ind] = likelihood
        return negll

    def compute_type2_negll(self, params, sub_ind=0, type2_noise_type=None, save_type=None):
        """
        Negative log likelihood minimization

        Parameters:
        -----------
        params : array-like of shape (nparams)
            Parameter array of the type 2 level.
        sub_ind : int
            Subject index (only valid for 2d multi-subject datasets)
        type2_noise_type : str
            Type 2 noise type: 'report', 'readout' or 'temperature'
        save_type : None | str
            If 'subject' or 'group', compute proper likelihoods and store latent variables and parameters.
            If 'mock', compute proper likelihoods but don't store latent variables and parameters.

        Returns:
        --------
        By default, the method returns the negative log likelihood. However, depending on the arguments various
        combinations of variables are returned (see Parameters).
        negll: float
            Negative (summed) log likelihood.
        """

        type2_noise_type = self.cfg.type2_noise_type if type2_noise_type is None else type2_noise_type

        # bl = self.cfg._paramset_type2.param_len_list
        # params_type2 = {p: params[int(np.sum(bl[:i]))] if n == 1 else [params[int(np.sum(bl[:i])) + j] for j in range(n)]
        #                 for i, (p, n) in enumerate(zip(self.cfg._paramset_type2.param_names, bl))}
        params_type2 = {p: params[ind] for p, ind in self.cfg._paramset_type2.param_ind.items()}

        if self.cfg.param_type2_criteria.enable or self.cfg.param_type2_criteria.preset is not None:
            likelihood_grid = self._compute_type2_likelihood_criteria(params_type2, sub_ind, type2_noise_type=type2_noise_type)
        else:
            likelihood_grid = self._compute_type2_likelihood_continuous(params_type2, sub_ind, type2_noise_type=type2_noise_type)

        # if save_type is not None and ('type2_criteria' in params_type2) and (np.any(params_type2['type2_criteria']) > 1.001):
        #     params_type2['type2_criteria'] = check_criteria(params_type2['type2_criteria'])

        if type2_noise_type == 'temperature':
            likelihood = likelihood_grid  # in case of the noisy-temp model there is no grid
            likelihood_grid = None
        else:
            if not self.cfg.type1_likel_incongr:
                likelihood_grid[self.modeldata.y_decval_grid_invalid[sub_ind]] = np.nan
            # compute log likelihood
            likelihood = np.nansum(self.modeldata.y_decval_pmf_grid[sub_ind] * likelihood_grid, axis=1)
            # This is equivalent:
            # type2_cum_likelihood2 = np.trapezoid(self.model.y_decval_grid_pdf * np.nan_to_num(likelihood_grid, 0),
            #                                      self.model.y_decval_grid)

        if self.cfg.min_type2_like_uni:
            # use an upper bound for the negative log likelihood based on a uniform 'guessing' model
            negll = min(self.modeldata.precomputed.uniform_type2_negll[sub_ind], -np.sum(np.log(np.maximum(likelihood, 1e-200))))
        else:
            negll = -np.sum(np.log(np.maximum(likelihood, self.cfg.min_type2_like)))

        if save_type is None:
            # We add likelihood punishments only during fitting, not for the final likelihood computation
            # Add likelihood punishment for invalid criteria
            if self.cfg.param_type2_criteria.enable and np.any((crit_abs := np.cumsum(params_type2['type2_criteria'])) > 1):
                negll *= crit_abs[crit_abs > 1].mean()

        # Add negative log likelihood of (fixed) Normal priors
        priors = [((params_type2[k] - p.prior[0])**2) / (2 * p.prior[1]**2) for k, p in
                  self.cfg._paramset_type2.parameters_flat.items() if isinstance(p.prior, tuple)]
        if len(priors) > 0:
            negll += np.sum(priors)

        if save_type is not None and (save_type != 'mock'):
            if type2_noise_type != 'temperature':
                if self.modeldata.c_conf_grid is None:
                    self.modeldata.c_conf_grid = empty_list(self.data.nsubjects, self.data.nsamples, self.cfg.type1_marg_steps)
                self.modeldata.c_conf_grid[sub_ind] = self._type1_evidence_to_confidence(self.modeldata.z1_type1_evidence_grid[sub_ind], params_type2, sub_ind)
                if not self.cfg.type1_likel_incongr:
                    self.modeldata.c_conf_grid[sub_ind][self.modeldata.y_decval_grid_invalid[sub_ind]] = np.nan
            if self.modeldata.c_conf is None:
                self.modeldata.c_conf = empty_list(self.data.nsubjects, self.data.nsamples)
            self.modeldata.c_conf[sub_ind] = self._type1_evidence_to_confidence(self.modeldata.z1_type1_evidence[sub_ind], params_type2, sub_ind)
            if self.modeldata.type2_likelihood is None:
                self.modeldata.type2_likelihood = empty_list(self.data.nsubjects, self.data.nsamples)
                self.modeldata.type2_likelihood_grid = empty_list(self.data.nsubjects, self.data.nsamples, self.cfg.type1_marg_steps)
            self.modeldata.type2_likelihood[sub_ind] = likelihood
            self.modeldata.type2_likelihood_grid[sub_ind] = likelihood_grid

            # Convert back to cumulative criteria!
            getattr(self.result.type2, save_type).params[sub_ind] = \
                {**params_type2, **dict(type2_criteria=np.minimum(1, np.cumsum(params_type2['type2_criteria'])))} \
                    if 'type2_criteria' in params_type2 else params_type2
            getattr(self.result.type2, save_type).loglik[sub_ind] = -negll

        return negll


    def _compute_type2_likelihood_continuous(self, params_type2, sub_ind, type2_noise_type):
        if self.cfg.type2_binsize_wrap:
            wrap_neg = (self.cfg.type2_binsize -
                        np.abs(np.minimum(1, self.data.c_conf_3d[sub_ind] + self.cfg.type2_binsize) - self.data.c_conf_3d[sub_ind]))
            wrap_pos = (self.cfg.type2_binsize -
                        np.abs(np.maximum(0, self.data.c_conf_3d[sub_ind] - self.cfg.type2_binsize) - self.data.c_conf_3d[sub_ind]))
            binsize_neg, binsize_pos = self.cfg.type2_binsize + wrap_neg, self.cfg.type2_binsize + wrap_pos
        else:
            binsize_neg, binsize_pos = self.cfg.type2_binsize, self.cfg.type2_binsize

        if type2_noise_type == 'temperature':
            if np.isclose(self.cfg.type2_binsize, 0):
                data = self.data.c_conf[sub_ind]
            else:
                data_lower = np.maximum(0, self.data.c_conf[sub_ind] - binsize_neg)
                data_upper = np.minimum(1, self.data.c_conf[sub_ind] + binsize_pos)
            dist = self.get_temperature_dist(params_type2)
        elif type2_noise_type == 'report':
            c_conf_grid = self._type1_evidence_to_confidence(self.modeldata.z1_type1_evidence_grid[sub_ind], params_type2, sub_ind)
            if np.isclose(self.cfg.type2_binsize, 0):
                data = self.data.c_conf_3d[sub_ind]
            else:
                data_lower = np.maximum(0, self.data.c_conf_3d[sub_ind] - binsize_neg)
                data_upper = np.minimum(1, self.data.c_conf_3d[sub_ind] + binsize_pos)
            dist = get_type2_dist(self.cfg.param_type2_noise.model, type2_center=c_conf_grid, type2_noise=params_type2['type2_noise'],
                                  type2_noise_type='report')
        elif type2_noise_type == 'readout':
            if np.isclose(self.cfg.type2_binsize, 0):
                data = self._confidence_to_type1_evidence(self.data.c_conf_3d[sub_ind], params_type2, sub_ind)
            else:
                data_lower = self._confidence_to_type1_evidence(
                    np.maximum(0, self.data.c_conf_3d[sub_ind] - binsize_neg), params_type2, sub_ind)
                data_upper = self._confidence_to_type1_evidence(
                    np.minimum(1, self.data.c_conf_3d[sub_ind] + binsize_pos), params_type2, sub_ind)
            dist = get_type2_dist(self.cfg.param_type2_noise.model, type2_center=self.modeldata.z1_type1_evidence_grid[sub_ind], type2_noise=params_type2['type2_noise'],
                                  type2_noise_type='readout')

        if np.isclose(self.cfg.type2_binsize, 0):
            type2_likelihood = dist.pdf(data)
        else:
            type2_likelihood = dist.cdf(data_upper) - dist.cdf(data_lower)

        return type2_likelihood

    def get_temperature_dist(self, params_type2, sub_ind=0, mask=None):

        def __init__(self_):

            self_.type2_dist = get_type2_dist(self.cfg.param_type2_noise.model, type2_center=self.result.type1.params[sub_ind]['type1_noise'], type2_noise=params_type2['type2_noise'],
                                              type2_noise_type='temperature')
            self_.params_type2 = params_type2
            # self_.type1_noise_range = self_.type2_dist.ppf(self.modeldata.precomputed_variables.quintiles_temperature)[:, None]
            # self_.type1_noise_range = np.maximum.accumulate(self_.type2_dist.ppf(self.modeldata.precomputed_variables.quintiles_temperature))[:, None]
            self_.type1_noise_range = self_.type2_dist.ppf(self.modeldata.precomputed.quintiles_temperature)
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

            type2_evidence_bias = self_.params_type2['type2_evidence_bias'] if self.cfg.param_type2_evidence_bias.enable \
                else self.cfg.param_type2_evidence_bias.default
            if self.cfg.param.type2_confidence_bias.enable:
                pbias = self_.params_type2['type2_confidence_bias']
                c_conf = np.minimum(1-1e-8, self.data.c_conf[sub_ind])
                if self.cfg.param_type1_noise.model == 'normal':
                    # normal: c = 2*ɸ(z) - 1 -> z = ɸ^-1(0.5*(c+1)) -> Jacobian = dz/dc = sqrt(π/2)*exp((erf^-1(c))**2)
                    jac = pbias*(c_conf**(pbias-1))*np.sqrt(np.pi / 2.0) * np.exp(erfinv(c_conf**pbias)**2) / type2_evidence_bias
                elif self.cfg.param_type1_noise.model == 'logistic':
                    # logistic: c = tanh(z * (π/2√3)) -> z = (2√3/π) * atanh(c) -> Jacobian = dz/dc = (2√3/π) / (1 - c**2)
                    jac = ((2 * np.sqrt(3) * pbias * (c_conf**(pbias-1))) / np.pi) / (1 - c_conf**(2*pbias)) / type2_evidence_bias
            else:
                jac = self.modeldata.self.precomputed.jacobian_temperature[sub_ind] / type2_evidence_bias

            if self.cfg.param_type1_noise.model == 'normal':
                # Normal (Gaussian) pdf at y with mean mu and std sigma
                z = (y - self.modeldata.y_decval[sub_ind]) / self.result.type1.params[sub_ind]['type1_noise']
                pdf_y = np.exp(-0.5 * z * z) / (self.result.type1.params[sub_ind]['type1_noise'] * np.sqrt(2.0 * np.pi))
            elif self.cfg.param_type1_noise.model == 'logistic':
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
            if self.cfg.param_type1_noise.model == 'normal':
                cdf_y = np.abs(ndtr(y_decval / self.result.type1.params[sub_ind]['type1_noise']) -
                               ndtr((y_decval - y) / self.result.type1.params[sub_ind]['type1_noise']))
            elif self.cfg.param_type1_noise.model == 'logistic':
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


    def _compute_type2_likelihood_criteria(self, params_type2, sub_ind, type2_noise_type):

        type2_likelihood = np.empty_like(self.modeldata.y_decval[sub_ind], float) if type2_noise_type == 'temperature' \
                      else np.empty_like(self.modeldata.z1_type1_evidence_grid[sub_ind], float)

        if self.cfg.param_type2_criteria.enable:
            criteria_ = np.hstack((params_type2['type2_criteria'], 1))
        elif self.cfg.param_type2_criteria.preset is not None:
            # Convert to criteria gaps
            criteria_ = np.hstack((np.diff(np.hstack((0, self.cfg.param_type2_criteria.preset))), 1))
        else:
            raise ValueError('Type2 criteria are neither fitted nor predefined.')

        for i, crit in enumerate(criteria_):
            cnd = self.data.c_conf_discrete[sub_ind] == i
            if cnd.sum():
                if i == 0:
                    lower_bin_edge = 0
                    upper_bin_edge = crit
                else:
                    lower_bin_edge = np.sum(criteria_[:i])
                    upper_bin_edge = np.sum(criteria_[:i+1])
                if type2_noise_type == 'report':
                    c_conf_grid = self._type1_evidence_to_confidence(self.modeldata.z1_type1_evidence_grid[sub_ind], params_type2, sub_ind)
                    data_lower = lower_bin_edge
                    data_upper = min(1, upper_bin_edge)
                    type2_noise_dist = get_type2_dist(
                        self.cfg.param_type2_noise.model, type2_center=c_conf_grid[cnd], type2_noise=params_type2['type2_noise'],
                        type2_noise_type='report'
                    )
                elif type2_noise_type == 'readout':
                    data_lower = self._confidence_to_type1_evidence(lower_bin_edge, params_type2, sub_ind, mask=cnd)
                    data_upper = self._confidence_to_type1_evidence(upper_bin_edge, params_type2, sub_ind, mask=cnd)
                    type2_noise_dist = get_type2_dist(
                        self.cfg.param_type2_noise.model, type2_center=self.modeldata.z1_type1_evidence_grid[sub_ind][cnd], type2_noise=params_type2['type2_noise'],
                        type2_noise_type='readout'
                    )
                elif type2_noise_type == 'temperature':
                    data_lower = lower_bin_edge
                    data_upper = min(1, upper_bin_edge)
                    type2_noise_dist = self.get_temperature_dist(params_type2, sub_ind, mask=cnd)

                type2_likelihood[cnd] = type2_noise_dist.cdf(data_upper) - type2_noise_dist.cdf(data_lower)  # noqa

        return type2_likelihood


    def _type1_evidence_to_confidence(self, z1_type1_evidence, params_type2, sub_ind=0):
        """
        Helper function to convert type 1 evidence to confidence
        """
        return type1_evidence_to_confidence(
            z1_type1_evidence=z1_type1_evidence, type1_dist=self.cfg.param_type1_noise.model,
            x_stim=self.data.x_stim[sub_ind],
            type1_noise_signal_dependency=self.cfg.param_type1_noise_heteroscedastic.model if self.cfg.param_type1_noise_heteroscedastic.enable else None,
            **self.result.type1.params[sub_ind], **params_type2
        )

    def _type1_noise_to_confidence(self, type1_noise, params_type2, sub_ind=0):
        """
        Helper function to convert type 1 noise to confidence
        """
        return type1_noise_to_confidence(
            y_decval=self.modeldata.y_decval_grid[sub_ind], type1_dist=self.cfg.param_type1_noise.model,
            x_stim=self.data.x_stim[sub_ind],
            type1_noise_signal_dependency=self.cfg.param_type1_noise_heteroscedastic.model if self.cfg.param_type1_noise_heteroscedastic.enable else None,
            **{**self.result.type1.params[sub_ind], **dict(type1_noise=type1_noise)}, **params_type2
        )

    def _confidence_to_type1_evidence(self, c_conf, params_type2, sub_ind=0, type1_noise=None,
                                      tile_on_type1_uncertainty=True, mask=None):
        """
        Helper function to convert confidence to type 1 evidence
        """
        return confidence_to_type1_evidence(
            c_conf=c_conf, type1_dist=self.cfg.param_type1_noise.model,
            x_stim=self.data.x_stim[sub_ind] if mask is None else self.data.x_stim[sub_ind][mask],
            y_decval=self.modeldata.y_decval_grid[sub_ind] if mask is None else self.modeldata.y_decval_grid[sub_ind][mask],
            type1_noise_signal_dependency=self.cfg.param_type1_noise_heteroscedastic.model if self.cfg.param_type1_noise_heteroscedastic.enable else None,
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
            c_conf=c_conf, type1_dist=self.cfg.param_type1_noise.model,
            x_stim=self.data.x_stim[sub_ind] if mask is None else self.data.x_stim[sub_ind][mask],
            y_decval=self.modeldata.y_decval_grid[sub_ind] if mask is None else self.modeldata.y_decval_grid[sub_ind][mask],
            type1_noise_signal_dependency=self.cfg.param_type1_noise_heteroscedastic.model if self.cfg.param_type1_noise_heteroscedastic.enable else None,
            **self.result.type1.params[sub_ind], **params_type2
        )

    def _compute_decision_values(self):
        """
        Compute type 1 decision values
        """

        range_ = np.linspace(0, self.cfg.type1_marg_z, int((self.cfg.type1_marg_steps + 1) / 2))[1:]
        yrange = np.hstack((-range_[::-1], 0, range_))
        self.modeldata.y_decval_grid = [np.empty((self.data.nsamples[s], yrange.shape[0])) for s in range(self.data.nsubjects)]
        self.modeldata.y_decval = [np.empty(self.data.nsamples[s]) for s in range(self.data.nsubjects)]
        if self.cfg.type2_noise_type != 'temperature':
            self.modeldata.y_decval_pmf_grid = [np.empty(self.modeldata.y_decval_grid[s].shape) for s in range(self.data.nsubjects)]
            if not self.cfg.type1_likel_incongr:
                self.modeldata.y_decval_grid_invalid = [np.empty(self.modeldata.y_decval_grid[s].shape, dtype=bool) for s in range(self.data.nsubjects)]
        if self.cfg.min_type2_like_uni:
            self.modeldata.precomputed.uniform_type2_negll = np.empty(self.data.nsubjects)
        for s in range(self.data.nsubjects):

            type1_noise_trialwise = compute_signal_dependent_type1_noise(
                x_stim=self.data.x_stim_3d[s],
                type1_noise_signal_dependency=self.cfg.param_type1_noise_heteroscedastic.model if self.cfg.param_type1_noise_heteroscedastic.enable else None,
                **self.result.type1.params[s]
            )
            type1_thresh = _check_param(self.result.type1.params[s]['type1_thresh'] if self.cfg.param_type1_thresh.enable else 0)
            type1_bias = _check_param(self.result.type1.params[s]['type1_bias'] if self.cfg.param_type1_bias.enable else 0)

            cond_neg, cond_pos = (self.data.x_stim_3d[s] < 0).squeeze(), (self.data.x_stim_3d[s] >= 0).squeeze()
            y_decval = np.empty(self.data.x_stim_3d[s].shape)
            y_decval[cond_neg] = (np.abs(self.data.x_stim_3d[s][cond_neg]) >= type1_thresh[0]) * \
                self.data.x_stim_3d[s][cond_neg] + type1_bias[0]
            y_decval[cond_pos] = (np.abs(self.data.x_stim_3d[s][cond_pos]) >= type1_thresh[1]) * \
                self.data.x_stim_3d[s][cond_pos] + type1_bias[1]
            self.modeldata.y_decval[s] = y_decval.flatten()

            self.modeldata.y_decval_grid[s][cond_neg] = y_decval[cond_neg] + yrange * np.mean(type1_noise_trialwise[cond_neg])
            self.modeldata.y_decval_grid[s][cond_pos] = y_decval[cond_pos] + yrange * np.mean(type1_noise_trialwise[cond_pos])

            if self.cfg.type2_noise_type != 'temperature':

                margin_neg = type1_noise_trialwise[cond_neg] * self.cfg.type1_marg_z / self.cfg.type1_marg_steps
                margin_pos = type1_noise_trialwise[cond_pos] * self.cfg.type1_marg_z / self.cfg.type1_marg_steps

                if self.cfg.param_type1_noise.model == 'normal':
                    self.modeldata.y_decval_pmf_grid[s][cond_neg] = (
                        ndtr((self.modeldata.y_decval_grid[s][cond_neg] + margin_neg - y_decval[cond_neg]) / type1_noise_trialwise[cond_neg]) -
                        ndtr((self.modeldata.y_decval_grid[s][cond_neg] - margin_neg - y_decval[cond_neg]) / type1_noise_trialwise[cond_neg])
                    )
                    self.modeldata.y_decval_pmf_grid[s][cond_pos] = (
                        ndtr((self.modeldata.y_decval_grid[s][cond_pos] + margin_pos - y_decval[cond_pos]) / type1_noise_trialwise[cond_pos]) -
                        ndtr((self.modeldata.y_decval_grid[s][cond_pos] - margin_pos - y_decval[cond_pos]) / type1_noise_trialwise[cond_pos])
                    )
                elif self.cfg.param_type1_noise.model == 'logistic':
                    self.modeldata.y_decval_pmf_grid[s][cond_neg] = (
                        logistic(self.modeldata.y_decval_grid[s][cond_neg] + margin_neg - y_decval[cond_neg], type1_noise_trialwise[cond_neg]) -
                        logistic(self.modeldata.y_decval_grid[s][cond_neg] - margin_neg - y_decval[cond_neg], type1_noise_trialwise[cond_neg])
                    )
                    self.modeldata.y_decval_pmf_grid[s][cond_pos] = (
                        logistic(self.modeldata.y_decval_grid[s][cond_pos] + margin_pos - y_decval[cond_pos], type1_noise_trialwise[cond_pos]) -
                        logistic(self.modeldata.y_decval_grid[s][cond_pos] - margin_pos - y_decval[cond_pos], type1_noise_trialwise[cond_pos])
                    )
                    # cdf_neg = logistic_dist(loc=y_decval[cond_neg], scale=type1_noise_trialwise[cond_neg] * np.sqrt(3) / np.pi)
                    # cdf_pos = logistic_dist(loc=y_decval[cond_pos], scale=type1_noise_trialwise[cond_pos] * np.sqrt(3) / np.pi)
                    # self.modeldata.y_decval_pmf_grid[s][cond_neg] = (cdf_neg.cdf(self.modeldata.y_decval_grid[s][cond_neg] + margin_neg) -
                    #                                                  cdf_neg.cdf(self.modeldata.y_decval_grid[s][cond_neg] - margin_neg))
                    # self.modeldata.y_decval_pmf_grid[s][cond_pos] = (cdf_pos.cdf(self.modeldata.y_decval_grid[s][cond_pos] + margin_pos) -
                    #                                                  cdf_pos.cdf(self.modeldata.y_decval_grid[s][cond_pos] - margin_pos))

                # normalize PMF
                self.modeldata.y_decval_pmf_grid[s] = self.modeldata.y_decval_pmf_grid[s] / self.modeldata.y_decval_pmf_grid[s].sum(axis=-1).reshape(-1, 1)
                # invalidate invalid decision values
                if not self.cfg.type1_likel_incongr:
                    self.modeldata.y_decval_grid_invalid[s] = np.sign(self.modeldata.y_decval_grid[s]) != \
                                                              np.sign(self.data.d_dec_3d[s] - 0.5)
                    self.modeldata.y_decval_pmf_grid[s][self.modeldata.y_decval_grid_invalid[s]] = np.nan

                if self.cfg.min_type2_like_uni:
                    # self.cfg.type2_binsize*2 is the probability for a given confidence rating assuming a uniform
                    # distribution for confidence. This 'confidence guessing model' serves as a upper bound for the
                    # negative log likelihood.
                    min_type2_likelihood_grid = self.cfg.type2_binsize * 2 * np.ones(self.modeldata.y_decval_pmf_grid[s].shape)
                    min_type2_likelihood = np.nansum(min_type2_likelihood_grid * self.modeldata.y_decval_pmf_grid[s], axis=1)
                    self.modeldata.precomputed.uniform_type2_negll[s] = -np.log(min_type2_likelihood).sum()

        if self.cfg.type2_noise_type != 'temperature':
            self.modeldata.z1_type1_evidence_grid = [np.abs(self.modeldata.y_decval_grid[s]) for s in range(self.data.nsubjects)]

        self.modeldata.z1_type1_evidence = [np.abs(self.modeldata.y_decval[s]) for s in range(self.data.nsubjects)]


    def _check_fit(self):
        if not self.type1_is_fitted and not self.type2_is_fitted:
            raise RuntimeError('Please fit the model before plotting.')
        elif not self.cfg.skip_type2 and (self.type1_is_fitted and not self.type2_is_fitted):
            raise RuntimeError('Only the type 1 level was fitted. Please also fit the type 2 level to plot'
                               'a link function.')

    def plot_psychometric(
        self,
        sub_ind: int = 0,
        model_prediction: bool = True,
        **kwargs
    ):
        """
        Invoke `remeta.plotting.plot_psychomtric` on the `ReMeta` instance.

        Usage:
            ```
            rem = ReMeta()
            rem.fit(stimuli, choices, confidence)
            rem.plot_psychometric()
            ```

        Args:
            sub_ind: subject index (only applicable if group data were fitted)
            model_prediction: if `True`, plot model predictions
            **kwargs: Pass parameters to `remeta.plotting.plot_psychomtric`
        """

        # self._check_fit()
        plot_psychometric(
            self.data.x_stim[sub_ind], self.data.d_dec[sub_ind],
            params=self.result.params[sub_ind] if model_prediction else None,
            cfg=self.cfg, **kwargs
        )

    def plot_stimulus_versus_confidence(
        self,
        sub_ind: int = 0,
        model_prediction: bool = True,
        **kwargs
    ):
        """
        Invoke `remeta.plotting.plot_stimulus_versus_confidence` on the `ReMeta` instance.

        Usage:
            ```
            rem = ReMeta()
            rem.fit(stimuli, choices, confidence)
            rem.plot_stimulus_versus_confidence()
            ```

        Args:
            sub_ind: subject index (only applicable if group data were fitted)
            model_prediction: if `True`, plot model predictions
            **kwargs: Pass parameters to `remeta.plotting.plot_stimulus_versus_confidence`
        """
        # self._check_fit()
        plot_stimulus_versus_confidence(
            self.data.x_stim[sub_ind], self.data.c_conf[sub_ind],self.data.d_dec[sub_ind],
            params=self.result.params[sub_ind] if model_prediction else None,
            model_prediction=model_prediction,
            cfg=self.cfg, **kwargs
        )

    def plot_confidence_histogram(
        self,
        sub_ind: int = 0,
        model_prediction: bool = True,
        **kwargs
    ):
        """
        Invoke `remeta.plotting.plot_confidence_histogram` on the `ReMeta` instance.

        Usage:
            ```
            rem = ReMeta()
            rem.fit(stimuli, choices, confidence)
            rem.plot_confidence_histogram()
            ```

        Args:
            sub_ind: subject index (only applicable if group data were fitted)
            model_prediction: if `True`, plot model predictions
            **kwargs: Pass parameters to `remeta.plotting.plot_confidence_histogram`
        """
        # self._check_fit()
        plot_confidence_histogram(
            self.data.c_conf[sub_ind], self.data.x_stim[sub_ind], self.data.d_dec[sub_ind],
            params=self.result.params[sub_ind] if model_prediction else None,
            cfg=self.cfg, **kwargs
        )


def load_dataset(name, verbosity=1, return_data_only=False):
    import gzip
    path = os.path.join(pathlib.Path(__file__).parent.resolve(), 'demo_data', f'example_data_{name}.pkl.gz')
    if os.path.exists(path):
        with gzip.open(path, 'rb') as f:
            dataset = pickle.load(f)
    else:
        raise FileNotFoundError(f'[Dataset does not exist!] No such file: {path}')

    if verbosity:
        print_dataset_characteristics(dataset)

    if return_data_only:
        return (dataset.x_stim, dataset.d_dec, dataset.params)
    else:
        return dataset