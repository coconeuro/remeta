import warnings

import numpy as np
from scipy.optimize import OptimizeResult
from numdifftools import Hessian
from scipy.stats import norm

from .util import (TAB, SP2, Struct, ReprMixin, create_struct_with_reprmixin, spearman2d, pearson2d, listlike,
                   empty_list, print_class_instance, cov_from_hessian, se_from_cov, compute_criterion_bias, compute_cov_criteria)


class Parameter(ReprMixin):
    """Definition of ReMeta parameter

    Usage:
        The Parameter class should only be used in the context of a
        [`Configuration`][remeta.configuration.Configuration] instance. This ensures that sensible defaults are used
        for unspecified attributes of the parameter.

        ```
        cfg = remeta.Configuration()
        ```

        Disable parameter: `cfg.param_type1_bias.enable = 0`

        Enable parameter: `cfg.param_type1_tresh.enable = 1`

        Change the initial guess: `cfg.param_type1_noise.guess = 0.8`

        Change parameter bounds: `cfg.param_type1_noise.bounds = (1e-2, 2)`

        Change values visited during grid search: `cfg.param_type1_noise.grid_range = np.arange(0.1, 0.5, 0.05)`

        Make a parameter a group-level (e.g. random-effects) parameter: `cfg.param_type1_thresh.group = 'random'`

        Create a parameter prior with (mean, SD): `cfg.param_type1_bias.prior = (0, 0.1)`

        Change the noise distribution: `cfg.param_type1_noise.model = 'logistic'`


    Args:
        enable:
            `0`: disabled;

            `integer > 0`: enabled (typically 1, but can be 2 for type 1 parameters if fitted
            separately to both stimulus categories; in case of param_type2_criteria, the number sets the
            number of confidence criteria (=number of discrete confidence ratings minus 1).
        guess:
            Initial guess for parameter optimization
        bounds:
            Parameter bounds of the form (lower bound, upper bound).
        grid_range:
            1-d grid for initial gridsearch in the parameter optimization procedure.
        group:
            `None`: no group-level estimate for the parameter

            `'fixed'`: fit parameter as a group fixed effect (i.e., single value for the group)

            `'random'`: fit parameter as a random effect (enforces shrinkage towards a group mean)
        prior:
            `None`: no prior for the parameter
            (group_mean, group_sd): apply a Normal prior defined by mean and standard deviation
        preset:
            (not yet supported) Instead of fitting a parameter, set it to a fixed value. Note that this
            automatically disables the parameter, i.e. parameter.enable will be set to 0.
        default:
            This an internal attribute, that should typically not be touched. It specifies a default value
            for a parameter that may be used if the parameter is not fitted.
        model:
            For noise parameters, specifies an appropriate sampling distribution.
            For other parameters, it may specify a function that is parameterized.
    """
    def __init__(
        self,
        enable: int = None,
        guess: float  = None,
        bounds: tuple[float, float] = None,
        grid_range: list[float] | np.typing.NDArray[float] = None,
        group: None | str = None,
        prior: None | tuple[float, float] = None,
        preset: None | float = None,
        default: None | float = None,
        model: None | str = None
    ):

        self.enable = enable if preset is None else 0
        self.guess = guess
        self.bounds = bounds
        self.grid_range = np.linspace(bounds[0], bounds[1], 4) if grid_range is None else grid_range
        self.group = group
        self.prior = prior
        self.preset = preset
        self.default = default
        self.model = model
        self._definition_changed = False

    def copy(self):
        new = self.__class__.__new__(self.__class__)
        new.__dict__ = {
            k: v.copy() if isinstance(v, (dict, np.ndarray))
            else v[:] if isinstance(v, list)
            else v
            for k, v in self.__dict__.items()
        }
        new._definition_changed = False
        return new

    def __copy__(self):
        return self.copy()

    def __setattr__(self, name, value):
        if name not in ('_definition_changed', 'enable', 'preset') and hasattr(self, name):
            old_value = getattr(self, name)
            if isinstance(old_value, np.ndarray) and isinstance(value, np.ndarray):
                changed = not np.array_equal(old_value, value)
            else:
                changed = old_value != value
            if changed:
                super().__setattr__("_definition_changed", True)

        super().__setattr__(name, value)

class ParameterSet(ReprMixin):
    def __init__(self, parameters, param_names, constraints=None):
        """
        Container class for all Parameters of a model.

        Parameters
        ----------
        parameters : dict[str, Parameter]
            The dictionary must have the form {parameter_name1: Parameter(..), parameter_name2: Parameter(..), ..}
        param_names: List[str]
            List of parameter names of a model.
        constraints: List[Dict]
            List of scipy minimize constraints. Each constraint is a dictionary with keys 'type' and 'fun', where
            'type' is ‘eq’ for equality and ‘ineq’ for inequality, and where fun is a function defining the constraint.
        """

        self.parameters = parameters
        self.param_names = param_names
        self.param_is_list = [isinstance(parameters[name], list) for name in param_names]
        self.param_len = {name: len(parameters[name]) if self.param_is_list[p] else 1 for p, name in enumerate(param_names)}  # noqa
        self.param_len_list = [len(parameters[name]) if self.param_is_list[p] else 1 for p, name in enumerate(param_names)]  # noqa
        self.param_names_flat = sum([[f'{name}_{i}' for i in range(len(parameters[name]))] if self.param_is_list[p]  # noqa
                          else [name] for p, name in enumerate(param_names)], [])
        parameters_flat_list_ = sum([[param[i] for i in range(len(param))] if self.param_is_list[p] else [param] for
                                     p, param in enumerate(parameters.values())], [])
        self.parameters_flat = {name: param for name, param in zip(self.param_names_flat, parameters_flat_list_)}
        self.guess = np.array(sum([[parameters[name][i].guess for i in range(len(parameters[name]))] if  # noqa
                                   self.param_is_list[p] else [parameters[name].guess] for p, name in enumerate(param_names)], []))
        self.bounds = np.array(sum([[parameters[name][i].bounds for i in range(len(parameters[name]))] if  # noqa
                                   self.param_is_list[p] else [parameters[name].bounds] for p, name in enumerate(param_names)], []))
        self.grid_range = sum([[parameters[name][i].grid_range for i in range(len(parameters[name]))] if  # noqa
                               self.param_is_list[p] else [parameters[name].grid_range] for p, name in enumerate(param_names)], [])
        self.constraints = constraints
        self.nparams = len(param_names)
        self.nparams_flat = len(self.param_names_flat)
        self.param_ind = {name: np.arange(int(np.sum(self.param_len_list[:i])), np.sum(self.param_len_list[:i + 1])).squeeze() for i, name in enumerate(self.param_names)}
        self.param_ind_list = [np.arange(int(np.sum(self.param_len_list[:i])), np.sum(self.param_len_list[:i + 1])).squeeze() for i in range(self.nparams)]
        self.param_revind_re = {k: np.arange(l := (0 if i == 0 else l), l := (l + self.param_len_list[j])) for i, (k, j) in
                                enumerate({k_: j_ for j_, (k_, param) in enumerate(self.parameters.items()) if (param[0] if self.param_is_list[j_] else param).group == 'random'}.items())}
        self.param_revind_fe = {k: np.arange(l := (0 if i == 0 else l), l := (l + self.param_len_list[j])) for i, (k, j) in
                                enumerate({k_: j_ for j_, (k_, param) in enumerate(self.parameters.items()) if (param[0] if self.param_is_list[j_] else param).group == 'fixed'}.items())}


class Data(ReprMixin):

    def __init__(self, cfg, stimuli=None, choices=None, confidence=None):
        """
        Container class for behavioral data.

        Parameters
        ----------
        cfg : configuration.Configuration
            Settings
        stimuli : array-like of shape (n_samples,) or array-like of shape (n_subjects, n_samples)
                    or list of array-like (variable n_samples per subject)
            Array of signed stimulus intensity values, where the sign codes the stimulus category (+: cat1; -: cat2) and
            the absolut value codes the intensity. The scale of the data is not relevant, as a normalisation to [-1; 1]
            is applied.
            Note: stimuli are automatically preprocessed and are made available as the Data attribute x_stim.
        choices : array-like of shape (n_samples,) or array-like of shape (n_subjects, n_samples)
                    or list of array-like (variable n_samples per subject)
            Array of choices coded as 0 (cat1) and 1 (cat2) for the two stimulus categories. See parameter 'stimuli'
            for the definition of cat1 and cat2.
            Note: choices are automatically preprocessed and are made available as the Data attribute d_dec.
        confidence : array-like of shape (n_samples,) or array-like of shape (n_subjects, n_samples)
                       or list of array-like (variable n_samples per subject)
            Confidence ratings; must be normalized to the range [0;1].
            Note: confidence is automatically preprocessed and is made available as the Data attribute c_conf.
        """

        self.cfg = cfg
        self._stimuli = stimuli
        self._choices = choices
        self._confidence = confidence

        self._x_stim = None
        self._d_dec = None
        self._c_conf = None
        if self.cfg.param_type2_criteria.enable or self.cfg.param_type2_criteria.preset is not None:
            self.c_conf_discrete = None

        self.x_stim_max = None
        self.x_stim_min = None

        self.x_stim_category = None
        self.x_stim_3d = None

        self.d_dec = None
        self.d_dec_3d = None
        self.d_dec_sign = None
        self.accuracy = None

        self.c_conf = None
        self.c_conf_3d = None

        # data_reference = self._confidence if self._stimuli is None else self._stimuli
        # self.nsubjects = 1 if data_reference.ndim == 1 else data_reference.shape[0]
        # self.nsamples = len(data_reference) if data_reference.ndim == 1 else data_reference.shape[1]
        self.nsubjects = None
        self.nsamples = None

        self.stats = create_struct_with_reprmixin('Stats')
        self.stats_accuracy = None
        self.stats_dprime = None
        self.stats_choice_bias = None
        self.stats_mean_confidence = None

        self.preproc_stim()
        self.preproc_dec()
        if hasattr(confidence, '__len__') and confidence[0] is not None:
            self.preproc_conf()

    @property
    def stimuli(self):
        return self._stimuli

    @stimuli.setter
    def stimuli(self, stimuli):
        self.preproc_stim(stimuli)

    @property
    def choices(self):
        return self._choices

    @choices.setter
    def choices(self, choices):
        self.preproc_dec(choices)

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, confidence):
        self.preproc_conf(confidence)

    @staticmethod
    def ensure_list_of_arrays(x):
        return [np.array(v) for v in x] if listlike(x[0]) else [np.array(x)]

    def preproc_stim(self, stimuli=None):

        self.x_stim = self.ensure_list_of_arrays(self._stimuli if stimuli is None else stimuli)
        self.nsubjects = len(self.x_stim)
        self.nsamples = np.array([len(v) for v in self.x_stim], int)

        _xstim_abs = np.abs(self.x_stim)
        self.x_stim_min = np.min(_xstim_abs)
        self.x_stim_max = np.max(_xstim_abs)
        # self.x_stim_min = min([np.abs(v).min() for v in self.x_stim])
        # self.x_stim_max = max([np.abs(v).max() for v in self.x_stim])

        if not self.cfg.normalize_stimuli_by_max and np.any(np.median(_xstim_abs, axis=1) > 2):
            warnings.warn('For at least one subject, the median of stimulus intensities is > 2. ReMeta is optimized '
                          'for stimuli that are roughly in the range [-1; 1]. ReMeta will ensure this automatically, '
                          'if you set cfg.normalize_stimuli_by_max = True.')

        self.x_stim_3d, self.x_stim_category = [None] * self.nsubjects, [None] * self.nsubjects
        for s in range(self.nsubjects):
            # Normalize stimuli
            if self.cfg.normalize_stimuli_by_max:
                self.x_stim[s] /= self.x_stim_max
            # else:
            #     # self.x_stim = self.stimuli
            #     if np.max(np.abs(self.x_stim[s])) > 1:
            #         raise ValueError('Stimuli are not normalized to the range [-1; 1].')
            self.x_stim_3d[s] = self.x_stim[s][..., np.newaxis]
            self.x_stim_category[s] = (np.sign(self.x_stim[s]) == 1).astype(int)

    def preproc_dec(self, choices=None):

        self.d_dec = self.ensure_list_of_arrays(self._choices if choices is None else choices)

        self.d_dec_sign, self.d_dec_3d, self.accuracy = [None] * self.nsubjects, [None] * self.nsubjects, [None] * self.nsubjects
        self.stats_accuracy, self.stats_dprime, self.stats_choice_bias = empty_list(self.nsubjects), empty_list(self.nsubjects), empty_list(self.nsubjects)
        for s in range(self.nsubjects):
            # convert to 0/1 scheme if choices are provides as -1's and 1's
            if np.array_equal(np.unique(self.d_dec[s][~np.isnan(self.d_dec[s])]), [-1, 1]):
                self.d_dec[s][self.d_dec[s] == -1] = 0
            self.d_dec_sign[s] = np.sign(self.d_dec[s] - 0.5)
            self.d_dec_3d[s] = self.d_dec[s][..., np.newaxis]
            self.accuracy[s] = (self.x_stim_category[s] == self.d_dec[s]).astype(int)
            self.stats_accuracy[s] = np.mean(self.accuracy[s]),
            self.stats_dprime[s] = norm.ppf(min(1 - 1e-3, max(1e-3, self.d_dec[s][self.x_stim_category[s] == 1].mean()))) - \
                                   norm.ppf(min(1 - 1e-3, max(1e-3, self.d_dec[s][self.x_stim_category[s] == 0].mean().mean()))),
            self.stats_choice_bias[s] = self.d_dec[s].mean() - self.x_stim_category[s].mean()
        self.stats.accuracy = np.nanmean(self.stats_accuracy)
        self.stats.dprime = np.nanmean(self.stats_dprime)
        self.stats.choice_bias = np.nanmean(self.stats_choice_bias)

    def preproc_conf(self, confidence=None):

        if not self.cfg.skip_type2 and self._confidence is not None or confidence is not None:
            self.c_conf = self.ensure_list_of_arrays(self._confidence if confidence is None else confidence)
            if self.c_conf is not None:
                self.nsubjects = len(self.c_conf)
                self.nsamples = np.array([len(v) for v in self.c_conf], int)
                self.c_conf_3d = [None] * self.nsubjects
                self.stats_mean_confidence = empty_list(self.nsubjects)
                for s in range(self.nsubjects):
                    self.c_conf_3d[s] = self.c_conf[s][..., np.newaxis]
                    self.stats_mean_confidence[s] = self.c_conf[s].mean()
                self.stats.mean_confidence = np.nanmean(self.stats_mean_confidence)
                if self.cfg.param_type2_criteria.enable or self.cfg.param_type2_criteria.preset is not None:
                    self.c_conf_discrete = [None] * self.nsubjects
                    for s in range(self.nsubjects):
                        self.c_conf_discrete[s] = np.digitize(
                            self.c_conf[s],
                            np.arange(1/self.cfg._n_conf_levels, 1-1e-10, 1/self.cfg._n_conf_levels)
                        )



class ModelResult():

    def __init__(self, level):
        self.level = level

    def store(self, stage, cfg, data, fun, params, params_se=None, params_cov=None, hessian=None,
              pop_mean_sd=None, execution_time=None, fit=None):
        self.nparams = cfg._paramset_type1.nparams
        self.nsubjects = data.nsubjects
        self.nsamples = data.nsamples
        self.loglik = np.empty(self.nsubjects)
        self.loglik_per_sample = np.empty(self.nsubjects)
        self.aic = np.empty(self.nsubjects)
        self.aic_per_sample = np.empty(self.nsubjects)
        self.bic = np.empty(self.nsubjects)
        self.bic_per_sample = np.empty(self.nsubjects)
        self.params = empty_list(self.nsubjects, None)
        self.params_extra = empty_list(self.nsubjects, None)
        self.params_se = empty_list(self.nsubjects, None)
        self.params_random_effect = None
        self.d = None
        self.execution_time = execution_time
        for s in range(self.nsubjects):
            fun(params[s], s, save_type=self.level)
            self.loglik_per_sample[s] = self.loglik[s] / self.nsamples[s]
            self.aic[s] = 2 * self.nparams - 2 * self.loglik[s]
            self.aic_per_sample[s] = self.aic[s] / self.nsamples[s]
            self.bic[s] = 2 * np.log(self.nsamples[s]) - 2 * self.loglik[s]
            self.bic_per_sample[s] = self.bic[s] / self.nsamples[s]

            # For subject-level fits, the Hessian will have been passed, for group-level random/fixed effects
            # the standard error (and the covariance in case of random effects).
            cov, se_params = None, None
            if (self.level == 'subject') and hessian is not None:
                cov = cov_from_hessian(hessian[s])
                se_params = se_from_cov(cov)
            elif (self.level == 'group') and params_se is not None:
                se_params = params_se[s]
                if params_cov is not None:
                    cov = params_cov[s]

            if (stage == 'type2') and ('type2_criteria' in self.params[s]) and cov is not None:
                cov_crit = compute_cov_criteria(cov, cfg._paramset_type2.param_ind['type2_criteria'])
                se_params[cfg._paramset_type2.param_ind['type2_criteria']] = se_from_cov(cov_crit)
                self.params_extra[s] = self._compute_parameters_extra(self.params[s], cov_crit)

            if se_params is not None:
                self.params_se[s] = {p: se_params[ind] for p, ind in getattr(cfg, f'_paramset_{stage}').param_ind.items()}

            if stage == 'type1':
                self.params_extra[s] = {f'{k}_unnorm': list(np.array(v) * data.x_stim_max) if listlike(v) else
                                            v * data.x_stim_max for k, v in self.params[s].items()}


        if pop_mean_sd is not None:
            self.params_random_effect = Struct()
            self.params_random_effect.mean = {k: p if hasattr(p:=pop_mean_sd[0][ind], '__len__') and len(p) > 1 else
                float(np.squeeze(p)) for k, ind in getattr(cfg, f'_paramset_{stage}').param_revind_re.items()}
            self.params_random_effect.std = {k: p if hasattr(p:=pop_mean_sd[1][ind], '__len__') and len(p) > 1 else
                float(np.squeeze(p)) for k, ind in getattr(cfg, f'_paramset_{stage}').param_revind_re.items()}

        if fit is not None:
            self.fit = fit

    def _compute_parameters_extra(self, params, cov_crit):

        bias_crit, bias_crit_se = compute_criterion_bias(params['type2_criteria'], cov_crit)

        params_extra = dict()
        params_extra['type2_criteria_bias'] = bias_crit
        params_extra['type2_criteria_bias_sem'] = bias_crit_se
        params_extra['type2_criteria_confidence_bias'] = -params_extra['type2_criteria_bias']
        params_extra['type2_criteria_confidence_bias_sem'] = bias_crit_se
        # params_extra['type2_criteria_absdev'] = np.abs(diff).mean()
        # params_extra['type2_criteria_bias'] = \
        #     np.mean(params['type2_criteria'])*(len(params['type2_criteria'])+1)-1
        for i in range(len(params['type2_criteria'])):
            params_extra[f'type2_criteria_{i}'] = params[f'type2_criteria'][i]
        return params_extra


    def __str__(self):
        txt = print_class_instance(self, attr_replace_string=dict(fit='Result(s) from scipy.minimize'))
        return txt

    def __repr__(self):
        return self.__str__()


class ModelResultContainer():
    def __init__(self, stage):
        self.stage = stage
        self.subject = ModelResult(level='subject')
        self.group = None
        self.nparams = None
        self.nsubjects = None
        self.nsamples = None
        self.loglik = None

    def init_group(self):
        self.group = ModelResult(level='group')

    def store(self, cfg, data, fun):
        self.nparams = cfg._paramset_type1.nparams
        self.nsubjects = data.nsubjects
        self.nsamples = data.nsamples

        result_vars = ['params', 'params_se', 'params_extra', 'params_random_effect', 'execution_time', 'loglik', 'loglik_per_sample', 'aic', 'aic_per_sample', 'bic', 'bic_per_sample']
        final_level = self.subject if self.group is None else self.group
        for var in result_vars:
            setattr(self, var, getattr(final_level, var))

        if cfg.true_params is not None:
            paramset = cfg._paramset_type1 if self.stage == 'type1' else cfg._paramset_type2
            if not np.all([p in cfg.true_params for p in paramset.param_names]):
                raise ValueError(f'Set of provided true parameters is incomplete (Stage {self.stage}).')
            self.loglik_true = np.empty(data.nsubjects)
            self.loglik_per_sample_true = np.empty(data.nsubjects)
            for s in range(data.nsubjects):
                tp = cfg.true_params.copy() if isinstance(cfg.true_params, dict) else cfg.true_params[s].copy()
                if (self.stage == 'type2') and 'type2_criteria' in tp:
                    # convert to criteria gap logic
                    tp['type2_criteria'] = np.diff(np.hstack((0, tp['type2_criteria'])))
                params_true = np.array(sum([list(tp[k]) if listlike(tp[k]) else [tp[k]] for k in tp if k in paramset.param_names], []))
                self.loglik_true[s] = -fun(params_true, s, save_type='mock')
                self.loglik_per_sample_true[s] = self.loglik_true[s] / self.nsamples[s]

    def _format_level(self, level, params):
        param = params[0] if isinstance(params, list) else params
        if level is None:
            if param.group is not None and param.prior is not None:
                return (f'group={param.group}+prior=N({param.prior[0]},{param.prior[1]})')
            elif param.group is not None:
                return f'group={param.group}'
            elif param.prior is not None:
                return (f'subject+prior=N({param.prior[0]},{param.prior[1]})')
            else:
                return 'subject'
        else:
            if param.prior is not None:
                return (f'{level}+prior=N({param.prior[0]},{param.prior[1]})')
            else:
                return level

    def report_fit(self, cfg):

        indent = f'{TAB}' if self.nsubjects == 1 else f'{TAB}{TAB}'

        def _print_parameters(params, params_se, loglik, loglik_per_sample, true_params, params_extra=None, level=None):
            parameters = cfg._paramset_type1.parameters if self.stage == 'type1' else cfg._paramset_type2.parameters
            for k, v in params.items():
                level_fmt = self._format_level(level, parameters[k][0] if isinstance(parameters[k], list) else parameters[k])
                true_string = '' if true_params is None or k not in true_params else \
                    (f" (true: [{', '.join([f'{p:.3f}' for p in true_params[k]])}])" if  # noqa
                     listlike(true_params[k]) else f' (true: {true_params[k]:.3f})')  # noqa
                se = params_se[k]
                value_string = f"[{', '.join([f'{p:.3f} ± {er:.3f}' for p, er in zip(v, se)])}]" if listlike(v) else f'{v:.3f} ± {se:.3f}'
                print(f'{indent}{TAB}[{level_fmt}] {k}: {value_string}{true_string}')

            if params_extra is not None:
                for p, v in params_extra.items():
                    if not p.split('_')[-1].isdigit() and not p.endswith('_sem'):
                        if f'{p}_sem' in params_extra:
                            se = params_extra[f'{p}_sem']
                            value_string =  f"[{', '.join([f'{p:.3f} ± {er:.3f}' for p, er in zip(v, se)])}]" if listlike(v) else f'{v:.3f} ± {se:.3f}'
                        else:
                            value_string =  f"[{', '.join([f'{p:.3f}' for p in v])}]" if listlike(v) else f'{v:.3f}'
                        if true_params is None:
                            true_string = ''
                        else:
                            v_ = true_params[p]
                            true_string = f" (true: [{', '.join([f'{p:.3f}' for p in v_])}])" if listlike(v_) else f' (true: {v_:.3f})'
                        print(f'{indent}{TAB}{TAB}[extra] {p}: {value_string}{true_string}')

            print(f'{indent}[{"final" if level is None else "subject"}] Log-likelihood: {loglik:.2f} (per sample: {loglik_per_sample:.4g})')

        print(f'{SP2}Final report')
        for s in range(self.nsubjects):
            if self.nsubjects > 1:
                print(f'{TAB}Subject {s + 1} / {self.nsubjects}')
            print(f'{indent}Parameters estimates (subject-level fit)')
            _print_parameters(
                self.subject.params[s], self.subject.params_se[s],
                self.subject.loglik[s], self.subject.loglik_per_sample[s],
                cfg.true_params[s] if isinstance(cfg.true_params, list) else cfg.true_params,
                None if self.stage == 'type1' else self.params_extra[s],
                level='subject'
            )
            if hasattr(self.subject.fit[s], 'execution_time'):
                print(f"{indent}[subject] Fitting time: {self.subject.fit[s].execution_time:.2f} secs")
            if self.group is not None:
                print(f'{indent}Parameters estimates (group-level fit)')
                _print_parameters(
                    self.group.params[s], self.group.params_se[s],
                    self.group.loglik[s], self.group.loglik_per_sample[s],
                    cfg.true_params[s] if isinstance(cfg.true_params, list) else cfg.true_params,
                    None if self.stage == 'type1' else self.params_extra[s]
                )
            if cfg.true_params is not None and hasattr(self, 'loglik_true'):
                print(f'{indent}Log-likelihood using true params: {self.loglik_true[s]:.2f} (per sample: {self.loglik_per_sample_true[s]:.4g})')

    def __str__(self):
        txt = print_class_instance(self, attr_class_only=('subject', 'group'))
        return txt

    def __repr__(self):
        return self.__str__()



class Summary(ReprMixin):

    def __init__(self, data, cfg):

        self.nsubjects = data.nsubjects
        self.nsamples = data.nsamples
        self.nparams = cfg._paramset_type1.nparams + (0 if cfg.skip_type2 else cfg._paramset_type2.nparams)

        self.type1 = ModelResultContainer(stage='type1')
        self.type2 = ModelResultContainer(stage='type2')
        self.subject = ModelResult(level='subject')
        self.group = ModelResult(level='group')

        self.stats = data.stats


    def store_combined_type1_type2(self, type1_source, type2_source, target):
        setattr(target, 'params', [{**type1_source.params[s], **type2_source.params[s]} for s in range(self.nsubjects)])
        setattr(target, 'params_se', [{**type1_source.params_se[s], **type2_source.params_se[s]} for s in range(self.nsubjects)])
        setattr(target, 'params_extra', [
            {**type1_source.params_extra[s], **({} if type2_source.params_extra[s] is None else type2_source.params_extra[s])}
            for s in range(self.nsubjects)]
                )
        if type1_source.params_random_effect is not None or type2_source.params_random_effect is not None:
            setattr(target, 'params_random_effect', Struct())
            getattr(target, 'params_random_effect').mean = \
                {**({} if type1_source.params_random_effect is None else type1_source.params_random_effect.mean),
                 **({} if type2_source.params_random_effect is None else type2_source.params_random_effect.mean)}
            getattr(target, 'params_random_effect').std = \
                {**({} if type1_source.params_random_effect is None else type1_source.params_random_effect.std),
                 **({} if type2_source.params_random_effect is None else type2_source.params_random_effect.std)}
        else:
            setattr(target, 'params_random_effect', None)
        # result_vars = ['loglik', 'loglik_per_sample', 'aic', 'aic_per_sample', 'bic', 'bic_per_sample']
        # for var in result_vars:
        #     setattr(target, var, getattr(type1_source, var) + getattr(type2_source, var))

    def shallow_result_copy(self, source, exclude=None):
        if exclude is None:
            exclude = ['loglik', 'loglik_per_sample', 'aic', 'aic_per_sample', 'bic', 'bic_per_sample', 'execution_time']
        source_copy = type(source).__new__(type(source))
        source_copy.__dict__.update({k: v for k, v in source.__dict__.items() if k not in exclude})
        return source_copy

    def store(self, store_type1_only=False):
        if store_type1_only:
            result_vars = ['params', 'params_se', 'params_extra', 'params_random_effect', 'execution_time', 'loglik', 'loglik_per_sample', 'aic', 'aic_per_sample', 'bic', 'bic_per_sample']
            for var in result_vars:
                setattr(self, var, getattr(self.type1, var))
            self.subject = self.type1.subject
            self.group = self.type1.group
        else:
            self.store_combined_type1_type2(self.type1, self.type2, self, )
            self.store_combined_type1_type2(self.type1.subject, self.type2.subject, self.subject, )
            if self.type1.group is not None and self.type2.group is None:
                # self.group = self.type1.group
                self.group = self.shallow_result_copy(self.type1.group)
            elif self.type2.group is not None and self.type1.group is None:
                # self.group = self.type2.group
                self.group = self.shallow_result_copy(self.type2.group)
            elif self.type1.group is not None and self.type2.group is not None:
                self.store_combined_type1_type2(self.type1.group, self.type2.group, self.group)

    def summary(self, c_conf_empirical=None, c_conf_generative=None, squeeze=True):

        from copy import deepcopy
        result = deepcopy(self)

        if c_conf_generative is not None:
            result.type2.confidence_gen_pearson = np.full(result.nsubjects, np.nan)
            result.type2.confidence_gen_spearman = np.full(result.nsubjects, np.nan)
            result.type2.confidence_gen_mae = np.full(result.nsubjects, np.nan)
            result.type2.confidence_gen_medae = np.full(result.nsubjects, np.nan)
            np.full(result.nsubjects, np.nan)
            for s in range(result.nsubjects):
                confidence_tiled = np.tile(c_conf_empirical[s], (c_conf_generative[s].shape[0], 1))
                with (warnings.catch_warnings()):
                    warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                    result.type2.confidence_gen_pearson[s] = \
                        np.nanmedian(pearson2d(c_conf_generative[s], confidence_tiled))
                    result.type2.confidence_gen_spearman[s] = \
                        np.nanmedian(spearman2d(c_conf_generative[s], confidence_tiled))
                result.type2.confidence_gen_mae[s] = np.nanmean(np.abs(c_conf_generative[s] - c_conf_empirical[s]))
                result.type2.confidence_gen_medae[s] = np.nanmedian(np.abs(c_conf_generative[s] - c_conf_empirical[s]))



        if squeeze:
            for target in (result, result.type1, result.type2, result.type1.subject, result.type1.group, result.type2.subject, result.type2.group):
                if target is not None:
                    for k, attr in target.__dict__.items():
                        if hasattr(attr, '__len__') and (len(attr) == 1):
                            setattr(target, k, attr[0])

        # print(self)
        return result


    def __str__(self):
        txt = print_class_instance(self, attr_class_only=('type1', 'type2'))
        # txt = f'{self.__class__.__name__}'
        # for k, v in self.__dict__.items():
        #     if k in ('type1', 'type2'):
        #         txt += f"\n\t{k}: {v.__class__.__name__}"
        #     elif isinstance(v, dict):
        #         txt += f"\n\t{k}: { {k_: np.array2string(np.array(v_), precision=4, threshold=20, separator=', ') for k_, v_ in v.items()} }"
        #     elif isinstance(v, list):
        #         if isinstance(v[0], dict):
        #             txt += f'\n\t{k}:[\n'
        #             for i in range(min(5, len(v))):
        #                 txt += '\t\t{' + ', '.join([f"'{k_}': {np.array2string(np.array(v_), precision=4, separator=', ')}" for k_, v_ in v[i].items()]) + '}\n'
        #             if len(v) > 10:
        #                 txt += '\t\t[...]\n'
        #             if len(v) > 5:
        #                 for i in range(max(-5, -len(v)), 0):
        #                     txt += '\t\t{' + ', '.join([f"'{k_}': {np.array2string(np.array(v_), precision=4, separator=', ')}" for k_, v_ in v[i].items()]) + '}\n'
        #             txt += '\t]'
        #         elif isinstance(v[0], float):
        #             txt += f"\n\t{k}: {np.array2string(np.array(v), precision=4, threshold=50, separator=', ')}"
        #     elif isinstance(v, np.ndarray):
        #         txt += f"\n\t{k}: {np.array2string(np.array(v), precision=4, threshold=50, separator=', ')}"
        #     else:
        #         txt += f"\n\t{k}: {v}"
        # print(txt)
        return txt

    def __repr__(self):
        return self.__str__()


class ModelData(ReprMixin):
    def __init__(self, cfg):
        """
        Container class for the model data.

        Parameters
        ----------
        cfg : configuration.Configuration
            Settings
        """
        self.cfg = cfg

        self.super_thresh = None
        self.y_decval = None
        self.y_decval_grid = None
        self.y_decval_grid_invalid = None
        self.y_decval_pmf_grid = None
        self.z1_type1_evidence = None
        self.z1_type1_evidence_grid = None
        self.c_conf_grid = None
        self.c_conf = None

        self.nsubjects = None
        self.nsamples = None

        self.type1_likelihood = None
        self.type1_posterior = None
        self.type2_likelihood = None
        self.type2_likelihood_grid = None

        self.precomputed = Struct()


class ModelFit(ReprMixin):
    fit_type1: OptimizeResult = None
    fit_type2: OptimizeResult = None
