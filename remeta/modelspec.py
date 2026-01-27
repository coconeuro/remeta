import warnings

import numpy as np
from scipy.optimize import OptimizeResult

from .util import TAB, SP2, Struct, ReprMixin, spearman2d, pearson2d, listlike, empty_list


class Parameter(ReprMixin):
    def __init__(self, guess=None, bounds=None, grid_range=None, group=None, prior=None):
        """
        Class that defines the fitting characteristics of a Parameter.

        Parameters
        ----------
        guess : None | float | np.floating
            Initial guess for the parameter value.
        bounds: None | array-like of length 2
            Parameter bounds. The first and second element indicate the lower and upper bound of the parameter.
        grid_range: None | array-like (1d)
            1-d grid for initial gridsearch in the parameter optimization procedure
        group: None | str
            None: no group-level estimate for the parameter
            'fixed': fit parameter as a group fixed effect (i.e., single value for the group)
            'random': fit parameter as a random effect (enforces shrinkage towards a group mean)
       prior: None | tuple[float, float]
            None: no prior for the parameter
            (group_mean, group_sd): apply a Normal prior defined by mean and standard deviation

        """
        self.guess = guess
        self.bounds = bounds
        self.grid_range = np.linspace(bounds[0], bounds[1], 4) if grid_range is None else grid_range
        self.group = group
        self.prior = prior
        self.default_changed = False

    def copy(self):
        return Parameter(self.guess, self.bounds, self.grid_range, self.group, self.prior)

    def __setattr__(self, name, value):
        if name != "default_changed" and hasattr(self, name):
            old_value = getattr(self, name)
            if isinstance(old_value, np.ndarray) and isinstance(value, np.ndarray):
                changed = not np.array_equal(old_value, value)
            else:
                changed = old_value != value
            if changed:
                super().__setattr__("default_changed", True)

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
        if self.cfg.type2_fitting_type == 'criteria':
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

        self.preproc_stim()
        self.preproc_dec()
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

        self.x_stim_min = min([np.abs(v).min() for v in self.x_stim])
        self.x_stim_max = max([np.abs(v).max() for v in self.x_stim])

        self.x_stim_3d, self.x_stim_category = [None] * self.nsubjects, [None] * self.nsubjects
        for s in range(self.nsubjects):
            # Normalize stimuli
            if self.cfg.normalize_stimuli_by_max:
                self.x_stim[s] /= self.x_stim_max
            else:
                # self.x_stim = self.stimuli
                if np.max(np.abs(self.x_stim[s])) > 1:
                    raise ValueError('Stimuli are not normalized to the range [-1; 1].')
            self.x_stim_3d[s] = self.x_stim[s][..., np.newaxis]
            self.x_stim_category[s] = (np.sign(self.x_stim[s]) == 1).astype(int)

    def preproc_dec(self, choices=None):

        self.d_dec = self.ensure_list_of_arrays(self._choices if choices is None else choices)

        self.d_dec_sign, self.d_dec_3d, self.accuracy = [None] * self.nsubjects, [None] * self.nsubjects, [None] * self.nsubjects
        for s in range(self.nsubjects):
            # convert to 0/1 scheme if choices are provides as -1's and 1's
            if np.array_equal(np.unique(self.d_dec[s][~np.isnan(self.d_dec[s])]), [-1, 1]):
                self.d_dec[s][self.d_dec[s] == -1] = 0
            self.d_dec_sign[s] = np.sign(self.d_dec[s] - 0.5)
            self.d_dec_3d[s] = self.d_dec[s][..., np.newaxis]
            self.accuracy[s] = (self.x_stim_category[s] == self.d_dec[s]).astype(int)

    def preproc_conf(self, confidence=None):

        if self._confidence is not None or confidence is not None:
            self.c_conf = self.ensure_list_of_arrays(self._confidence if confidence is None else confidence)
            if self.c_conf is not None:
                self.nsubjects = len(self.c_conf)
                self.nsamples = np.array([len(v) for v in self.c_conf], int)
                self.c_conf_discrete, self.c_conf_3d = [None] * self.nsubjects, [None] * self.nsubjects
                for s in range(self.nsubjects):
                    if self.cfg.type2_fitting_type == 'criteria':
                        self.c_conf_discrete[s] = np.digitize(self.c_conf[s], np.arange(1/self.cfg.n_discrete_confidence_levels, 1, 1/self.cfg.n_discrete_confidence_levels))
                    self.c_conf_3d[s] = self.c_conf[s][..., np.newaxis]



class ModelResult(ReprMixin):

    def __init__(self, level):
        self.level = level

    def store(self, cfg, data, params, fun, args, stage, pop_mean_sd=None, fit=None):
        self.nparams = cfg.paramset_type1.nparams
        self.nsubjects = data.nsubjects
        self.nsamples = data.nsamples
        self.negll = np.empty(self.nsubjects)
        self.negll_per_sample = np.empty(self.nsubjects)
        self.aic = np.empty(self.nsubjects)
        self.aic_per_sample = np.empty(self.nsubjects)
        self.bic = np.empty(self.nsubjects)
        self.bic_per_sample = np.empty(self.nsubjects)
        self.params = empty_list(self.nsubjects, None)
        self.params_extra = empty_list(self.nsubjects, None)
        self.params_random_effect = None
        for s in range(self.nsubjects):
            fun(params[s], s, *args, save_target=self.level)
            self.negll_per_sample[s] = self.negll[s] / self.nsamples[s]
            self.aic[s] = 2 * self.nparams + 2 * self.negll[s]
            self.aic_per_sample[s] = self.aic[s] / self.nsamples[s]
            self.bic[s] = 2 * np.log(self.nsamples[s]) + 2 * self.negll[s]
            self.bic_per_sample[s] = self.bic[s] / self.nsamples[s]
            if stage == 'type1':
                self.params_extra[s] = {f'{k}_unnorm': list(np.array(v) * data.x_stim_max) if listlike(v) else
                                            v * data.x_stim_max for k, v in self.params[s].items()}
            elif 'type2_criteria' in cfg.paramset_type2.param_names:
                self.params_extra[s] = self._compute_parameters_extra(self.params[s])

        if pop_mean_sd is not None:
            self.params_random_effect = Struct()
            self.params_random_effect.mean = {k: p if hasattr(p:=pop_mean_sd[0][ind], '__len__') and len(p) > 1 else
                float(np.squeeze(p)) for k, ind in getattr(cfg, f'paramset_{stage}').param_revind_re.items()}
            self.params_random_effect.std = {k: p if hasattr(p:=pop_mean_sd[1][ind], '__len__') and len(p) > 1 else
                float(np.squeeze(p)) for k, ind in getattr(cfg, f'paramset_{stage}').param_revind_re.items()}

        if fit is not None:
            self.fit = fit

    def _compute_parameters_extra(self, params):
        params_extra = dict()
        params_extra['type2_criteria_absolute'] = \
            [np.sum(params['type2_criteria'][:i+1]) for i in range(len(params['type2_criteria']))]
        params_extra['type2_criteria_bias'] = \
            np.mean(params['type2_criteria'])*(len(params['type2_criteria'])+1)-1
        for i in range(len(params['type2_criteria'])):
            params_extra[f'type2_criteria_absolute_{i}'] = params_extra[f'type2_criteria_absolute'][i]
        for i in range(len(params['type2_criteria'])):
            params_extra[f'type2_criteria_{i}'] = params[f'type2_criteria'][i]
        return params_extra


class ModelResultContainer(ReprMixin):
    def __init__(self, stage):
        self.stage = stage
        self.subject = ModelResult(level='subject')
        self.group = None
        self.nparams = None
        self.nsubjects = None
        self.nsamples = None
        self.negll = None

    def init_group(self):
        self.group = ModelResult(level='group')

    def store(self, cfg, data, fun, args):
        self.nparams = cfg.paramset_type1.nparams
        self.nsubjects = data.nsubjects
        self.nsamples = data.nsamples

        result_vars = ['params', 'params_extra', 'params_random_effect', 'negll', 'negll_per_sample', 'aic', 'aic_per_sample', 'bic', 'bic_per_sample']
        final_level = self.subject if self.group is None else self.group
        for var in result_vars:
            setattr(self, var, getattr(final_level, var))

        if cfg.true_params is not None:
            paramset = cfg.paramset_type1 if self.stage == 'type1' else cfg.paramset_type2
            if not np.all([p in cfg.true_params for p in paramset.param_names]):
                raise ValueError(f'Set of provided true parameters is incomplete (Stage {self.stage}).')
            self.negll_true = np.empty(data.nsubjects)
            for s in range(data.nsubjects):
                tp = cfg.true_params.copy() if isinstance(cfg.true_params, dict) else cfg.true_params[s].copy()
                params_true = sum([tp[k] if listlike(tp[k]) else [tp[k]] for k in tp if k in paramset.param_names], [])
                self.negll_true[s] = fun(params_true, s, *args)

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

        def _print_parameters(params, negll, true_params, params_extra=None, level=None):
            parameters = cfg.paramset_type1.parameters if self.stage == 'type1' else cfg.paramset_type2.parameters
            for k, v in params.items():
                level_fmt = self._format_level(level, parameters[k][0] if isinstance(parameters[k], list) else parameters[k])
                if k == 'type2_criteria':
                    for i in range(cfg.n_discrete_confidence_levels-1):
                        true_param_crit = None if true_params is None or k not in true_params else true_params[k][i]
                        true_string = '' if true_param_crit is None else f' (true: {true_param_crit:.3g})'
                        if i > 0:
                            criterion = np.sum(params[k][:i+1])
                            true_string_gap = '' if true_param_crit is None else f' (true: {np.sum(true_params[k][:i+1]):.3g})'
                            gap_string = f' = gap | criterion = {criterion:.3g}{true_string_gap}'
                        else:
                            gap_string = ''
                        print(f'{indent}{TAB}[{level_fmt}] {k}_{i}: {v[i]:.3g}{true_string}{gap_string}')
                else:
                    true_string = '' if true_params is None or k not in true_params else \
                        (f" (true: [{', '.join([f'{p:.3g}' for p in true_params[k]])}])" if  # noqa
                         listlike(true_params[k]) else f' (true: {true_params[k]:.3g})')  # noqa
                    value_string = f"[{', '.join([f'{p:.3g}' for p in v])}]" if listlike(v) else f'{v:.3g}'
                    print(f'{indent}{TAB}[{level_fmt}] {k}: {value_string}{true_string}')

            if params_extra is not None:
                for p, v in params_extra.items():
                    if not p.split('_')[-1].isdigit():
                        value_string =  f"[{', '.join([f'{p:.3g}' for p in v])}]" if listlike(v) else f'{v:.3g}'
                        if true_params is None:
                            true_string = ''
                        else:
                            v_ = true_params[p]
                            true_string = f" (true: [{', '.join([f'{p:.3g}' for p in v_])}])" if listlike(v_) else f' (true: {v_:.3g})'
                        print(f'{indent}{TAB}{TAB}[extra] {p}: {value_string}{true_string}')

            print(f'{indent}[{"final" if level is None else "subject"}] Neg. LL: {negll:.2f}')

        print(f'{SP2}Final report')
        for s in range(self.nsubjects):
            if self.nsubjects > 1:
                print(f'{TAB}Subject {s + 1} / {self.nsubjects}')
            print(f'{indent}Parameters estimates (subject-level fit)')
            _print_parameters(
                self.subject.params[s], self.subject.negll[s],
                cfg.true_params[s] if isinstance(cfg.true_params, list) else cfg.true_params,
                None if self.stage == 'type1' else self.params_extra[s],
                level='subject'
            )
            if hasattr(self.subject.fit[s], 'execution_time'):
                print(f"{indent}[subject] Fitting time: {self.subject.fit[s].execution_time:.2f} secs")
            if self.group is not None:
                print(f'{indent}Parameters estimates (group-level fit)')
                _print_parameters(
                    self.group.params[s], self.group.negll[s],
                    cfg.true_params[s] if isinstance(cfg.true_params, list) else cfg.true_params,
                    None if self.stage == 'type1' else self.params_extra[s]
                )
            if cfg.true_params is not None and hasattr(self, 'negll_true'):
                print(f'{indent}Neg. LL using true params: {self.negll_true[s]:.2f}')


class Summary(ReprMixin):

    def __init__(self, data, cfg):

        self.nsubjects = data.nsubjects
        self.nsamples = data.nsamples
        self.nparams = cfg.paramset_type1.nparams + (0 if cfg.skip_type2 else cfg.paramset_type2.nparams)

        self.type1 = ModelResultContainer(stage='type1')
        self.type2 = ModelResultContainer(stage='type2')

    def store(self, cfg):
        if cfg.skip_type2:
            result_vars = ['params', 'params_extra', 'params_random_effect', 'negll', 'negll_per_sample', 'aic', 'aic_per_sample', 'bic', 'bic_per_sample']
            for var in result_vars:
                setattr(self, var, getattr(self.type1, var))
        else:
            self.params = [{**self.type1.params[s], **self.type2.params[s]} for s in range(self.nsubjects)]
            self.params_extra = [
                {**self.type1.params_extra[s], **({} if self.type2.params_extra[s] is None else self.type2.params_extra[s])}
                for s in range(self.nsubjects)]
            if self.type1.params_random_effect is not None or self.type2.params_random_effect is not None:
                self.params_random_effect = Struct()
                self.params_random_effect.mean = \
                    {**({} if self.type1.params_random_effect is None else self.type1.params_random_effect.mean),
                     **({} if self.type2.params_random_effect is None else self.type2.params_random_effect.mean)}
                self.params_random_effect.std = \
                    {**({} if self.type1.params_random_effect is None else self.type1.params_random_effect.std),
                     **({} if self.type2.params_random_effect is None else self.type2.params_random_effect.std)}
            else:
                self.params_random_effect = None
            result_vars = ['negll', 'negll_per_sample', 'aic', 'aic_per_sample', 'bic', 'bic_per_sample']
            for var in result_vars:
                setattr(self, var, getattr(self.type1, var) + getattr(self.type2, var))

    def summary(self, c_conf_empirical=None, c_conf_generative=None, squeeze=True):

        from copy import deepcopy
        result = deepcopy(self)

        if c_conf_generative is not None:
            for s in range(result.nsubjects):
                confidence_tiled = np.tile(c_conf_empirical[s], (c_conf_generative[s].shape[0], 1))
                with (warnings.catch_warnings()):
                    warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                    result.type2.confidence_gen_pearson = \
                        np.nanmedian(pearson2d(c_conf_generative[s], confidence_tiled))
                    result.type2.confidence_gen_spearman = \
                        np.nanmedian(spearman2d(c_conf_generative[s], confidence_tiled))
                result.type2.confidence_gen_mae = np.nanmean(np.abs(c_conf_generative[s] - c_conf_empirical[s]))
                result.type2.confidence_gen_medae = np.nanmedian(np.abs(c_conf_generative[s] - c_conf_empirical[s]))



        if squeeze:
            for target in (result, result.type1, result.type2, result.type1.subject, result.type1.group, result.type2.subject, result.type2.group):
                if target is not None:
                    for k, attr in target.__dict__.items():
                        if hasattr(attr, '__len__') and (len(attr) == 1):
                            setattr(target, k, attr[0])

        # print(self)
        return result


    def __str__(self):
        txt = f'{self.__class__.__name__}'
        for k, v in self.__dict__.items():
            if k in ('type1', 'type2'):
                txt += f"\n\t{k}: {v.__class__.__name__}"
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
