from dataclasses import make_dataclass
import warnings

import numpy as np
from scipy.optimize import OptimizeResult

from .util import TAB, ReprMixin, spearman2d, pearson2d


class Parameter(ReprMixin):
    def __init__(self, guess=None, bounds=None, grid_range=None):
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
        """
        self.guess = guess
        self.bounds = bounds
        self.grid_range = np.linspace(bounds[0], bounds[1], 4) if grid_range is None else grid_range
        self.default_changed = False

    def copy(self):
        return Parameter(self.guess, self.bounds, self.grid_range)

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
        self.param_len = [len(parameters[name]) if self.param_is_list[p] else 1 for p, name in enumerate(param_names)]  # noqa
        self.param_names_flat = sum([[f'{name}_{i}' for i in range(len(parameters[name]))] if self.param_is_list[p]  # noqa
                          else [name] for p, name in enumerate(param_names)], [])
        self.guess = np.array(sum([[parameters[name][i].guess for i in range(len(parameters[name]))] if  # noqa
                                   self.param_is_list[p] else [parameters[name].guess] for p, name in enumerate(param_names)], []))
        self.bounds = np.array(sum([[parameters[name][i].bounds for i in range(len(parameters[name]))] if  # noqa
                                   self.param_is_list[p] else [parameters[name].bounds] for p, name in enumerate(param_names)], []))
        self.grid_range = np.array(sum([[parameters[name][i].grid_range for i in range(len(parameters[name]))] if  # noqa
                                   self.param_is_list[p] else [parameters[name].grid_range] for p, name in enumerate(param_names)], []),
                                   dtype=object)
        self.constraints = constraints
        self.nparams = len(param_names)
        self.nparams_flat = len(self.param_names_flat)


class Data(ReprMixin):

    def __init__(self, cfg, stimuli=None, choices=None, confidence=None):
        """
        Container class for behavioral data.

        Parameters
        ----------
        cfg : configuration.Configuration
            Settings
        stimuli : array-like of shape (n_samples)
            Array of signed stimulus intensity values, where the sign codes the stimulus category (+: cat1; -: cat2) and
            the absolut value codes the intensity. The scale of the data is not relevant, as a normalisation to [-1; 1]
            is applied.
            Note: stimuli are automatically preprocessed and are made available as the Data attribute x_stim.
        choices : array-like of shape (n_samples)
            Array of choices coded as 0 (cat1) and 1 (cat2) for the two stimulus categories. See parameter 'stimuli'
            for the definition of cat1 and cat2.
            Note: choices are automatically preprocessed and are made available as the Data attribute d_dec.
        confidence : array-like of shape (n_samples)
            Confidence ratings; must be normalized to the range [0;1].
            Note: confidence is automatically preprocessed and is made available as the Data attribute c_conf.
        """

        self.cfg = cfg
        self._stimuli = None if stimuli is None else np.array(stimuli)
        self._choices = None if choices is None else np.array(choices)
        self._confidence = None if confidence is None else np.array(confidence)

        self._x_stim = None
        self._d_dec = None
        self._c_conf = None
        if self.cfg.type2_fitting_type == 'criteria':
            self.c_conf_discrete = None

        self.stimuli_max = None
        self.stimuli_min = None

        self.x_stim_category = None
        self.x_stim_2d = None

        self.d_dec = None
        self.d_dec_2d = None
        self.d_dec_sign = None
        self.accuracy = None

        self.c_conf = None
        self.c_conf_2d = None

        self.nsamples = len(self._confidence) if self._stimuli is None else len(self._stimuli)

        self.jacobian_noisy_temperature = None

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

    def preproc_stim(self, stimuli=None):

        self.x_stim = self._stimuli if stimuli is None else stimuli

        self.stimuli_min = np.abs(self.stimuli).min()
        self.stimuli_max = np.max(np.abs(self.stimuli))
        # Normalize stimuli
        if self.cfg.normalize_stimuli_by_max:
            self.x_stim = self.stimuli / self.stimuli_max
        else:
            self.x_stim = self.stimuli
            if np.max(np.abs(self.x_stim)) > 1:
                raise ValueError('Stimuli are not normalized to the range [-1; 1].')
        self.x_stim_2d = self.x_stim.reshape(-1, 1)
        self.x_stim_category = (np.sign(self.x_stim) == 1).astype(int)

    def preproc_dec(self, choices=None):

        self.d_dec = self._choices if choices is None else choices

        # convert to 0/1 scheme if choices are provides as -1's and 1's
        if np.array_equal(np.unique(self.d_dec[~np.isnan(self.d_dec)]), [-1, 1]):
            self.d_dec[self.d_dec == -1] = 0
        self.d_dec_sign = np.sign(self.d_dec - 0.5)
        self.d_dec_2d = self.d_dec.reshape(-1, 1)
        self.accuracy = (self.x_stim_category == self.d_dec).astype(int)

    def preproc_conf(self, confidence=None):

        self.c_conf = self._confidence if confidence is None else confidence

        if self.c_conf is not None:
            if self.cfg.type2_fitting_type == 'criteria':
                self.c_conf_discrete = np.digitize(self.c_conf, np.arange(1/self.cfg.n_discrete_confidence_levels, 1, 1/self.cfg.n_discrete_confidence_levels))
            self.c_conf_2d = self.c_conf.reshape(-1, 1)

    def summary(self, full=False):
        desc = dict(
            nsamples=self.nsamples
        )
        if full:
            dict_extended = dict(
                stimuli=self.stimuli,
                x_stim=self.x_stim,
                x_stim_category=(self.x_stim >= 0).astype(int),
                choices=self.choices,
                d_dec=self.d_dec,
                confidence=self.confidence,
                c_conf=self.c_conf
            )
            data_extended = make_dataclass('DataExtended', dict_extended.keys())
            data_extended.__module__ = '__main__'
            desc.update(dict(data=data_extended(**dict_extended)))
        data_summary = make_dataclass('DataSummary', desc.keys())

        def _repr(self_):
            txt = f'{self_.__class__.__name__}\n'
            txt += '\n'.join([f"\t{k}: {'<not displayed>' if k == 'data' else v}"
                              for k, v in self_.__dict__.items()])
            return txt

        data_summary.__repr__ = _repr
        data_summary.__module__ = '__main__'
        return data_summary(**desc)


class Model(ReprMixin):
    def __init__(self, cfg):
        """
        Container class for behavioral data.

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
        self.nsamples = None

        self.type1_likelihood = None
        self.type1_posterior = None
        self.type2_likelihood = None
        self.type2_likelihood_grid = None
        self.uniform_type2_negll = None

        self.params = None

        self.params_type1 = None
        self.params_type1_full = None
        self.params_type1_unnorm = None

        self.params_type2 = None
        self.params_type2_extra = None
        self.type2_likelihood_dist = None

        self.type1_dist = None
        self.quintiles_noisy_temperature = None

        self.fit = ModelFit()

    def store_type1(self, negll_type1, params_type1, type1_likelihood, type1_posterior, stimuli_max):
        self.params_type1 = params_type1
        self.type1_likelihood = type1_likelihood
        self.type1_posterior = type1_posterior
        self.fit.fit_type1.negll = negll_type1
        self.nsamples = len(type1_posterior)
        self.params_type1_unnorm = {k: list(np.array(v) * stimuli_max) if hasattr(v, '__len__') else
            v * stimuli_max for k, v in params_type1.items()}

    def store_type2(self, negll_type2=None, params_type2=None, type2_likelihood=None, type2_likelihood_grid=None):
        self.params_type2 = params_type2
        if 'type2_criteria' in params_type2:
             self.params_type2_extra = dict(
                type2_criteria_absolute=[np.sum(params_type2['type2_criteria'][:i+1]) for i in range(len(params_type2['type2_criteria']))],
                type2_criteria_bias=np.mean(params_type2['type2_criteria'])*(len(params_type2['type2_criteria'])+1)-1
             )
             for i in range(len(params_type2['type2_criteria'])):
                 self.params_type2_extra[f'type2_criteria_absolute_{i}'] = self.params_type2_extra[f'type2_criteria_absolute'][i]
             for i in range(len(params_type2['type2_criteria'])):
                 self.params_type2_extra[f'type2_criteria_{i}'] = self.params_type2[f'type2_criteria'][i]
             if self.cfg.true_params is not None and 'type2_criteria' in self.cfg.true_params:
                 self.params_type2_extra.update(
                    type2_criteria_absolute_true=[np.sum(self.cfg.true_params['type2_criteria'][:i+1]) for i in range(len(self.cfg.true_params['type2_criteria']))],
                    type2_criteria_bias_true=np.mean(self.cfg.true_params['type2_criteria'])*(len(self.cfg.true_params['type2_criteria'])+1)-1
                 )
        self.type2_likelihood = type2_likelihood
        self.type2_likelihood_grid = type2_likelihood_grid
        self.fit.fit_type2.negll = negll_type2

    def report_fit_type1(self, verbose=True):
        if verbose:
            for k, v in self.params_type1.items():
                true_string = '' if self.cfg.true_params is None else \
                    (f" (true: [{', '.join([f'{p:.3g}' for p in self.cfg.true_params[k]])}])" if  # noqa
                     hasattr(self.cfg.true_params[k], '__len__') else f' (true: {self.cfg.true_params[k]:.3g})')  # noqa
                value_string = f"[{', '.join([f'{p:.3g}' for p in v])}]" if hasattr(v, '__len__') else f'{v:.3g}'
                print(f'{TAB}[final] {k}: {value_string}{true_string}')
            print(f'Final neg. LL: {self.fit.fit_type1.negll:.2f}')
            if self.cfg.true_params is not None and hasattr(self.fit.fit_type1, 'negll_true'):
                print(f'Neg. LL using true params: {self.fit.fit_type1.negll_true:.2f}')
            if hasattr(self.fit.fit_type1, 'execution_time'):
                print(f"Total fitting time: {self.fit.fit_type1.execution_time:.2g} secs")
            if hasattr(self.fit.fit_type1, 'best_solver'):
                print(f"(Best) Solver: {self.fit.fit_type1.best_solver:.2g} secs")

    def report_fit_type2(self, verbose=True):
        if verbose:
            for k, v in self.params_type2.items():
                if k == 'type2_criteria':
                    for i in range(self.cfg.n_discrete_confidence_levels-1):
                        true_param = None if self.cfg.true_params is None or k not in self.cfg.true_params else self.cfg.true_params[k][i]
                        true_string = '' if true_param is None else f' (true: {true_param:.3g})'
                        if i > 0:
                            criterion = np.sum(self.params_type2[k][:i+1])
                            true_string_gap = '' if true_param is None else f' (true: {np.sum(self.cfg.true_params[k][:i+1]):.3g})'
                            gap_string = f' = gap | criterion = {criterion:.3g}{true_string_gap}'
                        else:
                            gap_string = ''
                        print(f'{TAB}[final] {k}_{i}: {v[i]:.3g}{true_string}{gap_string}')
                else:
                    true_string = '' if self.cfg.true_params is None or k not in self.cfg.true_params else \
                        (f" (true: [{', '.join([f'{p:.3g}' for p in self.cfg.true_params[k]])}])" if
                         hasattr(self.cfg.true_params[k], '__len__') else f' (true: {self.cfg.true_params[k]:.3g})')
                    value_string = f"[{', '.join([f'{p:.3g}' for p in v])}]" if hasattr(v, '__len__') else f'{v:.3g}'
                    print(f'{TAB}[final] {k}: {value_string}{true_string}')
            if self.params_type2_extra is not None:
                for p, v in self.params_type2_extra.items():
                    if not p.endswith('_true') and not p.split('_')[-1].isdigit():
                        value_string =  f"[{', '.join([f'{p:.3g}' for p in v])}]" if hasattr(v, '__len__') else f'{v:.3g}'
                        if f'{p}_true' in self.params_type2_extra:
                            v_ = self.params_type2_extra[f'{p}_true']
                            true_string = f" (true: [{', '.join([f'{p:.3g}' for p in v_])}])" if hasattr(v_, '__len__') else f' (true: {v_:.3g})'
                        else:
                            true_string = ''
                        print(f'{TAB}[extra] {p}: {value_string}{true_string}')
            print(f'Final neg. LL: {self.fit.fit_type2.negll:.2f}')
            if self.cfg.true_params is not None:
                if verbose:
                    print(f'Neg. LL using true params: {self.fit.fit_type2.negll_true:.2f}')
            if hasattr(self.fit.fit_type2, 'execution_time'):
                print(f"Total fitting time: {self.fit.fit_type2.execution_time:.2g} secs")
            if hasattr(self.fit.fit_type2, 'best_solver'):
                print(f"(Best) Solver: {self.fit.fit_type2.best_solver}")

        # We delete possible "true" values from params_type2_extra, since they are only used for
        # the output and can be confusing to the user.
        if self.params_type2_extra is not None:
            true_keys = [k for k in self.params_type2_extra.keys() if k.endswith('_true')]
            for k in true_keys:
                self.params_type2_extra.pop(k, None)

    def summary(self, extended=False, fun_negll_type2=None, c_conf_empirical=None, c_conf_generative=None):

        if hasattr(self.fit, 'fit_type2') and self.fit.fit_type2 is not None:

            if c_conf_generative is not None:
                confidence_tiled = np.tile(c_conf_empirical, (c_conf_generative.shape[0], 1))
                with (warnings.catch_warnings()):
                    warnings.filterwarnings("ignore", message="All-NaN slice encountered", category=RuntimeWarning)
                    self.fit.fit_type2.confidence_gen_pearson = \
                        np.nanmedian(pearson2d(c_conf_generative, confidence_tiled))
                    self.fit.fit_type2.confidence_gen_spearman = \
                        np.nanmedian(spearman2d(c_conf_generative, confidence_tiled))
                self.fit.fit_type2.confidence_gen_mae = np.nanmean(np.abs(c_conf_generative - c_conf_empirical))
                self.fit.fit_type2.confidence_gen_medae = np.nanmedian(np.abs(c_conf_generative - c_conf_empirical))
            self.fit.fit_type2.negll_persample = self.fit.fit_type2.negll / self.nsamples
            self.fit.negll = self.fit.fit_type1.negll + self.fit.fit_type2.negll

        desc = dict(
            nsamples=self.nsamples,
            nparams_type1=self.cfg.paramset_type1.nparams,
            params_type1=self.params_type1,
            type1_model_evidence=dict(
                negll=self.fit.fit_type1.negll,
                aic=2 * self.cfg.paramset_type1.nparams + 2 * self.fit.fit_type1.negll,
                bic=2*np.log(self.nsamples) + 2*self.fit.fit_type1.negll
            ),
            params=self.params_type1,
            params_type1_unnorm=self.params_type1_unnorm,
            fit=self.fit
        )

        if self.cfg.true_params is not None:
            desc['type1_model_evidence'].update(
                negll_true=self.fit.fit_type1.negll_true,
                aic_true=2 * self.cfg.paramset_type1.nparams + 2 * self.fit.fit_type1.negll_true,
                bic_true=2*np.log(self.nsamples) + 2*self.fit.fit_type1.negll_true
            )

        if hasattr(self.fit, 'fit_type2') and self.fit.fit_type2 is not None:
            desc.update(dict(
                nparams_type2=self.cfg.paramset_type2.nparams,
                params_type2=self.params_type2,
                params_type2_extra={} if self.params_type2_extra is None else self.params_type2_extra,
                params={**self.params_type1, **self.params_type2},
                nparams=self.cfg.paramset_type1.nparams + self.cfg.paramset_type2.nparams,
                type2_model_evidence=dict(
                    negll=self.fit.fit_type2.negll,
                    aic=2 * self.cfg.paramset_type2.nparams + 2 * self.fit.fit_type2.negll,
                    bic=2*np.log(self.nsamples) + 2*self.fit.fit_type2.negll
                )
            ))
            if self.cfg.true_params is not None:
                desc['type2_model_evidence'].update(
                    negll_true=self.fit.fit_type2.negll_true,
                    aic_true=2 * self.cfg.paramset_type2.nparams + 2 * self.fit.fit_type2.negll_true,
                    bic_true=2*np.log(self.nsamples) + 2*self.fit.fit_type2.negll_true
                )

            if extended:
                dict_extended = dict(
                    type2_likelihood_grid=self.type2_likelihood_grid,
                    type2_likelihood=self.type2_likelihood,
                    c_conf_grid=self.c_conf_grid,
                    c_conf=self.c_conf,
                    z1_type1_evidence_grid=self.z1_type1_evidence_grid,
                    z1_type1_evidence=self.z1_type1_evidence,
                    y_decval_grid=self.y_decval_grid,
                    y_decval=self.y_decval,
                    y_decval_pmf_grid=self.y_decval_pmf_grid,
                    type1_likelihood=self.type1_likelihood,
                    type1_posterior=self.type1_posterior,
                )
                model_extended = make_dataclass('ModelExtended', dict_extended.keys())
                model_extended.__module__ = '__main__'
                desc.update(dict(extended=model_extended(**dict_extended)))

        model_summary = make_dataclass('ModelSummary', desc.keys())

        def _repr(self_):
            txt = f'{self_.__class__.__name__}'
            for k, v in self_.__dict__.items():
                if k in ('data', 'fit'):
                    txt += f"\n\t{k}: <not displayed>"
                elif k == 'extended':
                    txt += f"\n\t{k}: additional modeling results (attributes: " \
                           f"{', '.join([a for a in self_.extended.__dict__.keys()])})"
                else:
                    txt += f"\n\t{k}: {v}"
            return txt

        model_summary.__repr__ = _repr
        model_summary.__module__ = '__main__'
        return model_summary(**desc)


class ModelFit(ReprMixin):
    fit_type1: OptimizeResult = None
    fit_type2: OptimizeResult = None
