from __future__ import annotations
import warnings
from dataclasses import dataclass, fields, field

import numpy as np
from importlib.util import find_spec

from .modelspec import Parameter, ParameterSet
from .util import ReprMixin, reset_dataclass_on_init, listlike



@reset_dataclass_on_init
@dataclass
class Configuration(ReprMixin):
    """ Configuration for the ReMeta toolbox

    Usage:
        ```
        cfg = remeta.configuration
        cfg.<some_setting> = <some_value>
        rem = remeta.ReMeta(cfg)
        ```
    """

    ### Important settings

    normalize_stimuli_by_max: bool = field(default=False, metadata={'description': """ 
        If True, normalize provided stimuli by their maximum value to the range [-1; 1]. 
        Note that stimuli should be roughly in the range [-1; 1] for optimal parameter estimation."""
    })

    type2_noise_type: str = field(default='report', metadata={'description': """ 
        Whether the model considers noise at readout, report or for the estimation of type 1 noise ("temperature").
        Possible values: `'readout'`, `'report'`, `'temperature'`."""
    })

    skip_type2: bool = field(default=False, metadata={'description': """ 
        If `True`, only fit type 1 data. No confidence data needs to be passed to `fit()` in this case."""
    })



    ### Type 1 optimization

    optim_type1_gridsearch: bool = field(default=False, metadata={'description': """ 
        If `True`, perform an initial gridsearch search for type 1 parameter optimization, based on the `grid_range`
        attributes of Parameters."""
    })

    # optim_type1_fine_gridsearch: bool = field(default=False, metadata={'description': """
    #     Perform a fine-grained grid search for type 1 parameter optimization."""
    # })

    optim_type1_minimize_along_grid: bool = field(default=False, metadata={'description': """ 
        If `True`, do sqlqp minimization for type 1 parameter optimization at each grid point."""
    })

    optim_type1_global_minimization: str = field(default=None, metadata={'description': """ 
        Use one of 'shgo', 'dual_annealing' 'differential_evolution' to perform type 1 likelihood minimization with
        a global minimizer."""
    })

    optim_type1_scipy_solvers: str | list[str] | tuple[str, ...] = field(default='trust-constr', metadata={'description': """ 
        Set scipy.optimize.minimize solver method for type 1 parameter optimization..
        If provided as tuple/list, test different solvers and take the best."""
    })



    ### Type 2 optimization

    optim_type2_gridsearch: bool = field(default=False, metadata={'description': """ 
        If `True`, perform an initial gridsearch search for type 2 parameter optimization, based on the `grid_range`
        attributes of Parameters."""
    })

    # optim_type2_fine_gridsearch: bool = field(default=False, metadata={'description': """
    #     Perform a fine-grained grid search for type 2 parameter optimization."""
    # })

    optim_type2_minimize_along_grid: bool = field(default=False, metadata={'description': """ 
        If `True`, do sqlqp minimization for type 2 parameter optimization at each grid point."""
    })

    optim_type2_global_minimization: str = field(default=None, metadata={'description': """ 
        Use one of 'shgo', 'dual_annealing' 'differential_evolution' to perform type 2 likelihood minimization with
        a global minimizer."""
    })

    optim_type2_scipy_solvers: str | list[str] | tuple[str, ...] = field(default=('slsqp', 'Nelder-Mead'), metadata={'description': """ 
        Set scipy.optimize.minimize solver method for type 2 parameter optimization..
        If provided as tuple/list, test different solvers and take the best."""
    })

    optim_type2_slsqp_epsilon: float = field(default=None, metadata={'description': """ 
        Set parameter epsilon parameter for the SLSQP optimization method (type 2).
        If provided as tuple/list, test different eps parameters and take the best"""
    })

    optim_num_cores: int = field(default=1, metadata={'description': """ 
        Number of cores used for parameter estimation (-1 for all cores minus 1)."""
    })



    ### Parameters

    ## Type 1

    param_type1_noise: Parameter = field(
        default=Parameter(enable=1, guess=0.5, bounds=(0.001, 10), grid_range=np.linspace(0.1, 1, 8), default=0.01, model='normal'),
        metadata={'description': """ 
        Type 1 noise."""
    })
    param_type1_thresh: Parameter = field(
        default=Parameter(enable=0, guess=0, bounds=(0, 1), grid_range=np.linspace(0, 0.2, 5), default=0),
        metadata={'description': """ 
        Type 1 threshold."""
    })
    param_type1_bias: Parameter = field(
        default=Parameter(enable=1, guess=0, bounds=(-1, 1), grid_range=np.linspace(-0.2, 0.2, 8), default=0),
        metadata={'description': """ 
        Type 1 bias."""
    })
    param_type1_nonlinear_gain: Parameter = field(
        default=Parameter(enable=0, guess=0, bounds=(-8 / 9, 10), grid_range=np.linspace(-0.5, 1, 5), default=0),
        metadata={'description': """ 
        Gain parameter for nonlinear encoding (higher values -> stronger nonlinearity)."""
    })
    param_type1_nonlinear_scale: Parameter = field(
        default=Parameter(enable=0, guess=1, bounds=(0.01, 10), grid_range=np.linspace(0.01, 2, 5), default=None),
        metadata={'description': """ 
        Scale parameter for the nonlinearity (higher values -> non-linearity kicks in later)."""
    })
    param_type1_noise_heteroscedastic: Parameter = field(
        default=Parameter(enable=0, guess=0, bounds=(0, 10), grid_range=np.linspace(0, 1, 5), model='multiplicative', default=0),
        metadata={'description': """ 
        Signal-dependent type 1 noise. Specify the signal dependency via the `.model` attribute of the 
        parameter. Default is `'multiplicative'`, which corresponds to Weber's law with a noise floor. In this case, 
        `type1_noise` is the noise floor and `type1_noise_heteroscedastic` is the signal scaling factor."""
    })


    ## Type 2

    param_type2_noise: Parameter = field(
        default=Parameter(enable=1, guess=0.1, bounds=(0.05, 2), grid_range=np.linspace(0.1, 1, 8), default=0.01),
        metadata={'description': """ 
        Metacognitive noise. May characterize metacognitive noise of either a noisy-readout, noisy-report or 
        noisy-temperature model."""
    })
    param_type2_evidence_bias: Parameter = field(
        default=Parameter(enable=0, guess=1, bounds=(0.5, 2), grid_range=np.linspace(0.5, 2, 8), default=1),
        metadata={'description': """ 
        Parameter for a multiplicative metacognitive bias loading on evidence."""
    })
    param_type2_confidence_bias: Parameter = field(
        default=Parameter(enable=0, guess=1, bounds=(0.5, 2), grid_range=np.linspace(0.5, 2, 8), default=1),
        metadata={'description': """ 
        Parameter for a power-law metacognitive bias loading on confidence."""
    })
    param_type2_criteria: Parameter = field(
        default=Parameter(enable=3, guess='equispaced', grid_range='equispaced', default='equispaced', bounds=(1e-8, 1)),
        metadata={'description': """ 
        Confidence criteria."""
    })


    ### Likelihood computation

    min_type1_like: float = field(default=1e-10, metadata={'description': """ 
        Minimum probability used during the type 1 likelihood computation."""
                                                           })

    min_type2_like: float = field(default=1e-10, metadata={'description': """ 
        Minimum probability used during the type 2 likelihood computation."""
                                                           })

    min_type2_like_uni: bool =  field(default=False, metadata={'description': """ 
        Instead of using a minimum probability during the likelihood computation, use a maximum cumulative
        likelihood based on a uniform 'guessing' model. `min_type2_likelihood` is ignored in this case."""
                                                               })

    type2_binsize: float = field(default=0.01, metadata={'description': """ 
        Integration bin size for the computation of the likelihood around empirical confidence values.
        A setting of 0 means that the probability density is assesed instead."""
    })

    type2_binsize_wrap: bool = field(default=False, metadata={'description': """ 
        Ensure constant window size for likelihood integration at the bounds.
        Only applies if confidence criteria are disabled and type2_binsize > 0."""
    })

    type1_marg_z: int = field(default=5, metadata={'description': """ 
        Number of standard deviations around the mean considered for the marginalization of type 1 uncertainty."""
    })

    type1_marg_steps: int = field(default=101, metadata={'description': """ 
        Number of integration steps for the marginalization of type 1 uncertainty."""
    })

    temperature_marg_res: float = field(default=0.001, metadata={'description': """ 
        Quintile resolution for the marginalization of type 1 noise in case of type2_noise_type 'temperature'."""
    })

    type1_likel_incongr: bool = field(default=False, metadata={'description': """ 
        If `True`, include incongruent decision values (i.e., sign(actual choice) != sign(decision value)) for the type 2 
        likelihood computation."""
    })


    ### Useful for testing

    true_params: dict = field(default=None, metadata={'description': """ 
        Pass true (known) parameter values. This can be useful for testing to compare the likelihood of true and
        fitted parameters. The likelihood of true parameters is returned (and printed)."""
    })

    initilialize_fitting_at_true_params: bool = field(default=False, metadata={'description': """ 
        If `True`, initialize the parameter fitting procedure at the true parameters. True parameters must 
        have been passed via `true_params`."""
    })

    accept_mispecified_model: bool = field(default=False, metadata={'description': """ 
        If `True`, ignore warnings about user-specified settings."""
    })

    print_configuration: bool = field(default=False, metadata={'description': """ 
        If True, print the configuration at instatiation of the ReMeta class (useful for logging)."""
    })


    ### Private attributes (do not change)

    _param_type1_noise: Parameter | list[Parameter] = None
    _param_type1_noise_heteroscedastic: Parameter | list[Parameter] = None
    _param_type1_nonlinear_scale: Parameter | list[Parameter] = None
    _param_type1_nonlinear_gain: Parameter | list[Parameter] = None
    _param_type1_thresh: Parameter | list[Parameter] = None
    _param_type1_bias: Parameter | list[Parameter] = None
    _param_type2_noise: Parameter = None
    _param_type2_evidence_bias: Parameter = None
    _param_type2_confidence_bias: list[Parameter] = None
    _param_type2_criteria: list[Parameter] = None

    _paramset_type1: ParameterSet = None
    _paramset_type2: ParameterSet = None
    _paramset: ParameterSet = None

    _n_conf_levels: int = None

    _optim_num_cores: int = None

    _fields: set = None


    def __post_init__(self):
        # Define allowed fields
        self._fields = {f.name for f in fields(self)}


    def __setattr__(self, key, value):
        if self._fields is not None and key not in self._fields:
            raise AttributeError(f"'{type(self).__name__}' object has no attribute '{key}'")
        # Call the base class setattr to actually set the value
        super().__setattr__(key, value)


    def setup(self, generative_mode=False, silence_warnings=False):

        if (self.optim_num_cores > 1) and find_spec('multiprocessing_on_dill') is None:
            if not silence_warnings:
                warnings.warn(f'Multiprocessing on dill is not installed. Setting optim_num_cores to 1.')
            self.optim_num_cores = 1

        if self.optim_num_cores > 1:
            from multiprocessing import cpu_count
            self._optim_num_cores = max(1, (cpu_count() or 1) - 1) if self.optim_num_cores == -1 \
                else self.optim_num_cores
        else:
            self._optim_num_cores = 1

        self._prepare_params_type1()
        if self.skip_type2:
            if self.optim_type2_slsqp_epsilon is None:
                self.optim_type2_slsqp_epsilon = 1e-5
        else:

            if self.param_type1_thresh.enable and \
                (self.optim_type1_scipy_solvers == self.__dataclass_fields__['optim_type1_scipy_solvers'].default):
                self.optim_type1_scipy_solvers = ('trust-constr', 'Powell')


            if self.param_type2_noise.model is None:
                if self.type2_noise_type == 'report':
                    if (self.param_type2_criteria.enable):
                        self.param_type2_noise.model = 'beta_mode'
                    else:
                        self.param_type2_noise.model = 'truncated_normal_mode'
                elif (self.type2_noise_type == 'readout'):
                    if self.param_type2_criteria.enable:
                        self.param_type2_noise.model = 'lognormal_mode_std'
                    else:
                        self.param_type2_noise.model = 'truncated_normal_mode'
                elif self.type2_noise_type == 'temperature':
                    if self.param_type2_criteria.enable:
                        self.param_type2_noise.model = 'lognormal_mode_std'
                    else:
                        self.param_type2_noise.model = 'truncated_normal_mode'

                # if generative_mode and not silence_warnings:
                #         warnings.warn('In generative mode, you should to explicitly specify a type 2 noise distribution. '
                #                       f'Defaulting to "{self.param_type2_noise.model}"')

            self._prepare_params_type2()
            if self.optim_type2_slsqp_epsilon is None:
                self.optim_type2_slsqp_epsilon = 1e-5

            if self.type2_binsize is None:
                self.type2_binsize = 0.01

        self._prepare_params_all()

        self._check_compatibility(generative_mode=generative_mode, silence_warnings=silence_warnings)

        if self.print_configuration:
            self.print()
        # self.setup_called = True

    def _check_compatibility(self, generative_mode=False, silence_warnings=False):

        if not self.accept_mispecified_model:

            if not self.param_type1_noise.enable:
                raise ValueError("Type 1 noise must be enabled.")

            if not self.skip_type2:

                if self.param_type2_criteria.enable and self.param_type2_criteria.group is not None:
                    if not silence_warnings:
                        warnings.warn('It is not recommended to fit criteria as a random effect or a fixed group effect, '
                                      'for conceptual reasons, but also because standard errors are not reliable.')

                if not self.param_type2_noise.enable:
                    if not silence_warnings:
                        warnings.warn(f'Setting type2_param_noise.enable=False was provided -> type2_param_noise is set to its default value '
                                      f'({self._type2_param_noise_default}). You may change this value via the configuration.')

                if (self.type2_noise_type == 'temperature') and self.param_type2_noise._definition_changed and \
                    (self.param_type2_noise.bounds[0] < 1e-5):
                    if not silence_warnings:
                        warnings.warn('You manually changed the lower bound of the type 2 noise parameter for a '
                                      'noisy-temperature model to a very low value (<1e-5). Be warned that this may result '
                                      'in numerical instabilities that severely distort the likelihood computation.')

                if not generative_mode:
                    # If the configuration instance is used for generating data, we should not complain
                    # about fitting issues.

                    if self.param_type2_criteria.enable and self.param_type2_evidence_bias.enable:
                        if not silence_warnings:
                            warnings.warn(
                                'Fitting type2_param_criteria in combination with type2_param_evidence_bias.enable=1\n'
                                'can lead to biased parameter inferences. Use with caution.')

    def _prepare_params_type1(self):
        # if self.paramset_type1 is None:

            param_names_type1 = []
            params_type1 = ('noise', 'noise_heteroscedastic', 'nonlinear_gain', 'nonlinear_scale', 'thresh', 'bias')
            for param in params_type1:
                if getattr(self, f'param_type1_{param}').enable:
                    param_names_type1 += [f'type1_{param}']
                    # if getattr(self, f'_param_type1_{param}') is None:
                    param_definition = getattr(self, f'param_type1_{param}')
                    if getattr(self, f'param_type1_{param}').enable == 2:
                        setattr(self, f'_param_type1_{param}', [param_definition, param_definition])
                    else:
                        setattr(self, f'_param_type1_{param}', param_definition)
                    if self.true_params is not None and self.initilialize_fitting_at_true_params and f'type1_{param}' in self.true_params:
                        if (param_len := getattr(self, f'param_type1_{param}').enable) > 1:
                            for i in range(param_len):
                                getattr(self, f'_param_type1_{param}')[i].guess = self.true_params[f'type1_{param}'][i]
                        else:
                            getattr(self, f'_param_type1_{param}').guess = self.true_params[f'type1_{param}']

            parameters = {k: getattr(self, f"_param_{k}") for k in param_names_type1}
            self._paramset_type1 = ParameterSet(parameters, param_names_type1)

    def _prepare_params_type2(self):

        if self.param_type2_noise.enable and self._param_type2_noise is None and not self.param_type2_noise._definition_changed:

            lb = 0.05
            self.param_type2_noise.bounds = dict(
                report = dict(
                    beta_mean_std=(lb, 0.5),
                    beta_mode_std=(lb, 1 / np.sqrt(12)),
                    truncated_normal_mode_std=(lb, 1 / np.sqrt(12)),
                    truncated_gumbel_mode_std=(lb, 1 / np.sqrt(12)),
                    truncated_lognormal_mode_std=(lb, 1 / np.sqrt(12)),
                    beta_mode=(lb, 1),
                    truncated_normal_mode=(lb, 1),
                    truncated_gumbel_mode=(lb, 1),
                    truncated_lognormal_mode=(lb, 4),
                    truncated_lognormal_mean=(lb, 4),
                    truncated_lognorm=(lb, 4)
                ),
                readout = dict(
                    lognormal_mean=(lb, 1),
                    lognormal_mode=(lb, 1),
                    gamma_mean_std=(lb, 1),
                    lognormal_mean_std=(lb, 2),
                    lognormal_mode_std=(lb, 2),
                    lognormal_median_std=(lb, 2),
                    gamma_mean_cv=(lb, 2),
                    gamma_mean=(lb, 2),
                    gamma_mode_std=(lb, 2),
                    gamma_mode=(lb, 2),
                    betaprime_mean_std=(lb, 2),
                    truncated_normal_mode_std=(lb, 2),
                    truncated_normal_mode=(lb, 2),
                    truncated_gumbel_mode_std=(lb, 2),
                    truncated_gumbel_mode=(lb, 2)
                ),
                temperature = dict(
                    lognormal_mean=(lb, 1),
                    gamma_mean_std=(lb, 1),
                    lognormal_mean_std=(lb, 2),
                    lognormal_median_std=(lb, 2),
                    gamma_mean_cv=(lb, 2),
                    gamma_mean=(lb, 2),
                    gamma_mode_std=(lb, 2),
                    gamma_mode=(lb, 2),
                    betaprime_mean_std=(lb, 2),
                    truncated_normal_mode_std=(lb, 2),
                    truncated_normal_mode=(lb, 2),
                    truncated_gumbel_mode_std=(lb, 2),
                    truncated_gumbel_mode=(lb, 2),
                    lognormal_mode=(lb, 4),
                    lognormal_mode_std=(lb, 10),
                )
            )[self.type2_noise_type][self.param_type2_noise.model]
            self.param_type2_noise.grid_range = np.exp(np.linspace(np.log(self.param_type2_noise.bounds[0]),
                                                                   np.log(self.param_type2_noise.bounds[1]), 10)[1:-1])

        param_names_type2 = []
        params_type2 = ('noise', 'evidence_bias', 'confidence_bias')
        for param in params_type2:
            if getattr(self, f'param_type2_{param}').enable:
                param_names_type2 += [f'type2_{param}']
                # if getattr(self, f'_param_type2_{param}') is None:
                param_definition = getattr(self, f'param_type2_{param}')
                setattr(self, f'_param_type2_{param}', param_definition.copy())
                if self.true_params is not None and self.initilialize_fitting_at_true_params and f'type2_{param}' in self.true_params:
                    getattr(self, f'_param_type2_{param}').guess = self.true_params[f'type2_{param}']


        if self.param_type2_criteria.preset is not None:
            self.param_type2_criteria.enable = 0
            if listlike(self.param_type2_criteria.preset):
                self._n_conf_levels = len(self.param_type2_criteria.preset) + 1
            elif isinstance(self.param_type2_criteria.preset, int):
                self._n_conf_levels = self.param_type2_criteria.preset + 1
                self.param_type2_criteria.preset = np.arange(1/self._n_conf_levels, 1-1e-10, 1/self._n_conf_levels)
            else:
                raise ValueError('param_type2_criteria.preset must either be a list of criteria or '
                                 'an integer indicating the number of (equispaced) criteria.')


        if self.param_type2_criteria.enable:
            self._n_conf_levels = self.param_type2_criteria.enable + 1
            param_names_type2 += [f'type2_criteria']
            initialize_true = (self.initilialize_fitting_at_true_params and
                               self.true_params is not None and 'type2_criteria' in self.true_params)

            # internally, we handle criteria as criterion gaps!
            setattr(self, f'_param_type2_criteria',
                    [Parameter(
                       guess=self.true_params['type2_criteria'][i] if initialize_true
                                else (1 / self._n_conf_levels if self.param_type2_criteria.guess == 'equispaced'
                                      else self.param_type2_criteria_guesses[i]),
                       bounds=self.param_type2_criteria.bounds,
                       grid_range=np.linspace(0.05, 2 / self._n_conf_levels, 4) if
                       self.param_type2_criteria.grid_range == 'equispaced' else self.param_type2_criteria_grid_ranges[i],
                       default=1/self._n_conf_levels if self.param_type2_criteria.default == 'equispaced' else self.param_type2_criteria_default,
                    )
                     for i in range(self._n_conf_levels - 1)]
                    )
            if self.true_params is not None:
                if isinstance(self.true_params, dict):
                    # if 'type2_criteria' not in self.true_params:
                    #     raise ValueError('type2_criteria are missing from cfg.true_params')
                    if 'type2_criteria' in self.true_params:
                        self.true_params.update(
                            # type2_criteria_absolute=[np.sum(self.true_params['type2_criteria'][:i+1]) for i in range(len(self.true_params['type2_criteria']))],
                            type2_criteria_bias=np.mean(self.true_params['type2_criteria']) - 0.5,
                            type2_criteria_bias_sem=0,
                            type2_criteria_confidence_bias=0.5 - np.mean(self.true_params['type2_criteria']),
                            # type2_criteria_bias_mult=np.mean(self.true_params['type2_criteria']) / 0.5,
                            # type2_criteria_confidence_bias_mult=np.mean(self.true_params['type2_criteria']) / 0.5,
                            # type2_criteria_absdev=round(np.abs(np.array(self.true_params['type2_criteria']) -
                            #         np.arange(1/self._n_conf_levels, 1-1e-10, 1/self._n_conf_levels)).mean(), 10)
                        )
                elif isinstance(self.true_params, list):
                    for s in range(len(self.true_params)):
                        # if 'type2_criteria' not in self.true_params[s]:
                        #     raise ValueError(f'type2_criteria are missing from cfg.true_params (subject {s})')
                        if 'type2_criteria' in self.true_params[s]:
                            self.true_params[s].update(
                                # type2_criteria_absolute=[np.sum(self.true_params[s]['type2_criteria'][:i+1]) for i in range(len(self.true_params[s]['type2_criteria']))],
                                type2_criteria_bias=np.mean(self.true_params[s]['type2_criteria']) - 0.5,
                                type2_criteria_bias_sem=0,
                                type2_criteria_confidence_bias=0.5 - np.mean(self.true_params[s]['type2_criteria']),
                                # type2_criteria_absdev=round(np.abs(np.array(self.true_params[s]['type2_criteria']) -
                                #     np.arange(1/self._n_conf_levels, 1-1e-10, 1/self._n_conf_levels)).mean())
                            )

        parameters = {k: getattr(self, f"_param_{k}") for k in param_names_type2}
        self._paramset_type2 = ParameterSet(parameters, param_names_type2)


        self.check_type2_constraints()


    def _prepare_params_all(self):

        if self.skip_type2:
            self._paramset = self._paramset_type1
        else:
            parameters_all = {**self._paramset_type1.parameters, **self._paramset_type2.parameters}
            param_names_all = self._paramset_type1.param_names + self._paramset_type2.param_names
            self._paramset = ParameterSet(parameters_all, param_names_all)
            # for k, attr in self.paramset_type2.__dict__.items():
            #     attr_old = getattr(self.paramset, k)
            #     if isinstance(attr, list):
            #         attr_new = attr_old + attr
            #     elif isinstance(attr, dict):
            #         attr_new = {**attr_old, **attr}
            #     elif isinstance(attr, np.ndarray):
            #         if attr.ndim == 1:
            #             attr_new = np.hstack((attr_old, attr))
            #         else:
            #             attr_new = np.vstack((attr_old, attr))
            #     elif isinstance(attr, int):
            #         attr_new = attr_old + attr
            #     elif attr is None:
            #         if attr_old is None:
            #             attr_new = None
            #         else:
            #             raise ValueError(f'Type 2 attribute is None, but type 1 attribute is not.')
            #     else:
            #         raise ValueError(f'Unexpected type {type(attr)}')
            #     setattr(self.paramset, k, attr_new)




    def print(self):
        # print('***********************')
        print(f'{self.__class__.__name__}')
        for k, v in self.__dict__.items():
            # if not self.skip_type2 or ('type2' not in k):
            print('\n'.join([f'\t{k}: {v}']))
        # print('***********************')

    def __repr__(self):
        txt = f'{self.__class__.__name__}\n'
        txt += '\n'.join([f'\t{k}: {v}' for k, v in self.__dict__.items()])
        return txt

    def check_type2_constraints(self):
        pass