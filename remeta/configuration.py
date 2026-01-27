import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
from importlib.util import find_spec

from .modelspec import Parameter, ParameterSet
from .util import ReprMixin, reset_dataclass_on_init


@reset_dataclass_on_init
@dataclass
class Configuration(ReprMixin):
    """
    Configuration for the ReMeta toolbox

    Parameters
    ----------
    *** Basic definition of the model ***
    type2_fitting_type : str (default: 'criteria')
        Whether confidence is fitted with discrete *criteria* or as a continuous variable.
        Possible values: 'criteria', 'continuous'
    type2_noise_type : str (default: 'noisy-report)
        Whether the model considers noise at readout or report.
        Possible values: 'noisy_report', 'noisy_readout', 'noisy_temperature'
    type2_noise_dist : str
            (default: noisy-report + criteria -> 'truncated_norm_mode'
                      noisy-report + continuous -> 'truncated_norm_mode'
                      noisy-readout + criteria -> 'truncated_norm_mode'
                      noisy-readout + continuous -> 'truncated_norm_mode'
                      noisy-temperature + criteria -> 'lognorm_mode'
                      noisy-temperature + continuous -> 'truncated_norm_mode'
            )
        Metacognitive noise distribution.
        Possible values:
            noisy_report: 'beta_mean_std', 'beta_mode_std', 'beta_mode',
                          'truncated_norm_mode_std', 'truncated_norm_mode' (default),
                          'truncated_gumbel_mode_std', 'truncated_gumbel_mode',
                          'truncated_lognorm_mode_std', 'truncated_lognorm', 'truncated_lognorm_mode',
                          'truncated_lognorm_mean'
            noisy_readout: 'lognorm_median_std', 'lognorm_mean', 'lognorm_mode', 'lognorm_mode_std', 'lognorm_mean_std',
                           'gamma_mode_std', 'gamma_mean_std', 'gamma_mean', 'gamma_mode', 'gamma_mean_cv',
                           'betaprime_mean_std',
                           'truncated_norm_mode_std', 'truncated_norm_mode',
                           'truncated_gumbel_mode_std', 'truncated_gumbel_mode'
            noisy_temperature: same as noisy_readout


    *** Enable or disable specific parameters ***
    * Each setting can take the values 0, 1 or 2:
    *    0: Disable parameter.
    *    1: Enable parameter.
    *    2: Enable parameter and fit separate values for the negative and positive stimulus category
            (works only for type 1 parameters!)
    enable_type1_param_noise : int (default: 1)
        Fit separate type 1 noise parameters for both stimulus categories.
    enable_type1_param_noise_heteroscedastic : int (default: 0)
        Fit an additional type 1 noise parameter for signal-dependent type 1 noise (the type of dependency is
        defined via `type1_noise_signal_dependency`).
    enable_type1_param_nonlinear_encoding_gain : int (default: 0)
    enable_type1_param_nonlinear_encoding_transition : int (default: 0)
    enable_type1_param_thresh : int (default: 0)
        Fit a type 1 threshold.
    enable_type1_param_bias : int (default: 1)
        Fit a type 1 bias towards one of the stimulus categories.
    enable_type2_param_noise : int (default: 1)
        Fit a metacognitive noise parameter
    enable_type2_param_evidence_bias_mult : int (default: 0)
        Fit a multiplicative metacognitive bias loading on evidence.
    enable_type2_param_criteria : int (default: 0)
        Fit confidence criteria.

    *** Additional options to specify the nature of type 2 fitting ***
    n_discrete_confidence_levels : int (default: 5)
        Number of confidence criteria. Only applies in case of type2_fitting_type='criteria'.

    *** Define fitting characteristics of the parameters ***
    * The fitting of each parameter is characzerized as follows:
    *     1) An initial guess.
    *     2) Lower and upper bound.
    *     3) Grid range, i.e. list of values that are tested during the initial gridsearch search.
    * Sensible default values are provided for all parameters. To tweak those, one can either define an entire
    * ParameterSet, which is a container for a set of parameters, or each parameter individually. Note that the
    * parameters must be either defined as a Parameter instance or as List[Parameter] in case when separate values are
    * fitted for the positive and negative stimulus category/decision value).
    paramset_type1 : ParameterSet
        Parameter set for the type 1 stage.
    paramset_type2 : ParameterSet
        Parameter set for the type 2 stage.
    paramset : ParameterSet
        Parameter set for both stages.

    _type1_param_noise : Union[Parameter, List[Parameter]]  (default: 1)
        Parameter for type 1 noise.
    _type1_param_noise_heteroscedastic : Union[Parameter, List[Parameter]]  (default: 0)
        Parameter for signal-dependent type 1 noise.
    _type1_param_nonlinear_encoding_gain : Union[Parameter, List[Parameter]]  (default: 0)
        Gain parameter for nonlinear encoding (higher values -> stronger nonlinearity).
    _type1_param_nonlinear_encoding_transition : Union[Parameter, List[Parameter]]  (default: 0)
        Transition Parameter for nonlinear encoding ().
    type1_noise_signal_dependency: str (default: 'none')
        Can be one of 'none', 'multiplicative', 'power', 'exponential' or 'logarithm'.
    _type1_param_thresh : Union[Parameter, List[Parameter]] (default: 0)
        Parameter for the type 1 threshold.
    _type1_param_bias : Union[Parameter, List[Parameter]]  (default: 1)
        Parameter for the type 1 bias.
    _type2_param_noise : Union[Parameter, List[Parameter]]  (default: 1)
        Parameter for metacognitive noise.
    _type2_param_evidence_bias_mult : Union[Parameter, List[Parameter]]  (default: 0)
        Parameter for a multiplicative metacognitive bias loading on evidence.
    type2_param_confidence_criteria : List[Parameter]  (default: 1)
        List of parameter specifying the confidence criteria.

    *** Skip type 2 fitting ***
    skip_type2 : bool (default: False)
        If True, ignore type 2 settings in the setup of the model configuration & don't fit type 2 stage.

    *** Methodoligcal aspects of parameter fitting ***
    optim_type1_gridsearch : bool (default: False)
        If True, perform initial (usually coarse) gridsearch search for type 1 fitting, based on the gridsearch defined
        for a Parameter.
    optim_type1_fine_gridsearch : bool (default: False)
        If True, perform an iteratively finer gridsearch search for each parameter (type 1).
    optim_type1_minimize_along_grid : bool (default: False)
        If True, do sqlqp minimization for at each grid point (type 1).
    optim_type1_global_minimization : str (default: None)
        Use one of 'shgo', 'dual_annealing' 'differential_evolution' to start likelihood minimization with
        a global minimizer (type 1).
    optim_type1_scipy_solvers : str or Tuple/List (default: 'trust-constr')
        Set scipy.optimize.minimize gradient method (type 1)
        If provided as Tuple/List, test different gradient methods and take the best
    optim_type2_gridsearch : bool (default: True)
        If True, perform initial (usually coarse) gridsearch search for type 2 fitting, based on the gridsearch defined
        for a Parameter.
    optim_type2_fine_gridsearch : bool (default: False)
        If True, perform an iteratively finer gridsearch search for each parameter (type 2).
    optim_type2_minimize_along_grid : bool (default: False)
        If True, do sqlqp minimization for at each grid point (type 2).
    optim_type2_global_minimization : str (default: None)
        Use one of 'shgo', 'dual_annealing' 'differential_evolution' to start likelihood minimization with
        a global minimizer (type 2).
    optim_type2_scipy_solvers : str or Tuple/List (default: ('slsqp', 'Nelder-Mead'))
        Set scipy.optimize.minimize gradient method (type 2)
        If provided as Tuple/List, test different gradient methods and take the best
    optim_type2_slsqp_epsilon : float or Tuple/List (default: None)
        Set parameter epsilon parameter for the SLSQP optimization method (type 2).
        If provided as Tuple/List, test different eps parameters and take the best
    optim_grid_multiproc : bool (default: False)
        If True, use all available cores for the gridsearch search. If False, use a single core.

    *** Preprocessing ***
    normalize_stimuli_by_max : bool (default: True)
        If True, normalize provided stimuli by their maximum value.

    *** Parameters for the type 2 likelihood computation ***
    min_type1_likelihood : float
        Minimum probability used during the type 1 likelihood computation
    min_type2_likelihood : float
        Minimum probability used during the type 2 likelihood computation
    type2_binsize : float
        Integration bin size for the computation of the likelihood around empirical confidence values
    y_decval_range_nsds : int
        Number of standard deviations around the mean considered for type 1 uncertainty.
    y_decval_range_nbins : int
        Number of discrete decision values bins that are considered to represent type 1 uncertainty.
    resolution_noisy_temperature : float
        Quintile resolution for the marginalization of type 1 noise in case of type2_noise_type 'noisy_temperature'.
    experimental_min_uniform_type2_likelihood : bool
        Instead of using a minimum probability during the likelihood computation, use a maximum cumulative
        likelihood based on a 'guessing' model
    experimental_wrap_type2_integration_window : bool (default: False)
        Ensure constant window size for likelihood integration at the bounds.
        Only applies in case of type2_fitting_type='continuous' and experimental_disable_type2_binsize=False
    experimental_include_incongruent_y_decval : bool (default: False)
        Include incongruent decision values (i.e., sign(actual choice) != sign(decision value)) for the likelihood
        computation
    experimental_disable_type2_binsize : bool (default: None)
        Do not use an integegration window for likelihood computation.
        Only applies in case of type2_fitting_type='continuous'


    *** Other ***
    true_params : Dict
        Pass true (known) parameter values. This can be useful for testing to compare the likelihood of true and
        fitted parameters. The likelihood of true parameters is returned (and printed).
    initilialize_fitting_at_true_params : bool (default: False)
        Option to initialize the parameter fitting procedure at the true parameters; this can be helpful for testing.
    silence_configuration_warnings : bool (default: False)
        If True, ignore warnings about user-specified settings.
    print_configuration : bool (default: True)
        If True, print the configuration at instatiation of the ReMeta class.
    """

    type2_fitting_type: str = 'criteria'
    type2_noise_type: str = 'noisy_report'
    type2_noise_dist: str = None
        # noisy-report + criteria -> 'truncated_norm_mode'
        # noisy-report + continuous -> 'truncated_norm_mode'
        # noisy-readout + criteria -> 'truncated_norm_mode'
        # noisy-readout + continuous -> 'truncated_norm_mode'
        # noisy-temperature + criteria -> 'lognorm_mode'
        # noisy-temperature + continuous -> 'truncated_norm_mode'

    enable_type1_param_noise: int = 1
    enable_type1_param_thresh: int = 0
    enable_type1_param_bias: int = 1
    enable_type2_param_noise: int = 1
    enable_type2_param_evidence_bias_mult: int = 0
    enable_type2_param_criteria: int = 1
    # Experimental:
    enable_type1_param_noise_heteroscedastic: int = 0
    enable_type1_param_nonlinear_encoding_gain: int = 0
    enable_type1_param_nonlinear_encoding_transition: int = 0

    n_discrete_confidence_levels: int = 5

    paramset_type1: ParameterSet = None
    paramset_type2: ParameterSet = None
    paramset_all: ParameterSet = None

    type1_param_noise_heteroscedastic: Parameter = Parameter(guess=0, bounds=(0, 10), grid_range=np.linspace(0, 1, 5))
    type1_param_nonlinear_encoding_gain: Parameter = Parameter(guess=0, bounds=(-8/9, 10), grid_range=np.linspace(-0.5, 1, 5))
    type1_param_nonlinear_encoding_transition: Parameter = Parameter(guess=1, bounds=(0.01, 10), grid_range=np.linspace(0.01, 2, 5))
    type1_param_noise: Parameter = Parameter(guess=0.5, bounds=(0.001, 100), grid_range=np.linspace(0.1, 1, 8))
    type1_param_thresh: Parameter = Parameter(guess=0, bounds=(0, 1), grid_range=np.linspace(0, 0.2, 5))
    type1_param_bias: Parameter = Parameter(guess=0, bounds=(-1, 1), grid_range=np.linspace(-0.2, 0.2, 8))
    type2_param_noise: Parameter = Parameter(guess=0.1, bounds=(0.05, 2), grid_range=np.linspace(0.1, 1, 8))
    type2_param_evidence_bias_mult: Parameter = Parameter(guess=1, bounds=(0.5, 2), grid_range=np.linspace(0.5, 2, 8))
    type2_param_criteria: Parameter = Parameter(bounds=(1e-8, 1))
    type2_param_criteria_guesses: str | List[float] = 'equidistant'
    type2_param_criteria_grid_ranges: str | List[np.ndarray] = 'equidistant'

    type1_noise_signal_dependency: str = 'none'

    skip_type2 = False

    optim_type1_gridsearch: bool = False
    optim_type1_fine_gridsearch: bool = False
    optim_type1_minimize_along_grid: bool = False
    optim_type1_global_minimization: str = None
    _optim_type1_scipy_solvers_default = 'trust-constr'
    optim_type1_scipy_solvers: str | List[str] | Tuple[str, ...] = 'trust-constr'
    optim_type2_gridsearch: bool = True
    optim_type2_fine_gridsearch: bool = False
    optim_type2_minimize_along_grid: bool = False
    optim_type2_global_minimization: str = None
    optim_type2_scipy_solvers: str | List[str] | Tuple[str, ...] = ('slsqp', 'Nelder-Mead')
    optim_type2_slsqp_epsilon: float = None
    optim_grid_multiproc: bool = False

    normalize_stimuli_by_max: bool = True
    confidence_bounds_error: float = 0

    min_type2_likelihood: float = 1e-10
    min_type1_likelihood: float = 1e-10
    type2_binsize: float = 0.01
    y_decval_range_nsds: int = 5
    y_decval_range_nbins: int = 101
    resolution_noisy_temperature: float = 0.001

    experimental_min_uniform_type2_likelihood: bool = False
    experimental_wrap_type2_integration_window: bool = False
    experimental_include_incongruent_y_decval: bool = False
    experimental_disable_type2_binsize: bool = False

    true_params: Dict = None
    initilialize_fitting_at_true_params: bool = False
    silence_configuration_warnings: bool = False
    print_configuration: bool = False

    type2_param_noise_min: float = 0.001

    # setup_called = False

    _type1_param_noise: Parameter | List[Parameter] = None
    _type1_param_noise_heteroscedastic: Parameter | List[Parameter] = None
    _type1_param_nonlinear_encoding_transition: Parameter | List[Parameter] = None
    _type1_param_nonlinear_encoding_gain: Parameter | List[Parameter] = None
    _type1_param_thresh: Parameter | List[Parameter] = None
    _type1_param_bias: Parameter | List[Parameter] = None
    _type2_param_noise: Parameter = None
    _type2_param_evidence_bias_mult: Parameter = None
    _type2_param_criteria: List[Parameter] = None

    def setup(self, generative_mode=False):

        if find_spec('multiprocessing_on_dill') is None:
            warnings.warn(f'Multiprocessing on dill is not installed. Setting grid_multiproc is changed to False.')
            self.optim_grid_multiproc = False

        self._prepare_params_type1()
        if self.skip_type2:
            if self.optim_type2_slsqp_epsilon is None:
                self.optim_type2_slsqp_epsilon = 1e-5
        else:

            if self.enable_type1_param_thresh and \
                (self.optim_type1_scipy_solvers == self._optim_type1_scipy_solvers_default):
                self.optim_type1_scipy_solvers = ('trust-constr', 'Powell')


            if self.type2_noise_dist is None:
                if generative_mode:
                    raise ValueError('In generative mode, you need to explicitly specify a type 2 noise distribution.')
                else:
                    if self.type2_noise_type == 'noisy_report':
                        if (self.type2_fitting_type == 'criteria'):
                            self.type2_noise_dist = 'truncated_norm_mode'
                        else:
                            self.type2_noise_dist = 'truncated_norm_mode'
                    elif (self.type2_noise_type == 'noisy_readout'):
                        if self.type2_fitting_type == 'criteria':
                            self.type2_noise_dist = 'truncated_norm_mode'
                        else:
                            self.type2_noise_dist = 'truncated_norm_mode'
                    elif self.type2_noise_type == 'noisy_temperature':
                        if self.type2_fitting_type == 'criteria':
                            self.type2_noise_dist = 'lognorm_mode'
                        else:
                            self.type2_noise_dist = 'truncated_norm_mode'

            self._prepare_params_type2()
            if self.optim_type2_slsqp_epsilon is None:
                self.optim_type2_slsqp_epsilon = 1e-5

            if self.type2_binsize is None:
                self.type2_binsize = 0.01

        self._prepare_params_all()

        self._check_compatibility(generative_mode=generative_mode)

        if self.print_configuration:
            self.print()
        # self.setup_called = True

    def _check_compatibility(self, generative_mode=False):

        if not self.silence_configuration_warnings:

            if not self.skip_type2:
                if not self.enable_type2_param_noise:
                    warnings.warn(f'Setting enable_type2_param_noise=False was provided -> type2_param_noise is set to its default value '
                                  f'({self._type2_param_noise_default}). You may change this value via the configuration.')

                if (self.type2_noise_type == 'noisy_temperature') and self.type2_param_noise.default_changed and \
                    (self.type2_param_noise.bounds[0] < 1e-5):
                    warnings.warn('You manually changed the lower bound of the type 2 noise parameter for a '
                                  'noisy-temperature model to a very low value (<1e-5). Be warned that this may result '
                                  'in numerical instabilities that severely distort the likelihood computation.')

                if not generative_mode:
                    # If the configuration instance is used for generating data, we should not complain
                    # about fitting issues.

                    if self.enable_type2_param_criteria and self.enable_type2_param_evidence_bias_mult:
                        warnings.warn(
                            'enable_type2_param_criteria=True in combination with enable_type2_param_evidence_bias_mult=True\n'
                            'can lead to biased parameter inferences. Use with caution.')

                    if (self.type2_fitting_type == 'continuous') and self.enable_type2_param_criteria:
                        raise ValueError("Setting type2_fitting_type='continuous' conflicts with enable_type2_param_criteria=1.'")

                    if (self.type2_fitting_type == 'criteria') and not self.enable_type2_param_criteria:
                        warnings.warn("You selected type2_fitting_type='criteria', but did not enable type 2 criteria\n"
                                      "(enable_type2_param_criteria=0). This works, but be mindful that the model\n"
                                      "will assume equispaced ideal Bayesian observer criteria (respecting \n"
                                      "the setting n_discrete_confidence_levels).")

    def _prepare_params_type1(self):
        # if self.paramset_type1 is None:

            param_names_type1 = []
            params_type1 = ('noise', 'noise_heteroscedastic', 'nonlinear_encoding_gain', 'nonlinear_encoding_transition', 'thresh', 'bias')
            for param in params_type1:
                if getattr(self, f'enable_type1_param_{param}'):
                    param_names_type1 += [f'type1_{param}']
                    if getattr(self, f'_type1_param_{param}') is None:
                        param_definition = getattr(self, f'type1_param_{param}')
                        if getattr(self, f'enable_type1_param_{param}') == 2:
                            setattr(self, f'_type1_param_{param}', [param_definition, param_definition])
                        else:
                            setattr(self, f'_type1_param_{param}', param_definition)
                        if self.true_params is not None and self.initilialize_fitting_at_true_params and f'type1_{param}' in self.true_params:
                            getattr(self, f'_type1_param_{param}').guess = self.true_params[f'type1_{param}']

            parameters = {k: getattr(self, f"_type1_param_{k.split('type1_')[1]}") for k in param_names_type1}
            self.paramset_type1 = ParameterSet(parameters, param_names_type1)

    def _prepare_params_type2(self):

        # if self.paramset_type2 is None:

            if self.enable_type2_param_noise and self._type2_param_noise is None and not self.type2_param_noise.default_changed:

                lb = 0.05
                self.type2_param_noise.bounds = dict(
                    noisy_report = dict(
                        beta_mean_std=(lb, 0.5),
                        beta_mode_std=(lb, 1 / np.sqrt(12)),
                        truncated_norm_mode_std=(lb, 1 / np.sqrt(12)),
                        truncated_gumbel_mode_std=(lb, 1 / np.sqrt(12)),
                        truncated_lognorm_mode_std=(lb, 1 / np.sqrt(12)),
                        beta_mode=(lb, 1),
                        truncated_norm_mode=(lb, 1),
                        truncated_gumbel_mode=(lb, 1),
                        truncated_lognorm_mode=(lb, 4),
                        truncated_lognorm_mean=(lb, 4),
                        truncated_lognorm=(lb, 4)
                    ),
                    noisy_readout = dict(
                        lognorm_mean=(lb, 1),
                        lognorm_mode=(lb, 1),
                        gamma_mean_std=(lb, 1),
                        lognorm_mean_std=(lb, 2),
                        lognorm_mode_std=(lb, 2),
                        lognorm_median_std=(lb, 2),
                        gamma_mean_cv=(lb, 2),
                        gamma_mean=(lb, 2),
                        gamma_mode_std=(lb, 2),
                        gamma_mode=(lb, 2),
                        betaprime_mean_std=(lb, 2),
                        truncated_norm_mode_std=(lb, 2),
                        truncated_norm_mode=(lb, 2),
                        truncated_gumbel_mode_std=(lb, 2),
                        truncated_gumbel_mode=(lb, 2)
                    ),
                    noisy_temperature = dict(
                        lognorm_mean=(lb, 1),
                        gamma_mean_std=(lb, 1),
                        lognorm_mean_std=(lb, 2),
                        lognorm_median_std=(lb, 2),
                        gamma_mean_cv=(lb, 2),
                        gamma_mean=(lb, 2),
                        gamma_mode_std=(lb, 2),
                        gamma_mode=(lb, 2),
                        betaprime_mean_std=(lb, 2),
                        truncated_norm_mode_std=(lb, 2),
                        truncated_norm_mode=(lb, 2),
                        truncated_gumbel_mode_std=(lb, 2),
                        truncated_gumbel_mode=(lb, 2),
                        lognorm_mode=(lb, 4),
                        lognorm_mode_std=(lb, 10),
                    )
                )[self.type2_noise_type][self.type2_noise_dist]
                self.type2_param_noise.grid_range = np.exp(np.linspace(np.log(self.type2_param_noise.bounds[0]),
                                                                       np.log(self.type2_param_noise.bounds[1]), 10)[1:-1])

            param_names_type2 = []
            params_type2 = ('noise', 'evidence_bias_mult')
            for param in params_type2:
                if getattr(self, f'enable_type2_param_{param}'):
                    param_names_type2 += [f'type2_{param}']
                    if getattr(self, f'_type2_param_{param}') is None:
                        param_definition = getattr(self, f'type2_param_{param}')
                        setattr(self, f'_type2_param_{param}', param_definition.copy())
                        if self.true_params is not None and self.initilialize_fitting_at_true_params and f'type2_{param}' in self.true_params:
                            getattr(self, f'_type2_param_{param}').guess = self.true_params[f'type2_{param}']


            if self.enable_type2_param_criteria:
                param_names_type2 += [f'type2_criteria']
                initialize_true = (self.initilialize_fitting_at_true_params and
                                   self.true_params is not None and 'type2_criteria' in self.true_params)
                setattr(self, f'_type2_param_criteria',
                        [Parameter(
                           guess=self.true_params['type2_criteria'][i] if initialize_true
                                    else (1 / self.n_discrete_confidence_levels if self.type2_param_criteria_guesses == 'equidistant'
                                          else self.type2_param_criteria_guesses[i]),
                           bounds=self.type2_param_criteria.bounds,
                           grid_range=np.linspace(0.05, 2 / self.n_discrete_confidence_levels, 4) if
                                self.type2_param_criteria_grid_ranges == 'equidistant' else self.type2_param_criteria_grid_ranges[i]
                        )
                         for i in range(self.n_discrete_confidence_levels - 1)]
                        )
                if self.true_params is not None:
                    if isinstance(self.true_params, dict):
                        # if 'type2_criteria' not in self.true_params:
                        #     raise ValueError('type2_criteria are missing from cfg.true_params')
                        if 'type2_criteria' in self.true_params:
                            self.true_params.update(
                                type2_criteria_absolute=[np.sum(self.true_params['type2_criteria'][:i+1]) for i in range(len(self.true_params['type2_criteria']))],
                                type2_criteria_bias=np.mean(self.true_params['type2_criteria'])*(len(self.true_params['type2_criteria'])+1)-1
                            )
                    elif isinstance(self.true_params, list):
                        for s in range(len(self.true_params)):
                            # if 'type2_criteria' not in self.true_params[s]:
                            #     raise ValueError(f'type2_criteria are missing from cfg.true_params (subject {s})')
                            if 'type2_criteria' in self.true_params[s]:
                                self.true_params[s].update(
                                    type2_criteria_absolute=[np.sum(self.true_params[s]['type2_criteria'][:i+1]) for i in range(len(self.true_params[s]['type2_criteria']))],
                                    type2_criteria_bias=np.mean(self.true_params[s]['type2_criteria'])*(len(self.true_params[s]['type2_criteria'])+1)-1
                                )

            parameters = {k: getattr(self, f"_type2_param_{k.split('type2_')[1]}") for k in param_names_type2}
            self.paramset_type2 = ParameterSet(parameters, param_names_type2)


            self.check_type2_constraints()


    def _prepare_params_all(self):

        if self.skip_type2:
            self.paramset = self.paramset_type1
        else:
            parameters_all = {**self.paramset_type1.parameters, **self.paramset_type2.parameters}
            param_names_all = self.paramset_type1.param_names + self.paramset_type2.param_names
            self.paramset = ParameterSet(parameters_all, param_names_all)
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
        # if self.enable_type2_param_criteria:
        #     from scipy.optimize import NonlinearConstraint
        #
        #     def crit_order_fun_ineq(theta):
        #         crit = theta[-self.n_discrete_confidence_levels + 1:]
        #         return np.sum([-int(crit[i] <= (-1e-8 if i == 0 else crit[i - 1])) for i in range(len(crit))])
        #
        #     def crit_order_fun(theta):
        #         return np.diff(theta[-self.n_discrete_confidence_levels + 1:])  # [k2 - k1, k3 - k2, ...]
        #
        #     eps = 1e-4  # minimum spacing between criteria
        #     self.paramset_type2.constraints = [dict(
        #         type='ineq',
        #         fun=crit_order_fun_ineq,
        #         constraint=NonlinearConstraint(
        #             fun=crit_order_fun,
        #             lb=np.full(self.n_discrete_confidence_levels - 2, eps),  # diff >= eps
        #             ub=np.full(self.n_discrete_confidence_levels - 2, np.inf)
        #         )
        #     )]
