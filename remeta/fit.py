import timeit
import warnings
from functools import partial
from itertools import product

import numpy as np
try:  # only necessary if multiple cores should be used
    from multiprocessing_on_dill.pool import Pool as DillPool
except ModuleNotFoundError:
    pass
from datetime import datetime
import scipy.optimize as sciopt
from scipy.optimize._constraints import old_bound_to_new  # noqa

from .util import TAB, SP2
from .util import _slsqp_epsilon

def loop(fun, params, args, gridsize, minimize_along_grid, bounds, verbosity, param_id):
    # t0 = timeit.default_timer()
    if param_id and (verbosity > 1) and (np.mod(param_id, 1000) == 0):
        print(f'{TAB}{TAB}Grid iteration {param_id} / {gridsize}')
    negll = fun(params[param_id], *args)

    if minimize_along_grid:
        fit = sciopt.minimize(fun, params[param_id], bounds=bounds, args=tuple(args), method='slsqp')
        x = fit.x
        negll = fit.fun
    else:
        x = params[param_id]

    # print(timeit.default_timer() - t0)
    return negll, x


def fgrid(fun, valid, args, multiproc=False, multiproc_cores=1, minimize_along_grid=False, bounds=None, verbosity=1):
    if multiproc:
        with DillPool(multiproc_cores) as pool:
            result = pool.map(partial(loop, fun, valid, args, len(valid), minimize_along_grid, bounds, verbosity), range(len(valid)))
            negll_grid, x_grid = [res[0] for res in result], [res[1] for res in result]
    else:
        negll_grid = [None] * len(valid)
        x_grid = [None] * len(valid)
        for i, param in enumerate(valid):
            negll_grid[i], x_grid[i] = loop(fun, valid, args, len(valid), minimize_along_grid, bounds, verbosity, i)
    return negll_grid, x_grid


def fine_grid_search(x0, fun, args, param_set, ll_grid, valid, gridsearch_resolution, n_grid_iter, multiproc=False,
                     multiproc_cores=1, verbosity=1):
    n_grid_candidates_init = 3 * gridsearch_resolution
    gx0 = x0
    gll_min_grid = np.min(ll_grid)
    grid_candidates = [valid[i] for i in np.argsort(ll_grid)[:n_grid_candidates_init]]
    previous_grid_range = [param_set.grid_range for _ in range(n_grid_candidates_init)]
    candidate_ids = list(range(n_grid_candidates_init))
    counter = 0
    for i in range(n_grid_iter):
        if (verbosity > 1) and (np.mod(i, 10) == 0):
            print(f'{TAB}Grid iteration {i + 1} / {n_grid_iter}')
        gvalid = []
        valid_candidate_ids = []
        grid_range = [None] * len(grid_candidates) * 2
        for j, grid_candidate in enumerate(grid_candidates):
            grid_range[j] = [None] * param_set.nparams_flat
            for k, p in enumerate(grid_candidate):
                ind = np.where(previous_grid_range[candidate_ids[j]][k] == p)[0][0]
                lb = previous_grid_range[candidate_ids[j]][k][max(0, ind - 1)]  # noqa
                ub = previous_grid_range[candidate_ids[j]][k][  # noqa
                    min(len(previous_grid_range[candidate_ids[j]][k]) - 1, ind + 1)]  # noqa
                grid_range[j][k] = np.around(np.linspace(lb, ub, 5), decimals=10)  # noqa
            if param_set.constraints is None:
                gvalid_candidate = list(product(*grid_range[j]))
            else:
                gvalid_candidate = [p for p in product(*grid_range[j]) if  # noqa
                                    np.all([con['fun'](p) >= 0 for con in param_set.constraints])]
            gvalid += gvalid_candidate
            valid_candidate_ids += [j] * len(gvalid_candidate)
            if verbosity > 1:
                print(f'{TAB}{TAB}Candidate {j + 1}: {[(p[0], p[-1]) for p in grid_range[j]]}')  # noqa
        gll_grid = fgrid(fun, gvalid, args, multiproc, multiproc_cores, verbosity=0)[0]
        counter += len(gvalid)

        min_id = np.argmin(gll_grid)
        if gll_grid[min_id] < gll_min_grid:
            gll_min_grid = gll_grid[min_id]
            gx0 = gvalid[min_id]

        if verbosity:
            print(f'{TAB}{TAB}Best fit: {gx0} (LL={gll_min_grid})')

        if i != n_grid_iter - 1:
            grid_candidates, candidate_ids = [], []
            for j in np.argsort(gll_grid):
                if gvalid[j] not in grid_candidates:
                    grid_candidates += [gvalid[j]]
                    candidate_ids += [valid_candidate_ids[j]]
                    if len(candidate_ids) == gridsearch_resolution:
                        break
            previous_grid_range = grid_range
    fit = sciopt.OptimizeResult(success=True, x=gx0, fun=gll_min_grid, nfev=counter)

    return fit


def subject_estimation(fun, param_set, args=(), gridsearch=False, multiproc=True, multiproc_cores=1,
                       minimize_along_grid=False, global_minimization=None, fine_gridsearch=False, verbosity=1,
                       n_grid_candidates=10, n_grid_iter=3, scipy_solvers='slsqp', slsqp_epsilon=_slsqp_epsilon,
                       guess=None):

    t0 = timeit.default_timer()

    # if verbosity:
    #     negll_initial_guess = fun(param_set.guess, *args)
    #     print(f'{TAB}{TAB}Initial guess (neg. LL: {negll_initial_guess:.2f})')
    #     for i, p in enumerate(param_set.param_names_flat):
    #         if p.startswith('type2_criteria') and not p.endswith('0'):
    #             criterion_id = int(p.split('_')[-1])
    #             criterion = param_set.guess[i-criterion_id:i+1].sum()
    #             print(f'{TAB}{TAB}{TAB}[guess] {p}: {param_set.guess[i]:.4g} = gap | criterion = {criterion:.4g}')
    #         else:
    #             print(f'{TAB}{TAB}{TAB}[guess] {p}: {param_set.guess[i]:.4g}')

    bounds = sciopt.Bounds(*old_bound_to_new(param_set.bounds), keep_feasible=True)

    if guess is None:
        x0_guess = param_set.guess
    else:
        x0_guess = guess
    fit_guess = sciopt.OptimizeResult(success=True, x=x0_guess, fun=fun(x0_guess, *args), nfev=1)
    if gridsearch:
        if param_set.constraints is not None and len(param_set.constraints):
            valid = [p for p in product(*param_set.grid_range) if
                     np.all([con['fun'](p) >= 0 for con in param_set.constraints])]
        else:
            valid = list(product(*param_set.grid_range))
        if verbosity > 1:
            print(f"{TAB}{TAB}Grid search activated (grid size = {len(valid)})")
        t0 = timeit.default_timer()
        ll_grid, x_grid = fgrid(fun, valid, args, multiproc, multiproc_cores, minimize_along_grid, bounds, verbosity=verbosity)
        x0_grid = x_grid[np.argmin(ll_grid)]
        ll_min_grid = np.min(ll_grid)
        grid_time = timeit.default_timer() - t0
        if verbosity > 1:
            for i, p in enumerate(param_set.param_names_flat):
                if p.startswith('type2_criteria') and not p.endswith('0'):
                    criterion_id = int(p.split('_')[-1])
                    criterion = param_set.guess[i-criterion_id:i+1].sum()
                    print(f'{TAB}{TAB}{TAB}[grid] {p}: {param_set.guess[i]:.4g} = gap | criterion = {criterion:.4g}')
                else:
                    print(f'{TAB}{TAB}{TAB}[grid] {p}: {x0_grid[i]:.4g}')
            print(f"{TAB}{TAB}Grid neg. LL: {ll_min_grid:.1f}")
            print(f"{TAB}{TAB}Grid runtime: {grid_time:.2f} secs")
        fit_grid = sciopt.OptimizeResult(success=True, x=x0_grid, fun=ll_min_grid, nfev=len(valid))
        fit_best = fit_grid if ll_min_grid < fit_guess.fun else fit_guess
    else:
        fit_best = fit_guess
        x0_grid = None

    if fine_gridsearch:
        fit_fine = fine_grid_search(x0_grid if gridsearch else x0, fun, args, param_set, ll_grid, valid, n_grid_candidates,
                                    n_grid_iter, multiproc, multiproc_cores, verbosity=verbosity).x

    if global_minimization is not None:
        # if verbosity:
        #     print(f'{TAB}{TAB}Performing MLE (global minimization)')
        t_global = timeit.default_timer()
        # fit_global = basinhopping(fun, fit_best.x, take_step=RandomDisplacementBoundsConstraints(bounds, param_set.constraints),
        #                           accept_test=BoundsConstraints(bounds, param_set.constraints),
        #                           minimizer_kwargs=dict(method='Nelder-Mead', args=tuple(args)))
        if global_minimization == 'shgo':
            fit_global = sciopt.shgo(fun, bounds=bounds, args=tuple(args), minimizer_kwargs=dict(method='slsqp', bounds=bounds),
                                     options=dict(sampling_method='sobol', maxiter=50))
        elif global_minimization == 'differential_evolution':
            try:
                fit_global = sciopt.differential_evolution(fun, bounds=bounds, args=tuple(args))
            except ValueError as e:
                fit_global = sciopt.shgo(fun, bounds=bounds, args=tuple(args), minimizer_kwargs=dict(method='slsqp', bounds=bounds),
                                     options=dict(sampling_method='sobol', maxiter=50))
                print(e)

        elif global_minimization == 'dual_annealing':
            fit_global = sciopt.dual_annealing(fun, bounds=bounds, args=tuple(args),
                                               minimizer_kwargs={"method": "slsqp", "bounds": bounds})
        else:
            raise ValueError(f"Unknown global minimization {global_minimization}. Please choose one of 'shgo', "
                             f"'dual_annealing' or 'differential_evolution'.")

        execution_time_global = timeit.default_timer() - t_global
        x0_global = fit_global.x
        if verbosity:
            print(f'{TAB}{TAB}{TAB}.. global MLE finished ({execution_time_global:.1f} secs).')
    else:
        x0_global = None
        execution_time_global = None

    # if verbosity:
    #     print(f'{TAB}{TAB}Performing MLE (local minimization)')

    t_local = timeit.default_timer()

    scipy_solvers = [scipy_solvers] if isinstance(scipy_solvers, str) else scipy_solvers

    # We start both from x0_guess and x0_grid, as x0_grid is more prone to local minima
    x0s = (x0_guess, x0_grid, x0_global)

    best_solver = f"init_{'guess' if x0_grid is None else 'grid'}" if global_minimization is None else global_minimization
    for i, x0 in enumerate(x0s):
        if x0 is not None:
            for method in scipy_solvers:
                if method == 'slsqp':
                    slsqp_epsilon_ = slsqp_epsilon if hasattr(slsqp_epsilon, '__len__') else [slsqp_epsilon]
                    for eps in slsqp_epsilon_:
                        fit = sciopt.minimize(fun, x0, bounds=bounds, args=tuple(args), constraints=param_set.constraints,
                                              method='slsqp', options=dict(eps=eps))
                        if fit.fun < fit_best.fun:
                            fit_best = fit
                            best_solver = f"slsqp_{eps:.3g}_init{('guess', 'grid', 'global')[i]}"
                else:
                    fit = sciopt.minimize(fun, x0, bounds=bounds, args=tuple(args), constraints=param_set.constraints,
                                          method=method)
                    if fit.fun < fit_best.fun:
                        fit_best = fit
                        best_solver = f"{method}_init{('guess', 'grid', 'global')[i]}"
    fit_best.execution_time = timeit.default_timer() - t0  # noqa
    fit_best.execution_time_global = execution_time_global
    fit_best.execution_time_local = timeit.default_timer() - t_local
    fit_best.best_solver = best_solver
    fit_best.x0_grid = x0_grid
    # if verbosity:
    #     print(f'{TAB}{TAB}{TAB}.. local MLE finished ({fit_best.execution_time_local:.1f} secs).')

    return fit_best


def group_estimation(fun, nsubjects, params_init, bounds, idx_fe, idx_re,
                     multiproc=True, multiproc_cores=1, max_iter=30, sigma_floor=1e-3, damping=0.5, verbosity=1):

    params_init = np.array(params_init)

    if verbosity:
        print(f'\n{SP2}Group-level optimization (MLE / MAP)')
    t0 = timeit.default_timer()

    uparams_init = transform_to_unconstrained_space(params_init, bounds)

    if len(idx_fe) > 0:
        # Fixed effects init: take mean in (constrained) original space, invert-transform to unconstrained space
        params_fe_init = np.median(params_init, axis=0)
        uparams_fe = transform_to_unconstrained_space(params_fe_init, bounds)
    else:
        uparams_fe = None

    if len(idx_re) > 0:
        # init hyperparams from initial Z
        uparams_mean = uparams_init.mean(axis=0)
        uparams_sd = uparams_init.std(axis=0) + sigma_floor

    for it in range(max_iter):

        if len(idx_fe) > 0:
            def objective_fe_packed(x_packed):
                uparams_tmp = uparams_fe.copy()
                uparams_tmp[idx_fe] = x_packed
                return objective_group_fe(uparams_tmp, fun, uparams_init, bounds, idx_fe)

            x0 = uparams_fe[idx_fe].copy()
            uparams_fe[idx_fe] = sciopt.minimize(objective_fe_packed, x0=x0, method='L-BFGS-B').x

        if len(idx_re) > 0:

            def subject_loop(s):
                return sciopt.minimize(
                    objective_empirical_bayes,
                    x0=uparams_init[s],
                    args=(s, fun, bounds, idx_re, uparams_mean, uparams_sd, idx_fe, uparams_fe),
                    method='L-BFGS-B'
                ).x
            if multiproc:
                with DillPool(multiproc_cores) as pool:
                    uparams_init = np.array(pool.map(subject_loop, range(nsubjects)))
            else:
                for s in range(nsubjects):
                    uparams_init[s] = subject_loop(s)

            # for s in range(nsubjects):
            #     res = sciopt.minimize(
            #         objective_empirical_bayes,
            #         x0=uparams_init[s],
            #         args=(s, fun, bounds, idx_re, uparams_mean, uparams_sd, idx_fe, uparams_fe),
            #         method='L-BFGS-B'
            #     )
            #     uparams_init[s] = res.x

            # --- hyperparameter update (moment matching) ---
            uparams_mean_new = uparams_mean.copy()
            uparams_sd_new = uparams_sd.copy()

            uparams_mean_new[idx_re] = uparams_init[:, idx_re].mean(axis=0)
            uparams_sd_new[idx_re] = uparams_init[:, idx_re].std(axis=0)
            uparams_sd_new[idx_re] = np.maximum(uparams_sd_new[idx_re], sigma_floor)

            # damping for stability
            uparams_mean = (1 - damping) * uparams_mean + damping * uparams_mean_new
            uparams_sd = (1 - damping) * uparams_sd + damping * uparams_sd_new

            # optional: check convergence
            # if np.max(np.abs(uparams_re_init_mean - uparams_mean_new)) < 1e-4: break

        if verbosity and (np.mod(it, 10) == 0):
            convergence_str = f' (Convergence: {np.max(np.abs(uparams_mean - uparams_mean_new)):.8f})' if len(idx_re) > 0 else ''
            print(f"{TAB}{TAB}[{datetime.now().strftime('%H:%M:%S')}] Iteration {it+1} / {max_iter}{convergence_str}")

    if len(idx_re) > 0:
        params = transform_to_constrained_space(uparams_init, bounds)
    else:
        params = params_init
    if len(idx_fe) > 0:
        params[:, idx_fe] = transform_to_constrained_space(uparams_fe[idx_fe], bounds[idx_fe])

    result = sciopt.OptimizeResult(x=params)
    result.execution_time = timeit.default_timer() - t0
    if len(idx_re) > 0:
        result.x_re_pop_mean_sd = population_summary(uparams_mean, uparams_sd, bounds, idx_re)
    else:
        result.x_re_pop_mean_sd = None
    if verbosity:
        print(f'{TAB}.. finished ({result.execution_time:.1f} secs).')

    return result


def objective_empirical_bayes(uparams, sub_ind, fun, bounds, idx_re, uparams_re_mean, uparams_re_sd, idx_fe, uparams_fe):
    params = transform_to_constrained_space(uparams, bounds)
    if len(idx_fe) > 0:
        params_fe = transform_to_constrained_space(uparams_fe[idx_fe], bounds[idx_fe])
        params[idx_fe] = params_fe
    val = fun(params, sub_ind)
    # Gaussian prior penalty on random-effects
    if len(idx_re) > 0:
        val += 0.5 * np.sum((uparams[idx_re] - uparams_re_mean[idx_re])**2 / uparams_re_sd[idx_re]**2)
    return val

def objective_group_fe(uparams_fe, fun, uparams, bounds, idx_fe):
    # (uparams_tmp, fun, uparams_re_init, bounds, idx_fe)
    negll = 0.0
    params_fe = transform_to_constrained_space(uparams_fe, bounds)  # (K,)
    for s in range(uparams.shape[0]):
        params = transform_to_constrained_space(uparams[s], bounds)
        params[idx_fe] = params_fe[idx_fe]
        negll += fun(params, s)
    return negll


def population_summary(uparams_re_mean, uparams_re_sd, bounds, idx_re, n_draws=200000, seed=0):


    # params_re_popmean, params_re_popsd = population_summary(uparams_re_mean, uparams_re_sd, bounds, idx_re)

    """
    Compute group-level summaries in params-space using Monte Carlo sampling based on:
        z_j ~ Normal(mu_j, sigma_j)  (for j in re_idx)
        params = transform_constrained(uparams, bounds)

    Returns summaries only for re_idx parameters.
    """
    rng = np.random.default_rng(seed)

    # Draw uparams for RE dims only: (n_draws, |re_idx|)
    uparams_re = rng.normal(loc=uparams_re_mean[idx_re], scale=uparams_re_sd[idx_re], size=(n_draws, idx_re.size))

    # Put into full K-dim uparams with fixed values for non-RE dims (mu is a reasonable plug-in)
    uparams = np.tile(uparams_re_mean, (n_draws, 1))
    uparams[:, idx_re] = uparams_re

    # Transform to params-space
    params = transform_to_constrained_space(uparams, bounds)  # expects (n_draws, K) support

    # Summarize only RE dims
    params_re = params[:, idx_re]

    theta_mean, theta_sd = params_re.mean(axis=0), params_re.std(axis=0, ddof=0)

    return theta_mean, theta_sd


def transform_to_constrained_space(uparams, bounds):
    lower, upper = bounds.T
    uparams = np.asarray(uparams, dtype=float)

    params = np.empty_like(uparams, dtype=float)

    m_unbounded  = np.isinf(lower) & np.isinf(upper)
    m_lower_only = np.isfinite(lower) & np.isinf(upper)
    m_upper_only = np.isinf(lower) & np.isfinite(upper)
    m_bounded    = np.isfinite(lower) & np.isfinite(upper)

    # Use ... to apply along last axis
    params[..., m_unbounded]  = uparams[..., m_unbounded]
    params[..., m_lower_only] = lower[m_lower_only] + np.exp(uparams[..., m_lower_only])
    params[..., m_upper_only] = upper[m_upper_only] - np.exp(uparams[..., m_upper_only])

    params[..., m_bounded] = lower[m_bounded] + (upper[m_bounded] - lower[m_bounded]) / (1 + np.exp(-uparams[..., m_bounded]))
    return params


def transform_to_unconstrained_space(params, bounds, eps=1e-12):
    lower, upper = bounds.T
    params = np.asarray(params, dtype=float)

    uparams = np.empty_like(params, dtype=float)

    m_unbounded  = np.isinf(lower) & np.isinf(upper)
    m_lower_only = np.isfinite(lower) & np.isinf(upper)
    m_upper_only = np.isinf(lower) & np.isfinite(upper)
    m_bounded    = np.isfinite(lower) & np.isfinite(upper)

    uparams[..., m_unbounded] = params[..., m_unbounded]

    # Guard against exact-bound values
    uparams[..., m_lower_only] = np.log(np.maximum(params[..., m_lower_only] - lower[m_lower_only], eps))
    uparams[..., m_upper_only] = np.log(np.maximum(upper[m_upper_only] - params[..., m_upper_only], eps))

    x = (params[..., m_bounded] - lower[m_bounded]) / (upper[m_bounded] - lower[m_bounded])
    x = np.clip(x, eps, 1 - eps)
    uparams[..., m_bounded] = np.log(x) - np.log1p(-x)

    return uparams


class RandomDisplacementBoundsConstraints(object):
    def __init__(self, bounds, constraints, stepsize=0.5):
        self.xmin = bounds.lb
        self.xmax = bounds.ub
        self.constraints = constraints
        self.stepsize = stepsize

    def __call__(self, x):
        while True:
            xnew = x + np.random.uniform(-self.stepsize, self.stepsize, np.shape(x))
            if np.all(xnew < self.xmax) and np.all(xnew > self.xmin) and (self.constraints is None or (np.all(
                    [con['fun'](xnew) >= 0 for con in self.constraints]))):
                break
        return xnew


class BoundsConstraints(object):

    def __init__(self, bounds, constraints=None):
        self.xmin = bounds.lb
        self.xmax = bounds.ub
        self.constraints = constraints

    def __call__(self, **kwargs):
        x = kwargs["x_new"]
        tmax = bool(np.all(x <= self.xmax))
        tmin = bool(np.all(x >= self.xmin))
        cons = self.constraints is None or bool(np.all([con['fun'](x) >= 0 for con in self.constraints]))
        return tmax and tmin and cons
