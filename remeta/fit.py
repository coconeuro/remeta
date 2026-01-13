import timeit
import warnings
from functools import partial
from itertools import product
from multiprocessing import cpu_count

import numpy as np
try:  # only necessary if multiple cores should be used
    from multiprocessing_on_dill.pool import Pool as DillPool
except ModuleNotFoundError:
    pass
import scipy.optimize as sciopt
from scipy.optimize._constraints import old_bound_to_new  # noqa

from .util import TAB
from .util import _slsqp_epsilon

def loop(fun, params, args, gridsize, minimize_along_grid, bounds, verbose, param_id):
    # t0 = timeit.default_timer()
    if param_id and verbose and (np.mod(param_id, 1000) == 0):
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


def fgrid(fun, valid, args, grid_multiproc, minimize_along_grid=False, bounds=None, verbose=True):
    if grid_multiproc:
        with DillPool(cpu_count() - 1 or 1) as pool:
            result = pool.map(partial(loop, fun, valid, args, len(valid), minimize_along_grid, bounds, verbose), range(len(valid)))
            negll_grid, x_grid = [res[0] for res in result], [res[1] for res in result]
    else:
        negll_grid = [None] * len(valid)
        x_grid = [None] * len(valid)
        for i, param in enumerate(valid):
            negll_grid[i], x_grid[i] = loop(fun, valid, args, len(valid), minimize_along_grid, bounds, verbose, i)
    return negll_grid, x_grid


def fine_grid_search(x0, fun, args, param_set, ll_grid, valid, gridsearch_resolution, n_grid_iter, grid_multiproc, verbose=True):
    n_grid_candidates_init = 3 * gridsearch_resolution
    gx0 = x0
    gll_min_grid = np.min(ll_grid)
    grid_candidates = [valid[i] for i in np.argsort(ll_grid)[:n_grid_candidates_init]]
    previous_grid_range = [param_set.grid_range for _ in range(n_grid_candidates_init)]
    candidate_ids = list(range(n_grid_candidates_init))
    counter = 0
    for i in range(n_grid_iter):
        if verbose:
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
            if verbose:
                print(f'{TAB}{TAB}Candidate {j + 1}: {[(p[0], p[-1]) for p in grid_range[j]]}')  # noqa
        gll_grid = fgrid(fun, gvalid, args, grid_multiproc, verbose=False)[0]
        counter += len(gvalid)

        min_id = np.argmin(gll_grid)
        if gll_grid[min_id] < gll_min_grid:
            gll_min_grid = gll_grid[min_id]
            gx0 = gvalid[min_id]

        if verbose:
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


def fmincon(fun, param_set, args, gridsearch=False, grid_multiproc=True, minimize_along_grid=False,
            global_minimization=False, fine_gridsearch=False, verbose=True,
            n_grid_candidates=10, n_grid_iter=3, minimize_solver='slsqp', slsqp_epsilon=_slsqp_epsilon,
            guess=None):

    t0 = timeit.default_timer()

    if verbose:
        negll_initial_guess = fun(param_set.guess, *args)
        print(f'Initial guess (neg. LL: {negll_initial_guess:.2f})')
        for i, p in enumerate(param_set.param_names_flat):
            if p.startswith('type2_criteria') and not p.endswith('0'):
                criterion_id = int(p.split('_')[-1])
                criterion = param_set.guess[i-criterion_id:i+1].sum()
                print(f'{TAB}[guess] {p}: {param_set.guess[i]:.4g} = gap | criterion = {criterion:.4g}')
            else:
                print(f'{TAB}[guess] {p}: {param_set.guess[i]:.4g}')

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
        if verbose:
            print(f"Grid search activated (grid size = {len(valid)})")
        t0 = timeit.default_timer()
        ll_grid, x_grid = fgrid(fun, valid, args, grid_multiproc, minimize_along_grid, bounds, verbose=verbose)
        x0_grid = x_grid[np.argmin(ll_grid)]
        ll_min_grid = np.min(ll_grid)
        grid_time = timeit.default_timer() - t0
        if verbose:
            for i, p in enumerate(param_set.param_names_flat):
                if p.startswith('type2_criteria') and not p.endswith('0'):
                    criterion_id = int(p.split('_')[-1])
                    criterion = param_set.guess[i-criterion_id:i+1].sum()
                    print(f'{TAB}[guess] {p}: {param_set.guess[i]:.4g} = gap | criterion = {criterion:.4g}')
                else:
                    print(f'{TAB}[grid] {p}: {x0_grid[i]:.4g}')
            print(f"Grid neg. LL: {ll_min_grid:.1f}")
            print(f"Grid runtime: {grid_time:.2f} secs")
        fit_grid = sciopt.OptimizeResult(success=True, x=x0_grid, fun=ll_min_grid, nfev=len(valid))
        fit_best = fit_grid if ll_min_grid < fit_guess.fun else fit_guess
    else:
        fit_best = fit_guess
        x0_grid = None

    if fine_gridsearch:
        fit_fine = fine_grid_search(x0_grid if gridsearch else x0, fun, args, param_set, ll_grid, valid, n_grid_candidates,
                                    n_grid_iter, grid_multiproc, verbose=verbose).x

    if global_minimization is not None:
        if verbose:
            print('Performing global optimization')
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
    else:
        x0_global = None
        execution_time_global = None

    if verbose:
        print('Performing local optimization')

    minimize_solver = [minimize_solver] if isinstance(minimize_solver, str) else minimize_solver

    # We start both from x0_guess and x0_grid, as x0_grid is more prone to local minima
    x0s = (x0_guess, x0_grid, x0_global)

    best_solver = f"init_{'guess' if x0_grid is None else 'grid'}" if global_minimization is None else global_minimization
    for i, x0 in enumerate(x0s):
        if x0 is not None:
            for method in minimize_solver:
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
    fit_best.best_solver = best_solver
    fit_best.x0_grid = x0_grid

    return fit_best


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
