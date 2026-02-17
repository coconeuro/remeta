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
from scipy.special import expit
from numdifftools import Hessian

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


def fgrid(fun, valid, args, num_cores=1, minimize_along_grid=False, bounds=None, verbosity=1):
    if num_cores > 1:
        with DillPool(num_cores) as pool:
            result = pool.map(partial(loop, fun, valid, args, len(valid), minimize_along_grid, bounds, verbosity), range(len(valid)))
            negll_grid, x_grid = [res[0] for res in result], [res[1] for res in result]
    else:
        negll_grid = [None] * len(valid)
        x_grid = [None] * len(valid)
        for i, param in enumerate(valid):
            negll_grid[i], x_grid[i] = loop(fun, valid, args, len(valid), minimize_along_grid, bounds, verbosity, i)
    return negll_grid, x_grid


# def fine_grid_search(x0, fun, args, param_set, ll_grid, valid, gridsearch_resolution, n_grid_iter, multiproc=False,
#                      multiproc_cores=1, verbosity=1):
#     n_grid_candidates_init = 3 * gridsearch_resolution
#     gx0 = x0
#     gll_min_grid = np.min(ll_grid)
#     grid_candidates = [valid[i] for i in np.argsort(ll_grid)[:n_grid_candidates_init]]
#     previous_grid_range = [param_set.grid_range for _ in range(n_grid_candidates_init)]
#     candidate_ids = list(range(n_grid_candidates_init))
#     counter = 0
#     for i in range(n_grid_iter):
#         if (verbosity > 1) and (np.mod(i, 10) == 0):
#             print(f'{TAB}Grid iteration {i + 1} / {n_grid_iter}')
#         gvalid = []
#         valid_candidate_ids = []
#         grid_range = [None] * len(grid_candidates) * 2
#         for j, grid_candidate in enumerate(grid_candidates):
#             grid_range[j] = [None] * param_set.nparams_flat
#             for k, p in enumerate(grid_candidate):
#                 ind = np.where(previous_grid_range[candidate_ids[j]][k] == p)[0][0]
#                 lb = previous_grid_range[candidate_ids[j]][k][max(0, ind - 1)]  # noqa
#                 ub = previous_grid_range[candidate_ids[j]][k][  # noqa
#                     min(len(previous_grid_range[candidate_ids[j]][k]) - 1, ind + 1)]  # noqa
#                 grid_range[j][k] = np.around(np.linspace(lb, ub, 5), decimals=10)  # noqa
#             if param_set.constraints is None:
#                 gvalid_candidate = list(product(*grid_range[j]))
#             else:
#                 gvalid_candidate = [p for p in product(*grid_range[j]) if  # noqa
#                                     np.all([con['fun'](p) >= 0 for con in param_set.constraints])]
#             gvalid += gvalid_candidate
#             valid_candidate_ids += [j] * len(gvalid_candidate)
#             if verbosity > 1:
#                 print(f'{TAB}{TAB}Candidate {j + 1}: {[(p[0], p[-1]) for p in grid_range[j]]}')  # noqa
#         gll_grid = fgrid(fun, gvalid, args, multiproc, multiproc_cores, verbosity=0)[0]
#         counter += len(gvalid)
#
#         min_id = np.argmin(gll_grid)
#         if gll_grid[min_id] < gll_min_grid:
#             gll_min_grid = gll_grid[min_id]
#             gx0 = gvalid[min_id]
#
#         if verbosity:
#             print(f'{TAB}{TAB}Best fit: {gx0} (LL={gll_min_grid})')
#
#         if i != n_grid_iter - 1:
#             grid_candidates, candidate_ids = [], []
#             for j in np.argsort(gll_grid):
#                 if gvalid[j] not in grid_candidates:
#                     grid_candidates += [gvalid[j]]
#                     candidate_ids += [valid_candidate_ids[j]]
#                     if len(candidate_ids) == gridsearch_resolution:
#                         break
#             previous_grid_range = grid_range
#     fit = sciopt.OptimizeResult(success=True, x=gx0, fun=gll_min_grid, nfev=counter)
#
#     return fit


def subject_estimation(fun, param_set, args=(), gridsearch=False, num_cores=1,
                       minimize_along_grid=False, global_minimization=None, verbosity=1,
                       # n_grid_candidates=10, n_grid_iter=3,
                       scipy_solvers='slsqp', slsqp_epsilon=_slsqp_epsilon):

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

    x0_guess = param_set.guess
    fit_guess = sciopt.OptimizeResult(success=True, x=x0_guess, fun=fun(x0_guess, *args), nfev=1)
    if gridsearch:
        if param_set.constraints is not None and len(param_set.constraints):
            valid = np.array([p for p in product(*param_set.grid_range) if
                     np.all([con['fun'](p) >= 0 for con in param_set.constraints])])
        else:
            valid = np.array(list(product(*param_set.grid_range)))
        if verbosity > 1:
            print(f"{TAB}{TAB}Grid search activated (grid size = {len(valid)})")
        t0 = timeit.default_timer()
        ll_grid, x_grid = fgrid(fun, valid, args, num_cores, minimize_along_grid, bounds, verbosity=verbosity)
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

    # if fine_gridsearch:
    #     fit_grid = fine_grid_search(x0_grid if gridsearch else x0_guess, fun, args, param_set, ll_grid, valid, n_grid_candidates,
    #                                 n_grid_iter, multiproc, multiproc_cores, verbosity=verbosity)
    #     x0_grid = fit_grid.x
    #     fit_best = fit_grid if fit_grid.fun < fit_guess.fun else fit_guess

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
                    try:
                        fit = sciopt.minimize(fun, x0, bounds=bounds, args=tuple(args), constraints=param_set.constraints,
                                              method=method)
                    except ValueError as e:
                        if 'violates bound constraints' in str(e) and (method == 'trust-constr') and check_x0_within_bounds(x0, bounds):
                            # Catch this error in trust-constr which is a bug in scipy that has been fixed soon after
                            # the release of scipy 1.17.0:
                            # https://github.com/scipy/scipy/pull/24454
                            warnings.warn('SciPy trust-constr evaluated outside bounds despite a feasible x0. '
                                          'Using L-BFGS-B instead.')
                            method = 'L-BFGS-B'
                            fit = sciopt.minimize(fun, x0, bounds=bounds, args=tuple(args),
                                                  constraints=param_set.constraints, method=method)
                        else:
                            raise

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


    H = Hessian(fun)(fit_best.x)
    H = 0.5 * (H + H.T)  # make sure it is not asymmetric due to numerical errors
    # Logic: if not singular and all_eigenvalues > 0
    # if not (singular := (np.linalg.cond(H) > 1 / np.finfo(H.dtype).eps) and np.all(np.diag(np.linalg.inv(H)) > 0):
    # if not (singular := (np.linalg.svd(H, compute_uv=False)[-1] < np.finfo(H.dtype).eps)) and np.all(np.diag(np.linalg.inv(H)) > 0):
    if not (singular := np.linalg.cond(H) > 1e12) and np.all(np.diag(np.linalg.inv(H)) > 0):
        # positive definite
        fit_best.hessian = H
    else:
        warnings.warn(f'Hessian is {"singular" if singular else "not positive definite"}. Standard errors are computed '
                      f'with a ridged Hessian and may be unreliable for some parameters (check extreme values).')
        fit_best.hessian = ridge_hessian(H)

    return fit_best


# def ridge_hessian_se(H, eps_scale=1e-12, max_iter=60):
#     # Apply ridge regression until the Hessian H is positive definite
#     Hs = 0.5 * (H + H.T)
#     n = Hs.shape[0]
#
#     scale = max(np.linalg.norm(Hs, ord=np.inf), 1.0)
#     lam = eps_scale * scale
#     for _ in range(max_iter):
#         Hr = Hs + lam * np.eye(n)
#         try:
#             L = np.linalg.cholesky(Hr)
#             break
#         except np.linalg.LinAlgError:
#             lam *= 10
#     else:
#         warnings.warn('Could not determine uncertainty of the parameter estimate, even with a ridged Hessian.')
#
#     I = np.eye(n)
#     Y = np.linalg.solve(L, I)
#     X = np.linalg.solve(L.T, Y)
#     se = np.sqrt(np.diag(X))
#     # cond = np.linalg.cond(Hr)
#     # return se, Hr, lam, cond
#     return se


def ridge_hessian(H, eps_scale=1e-12, max_iter=60):
    # Apply ridge regression until the Hessian H is positive definite
    Hs = 0.5 * (H + H.T)
    n = Hs.shape[0]

    scale = max(np.linalg.norm(Hs, ord=np.inf), 1.0)
    lam = eps_scale * scale
    for _ in range(max_iter):
        Hr = Hs + lam * np.eye(n)
        try:
            np.linalg.cholesky(Hr)
            break
        except np.linalg.LinAlgError:
            lam *= 10
    else:
        warnings.warn('Could not determine uncertainty of the parameter estimate, even with a ridged Hessian.')
        return None

    return Hr


def check_x0_within_bounds(x0, bounds, tol=0.0):

    x0 = np.asarray(x0, dtype=float)

    if hasattr(bounds, 'lb'):  # scipy.optimize.Bounds
        lb = np.asarray(bounds.lb, dtype=float)
        ub = np.asarray(bounds.ub, dtype=float)
    else:  # list of (lb, ub)
        lb = np.array([b[0] for b in bounds], dtype=float)
        ub = np.array([b[1] for b in bounds], dtype=float)

    bad = (
        (x0 < lb - tol) |
        (x0 > ub + tol) |
        ~np.isfinite(x0)
    )

    return not np.any(bad)


def laplace_cov_from_hessian(H, eig_floor=1e-10):
    """
    Robust inverse of a small symmetric PD-ish Hessian.
    Returns covariance approx Sigma = H^{-1}.
    """
    H = 0.5 * (H + H.T)  # symmetrize

    # Cholesky path (fast + stable when PD)
    try:
        L = np.linalg.cholesky(H)
        I = np.eye(H.shape[0])
        Linv = np.linalg.solve(L, I)
        return Linv.T @ Linv
    except np.linalg.LinAlgError:
        # Eigenvalue-floor fallback
        w, Q = np.linalg.eigh(H)
        w = np.maximum(w, eig_floor)
        return (Q * (1.0 / w)) @ Q.T


def d_transform_constrained_du(u, bounds):
    """
    Elementwise derivative d(params)/d(uparams) for your transform_to_constrained_space.
    u: (..., K) or (K,)
    bounds: (K, 2) with columns [lower, upper]
    returns: same shape as u
    """
    lower, upper = bounds.T
    u = np.asarray(u, dtype=float)

    # broadcast masks across last axis
    m_unbounded  = np.isinf(lower) & np.isinf(upper)
    m_lower_only = np.isfinite(lower) & np.isinf(upper)
    m_upper_only = np.isinf(lower) & np.isfinite(upper)
    m_bounded    = np.isfinite(lower) & np.isfinite(upper)

    deriv = np.empty_like(u, dtype=float)

    deriv[..., m_unbounded] = 1.0

    # Match your exp clipping (avoid overflow)
    z = np.clip(u, -709.0, 709.0)
    ez = np.exp(z)

    deriv[..., m_lower_only] = ez[..., m_lower_only]          # d(l + exp(u))/du = exp(u)
    deriv[..., m_upper_only] = -ez[..., m_upper_only]         # d(u - exp(u))/du = -exp(u)

    s = expit(u[..., m_bounded])
    deriv[..., m_bounded] = (upper[m_bounded] - lower[m_bounded]) * s * (1.0 - s)

    return deriv


def compute_fixed_effect_se(fun, uparams_init, uparams_fe, bounds, idx_fe, eps=1e-4, eps_scale=True):
    """
    Compute global FE SEs (in constrained params-space) using Hessian of the group FE objective,
    conditional on current uparams_init.

    Returns:
      fe_se: fixed effects standard errors
    """

    def obj_fe_only(x_packed):
        u_tmp = uparams_fe.copy()
        u_tmp[idx_fe] = x_packed
        return objective_group_fe(u_tmp, fun, uparams_init, bounds, idx_fe)

    xhat_u = uparams_fe[idx_fe].copy()
    H = numerical_hessian(obj_fe_only, xhat_u, eps=eps, eps_scale=eps_scale)
    H = 0.5 * (H + H.T) + 1e-12 * np.eye(idx_fe.size)

    Sigma_u = laplace_cov_from_hessian(H, eig_floor=1e-10)
    var_u = np.maximum(np.diag(Sigma_u), 0.0)

    deriv_full = d_transform_constrained_du(uparams_fe, bounds)
    deriv_fe = deriv_full[idx_fe]
    se_theta = np.abs(deriv_fe) * np.sqrt(var_u)

    return se_theta



def fe_only_full_cov_per_subject(
    s, fun, uparams_init_final, uparams_fe_final, idx_fe, idx_free, bounds,
    eps=1e-4, eps_scale=True
):
    """
    FE-only case, assuming idx_fe and idx_free are disjoint and cover ALL params.

    Computes joint Laplace covariance for packed vector:
        z = [u_fe[idx_fe], u_s[idx_free]]
    using global objective sum_t fun(theta_t, t) with:
      - FE varying (via z[:d_fe])
      - subject s free params varying (via z[d_fe:])
      - other subjects held fixed at uparams_init_final

    Returns:
      Sigma_theta_full_s : (K,K) theta-space covariance for subject s,
      ordered in the original global parameter index order (0..K-1).
    """
    idx_fe = np.asarray(idx_fe, dtype=int)
    idx_free = np.asarray(idx_free, dtype=int)

    K = uparams_init_final.shape[1]
    d_fe = idx_fe.size
    d_free = idx_free.size

    # packed-to-full index order
    idx_z = np.concatenate([idx_fe, idx_free])
    # sanity: this function assumes coverage of all dims
    # (comment out if you don't want runtime checks)
    assert idx_z.size == K
    assert np.unique(idx_z).size == K

    # base copies used in objective
    U0 = uparams_init_final.copy()   # (N,K)
    ufe0 = uparams_fe_final.copy()   # (K,)

    def obj_z(z):
        # update FE in u-space
        ufe = ufe0.copy()
        ufe[idx_fe] = z[:d_fe]

        # update subject s free params in u-space
        Us = U0.copy()
        Us[s, idx_free] = z[d_fe:]

        # FE in theta-space (full K, but only idx_fe used)
        theta_fe_full = transform_to_constrained_space(ufe, bounds)

        total = 0.0
        for t in range(Us.shape[0]):
            theta_t = transform_to_constrained_space(Us[t], bounds)
            theta_t[idx_fe] = theta_fe_full[idx_fe]
            total += fun(theta_t, t)
        return total

    # starting point at converged values
    z0 = np.concatenate([uparams_fe_final[idx_fe], uparams_init_final[s, idx_free]])

    # Hessian in u-space
    H = numerical_hessian(obj_z, z0, eps=eps, eps_scale=eps_scale)
    H = 0.5 * (H + H.T) + 1e-12 * np.eye(d_fe + d_free)

    Sigma_u = laplace_cov_from_hessian(H, eig_floor=1e-10)
    Sigma_u = 0.5 * (Sigma_u + Sigma_u.T)

    # Jacobian diag for theta transform, aligned with z
    d_fe_u = d_transform_constrained_du(uparams_fe_final, bounds)[idx_fe]
    d_free_u = d_transform_constrained_du(uparams_init_final[s], bounds)[idx_free]
    d_z = np.concatenate([d_fe_u, d_free_u])  # (K,)

    # Delta method in packed order
    Sigma_theta_z = (d_z[:, None] * Sigma_u) * d_z[None, :]
    Sigma_theta_z = 0.5 * (Sigma_theta_z + Sigma_theta_z.T)

    # Permute packed covariance into global K-order
    Sigma_theta_full = np.empty((K, K), dtype=float)
    Sigma_theta_full[np.ix_(idx_z, idx_z)] = Sigma_theta_z

    return Sigma_theta_full



def group_estimation(fun, nsubjects, params_init, bounds, idx_fe, idx_re,
                     num_cores=1, max_iter=30,
                     sigma_floor=1e-3, damping_mu=0.5, damping_sig=0.5, verbosity=1):

    params_init = np.array(params_init)
    nparams = params_init.shape[1]

    # normalize indices once
    idx_fe = np.asarray(idx_fe, dtype=int) if len(idx_fe) > 0 else np.asarray([], dtype=int)
    idx_re = np.asarray(idx_re, dtype=int) if len(idx_re) > 0 else np.asarray([], dtype=int)

    if verbosity:
        print(f'\n{SP2}Group-level optimization (MLE / MAP)')
    t0 = timeit.default_timer()

    uparams_init = transform_to_unconstrained_space(params_init, bounds)

    # ----- FE init -----
    if len(idx_fe) > 0:
        params_fe_init = np.median(params_init, axis=0)
        uparams_fe = transform_to_unconstrained_space(params_fe_init, bounds)
    else:
        uparams_fe = None

    # ----- subject-level identifiable dims -----
    all_idx = np.arange(nparams, dtype=int)
    mask_free = np.ones(nparams, dtype=bool)
    if len(idx_fe) > 0:
        mask_free[idx_fe] = False
    idx_free = all_idx[mask_free]  # subject-level identifiable dims (optimized per subject)
    d_free = idx_free.size

    # ----- RE hyperparams init -----
    has_re = (len(idx_re) > 0)
    if has_re:
        uparams_mean = uparams_init.mean(axis=0)
        uparams_sd = uparams_init.std(axis=0) + sigma_floor
    else:
        uparams_mean = None
        uparams_sd = None

    # outputs (updated each iteration; final kept)
    x_se = np.full((nsubjects, nparams), np.nan, dtype=float)
    cov_theta_free = None  # list of (d_free,d_free) from last iteration

    def subject_loop_factory(uparams_init_ref, uparams_fe_ref, uparams_mean_ref, uparams_sd_ref):

        def subject_loop(s):
            # ---- pack/unpack on free block (exclude idx_fe directions entirely) ----
            def unpack_u(u_free):
                u_full = uparams_init_ref[s].copy()
                u_full[idx_free] = u_free
                return u_full

            def obj_free(u_free):
                u_full = unpack_u(u_free)

                # Likelihood with FE imposed inside objective_empirical_bayes
                val = objective_empirical_bayes(
                    u_full, s, fun, bounds,
                    idx_re, uparams_mean_ref, uparams_sd_ref,
                    idx_fe, uparams_fe_ref
                )

                # IMPORTANT: objective_empirical_bayes already includes the RE penalty when len(idx_re)>0.
                # In FE-only, idx_re is empty, so no penalty is added.
                return val

            # ---- 1) optimize free dims ----
            x0 = uparams_init_ref[s, idx_free].copy()
            res = sciopt.minimize(obj_free, x0=x0, method='L-BFGS-B')
            u_hat_free = res.x
            u_hat_full = unpack_u(u_hat_free)

            # ---- 2) Hessian / covariance in u-space on free block ----
            H = numerical_hessian(obj_free, u_hat_free, eps=1e-4, eps_scale=True)
            H = 0.5 * (H + H.T) + 1e-12 * np.eye(d_free)

            Sigma_u_free = laplace_cov_from_hessian(H, eig_floor=1e-10)
            Sigma_u_free = 0.5 * (Sigma_u_free + Sigma_u_free.T)

            # ---- 3) delta-method covariance in theta-space on free block ----
            deriv_full = d_transform_constrained_du(u_hat_full, bounds)  # (K,)
            d = deriv_full[idx_free]                                     # (d_free,)
            Sigma_theta_free = (d[:, None] * Sigma_u_free) * d[None, :]
            Sigma_theta_free = 0.5 * (Sigma_theta_free + Sigma_theta_free.T)

            # ---- 4) SEs from theta-cov diagonal ----
            var_theta_free = np.maximum(np.diag(Sigma_theta_free), 0.0)
            se_theta_free = np.sqrt(var_theta_free)  # (d_free,)

            # ---- 5) posterior var for RE dims (only needed if has_re) ----
            if has_re:
                # idx_re is subset of idx_free because idx_fe and idx_re should be disjoint
                pos_re_in_free = np.searchsorted(idx_free, idx_re)
                var_post_re = np.maximum(np.diag(Sigma_u_free)[pos_re_in_free], 0.0)  # (d_re,)
            else:
                var_post_re = None

            return u_hat_full, var_post_re, se_theta_free, Sigma_theta_free

        return subject_loop

    # -------- main loop --------
    for it in range(max_iter):

        # ---- (A) update FE given current subject parameters ----
        if len(idx_fe) > 0:
            def objective_fe_packed(x_packed):
                uparams_tmp = uparams_fe.copy()
                uparams_tmp[idx_fe] = x_packed
                return objective_group_fe(uparams_tmp, fun, uparams_init, bounds, idx_fe)

            x0 = uparams_fe[idx_fe].copy()
            uparams_fe[idx_fe] = sciopt.minimize(objective_fe_packed, x0=x0, method='L-BFGS-B').x

        # ---- (B) update subject parameters (ALWAYS) given current FE and (if present) RE hyperparams ----
        subject_loop = subject_loop_factory(
            uparams_init_ref=uparams_init,
            uparams_fe_ref=uparams_fe,
            uparams_mean_ref=uparams_mean,
            uparams_sd_ref=uparams_sd
        )

        if num_cores > 1:
            with DillPool(num_cores) as pool:
                out = list(pool.map(subject_loop, range(nsubjects)))
        else:
            out = [subject_loop(s) for s in range(nsubjects)]

        # unpack
        uparams_init = np.array([o[0] for o in out])            # (N,K)
        se_theta_free = np.array([o[2] for o in out])           # (N,d_free)
        cov_theta_free = [o[3] for o in out]                    # list of (d_free,d_free)

        # fill SEs for free dims (includes RE and "neither")
        x_se[:, idx_free] = se_theta_free

        # ---- (C) update RE hyperparams if present ----
        if has_re:
            var_post_re = np.array([o[1] for o in out])         # (N,d_re)

            uparams_mean_new = uparams_mean.copy()
            uparams_sd_new = uparams_sd.copy()

            # mu update
            uparams_mean_new[idx_re] = uparams_init[:, idx_re].mean(axis=0)

            # sigma^2 update with curvature correction
            centered2 = (uparams_init[:, idx_re] - uparams_mean_new[idx_re]) ** 2
            sigma2_new = (centered2 + var_post_re).mean(axis=0)

            uparams_sd_new[idx_re] = np.sqrt(np.maximum(sigma2_new, sigma_floor**2))
            uparams_sd_new[idx_re] = np.maximum(uparams_sd_new[idx_re], sigma_floor)

            # damping
            uparams_mean = (1 - damping_mu) * uparams_mean + damping_mu * uparams_mean_new
            uparams_sd   = (1 - damping_sig) * uparams_sd   + damping_sig * uparams_sd_new

        if verbosity and (np.mod(it, 10) == 0):
            if has_re:
                conv = np.max(np.abs(uparams_mean - uparams_mean_new))
                convergence_str = f' (Convergence: {conv:.8f})'
            else:
                convergence_str = ''
            print(f"{TAB}{TAB}[{datetime.now().strftime('%H:%M:%S')}] Iteration {it+1} / {max_iter}{convergence_str}")

    # -------- finalize params in constrained space --------
    params = transform_to_constrained_space(uparams_init, bounds)

    # impose FE at the end (subject-level uparams_init contains arbitrary values at idx_fe)
    if len(idx_fe) > 0:
        params[:, idx_fe] = transform_to_constrained_space(uparams_fe[idx_fe], bounds[idx_fe])

        # FE SEs (global) and broadcast across subjects
        fe_se = compute_fixed_effect_se(
            fun=fun, uparams_init=uparams_init, uparams_fe=uparams_fe, bounds=bounds, idx_fe=idx_fe
        )
        x_se[:, idx_fe] = fe_se

    # -------- build result --------
    result = sciopt.OptimizeResult(x=params)
    result.x_se = x_se

    # # always provide x_cov (for free block) even in FE-only
    # result.x_cov = np.full((nsubjects, nparams, nparams), np.nan, dtype=float)
    # if cov_theta_free is not None:
    #     cov_arr = np.stack(cov_theta_free, axis=0)  # (N,d_free,d_free)
    #     # optional: enforce symmetry
    #     cov_arr = 0.5 * (cov_arr + np.swapaxes(cov_arr, 1, 2))
    #     result.x_cov[:, *np.ix_(idx_free, idx_free)] = cov_arr


    result.x_cov = np.full((nsubjects, nparams, nparams), np.nan, dtype=float)
    if has_re:
        # keep your existing behavior in RE case (free-free only)
        cov_arr = np.stack(cov_theta_free, axis=0)  # (N,d_free,d_free)
        cov_arr = 0.5 * (cov_arr + np.swapaxes(cov_arr, 1, 2))
        result.x_cov[:, *np.ix_(idx_free, idx_free)] = cov_arr
    else:
        # FE-only: compute full KxK per subject (includes FE blocks + cross)
        if num_cores > 1:
            with DillPool(num_cores) as pool:
                cov_full_list = list(pool.map(
                    lambda s: fe_only_full_cov_per_subject(s, fun, uparams_init, uparams_fe, idx_fe, idx_free, bounds),
                    range(nsubjects)
                ))
        else:
            cov_full_list = [fe_only_full_cov_per_subject(s, fun, uparams_init, uparams_fe, idx_fe, idx_free, bounds)
                             for s in range(nsubjects)]

        result.x_cov = np.stack(cov_full_list, axis=0)  # (N,K,K)

    # RE population summary only if RE exists
    if has_re:
        result.x_re_pop_mean_sd = population_summary(uparams_mean, uparams_sd, bounds, idx_re)
    else:
        result.x_re_pop_mean_sd = None

    result.execution_time = timeit.default_timer() - t0
    if verbosity:
        print(f'{TAB}.. finished ({result.execution_time:.1f} secs).')

    return result


def numerical_hessian(phi, x, eps=1e-4, eps_scale=False):
    """
    Central-difference Hessian.

    eps:
      - float: fixed step size for all dims
      - array-like (d,): per-dim step sizes
    eps_scale:
      - if True and eps is a float, use eps_i = eps*(|x_i|+1)

    phi: callable R^d -> R
    x: (d,)
    """
    x = np.asarray(x, float)
    d = x.size
    H = np.zeros((d, d), float)
    fx = phi(x)

    if np.isscalar(eps):
        if eps_scale:
            eps_vec = eps * (np.abs(x) + 1.0)
        else:
            eps_vec = np.full(d, float(eps))
    else:
        eps_vec = np.asarray(eps, float)
        if eps_vec.shape != (d,):
            raise ValueError("eps must be scalar or shape (d,)")

    for i in range(d):
        ei = np.zeros(d); ei[i] = eps_vec[i]
        f_ip = phi(x + ei)
        f_im = phi(x - ei)
        H[i, i] = (f_ip - 2 * fx + f_im) / (eps_vec[i] ** 2)

        for j in range(i + 1, d):
            ej = np.zeros(d); ej[j] = eps_vec[j]
            f_pp = phi(x + ei + ej)
            f_pm = phi(x + ei - ej)
            f_mp = phi(x - ei + ej)
            f_mm = phi(x - ei - ej)
            H_ij = (f_pp - f_pm - f_mp + f_mm) / (4 * eps_vec[i] * eps_vec[j])
            H[i, j] = H_ij
            H[j, i] = H_ij

    return H


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

    # Optional: clip to avoid exp overflow in the one-sided cases too
    # (exp(709) ~ 8e307 is near float64 max)
    z = np.clip(uparams, -709.0, 709.0)

    params[..., m_lower_only] = lower[m_lower_only] + np.exp(z[..., m_lower_only])
    params[..., m_upper_only] = upper[m_upper_only] - np.exp(z[..., m_upper_only])

    # Stable bounded transform
    params[..., m_bounded] = lower[m_bounded] + (upper[m_bounded] - lower[m_bounded]) * expit(uparams[..., m_bounded])
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
