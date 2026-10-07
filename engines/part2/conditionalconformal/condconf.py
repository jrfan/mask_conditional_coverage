import cvxpy as cp
import numpy as np
import os
from functools import partial, lru_cache
from scipy.optimize import linprog
from sklearn.metrics.pairwise import pairwise_kernels
from typing import Callable
import warnings
FUNCTION_DEFAULTS = {'kernel': None, 'gamma': 1, 'lambda': 1}
_FALLBACK_WARNED = False

def solve_cvxpy_with_fallback(prob: cp.Problem, *, prefer: str='MOSEK', fallbacks: tuple[str, ...]=('OSQP', 'ECOS', 'SCS'), warm_start: bool=False, verbose: bool=False, abort_on_prefer_failure: bool=True):
    global _FALLBACK_WARNED
    installed = set(cp.installed_solvers())
    if abort_on_prefer_failure and prefer and (prefer not in installed):
        raise cp.error.SolverError(f"Preferred solver '{prefer}' is not installed.")
    if abort_on_prefer_failure:
        candidates: list[str | None] = [prefer] if prefer else [None]
    else:
        candidates = []
        if prefer:
            candidates.append(prefer)
        candidates.extend([s for s in fallbacks if s and s != prefer])
        candidates.append(None)
    last_err: Exception | None = None
    for solver in candidates:
        if solver is not None and solver not in installed:
            continue
        try:
            kwargs = dict(warm_start=warm_start, verbose=verbose)
            if solver == 'MOSEK':
                mosek_threads = os.environ.get('MOSEK_NUM_THREADS')
                if mosek_threads is not None:
                    kwargs['mosek_params'] = {'MSK_IPAR_NUM_THREADS': max(1, int(mosek_threads))}
            elif solver == 'OSQP':
                kwargs.update(dict(max_iter=100000, eps_abs=1e-05, eps_rel=1e-05))
            elif solver == 'SCS':
                kwargs.update(dict(max_iters=250000, eps=0.0001))
            elif solver == 'ECOS':
                kwargs.update(dict(max_iters=100000))
            if solver is None:
                prob.solve(**kwargs)
            else:
                prob.solve(solver=solver, **kwargs)
            if prob.status not in {cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE, cp.UNBOUNDED, cp.UNBOUNDED_INACCURATE}:
                return solver or 'CVXPY_DEFAULT'
            last_err = cp.error.SolverError(f"Solver {solver or 'CVXPY_DEFAULT'} returned status {prob.status}.")
            if abort_on_prefer_failure and solver == prefer:
                raise last_err
        except (cp.error.SolverError, Exception) as e:
            last_err = e
            if abort_on_prefer_failure and solver == prefer:
                raise
            if prefer == 'MOSEK' and solver == 'MOSEK' and (not _FALLBACK_WARNED):
                _FALLBACK_WARNED = True
                warnings.warn(f'MOSEK failed ({type(e).__name__}: {e}); falling back to other solvers.', RuntimeWarning)
            continue
    if last_err is None:
        raise cp.error.SolverError('No available solver could be used to solve the problem.')
    raise last_err

def build_penalty_partition(p: int, d_mask: int, intercept_col: int, penalize_mask: bool=False):
    if p <= 0:
        return (np.zeros(0, dtype=bool), np.zeros(0, dtype=bool))
    if intercept_col is not None:
        intercept_idx = int(intercept_col)
        if not 0 <= intercept_idx < p:
            intercept_idx = None
    else:
        intercept_idx = None
    assert d_mask >= 0 and d_mask <= p
    max_pen_stop = intercept_idx if intercept_idx is not None else p
    assert max_pen_stop <= p and max_pen_stop >= 0
    J_pen = np.zeros(p, dtype=bool)
    start_pen = 0 if penalize_mask else d_mask
    assert d_mask <= max_pen_stop
    assert start_pen >= 0 and start_pen <= p
    stop_pen = max_pen_stop
    if start_pen < stop_pen:
        J_pen[start_pen:stop_pen] = True
    if intercept_idx is not None:
        assert J_pen[intercept_idx] == False
    J_free = ~J_pen
    return (J_pen, J_free)

class CondConf:

    def __init__(self, score_fn: Callable, Phi_fn: Callable, quantile_fn: Callable=None, infinite_params: dict={}, seed: int=0, svd_compress: bool=False):
        self.score_fn = score_fn
        self.Phi_fn = Phi_fn
        self.quantile_fn = quantile_fn
        self.infinite_params = infinite_params
        self.rng = np.random.default_rng(seed=seed)
        self.svd_compress = svd_compress

    def setup_problem(self, x_calib: np.ndarray, y_calib: np.ndarray):
        self.x_calib = x_calib
        self.y_calib = y_calib
        phi_calib_raw = self.Phi_fn(x_calib)
        want_kernel = self.infinite_params.get('kernel', None)
        want_ridge = want_kernel == 'finite_ridge'
        do_svd = self.svd_compress and (not want_kernel) and (not want_ridge)
        if do_svd:
            _, s, Vt = np.linalg.svd(phi_calib_raw, full_matrices=False)
            tol = 1e-10
            r = int(np.sum(s > tol))
            if r < len(s):
                self.Phi_fn_orig = self.Phi_fn
                T = Vt.T[:, :r]
                self.Phi_fn = lambda x: self.Phi_fn_orig(x) @ T
                phi_calib = self.Phi_fn(x_calib)
            else:
                phi_calib = phi_calib_raw
        else:
            phi_calib = phi_calib_raw
        self.phi_calib = phi_calib
        self.scores_calib = self.score_fn(x_calib, y_calib)
        if self.quantile_fn is not None:
            self.quantile_calib = self.quantile_fn(x_calib).reshape(-1, 1)
        self.cvx_problem = setup_cvx_problem(self.x_calib, self.scores_calib, self.phi_calib, self.infinite_params)

    @lru_cache()
    def _get_calibration_solution(self, quantile: float):
        S = self.scores_calib.reshape(-1, 1)
        Phi = self.phi_calib.astype(float)
        zeros = np.zeros((Phi.shape[1],))
        if quantile is None:
            bounds = np.concatenate((self.quantile_calib - 1, self.quantile_calib), axis=1)
        else:
            bounds = np.asarray([quantile - 1, quantile])
            bounds = np.tile(bounds.reshape(1, -1), (len(S), 1))
        res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds, method='highs')
        primal_vars = -1 * res.eqlin.marginals.reshape(-1, 1)
        dual_vars = res.x.reshape(-1, 1)
        residuals = S - Phi @ primal_vars
        interpolated_pts = np.isclose(residuals, 0)
        if interpolated_pts.sum() < Phi.shape[1]:
            num_to_add = Phi.shape[1] - interpolated_pts.sum()
            for _ in range(num_to_add):
                candidate_pts = interpolated_pts.copy().flatten()
                Q, _ = np.linalg.qr(Phi[candidate_pts].T)
                projections = Phi @ Q @ Q.T
                norms = np.linalg.norm(Phi - projections, axis=1)
                thresholds = [1e-05, 1e-06, 1e-07, 1e-08, 1e-09, 1e-10, 1e-11, 1e-12, 1e-13]
                candidate_idx = None
                for thresh in thresholds:
                    idxs = np.where(norms > thresh)[0]
                    if len(idxs) > 0:
                        candidate_idx = idxs[0]
                        break
                if candidate_idx is None:
                    raise RuntimeError('No valid candidate_idx found — norms are too small.')
                candidate_pts[candidate_idx] = True
                gamma, _, _, _ = np.linalg.lstsq(Phi[candidate_pts], S[candidate_pts], rcond=None)
                direction = gamma.reshape(-1, 1) - primal_vars
                step_sizes = residuals / (Phi @ direction)
                positive_indices = np.where((step_sizes > 0) & ~interpolated_pts)[0]
                primal_vars += np.min(step_sizes[positive_indices]) * direction
                residuals = S - Phi @ primal_vars
                interpolated_pts = np.isclose(residuals, 0)
        return (dual_vars, primal_vars)

    def _compute_exact_cutoff(self, quantiles, primals, duals, phi_test, dual_threshold):

        def get_current_basis(primals, duals, Phi, S, quantiles):
            interp_bools = np.logical_and(~np.isclose(duals, quantiles - 1), ~np.isclose(duals, quantiles))
            if np.sum(interp_bools) == Phi.shape[1]:
                return interp_bools
            preds = (Phi @ primals).flatten()
            active_indices = np.where(interp_bools)[0]
            interp_indices = np.where(np.isclose(np.abs(S - preds), 0))[0]
            diff_indices = np.setdiff1d(interp_indices, active_indices)
            num_missing = Phi.shape[1] - np.sum(interp_bools)
            if num_missing < len(diff_indices):
                from itertools import combinations
                for cand_indices in combinations(diff_indices, num_missing):
                    cand_phi = Phi[np.concatenate((active_indices, cand_indices))]
                    if np.isfinite(np.linalg.cond(cand_phi)):
                        interp_bools[np.asarray(cand_indices)] = True
                        break
            else:
                interp_bools[diff_indices] = True
            if np.sum(interp_bools) != Phi.shape[1]:
                raise ValueError('Initial basis could not be found - retry with exact=False.')
            return interp_bools
        if np.allclose(phi_test, 0):
            return np.inf if quantiles[-1] >= 0.5 else -np.inf
        basis = get_current_basis(primals, duals, self.phi_calib, self.scores_calib, quantiles[:-1])
        S_test = phi_test @ primals
        duals = np.concatenate((duals.flatten(), [0]))
        basis = np.concatenate((basis.flatten(), [False]))
        phi = np.concatenate((self.phi_calib, phi_test.reshape(1, -1)), axis=0)
        S = np.concatenate((self.scores_calib.reshape(-1, 1), S_test.reshape(-1, 1)), axis=0)
        candidate_idx = phi.shape[0] - 1
        num_iters = 0
        while True:
            direction = -1 * np.linalg.solve(phi[basis].T, phi[candidate_idx].reshape(-1, 1)).flatten()
            active_indices = ~np.isclose(direction, 0)
            active_direction = direction[active_indices]
            active_basis = basis.copy()
            active_basis[np.where(basis)[0][~active_indices]] = False
            positive_step = True if duals[candidate_idx] <= 0 else False
            if candidate_idx == phi.shape[0] - 1:
                positive_step = True if dual_threshold >= 0 else False
            if positive_step:
                gap_to_bounds = np.maximum((quantiles[active_basis].flatten() - duals[active_basis]) / active_direction, (quantiles[active_basis].flatten() - 1 - duals[active_basis]) / active_direction)
                step_size = np.min(gap_to_bounds)
                departing_idx = np.where(active_basis)[0][np.argmin(gap_to_bounds)]
            else:
                gap_to_bounds = np.minimum((quantiles[active_basis].flatten() - duals[active_basis]) / active_direction, (quantiles[active_basis].flatten() - 1 - duals[active_basis]) / active_direction)
                step_size = np.max(gap_to_bounds)
                departing_idx = np.where(active_basis)[0][np.argmax(gap_to_bounds)]
            step_size_clip = np.clip(step_size, a_max=quantiles[candidate_idx] - duals[candidate_idx], a_min=quantiles[candidate_idx] - 1 - duals[candidate_idx])
            duals[basis] += step_size_clip * direction
            duals[candidate_idx] += step_size_clip
            if dual_threshold > 0 and duals[-1] > dual_threshold:
                break
            if dual_threshold < 0 and duals[-1] < dual_threshold:
                break
            if step_size_clip == step_size:
                basis[departing_idx] = False
                basis[candidate_idx] = True
            if np.isclose(duals[-1], dual_threshold):
                break
            reduced_A = np.linalg.solve(phi[basis].T, phi[~basis].T)
            reduced_costs = (S[~basis].T - S[basis].T @ reduced_A).flatten()
            bottom = reduced_A[-1]
            bottom[np.isclose(bottom, 0)] = np.inf
            req_change = reduced_costs / bottom
            if dual_threshold >= 0:
                ignore_entries = np.isclose(bottom, 0) | np.asarray(req_change <= 1e-05)
            else:
                ignore_entries = np.isclose(bottom, 0) | np.asarray(req_change >= -1e-05)
            if np.sum(~ignore_entries) == 0:
                S[-1] = np.inf if quantiles[-1] >= 0.5 else -np.inf
                break
            if dual_threshold >= 0:
                candidate_idx = np.where(~basis)[0][np.where(~ignore_entries, req_change, np.inf).argmin()]
                S[-1] += np.min(req_change[~ignore_entries])
            else:
                candidate_idx = np.where(~basis)[0][np.where(~ignore_entries, req_change, -np.inf).argmax()]
                S[-1] += np.max(req_change[~ignore_entries])
            num_iters += 1
            if num_iters > 10000:
                S[-1] = np.inf if dual_threshold > 0 else -1 * np.inf
        return S[-1]

    def predict(self, quantile: float, x_test: np.ndarray, score_inv_fn: Callable, S_min: float=None, S_max: float=None, randomize: bool=False, exact: bool=True, threshold: float=None):
        if quantile is None:
            quantile_test = self.quantile_fn(x_test).reshape(-1, 1)
            quantiles = np.concatenate((self.quantile_calib, quantile_test), axis=0)
        else:
            quantile_test = quantile
            quantiles = np.ones((len(self.scores_calib) + 1, 1)) * quantile
        if threshold is None:
            if randomize:
                threshold = self.rng.uniform(low=quantile_test - 1, high=quantile_test)
            elif quantile_test < 0.5:
                threshold = quantile_test - 1
            else:
                threshold = quantile_test
        if exact:
            if self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']):
                raise ValueError("Exact computation doesn't support RKHS quantile regression for now.")
            if np.allclose(quantiles[0], quantiles):
                naive_duals, naive_primals = self._get_calibration_solution(quantiles.flatten()[0])
            else:
                naive_duals, naive_primals = self._get_calibration_solution(None)
            score_cutoff = self._compute_exact_cutoff(quantiles, naive_primals, naive_duals, self.Phi_fn(x_test), threshold)
        else:
            _solve = partial(_solve_dual, gcc=self, x_test=x_test, quantiles=quantiles, threshold=threshold)
            if S_min is None:
                S_min = np.min(self.scores_calib)
            if S_max is None:
                S_max = np.max(self.scores_calib)
            lower, upper = binary_search(_solve, S_min, S_max * 3)
            if quantile < 0.5:
                score_cutoff = self._get_threshold(lower, x_test, quantiles)
            else:
                score_cutoff = self._get_threshold(upper, x_test, quantiles)
        return score_inv_fn(score_cutoff, np.atleast_2d(x_test))

    def _get_dual_solution(self, S: float, x: np.ndarray, quantiles: np.ndarray):
        if self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']):
            prob = finish_dual_setup(self.cvx_problem, S, x, quantiles[-1][0], self.Phi_fn(x), self.x_calib, self.infinite_params)
            solve_cvxpy_with_fallback(prob, prefer='MOSEK', warm_start=True, abort_on_prefer_failure=False)
            return prob.var_dict['weights'].value
        S = np.concatenate([self.scores_calib, [S]])
        Phi = np.concatenate([self.phi_calib, self.Phi_fn(x)], axis=0)
        zeros = np.zeros((Phi.shape[1],))
        bounds = np.concatenate((quantiles - 1, quantiles), axis=1)
        res = linprog(-1 * S, A_eq=Phi.T, b_eq=zeros, bounds=bounds, method='highs-ds', options={'presolve': False})
        return res.x

    def _get_primal_solution(self, S: float, x: np.ndarray, quantiles: np.ndarray):
        kernel = self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel'])
        if kernel == 'finite_ridge':
            prob = finish_dual_setup(self.cvx_problem, S, x, quantiles[-1][0], self.Phi_fn(x), self.x_calib, self.infinite_params)
            solve_cvxpy_with_fallback(prob, prefer='MOSEK', warm_start=True, abort_on_prefer_failure=False)
            eta = prob.var_dict['weights'].value.reshape(-1, 1)
            free_mask = getattr(prob, '_phi_free_mask', None)
            free_count = int(np.count_nonzero(free_mask)) if isinstance(free_mask, np.ndarray) else 0
            eq_constraint = getattr(prob, '_phi_eq_constraint', None)
            beta_free = np.zeros((free_count, 1), dtype=float)
            if eq_constraint is not None and eq_constraint.dual_value is not None:
                beta_free = np.asarray(eq_constraint.dual_value).reshape(-1, 1)
            return (beta_free, eta)
        elif kernel is None:
            S_vec = np.concatenate([self.scores_calib, [S]])
            Phi = np.concatenate([self.phi_calib, self.Phi_fn(x)], axis=0)
            zeros = np.zeros((Phi.shape[1],))
            bounds = np.concatenate((quantiles - 1, quantiles), axis=1)
            res = linprog(-1 * S_vec, A_eq=Phi.T, b_eq=zeros, bounds=bounds, method='highs-ds', options={'presolve': False})
            beta = (-1 * res.eqlin.marginals).reshape(-1, 1)
            weights = None
            return (beta, weights)
        else:
            prob = finish_dual_setup(self.cvx_problem, S, x, quantiles[-1][0], self.Phi_fn(x), self.x_calib, self.infinite_params)
            solve_cvxpy_with_fallback(prob, prefer='MOSEK', warm_start=True, abort_on_prefer_failure=False)
            weights = prob.var_dict['weights'].value
            eq_constraint = getattr(prob, '_phi_eq_constraint', None)
            beta = np.zeros((self.phi_calib.shape[1], 1), dtype=float)
            if eq_constraint is not None and eq_constraint.dual_value is not None:
                beta = np.asarray(eq_constraint.dual_value).reshape(-1, 1)
            return (beta, weights)

    def _get_threshold(self, S: float, x: np.ndarray, quantiles: np.ndarray):
        beta, weights = self._get_primal_solution(S, x, quantiles)
        phi_x = self.Phi_fn(x)
        kernel = self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel'])
        if kernel == 'finite_ridge':
            n, p = self.phi_calib.shape
            lam = float(self.infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda']))
            intercept_raw = self.infinite_params.get('intercept_col', p - 1)
            intercept_col = None if intercept_raw is None else int(intercept_raw)
            d_mask = int(self.infinite_params.get('d_mask', 0))
            penalize_mask = bool(self.infinite_params.get('penalize_mask', False))
            J_pen, J_free = build_penalty_partition(p=p, d_mask=d_mask, intercept_col=intercept_col, penalize_mask=penalize_mask)
            phi_free_x = phi_x[:, J_free]
            beta_free = beta
            part_free = float(phi_free_x @ beta_free)
            Z_cal = self.phi_calib[:, J_pen]
            Z_test = phi_x[:, J_pen]
            Z = np.vstack([Z_cal, Z_test])
            eta = weights.reshape(-1, 1)
            beta_pen = Z.T @ eta / lam / (n + 1)
            part_pen = float(Z_test @ beta_pen)
            threshold = part_free + part_pen
            return threshold
        elif kernel is None:
            return float(phi_x @ beta)
        else:
            K = pairwise_kernels(X=np.concatenate([self.x_calib, x.reshape(1, -1)], axis=0), Y=np.concatenate([self.x_calib, x.reshape(1, -1)], axis=0), metric=self.infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel']), gamma=self.infinite_params.get('gamma', FUNCTION_DEFAULTS['gamma']))
            return float((K @ weights)[-1] + phi_x @ beta)

def binary_search(func, min, max, tol=0.001):
    min, max = (float(min), float(max))
    assert max + tol > max
    while max - min > tol:
        mid = (min + max) / 2
        if func(mid) > 0:
            max = mid
        else:
            min = mid
    return (min, max)

def _solve_dual(S, gcc, x_test, quantiles, threshold=None):
    q_test = float(np.asarray(quantiles[-1]).reshape(-1)[0])
    if threshold is None:
        threshold = q_test if q_test >= 0.5 else q_test - 1.0
    kernel = gcc.infinite_params.get('kernel', None)
    if kernel is None:
        S_vec = np.concatenate([gcc.scores_calib, [float(S)]]).astype(float)
        Phi = np.concatenate([gcc.phi_calib, gcc.Phi_fn(x_test)], axis=0).astype(float)
        zeros = np.zeros(Phi.shape[1])
        bounds = np.c_[quantiles - 1.0, quantiles]
        res = linprog(-S_vec, A_eq=Phi.T, b_eq=zeros, bounds=bounds, method='highs', options={'presolve': False})
        eta = res.x
    else:
        prob = finish_dual_setup(gcc.cvx_problem, float(S), x_test, q_test, gcc.Phi_fn(x_test), gcc.x_calib, gcc.infinite_params)
        solve_cvxpy_with_fallback(prob, prefer='MOSEK', warm_start=True, abort_on_prefer_failure=False)
        eta = np.asarray(prob.var_dict['weights'].value).reshape(-1)
    return float(eta[-1] - float(threshold))

def setup_cvx_problem(x_calib, scores_calib, phi_calib, infinite_params={}):
    n_calib = len(scores_calib)
    if phi_calib is None:
        phi_calib = np.ones((n_calib, 1))
    eta = cp.Variable(name='weights', shape=n_calib + 1)
    quantile = cp.Parameter(name='quantile')
    scores_const = cp.Constant(scores_calib.reshape(-1, 1))
    scores_param = cp.Parameter(name='S_test', shape=(1, 1))
    scores = cp.vstack([scores_const, scores_param])
    Phi_calibration = cp.Constant(phi_calib)
    Phi_test = cp.Parameter(name='Phi_test', shape=(1, phi_calib.shape[1]))
    Phi = cp.vstack([Phi_calibration, Phi_test])
    kernel = infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel'])
    gamma = infinite_params.get('gamma', FUNCTION_DEFAULTS['gamma'])
    prob = None
    if kernel is None:
        constraints = [quantile - 1 <= eta, quantile >= eta, eta.T @ Phi == 0]
        prob = cp.Problem(cp.Minimize(-1 * cp.sum(cp.multiply(eta, cp.vec(scores)))), constraints)
        prob._phi_eq_constraint = constraints[2]
        prob._phi_free_mask = np.ones(phi_calib.shape[1], dtype=bool)
        prob._phi_pen_mask = np.zeros(phi_calib.shape[1], dtype=bool)
    elif kernel == 'finite_ridge':
        n, p = phi_calib.shape
        lam = float(infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda']))
        intercept_raw = infinite_params.get('intercept_col', p - 1)
        intercept_col = None if intercept_raw is None else int(intercept_raw)
        d_mask = int(infinite_params.get('d_mask', 0))
        penalize_mask = bool(infinite_params.get('penalize_mask', False))
        J_pen, J_free = build_penalty_partition(p=p, d_mask=d_mask, intercept_col=intercept_col, penalize_mask=penalize_mask)
        idx_pen = np.where(J_pen)[0]
        idx_free = np.where(J_free)[0]
        Phi_calibration = cp.Constant(phi_calib)
        Z_cal = Phi_calibration[:, idx_pen]
        Phi_free_cal = Phi_calibration[:, idx_free]
        Z_test = Phi_test[:, idx_pen]
        Phi_free_test = Phi_test[:, idx_free]
        Z = cp.vstack([Z_cal, Z_test])
        Phi_free = cp.vstack([Phi_free_cal, Phi_free_test])
        obj = 1.0 / (2.0 * lam) / (n + 1) * cp.sum_squares(Z.T @ eta) - cp.sum(cp.multiply(eta, cp.vec(scores)))
        cons = [eta >= quantile - 1.0, eta <= quantile]
        eq_constraint = None
        if J_free.any():
            eq_constraint = eta.T @ Phi_free == 0
            cons.append(eq_constraint)
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob._phi_eq_constraint = eq_constraint
        prob._phi_free_mask = J_free
        prob._phi_pen_mask = J_pen
    else:
        radius = cp.Parameter(name='radius', nonneg=True)
        _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
        L_11_const = cp.Constant(np.hstack([L_11, np.zeros((L_11.shape[0], 1))]))
        L_21_22_param = cp.Parameter(name='L_21_22', shape=(1, n_calib + 1))
        L = cp.vstack([L_11_const, L_21_22_param])
        C = radius / (n_calib + 1)
        constraints = [quantile - 1 <= eta, quantile >= eta, eta.T @ Phi == 0]
        prob = cp.Problem(cp.Minimize(0.5 * C * cp.sum_squares(L.T @ eta) - cp.sum(cp.multiply(eta, cp.vec(scores)))), constraints)
        prob._phi_eq_constraint = constraints[2]
        prob._phi_free_mask = np.ones(phi_calib.shape[1], dtype=bool)
        prob._phi_pen_mask = np.zeros(phi_calib.shape[1], dtype=bool)
    return prob

def _get_kernel_matrix(x_calib, kernel, gamma):
    K = pairwise_kernels(X=x_calib, metric=kernel, gamma=gamma) + 1e-05 * np.eye(len(x_calib))
    K_chol = np.linalg.cholesky(K)
    return (K, K_chol)

def finish_dual_setup(prob: cp.Problem, S: np.ndarray, X: np.ndarray, quantile: float, Phi: np.ndarray, x_calib: np.ndarray, infinite_params: dict={}):
    pd = prob.param_dict
    pd['S_test'].value = np.asarray([[float(S)]])
    pd['Phi_test'].value = Phi.reshape(1, -1)
    pd['quantile'].value = float(quantile)
    kernel = infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel'])
    gamma = infinite_params.get('gamma', FUNCTION_DEFAULTS['gamma'])
    if kernel is None or kernel == 'finite_ridge':
        return prob
    radius = 1.0 / float(infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda']))
    if 'K_12' in pd:
        K_12 = pairwise_kernels(X=np.concatenate([x_calib, X.reshape(1, -1)], axis=0), Y=X.reshape(1, -1), metric=kernel, gamma=gamma)
        pd['K_12'].value = K_12[:-1]
        pd['K_21'].value = K_12.T
    _, L_11 = _get_kernel_matrix(x_calib, kernel, gamma)
    K_12 = pairwise_kernels(X=np.concatenate([x_calib, X.reshape(1, -1)], axis=0), Y=X.reshape(1, -1), metric=kernel, gamma=gamma)
    K_22 = pairwise_kernels(X=X.reshape(1, -1), metric=kernel, gamma=gamma)
    L_21 = np.linalg.solve(L_11, K_12[:-1]).T
    L_22 = K_22 - L_21 @ L_21.T
    L_22[L_22 < 0] = 0
    L_22 = np.sqrt(L_22)
    pd['L_21_22'].value = np.hstack([L_21, L_22])
    pd['radius'].value = radius
    pd['quantile'].value = float(quantile)
    return prob

def setup_cvx_problem_calib(quantile, x_calib, scores_calib, phi_calib, infinite_params={}, penalty_weights=None):
    n_calib = len(scores_calib)
    if phi_calib is None:
        phi_calib = np.ones((n_calib, 1))
    eta = cp.Variable(name='weights', shape=n_calib)
    scores = cp.Constant(scores_calib.reshape(-1, 1))
    Phi = cp.Constant(phi_calib)
    kernel = infinite_params.get('kernel', FUNCTION_DEFAULTS['kernel'])
    gamma = infinite_params.get('gamma', FUNCTION_DEFAULTS['gamma'])
    if kernel is None:
        constraints = [quantile - 1 <= eta, quantile >= eta, eta.T @ Phi == 0]
        eq_free = constraints[2]
        prob = cp.Problem(cp.Minimize(-1 * cp.sum(cp.multiply(eta, cp.vec(scores)))), constraints)
        prob._phi_eq_constraint = eq_free
        prob._phi_free_mask = np.ones(phi_calib.shape[1], dtype=bool)
        prob._phi_pen_mask = np.zeros(phi_calib.shape[1], dtype=bool)
    elif kernel == 'finite_ridge':
        n, p = phi_calib.shape
        lam = float(infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda']))
        intercept_raw = infinite_params.get('intercept_col', p - 1)
        intercept_col = None if intercept_raw is None else int(intercept_raw)
        d_mask = int(infinite_params.get('d_mask', 0))
        penalize_mask = bool(infinite_params.get('penalize_mask', False))
        n, p = phi_calib.shape
        n_calib = len(scores_calib)
        scores = cp.Constant(scores_calib.reshape(-1, 1))
        Phi = np.asarray(phi_calib, float)
        J_pen, J_free = build_penalty_partition(p=p, d_mask=d_mask, intercept_col=intercept_col, penalize_mask=penalize_mask)
        Z = Phi[:, J_pen]
        Phi_free = Phi[:, J_free]
        eta = cp.Variable(name='weights', shape=n_calib)
        if Z.size:
            ridge_term = 1.0 / (2.0 * lam) / (n + 1) * cp.sum_squares(Z.T @ eta)
        else:
            ridge_term = cp.Constant(0.0)
        obj = ridge_term - cp.sum(cp.multiply(eta, cp.vec(scores)))
        cons = [eta >= quantile - 1.0, eta <= quantile]
        eq_free = None
        if J_free.any():
            eq_free = Phi_free.T @ eta == 0
            cons.append(eq_free)
        prob = cp.Problem(cp.Minimize(obj), cons)
        prob._phi_eq_constraint = eq_free
        prob._phi_free_mask = J_free
        prob._phi_pen_mask = J_pen
    else:
        radius = 1 / infinite_params.get('lambda', FUNCTION_DEFAULTS['lambda'])
        _, L = _get_kernel_matrix(x_calib, kernel, gamma)
        C = radius / (n_calib + 1)
        constraints = [quantile - 1 <= eta, quantile >= eta, eta.T @ Phi == 0]
        eq_free = constraints[2]
        prob = cp.Problem(cp.Minimize(0.5 * C * cp.sum_squares(L.T @ eta) - cp.sum(cp.multiply(eta, cp.vec(scores)))), constraints)
        prob._phi_eq_constraint = eq_free
        prob._phi_free_mask = np.ones(phi_calib.shape[1], dtype=bool)
        prob._phi_pen_mask = np.zeros(phi_calib.shape[1], dtype=bool)
    if prob is not None and (not hasattr(prob, '_phi_eq_constraint')):
        prob._phi_eq_constraint = None
        prob._phi_free_mask = np.ones(phi_calib.shape[1], dtype=bool)
        prob._phi_pen_mask = np.zeros(phi_calib.shape[1], dtype=bool)
    return prob
