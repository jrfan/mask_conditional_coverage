import json
import os
from pathlib import Path

import cvxpy as cp
import numpy as np
from sklearn.model_selection import RepeatedKFold


class NumericalFailure(RuntimeError):
    pass


def _solve(problem, stable=False, form="MSK_SOLVE_PRIMAL"):
    if "MOSEK" not in cp.installed_solvers():
        raise RuntimeError("MOSEK and a valid license are required.")
    params = {"MSK_IPAR_NUM_THREADS": int(os.environ.get("MOSEK_NUM_THREADS", os.environ.get("EXPERIMENT_THREADS", "1")))}
    if stable:
        params.update(MSK_IPAR_INTPNT_SOLVE_FORM=form,
                      MSK_DPAR_INTPNT_CO_TOL_PFEAS=1e-10,
                      MSK_DPAR_INTPNT_CO_TOL_DFEAS=1e-10,
                      MSK_DPAR_INTPNT_CO_TOL_REL_GAP=1e-10)
    try:
        problem.solve(solver="MOSEK", warm_start=False, mosek_params=params)
    except cp.error.SolverError as exc:
        message = str(exc).lower()
        if any(word in message for word in ("license", "licence", "not installed", "out of memory", "cannot allocate", "dll", "shared library")):
            raise
        raise NumericalFailure(str(exc)) from exc
    if problem.status != cp.OPTIMAL:
        raise NumericalFailure(f"Solver status: {problem.status}")


def certificate(phi, scores, beta, eta, lam, alpha, penalized):
    a = np.asarray(phi, dtype=np.longdouble)
    s = np.asarray(scores, dtype=np.longdouble).reshape(-1)
    b = np.asarray(beta, dtype=np.longdouble).reshape(-1)
    w = np.asarray(eta, dtype=np.longdouble).reshape(-1)
    if b.size != a.shape[1] or w.size != a.shape[0]:
        raise NumericalFailure("Invalid solution dimensions")
    if not np.isfinite(b).all() or not np.isfinite(w).all():
        raise NumericalFailure("Nonfinite solution")
    residual = s-a@b
    scale = np.longdouble(lam)*(len(s)+1)
    primal = np.sum((1-np.longdouble(alpha))*np.maximum(residual, 0)+np.longdouble(alpha)*np.maximum(-residual, 0))
    primal += scale/2*(b[penalized]@b[penalized])
    moment = a[:, penalized].T@w
    dual = s@w-(moment@moment)/(2*scale)
    equality = a[:, ~penalized].T@w
    gap = primal-dual
    return dict(primal=float(primal), dual=float(dual), gap=float(gap),
                relative_gap=float(abs(gap)/max(1, abs(primal), abs(dual))),
                equality_residual=float(np.max(abs(equality))) if equality.size else 0.,
                box_violation=float(max(0, np.max(w-(1-alpha)), np.max(-alpha-w))))


def _accepted(report):
    return (all(np.isfinite(v) for v in report.values()) and report["gap"] >= -1e-7
            and report["relative_gap"] <= 1e-6
            and report["equality_residual"] <= 1e-7 and report["box_violation"] <= 1e-7)


def _repair_witness(phi, eta, alpha, penalized):
    a = np.asarray(phi, dtype=np.longdouble)
    w = np.asarray(eta, dtype=np.longdouble).copy()
    w = np.clip(w, -np.longdouble(alpha), 1-np.longdouble(alpha))*(1-np.longdouble("1e-9"))
    huge = penalized & (np.max(abs(a), axis=0) > 1e6)
    categorical = all(len(np.unique(a[:, j])) <= 2 for j in np.flatnonzero(huge))
    if categorical and (huge.any() or (~penalized).any()):
        _, membership = np.unique(a[:, huge], axis=0, return_inverse=True)
        groups = [np.flatnonzero(membership == i) for i in np.unique(membership)]
        if all(np.all(a[ids][:, ~penalized] == a[ids[0], ~penalized]) for ids in groups):
            for ids in groups:
                for _ in range(2):
                    residual = w[ids].sum(dtype=np.longdouble)
                    room = w[ids]+alpha if residual > 0 else 1-alpha-w[ids]
                    w[ids[np.argmax(room)]] -= residual
            return w
    if (~penalized).any():
        free = np.asarray(a[:, ~penalized], dtype=float)
        free = free/np.maximum(1, np.max(abs(free), axis=0))
        variable = cp.Variable(len(w))
        problem = cp.Problem(cp.Minimize(cp.sum_squares(variable-np.asarray(w, dtype=float))),
                             [free.T@variable == 0, variable >= -alpha, variable <= 1-alpha])
        _solve(problem, stable=True)
        if variable.value is None:
            raise NumericalFailure("Missing dual certificate")
        w = np.asarray(variable.value, dtype=np.longdouble)
    return w


def primary_solve(phi, scores, lam, alpha, penalized, d_mask, intercept_col, penalize_mask):
    from conditionalconformal.condconf import setup_cvx_problem_calib
    problem = setup_cvx_problem_calib(quantile=1-alpha, x_calib=phi,
        scores_calib=scores, phi_calib=phi,
        infinite_params={"kernel":"finite_ridge", "lambda":float(lam),
                         "d_mask":d_mask, "intercept_col":intercept_col,
                         "penalize_mask":penalize_mask})
    _solve(problem)
    eta = problem.var_dict["weights"].value
    if eta is None:
        raise NumericalFailure("Missing dual solution")
    beta = np.zeros(phi.shape[1])
    beta[penalized] = phi[:, penalized].T@eta/(lam*(len(scores)+1))
    equality = getattr(problem, "_phi_eq_constraint", None)
    if (~penalized).any():
        if equality is None or equality.dual_value is None:
            raise NumericalFailure("Missing unpenalized coefficients")
        values = np.asarray(equality.dual_value).ravel()
        if len(values) != (~penalized).sum():
            raise NumericalFailure("Invalid unpenalized coefficients")
        beta[~penalized] = values
    report = certificate(phi, scores, beta, eta, lam, alpha, penalized)
    if not _accepted(report):
        raise NumericalFailure(f"Original-coordinate certificate failed: {report}")
    return beta, report


def stable_solve(phi, scores, lam, alpha, penalized):
    n, p = phi.shape
    scale = np.maximum(1., np.max(abs(phi), axis=0))
    errors = []
    for form in ("MSK_SOLVE_PRIMAL", "MSK_SOLVE_DUAL"):
        try:
            theta = cp.Variable(p)
            pos, neg = cp.Variable(n, nonneg=True), cp.Variable(n, nonneg=True)
            equality = scores-(phi/scale)@theta == pos-neg
            objective = cp.sum((1-alpha)*pos+alpha*neg)
            if penalized.any():
                objective += lam*(n+1)/2*cp.sum_squares(cp.multiply(1/scale[penalized], theta[penalized]))
            inactive = penalized & (np.max(abs(phi), axis=0) == 0)
            constraints = [equality]
            if inactive.any():
                constraints.append(theta[inactive] == 0)
            problem = cp.Problem(cp.Minimize(objective), constraints)
            _solve(problem, stable=True, form=form)
            if theta.value is None or equality.dual_value is None:
                raise NumericalFailure("Missing scaled-primal solution")
            beta = np.asarray(theta.value).ravel()/scale
            beta[inactive] = 0.
            eta = np.asarray(equality.dual_value).ravel()
            report = certificate(phi, scores, beta, eta, lam, alpha, penalized)
            if not _accepted(report):
                eta = _repair_witness(phi, eta, alpha, penalized)
                report = certificate(phi, scores, beta, eta, lam, alpha, penalized)
            if not _accepted(report):
                raise NumericalFailure(f"Scaled-primal certificate failed: {report}")
            return beta, dict(report, form=form)
        except NumericalFailure as exc:
            errors.append(str(exc))
    raise NumericalFailure("Both stable formulations failed: "+"; ".join(errors))


def _write_report(record):
    name = os.environ.get("NUMERICAL_REPORT_PATH")
    if name:
        with Path(name).open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, allow_nan=False)+"\n")


def runCV_ridge_v2(XDev, scoresDev, kernel, gamma, alpha, k,
                   minRad, maxRad, numRad, phiDev, d_mask, intercept_col=None,
                   penalize_mask=False, n_repeats=5):
    from conditionalconformal.condconf import build_penalty_partition
    phi = np.asarray(phiDev, dtype=float)
    scores = np.asarray(scoresDev, dtype=float).reshape(-1)
    if kernel != "finite_ridge":
        raise ValueError("This CV routine supports finite_ridge only")
    if phi.ndim != 2 or phi.shape[0] != len(scores) or len(XDev) != len(scores) or phi.shape[1] < 1:
        raise ValueError("Inconsistent CV input dimensions")
    if not np.isfinite(phi).all() or not np.isfinite(scores).all():
        raise ValueError("CV features and scores must be finite")
    if not 0 < alpha < 1 or not 0 < minRad <= maxRad or numRad < 1:
        raise ValueError("Invalid CV parameters")
    if not 0 <= d_mask <= phi.shape[1] or (intercept_col is not None and not 0 <= intercept_col < phi.shape[1]):
        raise ValueError("Invalid penalty layout")
    penalized, _ = build_penalty_partition(phi.shape[1], d_mask, intercept_col, penalize_mask)
    folds = list(RepeatedKFold(n_splits=k, n_repeats=n_repeats, random_state=42).split(XDev))
    lambdas = np.logspace(np.log10(minRad), np.log10(maxRad), numRad)
    attempts = []
    for route in ("original", "scaled_primal"):
        losses = np.empty((len(lambdas), len(folds)))
        certificates = []
        try:
            for i, lam in enumerate(lambdas):
                for j, (train, valid) in enumerate(folds):
                    if route == "original":
                        beta, report = primary_solve(phi[train], scores[train], lam, alpha, penalized,
                                                     d_mask, intercept_col, penalize_mask)
                    else:
                        beta, report = stable_solve(phi[train], scores[train], lam, alpha, penalized)
                    if route == "original":
                        prediction = phi[valid][:, penalized]@beta[penalized] + phi[valid][:, ~penalized]@beta[~penalized]
                    else:
                        prediction = phi[valid]@beta
                    if not np.isfinite(prediction).all():
                        raise NumericalFailure("Nonfinite validation prediction")
                    residual = scores[valid]-prediction
                    losses[i, j] = np.mean(.5*abs(residual)+(1-alpha-.5)*residual)
                    if not np.isfinite(losses[i, j]):
                        raise NumericalFailure("Nonfinite validation loss")
                    certificates.append(report)
            means = losses.mean(axis=1)
            ses = losses.std(axis=1, ddof=1)/np.sqrt(len(folds))
            _write_report(dict(status="completed", route=route, attempts=attempts,
                solves=int(losses.size), lambdas=lambdas.tolist(), means=means.tolist(), ses=ses.tolist(),
                solver_forms={form: sum(x.get("form", "original") == form for x in certificates)
                              for form in sorted({x.get("form", "original") for x in certificates})},
                maximum_relative_gap=max(x["relative_gap"] for x in certificates),
                maximum_equality_residual=max(x["equality_residual"] for x in certificates),
                maximum_box_violation=max(x["box_violation"] for x in certificates)))
            return means, ses, lambdas
        except NumericalFailure as exc:
            attempts.append(dict(route=route, lambda_index=i, fold_index=j,
                                 completed_solves=len(certificates), reason=str(exc)))
            if route == "scaled_primal":
                _write_report(dict(status="failed", attempts=attempts))
                raise
            _write_report(dict(status="recovering", attempts=attempts))
            print("Numerical recovery: restarting the complete CV grid with an equivalent scaled formulation.", flush=True)
