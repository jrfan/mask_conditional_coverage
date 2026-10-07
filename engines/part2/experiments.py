import os
import json
import pandas as pd
import numpy as np
import optuna
from scipy.stats import norm
from scipy.optimize import linprog
from collections import defaultdict
from functools import partial
from models import fit_basemodel, make_score_fns
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from utils import pinball_loss, evaluate_model, get_tqdm, make_phi_fn_with_norm_new
from utils import oracle_len_pattern, create_patterns
from crossval import runCV_ridge_v2
from conditionalconformal import CondConf
from conditionalconformal.condconf import setup_cvx_problem_calib, build_penalty_partition, solve_cvxpy_with_fallback
from threadpoolctl import threadpool_limits
import hashlib
from models import get_quantile_adapter

def training_validation_indices(n_samples, replication_idx, val_fraction=0.2):
    if n_samples < 2:
        raise ValueError('At least two training-score samples are required for tuning.')
    n_val = min(n_samples - 1, max(1, int(round(val_fraction * n_samples))))
    permutation = np.random.default_rng(42 + replication_idx).permutation(n_samples)
    return (permutation[n_val:], permutation[:n_val])

def replication_slice(values, replication_idx):
    values = np.asarray(values)
    if values.ndim <= 1:
        return values
    return values[replication_idx]

class ConformalExperiment:

    def __init__(self, args, filename, data, output_dir='./results'):
        self.args = args
        self.data = data
        self.output_dir = output_dir
        self.one_hot_encode_batch, self.mask_dim = get_one_hot_encoder(single_M=args.single_M, dim=args.dim)
        os.makedirs(output_dir, exist_ok=True)
        self.file_name = filename
        self.out_file = os.path.join(output_dir, self.file_name + '.csv')
        self.out_file_pkl = os.path.join(output_dir, self.file_name + '.pkl')
        self.out_file_phi_pkl = os.path.join(output_dir, self.file_name + '_phi_err.pkl')
        self.model_type = args.model_type
        self.all_result = defaultdict(list)
        self.all_results = []
        self.all_betas = []
        self.m_stats_accum = []
        self.phi_err_history = []
        self._phi_err_pool = defaultdict(lambda: {'ss_rho': 0.0, 'n': 0, 'pi_num': 0, 'pi_den': 0})
        self.data_name_hash = self.generate_unique_code(self.data.unique_name)

    @staticmethod
    def generate_unique_code(data):
        data = str(data)
        encoded_data = data.encode('utf-8')
        hash_object = hashlib.md5(encoded_data)
        hex_digest = hash_object.hexdigest()
        return hex_digest

    def tune_histgbdt(self, x_fit, y_fit, x_val, y_val, alpha, n_trials=30):
        print(f'[tune_histgbdt] starting (n_trials={n_trials})...')
        with threadpool_limits(limits=int(os.environ.get('EXPERIMENT_THREADS', '1'))):

            def objective(trial):
                params = {'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True), 'max_iter': trial.suggest_int('max_iter', 50, 400), 'max_depth': trial.suggest_int('max_depth', 2, 6), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 80), 'l2_regularization': trial.suggest_float('l2_regularization', 1e-06, 10.0, log=True), 'max_bins': trial.suggest_int('max_bins', 128, 255), 'random_state': 42}
                model = fit_basemodel(x_fit, y_fit, target='Quantiles', basemodel='histgbdt', alpha=alpha, params_basemodel=params)
                loss = evaluate_model(model, x_val, y_val, alpha)
                return loss

            def _log_progress(study, trial):
                if (trial.number + 1) % 10 == 0 or trial.number == 0:
                    print(f'  [tune_histgbdt] trial {trial.number + 1}/{n_trials}')
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials, callbacks=[_log_progress])
            best_params = study.best_params
            best_model = fit_basemodel(x_fit, y_fit, target=self.model_type, basemodel='histgbdt', alpha=alpha, params_basemodel=best_params)
        print('[tune_histgbdt] done.')
        return (best_model, best_params, study)

    def _update_phi_err_pool_from_phi(self, Phi, mask_keys, alpha=0.1, ridge=1e-06):
        Phi = np.asarray(Phi, dtype=float)
        n, p = Phi.shape
        G = Phi.T @ Phi
        G.flat[::p + 1] += ridge
        Ginv = np.linalg.pinv(G, rcond=1e-10)
        PhiT = Phi.T
        mk_arr = np.asarray(mask_keys, dtype=object)
        records = []
        c_alpha = max(alpha, 1 - alpha)
        n_total = int(n)
        uniq, cnts = np.unique(mk_arr, return_counts=True)
        for key, c in zip(uniq, cnts):
            y = (mk_arr == key).astype(float)
            beta = Ginv @ (PhiT @ y)
            resid = y - Phi @ beta
            ss = float(np.sum(resid ** 2))
            slot = self._phi_err_pool[key]
            slot['ss_rho'] += ss
            slot['n'] += n
            slot['pi_num'] += int(c)
            slot['pi_den'] += n
            pi_hat = float(c) / max(n, 1e-12)
            Delta_rho = float(np.sqrt(ss / max(n, 1e-12)))
            lb_rho = max(0.0, min(1.0, 1.0 - alpha - c_alpha * Delta_rho))
            records.append({'Mask': key, 'ss_rho': ss, 'n': n_total, 'pi_num': int(c), 'pi_den': n_total, 'pi_hat': pi_hat, 'Delta_rho_phi': Delta_rho, 'LB_rho_only_phi': lb_rho})
        return pd.DataFrame(records)

    def _update_phi_err_pool(self, X_all, M_all, phi_fn, alpha=0.1, ridge=1e-06):
        Phi = phi_fn(X_all)
        if Phi.ndim != 2:
            Phi = np.atleast_2d(Phi)
        n = Phi.shape[0]
        Phi_aug = np.concatenate([Phi, np.ones((n, 1))], axis=1)
        p_aug = Phi_aug.shape[1]
        XtX = Phi_aug.T @ Phi_aug
        XtX_reg = XtX + ridge * np.eye(p_aug)
        XtX_reg_inv = np.linalg.pinv(XtX_reg, rcond=1e-08)
        Xty_op = XtX_reg_inv @ Phi_aug.T
        keys = [''.join(map(str, map(int, r))) for r in M_all]
        uniq, cnts = np.unique(keys, return_counts=True)
        count_map = {u: int(c) for u, c in zip(uniq, cnts)}
        keys_arr = np.array(keys)
        for key in uniq:
            y = (keys_arr == key).astype(float)
            beta = Xty_op @ y
            resid = y - Phi_aug @ beta
            ss = float(np.sum(resid ** 2))
            slot = self._phi_err_pool[key]
            slot['ss_rho'] += ss
            slot['n'] += n
            slot['pi_num'] += count_map[key]
            slot['pi_den'] += n

    def run(self):
        args = self.args
        data = self.data
        import torch
        torch.set_num_threads(int(os.environ.get('OMP_NUM_THREADS', '1')))
        alpha = self.args.alpha
        exact = False
        replication_index = os.environ.get('REPLICATION_INDEX')
        replication_start = os.environ.get('REPLICATION_START')
        if replication_index is not None and replication_start is not None:
            raise ValueError('Set only one of REPLICATION_INDEX and REPLICATION_START.')
        if replication_index is not None:
            k_only = int(replication_index)
            if not 0 <= k_only < args.reps:
                raise ValueError(f'REPLICATION_INDEX must be in [0, {args.reps - 1}], got {k_only}.')
            replication_indices = [k_only]
        elif replication_start is not None:
            k_start = int(replication_start)
            if not 0 <= k_start < args.reps:
                raise ValueError(f'REPLICATION_START must be in [0, {args.reps - 1}], got {k_start}.')
            if args.split_train < 1.0:
                for skipped_k in range(k_start):
                    skipped_train = replication_slice(data['X_missing']['Train'], skipped_k)
                    np.random.permutation(skipped_train.shape[0])
                print(f'Advanced the training-split RNG for {k_start} skipped replications.')
            replication_indices = range(k_start, args.reps)
        else:
            replication_indices = range(args.reps)
        for k in replication_indices:
            print(f'\n===== Replication {k + 1}/{args.reps} =====')
            x_train = replication_slice(data['X_missing']['Train'], k)
            y_train = replication_slice(data['Y']['Train'], k)
            x_cal = replication_slice(data['X_missing']['Cal'], k)
            y_cal = replication_slice(data['Y']['Cal'], k)
            x_test = replication_slice(data['X_missing']['Test']['fixed_nb_sample_pattern'], k)
            y_test = replication_slice(data['Y']['Test']['fixed_nb_sample_pattern'], k)
            m_cal = replication_slice(data['M']['Cal'], k)
            m_test = replication_slice(data['M']['Test']['fixed_nb_sample_pattern'], k)
            m_train = replication_slice(data['M']['Train'], k)
            if args.split_train < 1.0:
                n_total = x_train.shape[0]
                n_sub = int(args.split_train * n_total)
                idx = np.random.permutation(n_total)
                train_idx, val_idx = (idx[:n_sub], idx[n_sub:])
                x_base, y_base = (x_train[train_idx], y_train[train_idx])
                x_score, y_score = (x_train[val_idx], y_train[val_idx])
            else:
                x_base, y_base = (x_train, y_train)
                x_score, y_score = (x_train, y_train)
            print('Tuning protocol: training_only_v1')
            print('Calibration responses are used only after all hyperparameters are frozen.')
            reg, base_best_params, _ = self.tune_histgbdt(x_base, y_base, x_score, y_score, alpha=alpha, n_trials=args.tuning_trials)
            score_fn, score_inv_lb, score_inv_ub = make_score_fns(reg, model_type=self.model_type, alpha=alpha)
            scores_train = score_fn(x_score, y_score)
            embed_fit_idx, embed_val_idx = training_validation_indices(len(x_score), replication_idx=k)
            _xs = np.asarray(x_score, dtype=np.float64)
            X_embed_fit = _xs
            embed_type = getattr(args, 'embed_type', 'tree')
            if embed_type == 'tree':
                if args.n_treespace > 0:
                    if not args.trained_on_train:
                        raise ValueError('trained_on_train=False is incompatible with the calibration-clean theoretical pipeline.')
                    q_model_name = args.q_model_name
                    suggest, fit_once, pred_fn = get_quantile_adapter(q_model_name, quantile_alpha=1 - alpha)

                    def objective(trial, x1, y1, x2, y2):
                        params = suggest(trial)
                        nrounds = args.n_treespace if args.n_treespace_max == -1 else args.n_treespace_max
                        model = fit_once(x1, y1, params, nrounds=nrounds)
                        pred = pred_fn(model, x2)
                        return float(np.mean(pinball_loss(y2, pred, 1 - alpha)))
                    wrapped_obj = partial(objective, x1=x_score[embed_fit_idx], y1=scores_train[embed_fit_idx], x2=x_score[embed_val_idx], y2=scores_train[embed_val_idx])
                    study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42))

                    def _log_tree_trial(study, trial):
                        if (trial.number + 1) % 10 == 0 or trial.number == 0:
                            print(f'  [tree quantile tune] trial {trial.number + 1}/100')
                    study.optimize(wrapped_obj, n_trials=args.tuning_trials, show_progress_bar=False, callbacks=[_log_tree_trial])
                    embedding_best_params = study.best_trial.params

                    def train_quantile_model(X, y, quantile):
                        _, fit_final, _ = get_quantile_adapter(q_model_name, quantile_alpha=quantile)
                        return fit_final(X, y, embedding_best_params, nrounds=args.n_treespace)
                    model_09 = train_quantile_model(x_score, scores_train, 1 - alpha)
                else:
                    model_09 = None
                print(f'[replication] building phi_fn (embed_type=tree, n_treespace={args.n_treespace})...')
                phi_fn, phi_mu, phi_std, phi_meta = make_phi_fn_with_norm_new(model=model_09, n_treespace=args.n_treespace, fit_X=x_cal, one_hot_encode_batch=self.one_hot_encode_batch, concat_mask=args.concat_vector, scale_mask=True, use_mask_patterns=args.use_mask_patterns, M_calib=m_cal, include_other_pattern=False)
            elif embed_type == 'mice':
                print('[replication] building phi_fn (embed_type=mice: mask + MICE(X), no intercept)...')
                mice_imputer = IterativeImputer(missing_values=np.nan, sample_posterior=False, max_iter=50, tol=0.01, random_state=42)
                mice_imputer.fit(X_embed_fit)
                phi_fn, phi_mu, phi_std, phi_meta = make_phi_fn_with_norm_new(model=None, n_treespace=0, fit_X=x_cal, one_hot_encode_batch=self.one_hot_encode_batch, covariate_imputer=mice_imputer, concat_mask=args.concat_vector, scale_mask=True, use_mask_patterns=args.use_mask_patterns, M_calib=m_cal, include_other_pattern=False)
            else:
                raise ValueError(f"Unsupported embed_type '{embed_type}'.")
            print('[replication] computing training representations for lambda selection...')
            phi_train = phi_fn(x_score)
            p = phi_train.shape[1]
            intercept_raw = phi_meta['intercept_col']
            intercept_col = None if intercept_raw is None else int(intercept_raw)
            d_mask = int(phi_meta.get('d_mask', args.dim))
            penalize_mask = bool(args.penalize_mask)
            if args.penalization:
                if args.ridge_lambda == -1:
                    print('Running cross-validation for ridge lambda selection...')
                    cv_means, cv_ses, lambdas = runCV_ridge_v2(XDev=x_score, scoresDev=scores_train, kernel='finite_ridge', gamma='not_used', alpha=alpha, k=5, minRad=0.0001, maxRad=1000.0, numRad=args.lambda_grid_size, phiDev=phi_train, d_mask=d_mask, intercept_col=intercept_col, penalize_mask=penalize_mask, n_repeats=args.cv_repeats)
                    idx_min = np.argmin(cv_means)
                    best_lambda = lambdas[idx_min]
                    print(f'Min Loss Lambda: {lambdas[idx_min]:.5f}')
                else:
                    best_lambda = args.ridge_lambda
                    print(f'Denoted lambda: {best_lambda:.5f}')
            else:
                best_lambda = 1
            print('[replication] all hyperparameters frozen; starting conformal calibration.')
            protocol_metadata = dict(protocol='training_only_v1', replication=k, base_best_params=base_best_params, embedding_best_params=locals().get('embedding_best_params', {}), best_lambda=float(best_lambda), normalization_split='x_cal', calibration_used_for_hyperparameter_selection=False, penalize_mask=penalize_mask, penalize_intercept=False, has_intercept=bool(phi_meta['has_intercept']), actual_sizes=dict(training_pool=len(x_train), base_fit=len(x_base), selection=len(x_score), calibration=len(x_cal), test=len(x_test)))
            with open(os.path.join(self.output_dir, 'protocol.json'), 'w') as f:
                json.dump(protocol_metadata, f, indent=2)
            scores_cal = score_fn(x_cal, y_cal)
            phi_cal = phi_fn(x_cal)
            if embed_type == 'tree' and args.n_treespace == 0:
                kernel_type = None
            elif not args.penalization:
                kernel_type = None
            else:
                kernel_type = 'finite_ridge'
            infinite_params = {'lambda': best_lambda, 'kernel': kernel_type, 'gamma': 'None', 'd_mask': d_mask, 'intercept_col': intercept_col, 'penalize_mask': penalize_mask}
            calib_quantile = 1 - alpha
            kernel = infinite_params.get('kernel', None)
            beta = None
            if kernel is None:
                S = scores_cal.reshape(-1, 1)
                Phi = phi_cal.astype(float)
                bounds = np.tile(np.asarray([calib_quantile - 1, calib_quantile]).reshape(1, -1), (len(S), 1))
                res = linprog(-1 * S, A_eq=Phi.T, b_eq=np.zeros((Phi.shape[1],)), bounds=bounds, method='highs')
                beta = (-1 * res.eqlin.marginals).reshape(-1)
            else:
                prob = setup_cvx_problem_calib(quantile=calib_quantile, x_calib=x_cal, scores_calib=scores_cal, phi_calib=phi_cal, infinite_params=infinite_params)
                solve_cvxpy_with_fallback(prob, prefer='MOSEK', warm_start=False, abort_on_prefer_failure=False)
                if kernel == 'finite_ridge':
                    lam = float(best_lambda)
                    J_pen, J_free = build_penalty_partition(p=p, d_mask=d_mask, intercept_col=intercept_col, penalize_mask=penalize_mask)
                    Z = phi_cal[:, J_pen]
                    eta = np.asarray(prob.var_dict['weights'].value).reshape(-1, 1)
                    beta_pen = Z.T @ eta / (lam * (len(Z) + 1)) if Z.size else np.zeros((J_pen.sum(), 1), dtype=float)
                    eq_constraint = getattr(prob, '_phi_eq_constraint', None)
                    beta_free = np.zeros((J_free.sum(), 1), dtype=float)
                    if eq_constraint is not None and getattr(eq_constraint, 'dual_value', None) is not None:
                        beta_free = np.asarray(eq_constraint.dual_value).reshape(-1, 1)
                        if beta_free.size != J_free.sum():
                            beta_free = beta_free[:J_free.sum()].reshape(-1, 1)
                    beta_full = np.zeros((p, 1), dtype=float)
                    beta_full[J_pen] = beta_pen
                    beta_full[J_free] = beta_free
                    beta = beta_full.reshape(-1)
                else:
                    raise ValueError(f"Unsupported kernel '{kernel}' for calibration beta.")
            self.all_betas.append(beta)
            lbs, ubs = (np.zeros(len(x_test)), np.zeros(len(x_test)))
            cond_conf = CondConf(score_fn, phi_fn, infinite_params=infinite_params)
            cond_conf.setup_problem(x_cal, y_cal)
            test_paral = int(os.environ.get('TEST_PARALLEL_JOBS', '0'))
            if test_paral > 0:
                from joblib import Parallel, delayed

                def compute_interval(x_t, model_type, cond_conf, alpha, score_inv_lb, score_inv_ub, exact):
                    res = cond_conf.predict(1 - alpha, x_t, score_inv_lb, exact=exact, randomize=False)
                    return (res[0], res[1])
                results = Parallel(n_jobs=test_paral, verbose=1, batch_size=10)((delayed(compute_interval)(x_t, self.model_type, cond_conf, alpha, score_inv_lb, score_inv_ub, exact) for x_t in x_test))
                for i, (l_val, u_val) in enumerate(results):
                    lbs[i] = l_val
                    ubs[i] = u_val
            else:
                for i, x_t in enumerate(get_tqdm(x_test)):
                    res = cond_conf.predict(1 - alpha, x_t, score_inv_lb, exact=exact, randomize=False)
                    lbs[i], ubs[i] = (res[0], res[1])
            coverage = np.mean((y_test >= lbs) & (y_test <= ubs))
            avg_len = np.mean(ubs - lbs)
            print(f'Coverage={coverage:.3f}, Length={avg_len:.3f}')
            rep_rows = self._aggregate_results(k, m_cal, m_test, lbs, ubs, y_test, m_train=m_train, phi_meta=phi_meta)
            phi_train = phi_fn(x_train)
            phi_iid = phi_fn(self.data['X_missing']['Test']['iid'][k])
            phi_stats = np.vstack([phi_train, phi_cal, phi_iid])
            m_iid = data['M']['Test']['iid'][k]
            mask_keys_stats = self._mask_rows_to_keys(np.vstack([m_train, m_cal, m_iid]))
            phi_err_rep = self._update_phi_err_pool_from_phi(phi_stats, mask_keys_stats, alpha=alpha, ridge=1e-06)
            phi_err_rep['epoch'] = k
            self.phi_err_history.append(phi_err_rep)
            mask_df = self._build_mask_df(epoch=k)
            rep_df = self._build_mask_df_from_rows(rep_rows, epoch=k, replication=k)
            df_pool = self._finalize_phi_err_pool(alpha=alpha, use_r=False)
            mask_df = mask_df.merge(df_pool, on='Mask', how='left')
            rep_df = rep_df.merge(df_pool, on='Mask', how='left')
            mask_df = self._oracle_lengths(mask_df)
            rep_df = self._oracle_lengths(rep_df)
            self._save_artifacts(mask_df, phi_mu=phi_mu, phi_std=phi_std, phi_meta=phi_meta, rep_df=rep_df, replication=k)
            print(f'Results saved to {self.out_file}, k:{k}')

    def _mask_rows_to_keys(self, M_rows):
        return [''.join(map(str, map(int, r))) for r in M_rows]

    def _ensure_phi_pool_columns(self, mask_df):
        if 'Delta_rho_phi_avg' not in mask_df.columns:
            mask_df['Delta_rho_phi_avg'] = np.nan
        if 'LB_rho_only_phi_avg' not in mask_df.columns:
            mask_df['LB_rho_only_phi_avg'] = np.nan
        return mask_df

    def _finalize_phi_err_pool(self, alpha=0.1, use_rho=True, use_r=False):
        import pandas as pd
        c_alpha = max(alpha, 1 - alpha)
        recs = []
        for key, v in self._phi_err_pool.items():
            if v['n'] <= 0:
                continue
            mse_rho = v['ss_rho'] / v['n']
            Delta_rho = float(mse_rho ** 0.5)
            out = {'Mask': key, 'Delta_rho_phi_avg': Delta_rho, 'LB_rho_only_phi_avg': max(0.0, min(1.0, 1.0 - alpha - c_alpha * Delta_rho))}
            if use_r:
                pi_hat_pool = v['pi_num'] / max(v['pi_den'], 1e-09)
                Delta_r = Delta_rho / max(pi_hat_pool, 1e-09)
                out.update({'pi_hat_phi_pool': pi_hat_pool, 'Delta_r_phi_avg': Delta_r, 'LB_r_only_phi_avg': max(0.0, min(1.0, 1.0 - alpha - c_alpha * Delta_r))})
            recs.append(out)
        return pd.DataFrame(recs)

    def _add_delta_bounds_via_projection(self, mask_df, phi_data, mask_keys, alpha=0.1, normalize='rho', ridge=1e-06):
        n, p = phi_data.shape
        Phi = np.asarray(phi_data, dtype=float)
        G = Phi.T @ Phi
        G.flat[::p + 1] += ridge
        Ginv = np.linalg.pinv(G, rcond=1e-10)
        PhiT = Phi.T
        recs = []
        c_alpha = max(alpha, 1 - alpha)
        unique_keys = mask_df['Mask'].values.tolist()
        mk_arr = np.array(mask_keys, dtype=object)
        for key in unique_keys:
            y = (mk_arr == key).astype(float)
            pi_hat = float(y.mean())
            if pi_hat <= 0.0:
                recs.append({'Mask': key, 'pi_hat_proj': np.nan, 'Delta_phi': np.nan, 'Delta_term_phi': np.nan, 'LB_phi': np.nan})
                continue
            if normalize == 'r':
                y = y / max(pi_hat, 1e-08)
            w_hat = Ginv @ (PhiT @ y)
            resid = y - Phi @ w_hat
            Delta_hat = float(np.sqrt(np.mean(resid ** 2)))
            if normalize == 'r':
                Delta_term = c_alpha * Delta_hat / max(pi_hat, 1e-08)
            else:
                Delta_term = c_alpha * Delta_hat
            LB = max(0.0, min(1.0, 1.0 - alpha - Delta_term))
            recs.append({'Mask': key, 'pi_hat_proj': pi_hat, 'Delta_phi': Delta_hat, 'Delta_term_phi': Delta_term, 'LB_phi': LB})
        est_df = pd.DataFrame(recs)
        return mask_df.merge(est_df, on='Mask', how='left')

    def _collect_masks_for_stats(self, k, accumulate=True):
        mats = []
        M = self.data['M']
        mats.append(M['Train'][k])
        mats.append(M['Cal'][k])
        mats.append(M['Test']['iid'][k])
        if not mats:
            raise ValueError('No masks found to aggregate for stats.')
        m_all = np.vstack(mats).astype(int)
        if accumulate:
            self.m_stats_accum.append(m_all)
        return m_all

    def _aggregate_results(self, k, m_calib, m_test, lbs, ubs, y_test, m_train=None, phi_meta=None):
        np.savez_compressed(os.path.join(self.output_dir, 'predictions.npz'), y=np.asarray(y_test), lower=np.asarray(lbs), upper=np.asarray(ubs), masks=np.asarray(m_test), calibration_masks=np.asarray(m_calib), training_masks=np.asarray(m_train))
        coverage_all = (y_test >= lbs) & (y_test <= ubs)
        length_all = ubs - lbs
        mask_cov, mask_len = (defaultdict(list), defaultdict(list))
        mask_count_cal = defaultdict(int)
        mask_count_train = defaultdict(int)
        rep_rows = []
        for key in self._mask_rows_to_keys(m_calib):
            mask_count_cal[key] += 1
        if m_train is not None:
            for key in self._mask_rows_to_keys(m_train):
                mask_count_train[key] += 1
        test_keys = self._mask_rows_to_keys(m_test)
        for i, key in enumerate(test_keys):
            mask_cov[key].append(coverage_all[i])
            mask_len[key].append(length_all[i])
        phi_mask_patterns = None
        phi_has_mask = None
        if phi_meta is not None:
            if phi_meta.get('mask_encoding') == 'pattern' and phi_meta.get('mask_patterns') is not None:
                phi_mask_patterns = set(phi_meta.get('mask_patterns'))
            else:
                phi_has_mask = bool(phi_meta.get('d_mask', 0) > 0)
        all_keys = set(mask_cov)
        if bool(getattr(self.args, 'save_all_masks', False)):
            all_keys.update(mask_count_cal.keys())
            all_keys.update(mask_count_train.keys())
        for key in sorted(all_keys):
            cov_vals = mask_cov.get(key, [])
            len_vals = mask_len.get(key, [])
            if phi_mask_patterns is not None:
                phi_used = 1.0 if key in phi_mask_patterns else 0.0
            elif phi_has_mask is not None:
                phi_used = 1.0 if phi_has_mask else 0.0
            else:
                phi_used = np.nan
            mask_bits = [int(c) for c in key]
            self.all_result[key].append([key, float(np.nanmean(cov_vals)), float(np.nanstd(cov_vals)), float(np.nanmean(len_vals)), float(np.nanstd(len_vals)), len(cov_vals), mask_count_cal.get(key, 0), mask_count_train.get(key, 0), phi_used, *mask_bits])
            rep_rows.append([key, float(np.nanmean(cov_vals)), float(np.nanstd(cov_vals)), float(np.nanmean(len_vals)), float(np.nanstd(len_vals)), len(cov_vals), mask_count_cal.get(key, 0), mask_count_train.get(key, 0), phi_used, *mask_bits])
        self._append_missing_feature_results(m_test=m_test, m_calib=m_calib, m_train=m_train, coverage_all=coverage_all, length_all=length_all, phi_meta=phi_meta, rep_rows=rep_rows)
        self.all_results.append([mask_cov, mask_len, mask_count_cal])
        return rep_rows

    def _build_mask_df(self, epoch=0):
        mask_df_rows = []
        n_mask_bits = None
        for key in self.all_result:
            rows = self.all_result[key]
            value = np.array([row[1:] for row in rows], dtype=float)
            current_bits = value.shape[1] - 8
            if n_mask_bits is None:
                n_mask_bits = current_bits
            elif current_bits != n_mask_bits:
                raise ValueError('Inconsistent mask bit dimension in all_result.')
            mask_bits = []
            if current_bits > 0:
                mask_bits = np.nanmean(value[:, 8:], axis=0).tolist()
            mask_df_rows.append([key, np.nanmean(value[:, 0]), np.nanstd(value[:, 0]), np.nanmean(value[:, 2]), np.nanstd(value[:, 2]), np.nanmean(value[:, 4]), np.nanmean(value[:, 5]), np.nanmean(value[:, 6]), np.nanmean(value[:, 7]), *mask_bits, epoch])
        if n_mask_bits is None:
            n_mask_bits = 0
        mask_df = pd.DataFrame(mask_df_rows, columns=['Mask', 'Coverage Mean', 'Coverage Std', 'Length Mean', 'Length Std', 'Test Count', 'Cal Count', 'Train Count', 'Phi Used', *[f'M_{i + 1}' for i in range(n_mask_bits)], 'epoch'])
        return mask_df

    def _build_mask_df_from_rows(self, rows, epoch=0, replication=None):
        if not rows:
            cols = ['Mask', 'Coverage Mean', 'Coverage Std', 'Length Mean', 'Length Std', 'Test Count', 'Cal Count', 'Train Count', 'Phi Used', 'epoch']
            if replication is not None:
                cols.append('replication')
            return pd.DataFrame(columns=cols)
        n_mask_bits = max((max(len(row) - 9, 0) for row in rows))
        out_rows = []
        for row in rows:
            row_vals = list(row)
            pad = n_mask_bits - (len(row_vals) - 9)
            if pad > 0:
                row_vals.extend([np.nan] * pad)
            row_vals.append(epoch)
            if replication is not None:
                row_vals.append(replication)
            out_rows.append(row_vals)
        columns = ['Mask', 'Coverage Mean', 'Coverage Std', 'Length Mean', 'Length Std', 'Test Count', 'Cal Count', 'Train Count', 'Phi Used', *[f'M_{i + 1}' for i in range(n_mask_bits)], 'epoch']
        if replication is not None:
            columns.append('replication')
        return pd.DataFrame(out_rows, columns=columns)

    def _append_missing_feature_results(self, m_test, m_calib, m_train, coverage_all, length_all, phi_meta=None, rep_rows=None):
        m_test = np.asarray(m_test)
        if m_test.ndim != 2:
            raise ValueError('m_test must be 2D for missing-feature stats.')
        d = m_test.shape[1]
        mask_bits = [np.nan] * d
        phi_used = np.nan
        if phi_meta is not None:
            phi_used = 1.0 if int(phi_meta.get('d_mask', 0)) > 0 else 0.0
        m_calib_arr = None
        if m_calib is not None:
            m_calib_arr = np.asarray(m_calib)
        m_train_arr = None
        if m_train is not None:
            m_train_arr = np.asarray(m_train)
        for i in range(d):
            key = f'MARGINAL_M{i + 1}'
            idx = m_test[:, i].astype(bool)
            count_test = int(np.sum(idx))
            if count_test > 0:
                cov_vals = coverage_all[idx]
                len_vals = length_all[idx]
                cov_mean = float(np.mean(cov_vals))
                cov_std = float(np.std(cov_vals))
                len_mean = float(np.mean(len_vals))
                len_std = float(np.std(len_vals))
            else:
                cov_mean = np.nan
                cov_std = np.nan
                len_mean = np.nan
                len_std = np.nan
            count_cal = int(np.sum(m_calib_arr[:, i])) if m_calib_arr is not None else 0
            count_train = int(np.sum(m_train_arr[:, i])) if m_train_arr is not None else 0
            self.all_result[key].append([key, cov_mean, cov_std, len_mean, len_std, count_test, count_cal, count_train, phi_used, *mask_bits])
            if rep_rows is not None:
                rep_rows.append([key, cov_mean, cov_std, len_mean, len_std, count_test, count_cal, count_train, phi_used, *mask_bits])

    def _add_delta_mask_bounds(self, mask_df, m_data, alpha=0.1, delta_pi_eps=0.001):
        m_data = np.asarray(m_data, dtype=float)
        n_cal, d = m_data.shape
        mu = m_data.mean(axis=0)
        if n_cal > 1:
            Xc = m_data - mu
            Sigma = Xc.T @ Xc / float(n_cal)
        else:
            Sigma = np.zeros((d, d))
        Sigma += 1e-10 * np.eye(d)
        Sigma_pinv = np.linalg.pinv(Sigma, rcond=1e-08)
        keys = [''.join(map(str, map(int, r))) for r in m_data]
        uniq, cnts = np.unique(keys, return_counts=True)
        n = float(n_cal)
        pi_map = {u: c / n for u, c in zip(uniq, cnts)}

        def strmask_to_vec(s):
            return np.fromiter((int(c) for c in s), dtype=float, count=len(s))
        recs = []
        for key in mask_df['Mask'].values:
            pi_hat = float(pi_map.get(key, 0.0))
            if pi_hat <= 0.0:
                recs.append({'Mask': key, 'pi_hat': np.nan, 'Delta_mask': np.nan, 'Delta_term': np.nan, 'LB_delta_only': np.nan})
                continue
            diff = strmask_to_vec(key) - mu
            val = pi_hat * (1.0 - pi_hat) - pi_hat ** 2 * float(diff @ Sigma_pinv @ diff)
            val = max(val, 0.0)
            Delta_mask = max(alpha, 1 - alpha) * np.sqrt(val)
            Delta_term = Delta_mask / max(pi_hat, delta_pi_eps)
            LB_delta_only = max(0.0, min(1.0, 1.0 - alpha - Delta_term))
            recs.append({'Mask': key, 'pi_hat': pi_hat, 'Delta_mask': Delta_mask, 'Delta_term': Delta_term, 'LB_delta_only': LB_delta_only})
        est_df = pd.DataFrame(recs)
        return mask_df.merge(est_df, on='Mask', how='left')

    def _oracle_lengths(self, mask_df):
        dataset_name = self.args.dataset_name
        dataset_name = dataset_name.lower()
        if dataset_name != 'synthetic':
            mask_df = mask_df.copy()
            mask_df['oracle_lent'] = 0.0
            return mask_df
        mech = self.args.mechanism
        dim = self.args.dim
        alpha = 0.1
        var_missing = getattr(self.data, 'var_missing', None)
        if var_missing is None:
            var_missing = getattr(self.args, 'var_missing', None)
        if var_missing is None:
            var_missing = np.ones(dim, dtype=int)
        else:
            var_missing = np.asarray(var_missing).astype(int).ravel()
            if var_missing.size == 1 and var_missing.item() == -1:
                var_missing = np.ones(dim, dtype=int)
        if hasattr(self.data, 'get_test_patterns'):
            patterns = self.data.get_test_patterns(dim)
        else:
            patterns = create_patterns(dim, var_missing)
        pattern_keys = [''.join(map(str, map(int, p))) for p in patterns]
        if mech == 'MCAR' and (not hasattr(self.data, 'get_cov_for_pattern')):
            default_beta = np.array([1, 2, -1, 3, -0.5, -1, 0.3, 1.7, 0.4, -0.3])
            beta = getattr(self.args, 'beta', None)
            if beta is None:
                beta = getattr(self.data, 'beta', None)
            if beta is None:
                beta = default_beta
            beta = np.asarray(beta).ravel()
            if beta.size < dim:
                pad = np.ones(dim - beta.size)
                beta = np.concatenate([beta, pad])
            else:
                beta = beta[:dim]
            phi = getattr(self.args, 'phi', 0.8)
            cov = np.full((dim, dim), phi) + (1 - phi) * np.eye(dim)
            oracle_lent = [oracle_len_pattern(p, beta, cov, alpha=alpha) for p in patterns]
            oracle_df = pd.DataFrame({'Mask': pattern_keys, 'oracle_lent': np.asarray(oracle_lent, dtype=float)})
            return mask_df.merge(oracle_df, on='Mask', how='left')
        if hasattr(self.data, 'get_cov_for_pattern') or mech == 'PMIX':
            beta = np.asarray(getattr(self.data, 'beta', None))
            if beta is None:
                raise NotImplementedError
            else:
                beta = beta[:dim]
            noise_std = float(getattr(self.data, 'noise_std', 1.0))
            oracle_lent = []
            for p in patterns:
                cov_m = self.data.get_cov_for_pattern(p)
                length_m = oracle_len_pattern_pmix(pattern=p, beta=beta, cov_m=cov_m, alpha=alpha, noise_std=noise_std)
                oracle_lent.append(float(length_m))
            oracle_df = pd.DataFrame({'Mask': pattern_keys, 'oracle_lent': np.asarray(oracle_lent, dtype=float)})
            return mask_df.merge(oracle_df, on='Mask', how='left')
        oracle_lent = []
        for p in patterns:
            val = self.data.simulate_oracle_length(pattern=p, n_sim=5000, alpha=alpha, n_bins=100, zoom=10, random_state=0)
            oracle_lent.append(float(val))
        oracle_df = pd.DataFrame({'Mask': pattern_keys, 'oracle_lent': np.asarray(oracle_lent, dtype=float)})
        return mask_df.merge(oracle_df, on='Mask', how='left')

    def _save_artifacts(self, mask_df, phi_mu=None, phi_std=None, phi_meta=None, rep_df=None, replication=None):
        (rep_df if rep_df is not None else mask_df).to_csv(os.path.join(self.output_dir, 'patterns.csv'), index=False)
        if self.phi_err_history:
            pd.concat(self.phi_err_history, ignore_index=True).to_csv(os.path.join(self.output_dir, 'representation.csv'), index=False)

def get_one_hot_encoder(single_M: bool, dim: int):
    if single_M:

        def one_hot_encode_batch(X):
            mask_part = np.isnan(X).astype(int)
            return mask_part
        mask_dim = dim
    else:

        def one_hot_encode_batch(X):
            mask_part = np.isnan(X).astype(int)
            mask_part_reverse = (mask_part == 0).astype(int)
            mask_concat = np.column_stack((mask_part, mask_part_reverse))
            return mask_concat
        mask_dim = dim * 2
    return (one_hot_encode_batch, mask_dim)

def oracle_len_pattern_pmix(pattern, beta, cov_m, alpha=0.1, noise_std=1.0):
    pattern = np.array(list(map(bool, pattern)))
    beta = np.asarray(beta, dtype=float)
    cov_m = np.asarray(cov_m, dtype=float)
    d = beta.shape[0]
    assert cov_m.shape == (d, d)
    if not pattern.any():
        var_y_given = noise_std ** 2
    else:
        idx_mis = pattern
        idx_obs = ~pattern
        beta_mis = beta[idx_mis]
        cov_oo = cov_m[np.ix_(idx_obs, idx_obs)]
        cov_mm = cov_m[np.ix_(idx_mis, idx_mis)]
        cov_mo = cov_m[np.ix_(idx_mis, idx_obs)]
        cov_om = cov_m[np.ix_(idx_obs, idx_mis)]
        cov_oo_inv = np.linalg.pinv(cov_oo)
        cov_mm_given_o = cov_mm - cov_mo @ cov_oo_inv @ cov_om
        var_y_given = float(beta_mis @ cov_mm_given_o @ beta_mis) + noise_std ** 2
    z = norm.ppf(1 - alpha / 2.0)
    length = 2.0 * z * np.sqrt(var_y_given)
    return length
