import base64
import hashlib
import numpy as np
import os
import functools
from scipy.optimize import bisect
from tqdm import tqdm
import sys

def _sanitize_token(value):
    token = str(value)
    cleaned = ''.join((ch if ch.isalnum() or ch in ('.', '-', '+') else '-' for ch in token))
    return cleaned.strip('-') or 'na'

def get_tqdm(*args, **kwargs):
    return tqdm(*args, file=sys.stdout, dynamic_ncols=True, ascii=True, disable=False, **kwargs)

def pinball_loss(y_true, y_pred, tau):
    u = y_true - y_pred
    return np.where(u >= 0, tau * u, (tau - 1) * u)

def interval_pinball_loss(y, pred_low, pred_high, alpha):
    loss_low = pinball_loss(y, pred_low, tau=alpha / 2)
    loss_high = pinball_loss(y, pred_high, tau=1 - alpha / 2)
    loss_per_sample = loss_low ** 2 + loss_high ** 2
    return np.mean(loss_per_sample)

def evaluate_model(model, X_val, y_val, alpha):
    pred_low = model['q_low'].predict(X_val)
    pred_high = model['q_high'].predict(X_val)
    score = interval_pinball_loss(y_val, pred_low, pred_high, alpha)
    return np.mean(score)

def make_phi_fn_with_norm_new(model, n_treespace, fit_X, one_hot_encode_batch, concat_mask=True, scale_mask=True, special_mask=0, use_mask_patterns=False, M_calib=None, min_pattern_count=1, include_other_pattern=False, mask_scale_factor=1.0, tree_scale_factor=1.0, covariate_imputer=None):
    import numpy as np
    fit_X = np.asarray(fit_X)
    if fit_X.ndim != 2:
        raise ValueError('fit_X must be a 2D array')
    n_fit, d = fit_X.shape
    pattern_key_to_col = None
    pattern_keys_ordered = None
    other_col = None

    def _mask_rows_to_keys(M_rows):
        return [''.join(map(str, map(int, r))) for r in M_rows]
    if concat_mask:
        if use_mask_patterns:
            assert M_calib is not None
            M_ref = np.asarray(M_calib, dtype=int)
            if M_ref.ndim != 2:
                raise ValueError('M_calib must be 2D')
            if M_ref.shape[1] != d:
                raise ValueError(f'M_calib.shape[1]={M_ref.shape[1]} != fit_X.shape[1]={d}')
            keys_ref = _mask_rows_to_keys(M_ref)
            uniq_keys, counts = np.unique(keys_ref, return_counts=True)
            keep = counts >= min_pattern_count
            uniq_keys = uniq_keys[keep]
            if uniq_keys.size == 0:
                raise ValueError('No mask patterns with count >= min_pattern_count in calibration set.')
            pattern_keys_ordered = list(uniq_keys)
            pattern_key_to_col = {key: j for j, key in enumerate(pattern_keys_ordered)}
            mask_dim = len(pattern_keys_ordered)
            if include_other_pattern:
                other_col = mask_dim
                mask_dim += 1

            def encode_mask_batch(X):
                X = np.asarray(X)
                M = np.isnan(X).astype(int)
                keys = _mask_rows_to_keys(M)
                out = np.zeros((len(keys), mask_dim), dtype=float)
                for i, k in enumerate(keys):
                    j = pattern_key_to_col.get(k)
                    if j is not None:
                        out[i, j] = 1.0
                    elif include_other_pattern:
                        out[i, other_col] = 1.0
                return out
            d_mask = mask_dim
        else:

            def encode_mask_batch(X):
                return one_hot_encode_batch(X).astype(float, copy=False)
            d_mask = one_hot_encode_batch(fit_X[:1]).shape[1]
    else:

        def encode_mask_batch(X):
            X = np.asarray(X)
            return np.zeros((X.shape[0], 0), dtype=float)
        d_mask = 0

    def raw_phi(X):
        X = np.atleast_2d(X)
        if covariate_imputer is not None:
            mask = encode_mask_batch(X)
            return np.concatenate([mask, covariate_imputer.transform(X)], axis=1)
        if n_treespace == 0 or model is None:
            mask = encode_mask_batch(X) if concat_mask else np.zeros((X.shape[0], 0), dtype=float)
            if special_mask > 0 and mask.shape[1] > special_mask:
                mask = mask[:, :special_mask]
            return mask.astype(float, copy=False)
        trees = extract_tree_outputs(X, model, n_treespace)
        if concat_mask:
            mask = encode_mask_batch(X)
            if special_mask > 0 and mask.shape[1] > special_mask:
                mask = mask[:, :special_mask]
            Phi = np.concatenate([mask.astype(float, copy=False), trees.astype(float, copy=False)], axis=1)
        else:
            Phi = trees.astype(float, copy=False)
        return Phi
    Phi_fit = raw_phi(fit_X)
    p = Phi_fit.shape[1]
    itp = covariate_imputer is None and (not (n_treespace == 0 or model is None))
    itp_col = p - 1 if itp and p > 0 else None
    feature_indices = np.arange(p, dtype=int)
    if itp_col is not None:
        feature_indices = feature_indices[feature_indices != itp_col]
    if not scale_mask and d_mask > 0:
        feature_indices = feature_indices[feature_indices >= d_mask]
    mu_full = np.zeros(p, dtype=float)
    std_full = np.ones(p, dtype=float)
    if feature_indices.size > 0:
        mu_full[feature_indices] = Phi_fit[:, feature_indices].mean(axis=0)
        std_full[feature_indices] = Phi_fit[:, feature_indices].std(axis=0) + 1e-12
        std_full[feature_indices[std_full[feature_indices] <= 1e-12]] = 1.0
    feature_indices = feature_indices.astype(int, copy=False)

    def phi_fn(X):
        Phi = raw_phi(X).astype(float, copy=False)
        if feature_indices.size > 0:
            Phi[:, feature_indices] = (Phi[:, feature_indices] - mu_full[feature_indices]) / std_full[feature_indices]
        if d_mask > 0 and mask_scale_factor != 1.0:
            Phi[:, :d_mask] *= mask_scale_factor
        if tree_scale_factor != 1.0 and p - d_mask > 0:
            tree_cols = np.arange(d_mask, p, dtype=int)
            if itp_col is not None:
                tree_cols = tree_cols[tree_cols != itp_col]
            if tree_cols.size > 0:
                Phi[:, tree_cols] *= tree_scale_factor
        return Phi
    meta = dict(d_mask=d_mask, intercept_col=itp_col, has_intercept=itp, normalized_cols=feature_indices.tolist(), mask_encoding='pattern' if use_mask_patterns else 'coord', mask_scale_factor=mask_scale_factor, tree_scale_factor=tree_scale_factor)
    if use_mask_patterns:
        meta.update(mask_patterns=pattern_keys_ordered, min_pattern_count=min_pattern_count, include_other_pattern=include_other_pattern)
    return (phi_fn, mu_full[feature_indices], std_full[feature_indices], meta)

def extract_tree_outputs(X, model, n_treespace):
    X = np.atleast_2d(X)
    if n_treespace == 0:
        n_samples = X.shape[0]
        return np.zeros((n_samples, 0), dtype=np.float32)
    module_name = getattr(type(model), '__module__', '')
    n_samples = X.shape[0]

    def _diff_cumulative(pred_fn, n_trees):
        tree_feats = np.ones((n_samples, n_trees + 1), dtype=np.float32)
        prev = np.zeros(n_samples, dtype=np.float32)
        for i in range(n_trees):
            current = np.asarray(pred_fn(i + 1), dtype=np.float32).reshape(n_samples)
            tree_feats[:, i] = current - prev
            prev = current
        return tree_feats
    if 'lightgbm' in module_name:
        available = int(model.num_trees())
        if available != n_treespace:
            print('error!!! available not equal to n_treespace?', available, n_treespace)
        n_trees = min(n_treespace, available)

        def _lgb_pred(num_iter):
            return model.predict(X, raw_score=True, num_iteration=num_iter)
        return _diff_cumulative(_lgb_pred, n_trees)
    if 'catboost' in module_name:
        available = getattr(model, 'tree_count_', n_treespace)
        assert available == n_treespace
        n_trees = n_treespace

        def _cat_pred(num_iter):
            return model.predict(X, prediction_type='RawFormulaVal', ntree_start=0, ntree_end=num_iter, thread_count=int(os.environ.get('EXPERIMENT_THREADS', '1')))
        return _diff_cumulative(_cat_pred, n_trees)
    if 'xgboost' in module_name:
        import xgboost as xgb
        available = int(model.num_boosted_rounds())
        assert available == n_treespace
        n_trees = min(n_treespace, available)
        dmatrix = xgb.DMatrix(X, nthread=int(os.environ.get('EXPERIMENT_THREADS', '1')))

        def _xgb_pred(num_iter):
            return model.predict(dmatrix, output_margin=True, iteration_range=(0, num_iter))
        return _diff_cumulative(_xgb_pred, n_trees)
    booster = model.dump_model()
    trees = booster['tree_info'][:n_treespace]
    n_trees = len(trees)
    n_samples = X.shape[0]
    tree_outputs = np.ones((n_samples, n_trees + 1), dtype=np.float32)
    for i, tree in enumerate(trees):
        tree_structure = tree['tree_structure']
        tree_outputs[:, i] = np.array([predict_single_tree(tree_structure, x) for x in X])
    return tree_outputs

def predict_single_tree(tree, x):
    while 'leaf_index' not in tree:
        split_feature = tree['split_feature']
        threshold = tree['threshold']
        if x[split_feature] <= threshold:
            tree = tree['left_child']
        else:
            tree = tree['right_child']
    return tree['leaf_value']

def pick_coeffs(X, idxs_obs=None, idxs_nas=None, self_mask=False):
    n, d = X.shape
    if self_mask:
        coeffs = np.random.randn(d)
        Wx = X * coeffs
        std = np.std(Wx, axis=0)
        std[std == 0] = 1.0
        coeffs /= std
    else:
        d_obs = len(idxs_obs)
        d_na = len(idxs_nas)
        coeffs = np.random.randn(d_obs, d_na)
        Wx = X[:, idxs_obs] @ coeffs
        coeffs /= np.std(Wx, axis=0, keepdims=True)
    return coeffs

def fit_intercepts(X, coeffs, p, self_mask=False):
    if self_mask:
        X = np.asarray(X, dtype=float)
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        d = coeffs.size
        if X.shape[1] < d:
            raise ValueError('X must have at least as many columns as coeffs when self_mask=True.')
        intercepts = np.zeros(d)
        for j in range(d):
            col = X[:, j]
            cj = coeffs[j]

            def f(x):
                return np.mean(1 / (1 + np.exp(-(col * cj + x)))) - p
            intercepts[j] = bisect(f, -50, 50)
    else:
        d_obs, d_na = coeffs.shape
        intercepts = np.zeros(d_na)
        for j in range(d_na):

            def f(x):
                return np.mean(1 / (1 + np.exp(-(X @ coeffs[:, j] + x)))) - p
            intercepts[j] = bisect(f, -50, 50)
    return intercepts

def MAR_mask(X, p, incomplete_vars, coeffs=None, intercepts=None, depend_vars=None, return_vars=False):
    n, d = X.shape
    mask = np.zeros((n, d), dtype=bool)
    idxs_nas = np.array([i for i in range(d) if incomplete_vars[i] == 1])
    if depend_vars is None:
        idxs_obs = np.array([i for i in range(d) if i not in idxs_nas])
    else:
        idxs_obs = np.array(depend_vars)
    if coeffs is None:
        coeffs = pick_coeffs(X, idxs_obs, idxs_nas)
    if intercepts is None:
        intercepts = fit_intercepts(X[:, idxs_obs], coeffs, p)
    ps = 1 / (1 + np.exp(-(X[:, idxs_obs] @ coeffs + intercepts)))
    ber = np.random.uniform(size=(n, len(idxs_nas)))
    mask[:, idxs_nas] = ber < ps
    if return_vars:
        return (mask, coeffs, intercepts)
    return mask
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import SimpleImputer, IterativeImputer

class No_impute:

    def __init__(self):
        pass

    def fit(self, x):
        return self

    def transform(self, x):
        return x

def get_impute(imputation, iteration=50, device='cpu', last_iterative=False):
    if imputation in ['mean', 'constant']:
        imputer = SimpleImputer(missing_values=np.nan, strategy=imputation)
    elif imputation == 'MICE':
        if last_iterative:
            imputer = IterativeImputerWithLastModel(missing_values=np.nan, sample_posterior=True, max_iter=iteration, tol=0.01)
        else:
            imputer = IterativeImputer(missing_values=np.nan, sample_posterior=True, max_iter=iteration, tol=0.01)
    elif imputation == 'iterative_ridge':
        if last_iterative:
            imputer = IterativeImputerWithLastModel(missing_values=np.nan, sample_posterior=False, max_iter=iteration, tol=0.01)
        else:
            imputer = IterativeImputer(missing_values=np.nan, sample_posterior=False, max_iter=iteration, tol=0.01)
    elif imputation == 'None' or imputation is None:
        imputer = No_impute()
    else:
        raise ValueError(f'Unsupported imputation: {imputation}')
    return imputer

class IterativeImputerWithLastModel(IterativeImputer):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def fit_transform(self, X, y=None, **params):
        super().fit_transform(X, y, **params)
        imputations_per_round = len(self.imputation_sequence_) // self.n_iter_
        last_iteration_models = self.imputation_sequence_[-imputations_per_round:]
        self.imputation_sequence_ = last_iteration_models * self.n_iter_
        return super().transform(X)

def check_condition(A, B):
    valid_positions = ~A | B
    row_satisfies_condition = np.all(valid_positions, axis=1)
    return row_satisfies_condition

def quantile_corrected(x, alpha):
    n_x = len(x)
    if (1 - alpha) * (1 + 1 / n_x) > 1:
        return np.inf
    else:
        return np.quantile(x, (1 - alpha) * (1 + 1 / n_x), method='higher')

def weighted_quantile_matrix(X, W, q):
    X = np.asarray(X)
    W = np.asarray(W)
    if X.shape != W.shape:
        raise ValueError('X and W must have the same shape')
    quantiles = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        sorter = np.argsort(X[i])
        x_sorted = X[i][sorter]
        w_sorted = W[i][sorter]
        cdf = np.cumsum(w_sorted) / np.sum(w_sorted)
        idx = np.searchsorted(cdf, q, side='right')
        quantiles[i] = x_sorted[idx]
    return quantiles
from scipy.stats import norm

def pattern_to_id(m):
    return int(''.join(map(str, m)), 2)

def oracle_len_pattern(pattern, beta, cov, alpha=0.1):
    pattern = np.array(list(map(bool, pattern)))
    beta_mis = beta[pattern]
    cov_obs = cov[~pattern][:, ~pattern]
    cov_obs_inv = np.linalg.pinv(cov_obs)
    cov_mis = cov[pattern][:, pattern]
    cov_mis_obs = cov[pattern][:, ~pattern]
    cov_mis_given_obs = cov_mis - np.dot(cov_mis_obs, np.dot(cov_obs_inv, cov_mis_obs.T))
    length = 2 * norm.ppf(1 - alpha / 2) * np.sqrt(np.dot(beta_mis, np.dot(cov_mis_given_obs, beta_mis.T)) + 1)
    return length

def bin_to_vec(bin_pattern, d, var_missing=None):
    bin_pattern = bin_pattern[2:]
    l = len(bin_pattern)
    if var_missing is None:
        nb_missing = d
    else:
        nb_missing = np.sum(var_missing)
    if l < nb_missing:
        for i in range(nb_missing - l):
            bin_pattern = '0' + bin_pattern
    vec_bin = [int(x) for x in bin_pattern]
    if nb_missing < d:
        vec_pattern = [0] * d
        vec_pattern = np.array(vec_pattern)
        vec_pattern[list(np.where(var_missing == 1)[0])] = vec_bin
        vec_pattern = list(vec_pattern)
    else:
        vec_pattern = vec_bin
    return vec_pattern

def create_patterns(d, var_missing):
    nb_var_missing = np.sum(var_missing)
    if nb_var_missing == d:
        keys_patterns = np.arange(0, 2 ** d)
        bin_patterns = list(map(bin, keys_patterns))
        vec_patterns = list(map(functools.partial(bin_to_vec, d=d), bin_patterns))
    else:
        keys_patterns = np.arange(0, 2 ** nb_var_missing)
        bin_patterns = list(map(bin, keys_patterns))
        vec_patterns = list(map(functools.partial(bin_to_vec, d=d, var_missing=var_missing), bin_patterns))
    return vec_patterns

def sample_patterns(d, var_missing, num_patterns, seed=None):
    var_missing = np.asarray(var_missing)
    idx_missing = np.where(var_missing == 1)[0]
    nb_var_missing = len(idx_missing)
    if nb_var_missing == 0:
        return [np.zeros(d, dtype=int).tolist()]
    max_unique = 2 ** nb_var_missing
    if num_patterns >= max_unique:
        return create_patterns(d, var_missing)
    rng = np.random.default_rng(seed)
    patterns = []
    seen = set()
    while len(patterns) < num_patterns:
        bits = rng.integers(0, 2, size=nb_var_missing)
        pattern = np.zeros(d, dtype=int)
        pattern[idx_missing] = bits
        tup = tuple(pattern.tolist())
        if tup in seen:
            continue
        seen.add(tup)
        patterns.append(pattern.tolist())
    return patterns

def MNAR_mask_self_logistic(X, p, incomplete_vars, coeffs=None, intercepts=None, return_vars=False):
    n, d = X.shape
    incomplete_vars = np.asarray(incomplete_vars)
    idxs_nas = np.where(incomplete_vars == 1)[0]
    if idxs_nas.size == 0:
        mask = np.zeros((n, d), dtype=bool)
        zeros = np.zeros(d)
        if return_vars:
            return (mask, zeros, zeros)
        return mask

    def _select_params(arr):
        arr = np.asarray(arr, dtype=float).ravel()
        if arr.size == d:
            return arr[idxs_nas]
        if arr.size == idxs_nas.size:
            return arr
        raise ValueError('Coefficient/intercept array has incompatible shape.')
    if coeffs is None:
        coeffs_sub = pick_coeffs(X[:, idxs_nas], self_mask=True)
    else:
        coeffs_sub = _select_params(coeffs)
    if intercepts is None:
        intercepts_sub = fit_intercepts(X[:, idxs_nas], coeffs_sub, p, self_mask=True)
    else:
        intercepts_sub = _select_params(intercepts)
    logits = X[:, idxs_nas] * coeffs_sub + intercepts_sub
    ps = 1.0 / (1.0 + np.exp(-logits))
    mask = np.zeros((n, d), dtype=bool)
    mask[:, idxs_nas] = np.random.uniform(size=ps.shape) < ps
    if return_vars:
        coeffs_full = np.zeros(d)
        intercepts_full = np.zeros(d)
        coeffs_full[idxs_nas] = coeffs_sub
        intercepts_full[idxs_nas] = intercepts_sub
        return (mask, coeffs_full, intercepts_full)
    return mask

def _format_mask_token(mask, threshold: int=10) -> str:
    arr = np.asarray(mask, dtype=int).ravel()
    if arr.size == 0:
        return 'na'
    arr = np.where(arr != 0, 1, 0).astype(np.uint8, copy=False)
    if arr.size > threshold:
        ones = int(arr.sum())
        zeros = int(arr.size - ones)
        if zeros == 0:
            return f'{arr.size}x1'
        if ones == 0:
            return f'{arr.size}x0'
        try:
            packed = np.packbits(arr, bitorder='big')
        except TypeError:
            packed = np.packbits(arr)
        enc = base64.b32encode(packed).decode('ascii').rstrip('=').lower()
        sig = hashlib.sha1(packed.tobytes()).hexdigest()[:8]
        return f'{ones}x1-{zeros}x0-b{arr.size}-{enc}-h{sig}'
    return ''.join((str(int(v)) for v in arr.tolist()))
