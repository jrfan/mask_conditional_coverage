import os
import pandas as pd
import numpy as np
import optuna
from scipy.stats import norm
from scipy.optimize import linprog
from collections import defaultdict
from functools import partial
from models import fit_basemodel, make_score_fns, Data_for_mask, Miss_proba_simple_model, predict_basemodel
import utils
from utils import pinball_loss, evaluate_model, get_tqdm, make_phi_fn_with_norm_new
from utils import oracle_len_pattern, create_patterns
from crossval import runCV_ridge_v2
from conditionalconformal import CondConf
from conditionalconformal.condconf import setup_cvx_problem_calib, build_penalty_partition, solve_cvxpy_with_fallback
from threadpoolctl import threadpool_limits
import hashlib
from models import get_quantile_adapter
from tqdm import tqdm
import copy
import json

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

def cache_test_features(phi_fn, x_test):
    features = phi_fn(x_test)
    lookup = {np.ascontiguousarray(row).tobytes(): features[i:i + 1] for i, row in enumerate(x_test)}

    def cached(x):
        rows = np.atleast_2d(x)
        if len(rows) == 1:
            found = lookup.get(np.ascontiguousarray(rows[0]).tobytes())
            if found is not None:
                return found
        return phi_fn(x)
    return cached

def get_sota_default_config():
    return dict(pre_impute=False, pre_imputation='MICE', model_imputation='iterative_ridge', RN_model='histgbdt', RN_train_with_cal=False, generalized_impute_train=False, generalize_pre_impute_iteration=0, basemodel='histgbdt', cores=8, other_model_parameters=None, ocacle_pre_impute=False, weighted_search_range=0.5, weighted_grid_number=30, split_train=1.0)

def resolve_sota_flags(method):
    method = (method or '').strip().lower()
    if method == 'exact':
        return (True, False, False)
    if method == 'nested':
        return (False, False, False)
    if method == 'weightedcp':
        return (True, True, False)
    if method == 'arccp':
        return (True, False, True)
    raise ValueError(f'Unknown SOTA method: {method}')

class Sota_experiments_to_compare:

    def __init__(self, data, pre_impute, pre_imputation, model_imputation, accept_reject, reweight, RN_model, RN_train_with_cal, generalized_impute_train, generalize_pre_impute_iteration, basemodel, alpha, cores, exact, other_model_parameters=None, ocacle_pre_impute=False, weighted_search_range=0.5, weighted_grid_number=30, split_train=1.0, replication_index=None, tuning_trials=100):
        self.data = data
        self.dataset_name = data.name
        self.pre_impute = pre_impute
        self.n_rep = self.data.n_rep
        self.is_tqdm = False
        self.pre_impute = pre_impute
        self.pre_imputation = pre_imputation
        self.model_imputation = model_imputation
        self.generalized_pre_impute = pre_impute
        self.generalized_impute_train = generalized_impute_train
        self.generalize_pre_impute_iteration = generalize_pre_impute_iteration
        self.accept_reject = accept_reject
        self.reweight = reweight
        self.RN_model = RN_model
        self.RN_train_with_cal = RN_train_with_cal
        self.basemodel = basemodel
        self.alpha = alpha
        self.params_basemodel = {}
        if basemodel != 'histgbdt':
            self.params_basemodel['cores'] = cores
        if other_model_parameters is not None:
            self.params_basemodel.update(other_model_parameters)
        self.exact = exact
        self.ocacle_pre_impute = ocacle_pre_impute
        self.weighted_search_range = weighted_search_range
        self.weighted_grid_number = weighted_grid_number
        self.split_train = split_train
        self.replication_indices = list(range(self.n_rep)) if replication_index is None else [replication_index]
        self.tuning_trials = tuning_trials
        self.model_type = 'Quantiles'
        self.protocol_by_rep = {}
        self.results_by_rep = {}

    def run_experiments(self):
        results = None
        for k in tqdm(self.replication_indices, disable=not self.is_tqdm):
            np.random.seed(42000 + k)
            self.k = k
            if not self.is_tqdm:
                print('{} rep over {}'.format(k, self.n_rep))
            X_mis_train = self.data['X_missing']['Train'][k, :, :]
            X_mis_cal = self.data['X_missing']['Cal'][k, :, :]
            M_train = self.data['M']['Train'][k, :, :]
            M_train_unchanged = M_train.copy()
            M_cal = self.data['M']['Cal'][k, :, :]
            M_cal_unchanged = M_cal.copy()
            Y_train = self.data['Y']['Train'][k, :]
            Y_cal = self.data['Y']['Cal'][k, :]
            keys_test = list(self.data['X_missing']['Test'].keys())
            X_mis_test = dict.fromkeys(keys_test)
            M_test = dict.fromkeys(keys_test)
            Y_test = dict.fromkeys(keys_test)
            for key in keys_test:
                X_mis_test[key] = self.data['X_missing']['Test'][key][k, :, :]
                M_test[key] = self.data['M']['Test'][key][k, :, :]
                Y_test[key] = self.data['Y']['Test'][key][k, :]
            target = 'Quantiles'
            if self.pre_impute and self.pre_imputation != 'oracle':
                pre_imputer = utils.get_impute(self.pre_imputation)
            else:
                pre_imputer = utils.get_impute('None')
            model_imputer = utils.get_impute(self.model_imputation)
            model_imputer.fit(X_mis_train)
            if self.pre_impute and self.pre_imputation == 'oracle':
                print('oracle imputation')

                class Oracle_pre_imputer:

                    def __init__(self, parent, k):
                        self.parent = parent
                        self.k = k
                        print(self.parent.data['X']['Train'].shape)
                        assert self.parent.data['X']['Train'].shape[1] != self.parent.data['X']['Cal'].shape[1]

                    def fit(self, X):
                        pass

                    def transform(self, X):
                        if X.shape[0] == self.parent.data['X']['Train'].shape[1]:
                            return self.parent.data['X']['Train'][self.k]
                        elif X.shape[0] == self.parent.data['X']['Cal'].shape[1]:
                            return self.parent.data['X']['Cal'][self.k]
                        else:
                            raise NotImplementedError
                pre_imputer = Oracle_pre_imputer(self, self.k)
            elif self.generalized_pre_impute:
                pre_imputer.fit(np.column_stack((X_mis_train, Y_train)))
            else:
                pre_imputer.fit(X_mis_train)
            X_imp_train = model_imputer.transform(X_mis_train)
            if self.generalized_pre_impute and self.generalized_impute_train:
                X_imp_train_with_Y = pre_imputer.transform(np.column_stack((X_mis_train, Y_train)))[:, :-1]
            X_imp_test = dict.fromkeys(keys_test)
            if self.pre_impute:
                if self.generalized_pre_impute:
                    if self.generalize_pre_impute_iteration > 0:
                        x_cal_tmp = []
                        m_cal_tmp = []
                        y_cal_tmp = []
                        for i in range(self.generalize_pre_impute_iteration):
                            X_mis_cal = pre_imputer.transform(np.column_stack((X_mis_cal, Y_cal)))[:, :-1]
                            M_cal = np.zeros_like(M_cal)
                            x_cal_tmp.append(X_mis_cal)
                            m_cal_tmp.append(M_cal)
                            y_cal_tmp.append(Y_cal)
                        X_mis_cal = np.concatenate(x_cal_tmp, axis=0)
                        M_cal = np.concatenate(m_cal_tmp, axis=0)
                        Y_cal = np.concatenate(y_cal_tmp, axis=0)
                    else:
                        X_mis_cal = pre_imputer.transform(np.column_stack((X_mis_cal, Y_cal)))[:, :-1]
                        M_cal = np.zeros_like(M_cal)
                else:
                    X_mis_cal = pre_imputer.transform(X_mis_cal)
                    M_cal = np.zeros_like(M_cal)
            self.model_RN_derivative = None
            assert not (self.accept_reject and self.reweight), f'Invalid state: bool_var1={self.accept_reject}, bool_var2={self.reweight}'
            self.mask_class_model = None
            if self.accept_reject or self.reweight:
                if self.generalized_pre_impute:
                    train_imputed_train_C = pre_imputer.transform(np.column_stack((X_mis_train, Y_train)))[:, :-1]
                else:
                    train_imputed_train_C = pre_imputer.transform(X_mis_train)
                train_concat = np.column_stack([train_imputed_train_C, Y_train])
                cal_concat = np.column_stack([X_mis_cal, Y_cal])
                if self.RN_train_with_cal:
                    train_concat = np.vstack((train_concat, cal_concat))
                    M_train_concat = np.vstack((M_train, M_cal_unchanged))
                else:
                    M_train_concat = M_train.copy()
                self.mask_class_data = Data_for_mask(train_concat, M_train_concat, batch_size=128, contain_y=True, dataset=self.dataset_name)
                assert self.RN_model.lower() in ['histgbdt', 'xgboost', 'logistic'], '{} not correct'.format(self.RN_model)
                self.mask_class_model = Miss_proba_simple_model(self.mask_class_data, model=self.RN_model, epoch=1000)
                self.mask_class_model.train()
            for key in keys_test:
                X_imp_test[key] = model_imputer.transform(X_mis_test[key])
            if self.generalized_pre_impute and self.generalized_impute_train:
                features_train = X_imp_train_with_Y
            else:
                features_train = X_imp_train
            features_test = X_imp_test
            if self.split_train < 1.0:
                n_total = features_train.shape[0]
                n_sub = int(self.split_train * n_total)
                idx = np.random.RandomState(42 + k).permutation(n_total)
                train_idx, val_idx = (idx[:n_sub], idx[n_sub:])
                x_base, y_base = (features_train[train_idx], Y_train[train_idx])
                x_score, y_score = (features_train[val_idx], Y_train[val_idx])
            else:
                x_base, y_base = (features_train, Y_train)
                x_score, y_score = (features_train, Y_train)
            trained_model, base_best_params, _ = ConformalExperiment.tune_histgbdt(self, x_base, y_base, x_score, y_score, alpha=self.alpha, n_trials=self.tuning_trials)
            self.protocol_by_rep[k] = {'protocol': 'training_only_v2', 'replication': k, 'split_seed': 42 + k, 'random_seed': 42000 + k, 'base_best_params': base_best_params, 'base_fit_size': len(x_base), 'selection_size': len(x_score), 'calibration_used_for_tuning': False, 'model_imputation': self.model_imputation, 'pre_imputation': self.pre_imputation if self.pre_impute else None, 'note': 'Same HistGBDT architecture/search budget and paired row split; baseline-specific imputation is retained.'}
            groups_test = dict.fromkeys(keys_test)
            for key in keys_test:
                groups_test[key] = list(map(utils.pattern_to_id, M_test[key].astype(int)))
            preds = dict.fromkeys(keys_test)
            for key in keys_test:
                pred = self.calibrate_predict(fitted_basemodel=trained_model, model_imputer=model_imputer, X_cal=X_mis_cal, M_cal=M_cal, Y_cal=Y_cal, X_mis_test=X_mis_test[key], features_test=features_test[key], M_test=M_test[key], groups_test=groups_test[key], target=target, M_cal_unchanged=M_cal_unchanged)
                preds[key] = pred
            self.results_by_rep[k] = preds
            if results is None:
                results = dict.fromkeys(keys_test)
                for key in keys_test:
                    results[key] = {'Y_inf': np.array(preds[key]['y_inf']), 'Y_sup': np.array(preds[key]['y_sup'])}
            else:
                for key in keys_test:
                    results[key]['Y_inf'] = np.vstack((results[key]['Y_inf'], np.array(preds[key]['y_inf'])))
                    results[key]['Y_sup'] = np.vstack((results[key]['Y_sup'], np.array(preds[key]['y_sup'])))
        self.results = results
        return results

    def calibrate_predict(self, fitted_basemodel, model_imputer, X_cal, M_cal, Y_cal, X_mis_test, features_test, M_test, groups_test, target='Quantiles', M_cal_unchanged=None, reweight_with_y=False):
        exact = self.exact
        alpha = self.alpha
        mechanism = self.data['mechanism']
        accept_reject = self.accept_reject
        reweight = self.reweight
        basemodel = self.basemodel
        mask_class_model = self.mask_class_model
        assert target in ['Quantiles']
        assert basemodel in ['Linear', 'RF', 'NNet', 'XGBoost', 'histgbdt', 'NNet_missing']
        impute_and_mask_conditional = False
        if impute_and_mask_conditional:
            M_test = np.zeros_like(M_test)
            M_cal = np.zeros_like(M_cal)
            X_cal = model_imputer.transform(X_cal.copy())
            X_test = model_imputer.transform(X_mis_test.copy())
            print(X_test[0], features_test[0])
            assert np.array_equal(X_test, features_test), 'Arrays are not equal'
        patterns = np.unique(M_test, axis=0)
        ids = list(map(utils.pattern_to_id, patterns.astype(int)))
        n_test = features_test.shape[0]
        q_scores_test = np.empty(n_test)
        if not exact:
            pred_test = {'y_inf': np.empty(n_test), 'y_sup': np.empty(n_test)}
        if reweight:
            pred_test_reweight = {'y_inf': np.ones(n_test) * np.inf, 'y_sup': np.ones(n_test) * -np.inf}
        for idp, id_pattern in enumerate(ids):
            test_index = (np.array(groups_test) == id_pattern).flatten()
            test_index_list = np.where(test_index)[0]
            X_imp_cal_masking = copy.deepcopy(X_cal)
            M_cal_masking = copy.deepcopy(M_cal)
            Y_cal_masking = copy.deepcopy(Y_cal)
            pattern = patterns[idp]
            empty = False
            if exact:
                ind_subsample = np.all(M_cal[:, pattern == 0] == 0, axis=1)
                if np.sum(ind_subsample) == 0:
                    empty = True
                X_imp_cal_masking = X_imp_cal_masking[ind_subsample, :]
                M_cal_masking = M_cal_masking[ind_subsample, :]
                Y_cal_masking = Y_cal_masking[ind_subsample]
            if X_imp_cal_masking.shape[1] > len(pattern):
                nb = X_imp_cal_masking.shape[1] - len(pattern)
                pattern_ext = np.append(pattern, np.full(nb, 0))
            else:
                pattern_ext = pattern
            X_imp_cal_masking[:, pattern_ext == 1] = np.nan
            M_cal_masking[:, pattern == 1] = 1
            if reweight:
                shift_scores = mask_class_model.eval(np.column_stack([X_imp_cal_masking, Y_cal]))
                X_ampute_test = features_test.copy()
                X_ampute_test[M_test == 1] = np.nan
                X_ampute_test = X_ampute_test[test_index]
            if accept_reject:
                shift_scores = mask_class_model.normalize_eval(np.column_stack([X_imp_cal_masking, Y_cal]))
                random_uniform = np.random.uniform(0, 1, size=len(shift_scores))
                if mechanism == 'MCAR':
                    include_mask = utils.check_condition(M_cal_unchanged, pattern_ext)
                else:
                    include_mask = np.zeros_like(random_uniform, dtype=bool)
                accepted_indices = np.where((random_uniform < shift_scores) | include_mask)[0]
                X_imp_cal_masking = X_imp_cal_masking[accepted_indices]
                M_cal_masking = M_cal_masking[accepted_indices]
                Y_cal_masking = Y_cal_masking[accepted_indices]
            if not empty:
                X_imp_cal_masking = model_imputer.transform(X_imp_cal_masking)
                features_cal = X_imp_cal_masking
                cal_predictions = predict_basemodel(fitted_basemodel, features_cal, target, basemodel, alpha)
                scores = np.maximum(cal_predictions['y_inf'] - Y_cal_masking, Y_cal_masking - cal_predictions['y_sup'])
            if exact == True:
                if not empty:
                    q_scores_cal = utils.quantile_corrected(scores, alpha=alpha)
                    if reweight:
                        X_to_pred = copy.deepcopy(X_mis_test[(np.array(groups_test) == id_pattern).flatten(), :])
                        X_to_pred_masking = copy.deepcopy(X_to_pred)
                        X_to_pred_masking = model_imputer.transform(X_to_pred_masking)
                        features_test_pattern = X_to_pred_masking
                        preds_this_test = predict_basemodel(fitted_basemodel, features_test_pattern, target, basemodel, alpha)
                        min_score = min(scores) - self.weighted_search_range
                        max_score = max(scores) + self.weighted_search_range
                        grid_number = self.weighted_grid_number
                        each_grid = (-min_score + max_score) / grid_number
                        for grid in np.linspace(-max_score, -min_score, grid_number):
                            y_inf = preds_this_test['y_inf'] + grid
                            shift_scores_y = mask_class_model.eval(np.column_stack([X_ampute_test, y_inf.reshape(-1, 1)]))
                            test_len = shift_scores_y.shape[0]
                            array_n = np.asarray(shift_scores).reshape(1, -1)
                            array_m = np.asarray(shift_scores_y).reshape(-1, 1)
                            combined_y = np.hstack([array_n.repeat(len(array_m), axis=0), array_m])
                            weights_y = combined_y / combined_y.sum(axis=1, keepdims=True)
                            expanded_array_y = np.tile(scores, (test_len, 1))
                            expanded_array_y = np.hstack((expanded_array_y, np.full((test_len, 1), np.nan)))
                            quantiles_y = utils.weighted_quantile_matrix(expanded_array_y, weights_y, q=1 - alpha)
                            score_y = np.maximum(preds_this_test['y_inf'] - y_inf, y_inf - preds_this_test['y_sup'])
                            indices = np.where(score_y <= quantiles_y)[0]
                            pred_test_reweight['y_inf'][test_index_list[indices]] = np.minimum(pred_test_reweight['y_inf'][test_index_list[indices]], y_inf[indices])
                            y_sup = preds_this_test['y_sup'] - grid
                            shift_scores_y = mask_class_model.eval(np.column_stack([X_ampute_test, y_sup.reshape(-1, 1)]))
                            test_len = shift_scores_y.shape[0]
                            array_n = np.asarray(shift_scores).reshape(1, -1)
                            array_m = np.asarray(shift_scores_y).reshape(-1, 1)
                            combined_y = np.hstack([array_n.repeat(len(array_m), axis=0), array_m])
                            weights_y = combined_y / combined_y.sum(axis=1, keepdims=True)
                            expanded_array_y = np.tile(scores, (test_len, 1))
                            expanded_array_y = np.hstack((expanded_array_y, np.full((test_len, 1), np.nan)))
                            quantiles_y = utils.weighted_quantile_matrix(expanded_array_y, weights_y, q=1 - alpha)
                            score_y = np.maximum(preds_this_test['y_inf'] - y_sup, y_sup - preds_this_test['y_sup'])
                            indices = np.where(score_y <= quantiles_y)[0]
                            pred_test_reweight['y_sup'][test_index_list[indices]] = np.maximum(pred_test_reweight['y_sup'][test_index_list[indices]], y_sup[indices])
                        index_inf_to_fill = np.where(pred_test_reweight['y_inf'][test_index_list] == np.inf)[0]
                        pred_test_reweight['y_inf'][test_index_list[index_inf_to_fill]] = y_inf[index_inf_to_fill]
                        index_sup_to_fill = np.where(pred_test_reweight['y_sup'][test_index_list] == -np.inf)[0]
                        pred_test_reweight['y_sup'][test_index_list[index_sup_to_fill]] = y_sup[index_sup_to_fill]
                        pred_test_reweight['y_sup'][test_index_list] = pred_test_reweight['y_sup'][test_index_list] + each_grid
                        pred_test_reweight['y_inf'][test_index_list] = pred_test_reweight['y_inf'][test_index_list] - each_grid
                    else:
                        q_scores_test[(np.array(groups_test) == id_pattern).flatten()] = q_scores_cal
                else:
                    q_scores_test[(np.array(groups_test) == id_pattern).flatten()] = np.inf
            else:
                assert not exact
                X_to_pred = copy.deepcopy(X_mis_test[(np.array(groups_test) == id_pattern).flatten(), :])
                M_to_pred = copy.deepcopy(M_test[(np.array(groups_test) == id_pattern).flatten(), :])
                n_current = X_to_pred.shape[0]
                patterns_cal = np.unique(M_cal_masking, axis=0)
                ids_cal = list(map(utils.pattern_to_id, patterns_cal.astype(int)))
                groups_cal_masking = list(map(utils.pattern_to_id, M_cal_masking.astype(int)))
                nb_cal = len(scores)
                all_preds = {'y_inf': np.empty((n_current, nb_cal)), 'y_sup': np.empty((n_current, nb_cal))}
                for idp_cal, id_pattern_cal in enumerate(ids_cal):
                    idx_cal_masking = (np.array(groups_cal_masking) == id_pattern_cal).flatten()
                    nb_mask = np.sum(idx_cal_masking)
                    all_preds['y_inf'][:, idx_cal_masking] = np.tile(scores[idx_cal_masking], (n_current, 1))
                    all_preds['y_sup'][:, idx_cal_masking] = np.tile(scores[idx_cal_masking], (n_current, 1))
                    pattern_masking = patterns_cal[idp_cal]
                    X_to_pred_masking = copy.deepcopy(X_to_pred)
                    M_to_pred_masking = copy.deepcopy(M_to_pred)
                    X_to_pred_masking[:, pattern_masking == 1] = np.nan
                    M_to_pred_masking[:, pattern_masking == 1] = 1
                    X_to_pred_masking = model_imputer.transform(X_to_pred_masking)
                    features_test_pattern = X_to_pred_masking
                    preds_k = predict_basemodel(fitted_basemodel, features_test_pattern, target, basemodel, alpha)
                    all_preds['y_inf'][:, idx_cal_masking] = -np.tile(preds_k['y_inf'], (nb_mask, 1)).T + all_preds['y_inf'][:, idx_cal_masking]
                    all_preds['y_sup'][:, idx_cal_masking] = np.tile(preds_k['y_sup'], (nb_mask, 1)).T + all_preds['y_sup'][:, idx_cal_masking]
                if (1 - alpha) * (1 + 1 / nb_cal) > 1:
                    pred_test['y_inf'][(np.array(groups_test) == id_pattern).flatten()] = [-np.inf] * n_current
                    pred_test['y_sup'][(np.array(groups_test) == id_pattern).flatten()] = [np.inf] * n_current
                else:
                    pred_test['y_inf'][(np.array(groups_test) == id_pattern).flatten()] = -np.quantile(all_preds['y_inf'], (1 - alpha) * (1 + 1 / nb_cal), axis=1)
                    pred_test['y_sup'][(np.array(groups_test) == id_pattern).flatten()] = np.quantile(all_preds['y_sup'], (1 - alpha) * (1 + 1 / nb_cal), axis=1)
        if exact == True:
            if reweight:
                if reweight_with_y:
                    pred_test = predict_basemodel(fitted_basemodel, features_test, target, basemodel, alpha)
                    interval_predictions = {'y_inf': pred_test['y_inf'] - q_scores_test, 'y_sup': pred_test['y_sup'] + q_scores_test}
                else:
                    interval_predictions = {'y_inf': pred_test_reweight['y_inf'], 'y_sup': pred_test_reweight['y_sup']}
            else:
                pred_test = predict_basemodel(fitted_basemodel, features_test, target, basemodel, alpha)
                interval_predictions = {'y_inf': pred_test['y_inf'] - q_scores_test, 'y_sup': pred_test['y_sup'] + q_scores_test}
        else:
            interval_predictions = {'y_inf': pred_test['y_inf'], 'y_sup': pred_test['y_sup']}
        return interval_predictions

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
        with threadpool_limits(limits=int(os.environ.get('EXPERIMENT_THREADS', '1'))):

            def objective(trial):
                params = {'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True), 'max_iter': trial.suggest_int('max_iter', 50, 400), 'max_depth': trial.suggest_int('max_depth', 2, 6), 'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 80), 'l2_regularization': trial.suggest_float('l2_regularization', 1e-06, 10.0, log=True), 'max_bins': trial.suggest_int('max_bins', 128, 255), 'random_state': 42}
                model = fit_basemodel(x_fit, y_fit, target='Quantiles', basemodel='histgbdt', alpha=alpha, params_basemodel=params)
                loss = evaluate_model(model, x_val, y_val, alpha)
                return loss
            sampler = optuna.samplers.TPESampler(seed=42)
            study = optuna.create_study(direction='minimize', sampler=sampler)
            study.optimize(objective, n_trials=n_trials)
            best_params = study.best_params
            best_params['random_state'] = 42
            best_model = fit_basemodel(x_fit, y_fit, target=self.model_type, basemodel='histgbdt', alpha=alpha, params_basemodel=best_params)
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
        alpha = self.args.alpha
        exact = False
        rep_indices = range(args.reps) if args.replication_index is None else [args.replication_index]
        for k in rep_indices:
            if not 0 <= k < args.reps:
                raise ValueError('replication_index must be in [0, reps).')
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
                idx = np.random.RandomState(42 + k).permutation(n_total)
                train_idx, val_idx = (idx[:n_sub], idx[n_sub:])
                x_base, y_base = (x_train[train_idx], y_train[train_idx])
                x_score, y_score = (x_train[val_idx], y_train[val_idx])
            else:
                x_base, y_base = (x_train, y_train)
                x_score, y_score = (x_train, y_train)
            print('Tuning protocol: training_only_v2 (normalization also training-only)')
            print('Calibration responses are used only after all hyperparameters are frozen.')
            reg, base_best_params, _ = self.tune_histgbdt(x_base, y_base, x_score, y_score, alpha=alpha, n_trials=args.tuning_trials)
            score_fn, score_inv_lb, score_inv_ub = make_score_fns(reg, model_type=self.model_type, alpha=alpha)
            scores_train = score_fn(x_score, y_score)
            embed_fit_idx, embed_val_idx = training_validation_indices(len(x_score), replication_idx=k)
            embed_type = getattr(args, 'embed_type', 'tree')
            embedding_best_params = {}
            model_09 = None
            mice_imputer = None
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
                    study.optimize(wrapped_obj, n_trials=args.tuning_trials, show_progress_bar=False)
                    embedding_best_params = study.best_trial.params

                    def train_quantile_model(X, y, quantile):
                        _, fit_final, _ = get_quantile_adapter(q_model_name, quantile_alpha=quantile)
                        return fit_final(X, y, embedding_best_params, nrounds=args.n_treespace)
                    model_09 = train_quantile_model(x_score, scores_train, 1 - alpha)
                else:
                    model_09 = None
                    embedding_best_params = {}
                phi_fn, phi_mu, phi_std, phi_meta = make_phi_fn_with_norm_new(model=model_09, n_treespace=args.n_treespace, fit_X=x_score, one_hot_encode_batch=self.one_hot_encode_batch, concat_mask=args.concat_vector, scale_mask=True, use_mask_patterns=args.use_mask_patterns, M_calib=np.isnan(x_score), include_other_pattern=False)
            elif embed_type == 'mice':
                from sklearn.experimental import enable_iterative_imputer
                from sklearn.impute import IterativeImputer
                mice_imputer = IterativeImputer(sample_posterior=False, max_iter=50, tol=0.01, random_state=42, keep_empty_features=True).fit(x_score)
                phi_fn, phi_mu, phi_std, phi_meta = make_phi_fn_with_norm_new(model=None, n_treespace=0, fit_X=x_score, one_hot_encode_batch=self.one_hot_encode_batch, concat_mask=args.concat_vector, scale_mask=True, use_mask_patterns=args.use_mask_patterns, M_calib=np.isnan(x_score), covariate_imputer=mice_imputer)
            else:
                raise ValueError(f"Unsupported embed_type '{embed_type}'.")
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
            if args.use_mask_patterns and embed_type in ('tree', 'mice'):
                phi_fn, phi_mu, phi_std, phi_meta = make_phi_fn_with_norm_new(model=model_09, n_treespace=args.n_treespace, fit_X=x_score, one_hot_encode_batch=self.one_hot_encode_batch, concat_mask=args.concat_vector, scale_mask=True, use_mask_patterns=True, M_calib=m_cal, covariate_imputer=mice_imputer)
                p = phi_fn(x_score[:1]).shape[1]
                intercept_col = phi_meta['intercept_col']
                d_mask = int(phi_meta['d_mask'])
            protocol_metadata = {'protocol': 'training_only_v2', 'replication': k, 'split_seed': 42 + k, 'base_best_params': base_best_params, 'embedding_best_params': embedding_best_params, 'best_lambda': float(best_lambda), 'normalization_split': 'x_score', 'calibration_used_for_tuning': False, 'mask_dictionary_from_calibration': bool(args.use_mask_patterns), 'lambda_tuning_dictionary': 'training_patterns' if args.use_mask_patterns else 'raw_coordinates', 'penalize_mask': penalize_mask, 'penalize_intercept': False, 'has_intercept': bool(phi_meta['has_intercept']), 'actual_sizes': {'training_pool': len(x_train), 'base_fit': len(x_base), 'selection': len(x_score), 'calibration': len(x_cal), 'test': len(x_test)}, 'args': vars(args)}
            with open(os.path.join(self.output_dir, 'protocol.json'), 'w') as f:
                json.dump(protocol_metadata, f, indent=2, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))
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
                solve_cvxpy_with_fallback(prob, prefer='MOSEK', warm_start=False)
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
            cond_conf = CondConf(score_fn, cache_test_features(phi_fn, x_test), infinite_params=infinite_params)
            cond_conf.setup_problem(x_cal, y_cal)
            test_paral = int(os.environ.get('TEST_PARALLEL_JOBS', '0'))
            if test_paral > 0 and self.model_type != 'Quantiles':
                raise ValueError('Parallel prediction is enabled only for deterministic Quantiles mode.')
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

    def run_sota(self, method):
        args = self.args
        data = self.data
        alpha = args.alpha
        exact, reweight, accept_reject = resolve_sota_flags(method)
        config = get_sota_default_config()
        config.update(alpha=alpha, accept_reject=accept_reject, reweight=reweight, exact=exact)
        config['pre_impute'] = bool(accept_reject or reweight)
        config['split_train'] = args.split_train
        config['replication_index'] = args.replication_index
        config['tuning_trials'] = args.tuning_trials
        sota_exp = Sota_experiments_to_compare(data=data, **config)
        results = sota_exp.run_experiments()
        phi_meta = {'d_mask': 0, 'mask_encoding': None, 'mask_patterns': None, 'include_other_pattern': False}
        for k in sota_exp.replication_indices:
            print(f'\n===== Replication {k + 1}/{args.reps} =====')
            m_train = data['M']['Train'][k]
            m_cal = data['M']['Cal'][k]
            m_test = data['M']['Test']['fixed_nb_sample_pattern'][k]
            y_test = data['Y']['Test']['fixed_nb_sample_pattern'][k]
            preds = sota_exp.results_by_rep[k]['fixed_nb_sample_pattern']
            lbs, ubs = (np.asarray(preds['y_inf']), np.asarray(preds['y_sup']))
            with open(os.path.join(self.output_dir, 'protocol.json'), 'w') as f:
                json.dump(dict(sota_exp.protocol_by_rep[k], args=vars(args)), f, indent=2, default=lambda v: v.tolist() if hasattr(v, 'tolist') else str(v))
            coverage = np.mean((y_test >= lbs) & (y_test <= ubs))
            avg_len = np.mean(ubs - lbs)
            print(f'Coverage={coverage:.3f}, Length={avg_len:.3f}')
            rep_rows = self._aggregate_results(k, m_cal, m_test, lbs, ubs, y_test, m_train=m_train, phi_meta=phi_meta)
            mask_df = self._build_mask_df(epoch=k)
            rep_df = self._build_mask_df_from_rows(rep_rows, epoch=k, replication=k)
            mask_df = self._ensure_phi_pool_columns(mask_df)
            rep_df = self._ensure_phi_pool_columns(rep_df)
            mask_df = self._oracle_lengths(mask_df)
            rep_df = self._oracle_lengths(rep_df)
            self._save_artifacts(mask_df, phi_mu=None, phi_std=None, phi_meta=None, rep_df=rep_df, replication=k)
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
