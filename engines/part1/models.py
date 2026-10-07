import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import log_loss
import torch
from sklearn.preprocessing import StandardScaler
import utils

def get_quantile_adapter(lib_name: str, quantile_alpha: float):
    lib = lib_name.lower()
    if lib in ('lgbm', 'lightgbm', 'lightgbm_reg'):
        import lightgbm as lgb

        def suggest(trial):
            return {'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True), 'num_leaves': trial.suggest_int('num_leaves', 8, 64), 'max_depth': trial.suggest_int('max_depth', 3, 10), 'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 50), 'feature_fraction': trial.suggest_float('feature_fraction', 0.8, 1.0), 'bagging_fraction': trial.suggest_float('bagging_fraction', 0.8, 1.0), 'bagging_freq': trial.suggest_int('bagging_freq', 0, 5), 'lambda_l1': trial.suggest_float('lambda_l1', 1e-08, 0.1, log=True), 'lambda_l2': trial.suggest_float('lambda_l2', 1e-08, 1.0, log=True)}

        def fit_once(X, y, params, nrounds):
            p = dict(params)
            p.update({'objective': 'quantile', 'alpha': float(quantile_alpha), 'metric': 'quantile', 'verbosity': -1, 'num_threads': int(os.environ.get('EXPERIMENT_THREADS', '1'))})
            dtrain = lgb.Dataset(X, label=y)
            return lgb.train(p, dtrain, num_boost_round=nrounds)

        def predict(model, X):
            return model.predict(X)
        return (suggest, fit_once, predict)
    elif lib in ('catboost', 'cat', 'cbr'):
        from catboost import CatBoostRegressor

        def suggest(trial):
            return {'depth': trial.suggest_int('depth', 3, 10), 'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.2, log=True), 'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 20.0, log=True), 'random_strength': trial.suggest_float('random_strength', 0.0, 1.0), 'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 1.0), 'border_count': trial.suggest_int('border_count', 64, 255)}

        def fit_once(X, y, params, nrounds):
            model = CatBoostRegressor(loss_function=f'Quantile:alpha={float(quantile_alpha)}', iterations=int(nrounds), verbose=False, thread_count=int(os.environ.get('EXPERIMENT_THREADS', '1')), allow_writing_files=False, **params)
            model.fit(X, y, verbose=False)
            return model

        def predict(model, X):
            return model.predict(X, thread_count=int(os.environ.get('EXPERIMENT_THREADS', '1')))
        return (suggest, fit_once, predict)
    elif lib in ('xgboost', 'xgb'):
        import xgboost as xgb

        def suggest(trial):
            return {'eta': trial.suggest_float('eta', 0.001, 0.3, log=True), 'max_depth': trial.suggest_int('max_depth', 3, 16), 'min_child_weight': trial.suggest_float('min_child_weight', 1.0, 20.0, log=True), 'subsample': trial.suggest_float('subsample', 0.5, 1.0), 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0), 'alpha': trial.suggest_float('alpha', 0.0, 1.0), 'lambda': trial.suggest_float('lambda', 0.0, 5.0), 'tree_method': 'hist'}

        def fit_once(X, y, params, nrounds):
            p = dict(params)
            p.update({'objective': 'reg:quantileerror', 'quantile_alpha': float(quantile_alpha), 'verbosity': 0, 'nthread': int(os.environ.get('EXPERIMENT_THREADS', '1'))})
            dtrain = xgb.DMatrix(X, label=y, nthread=int(os.environ.get('EXPERIMENT_THREADS', '1')))
            return xgb.train(p, dtrain, num_boost_round=int(nrounds))

        def predict(model, X):
            dtest = xgb.DMatrix(X, nthread=int(os.environ.get('EXPERIMENT_THREADS', '1')))
            return model.predict(dtest)
        return (suggest, fit_once, predict)
    else:
        raise ValueError(f"Unknown q_model: {lib_name}. Use one of ['lightgbm','catboost','xgboost'].")

def fit_basemodel(X_train, Y_train, target='Mean', basemodel='Linear', alpha=0.1, params_basemodel={}, params_lgm={}):
    if target != 'Quantiles' or basemodel != 'histgbdt':
        raise ValueError('The experiment base model is HistGBDT quantile regression')
    a_low, a_high = (alpha / 2, 1 - alpha / 2)
    return {'q_low': HistGradientBoostingRegressor(loss='quantile', quantile=a_low, **params_basemodel).fit(X_train, Y_train), 'q_high': HistGradientBoostingRegressor(loss='quantile', quantile=a_high, **params_basemodel).fit(X_train, Y_train)}

class Data_for_mask:

    def __init__(self, X, M, numerical_feature=None, batch_size=128, test=0.2, contain_y=False, dataset=None):
        self.N = X.shape[0]
        self.d = X.shape[1]
        self.contain_y = contain_y
        if contain_y:
            self.d_x = X.shape[1] - 1
        else:
            self.d_x = X.shape[1]
        if dataset == 'bike':
            self.d_x = 4
        if numerical_feature is None:
            self.numerical_feature = list(range(self.d))
        else:
            self.numerical_feature = numerical_feature
        self.batch_size = batch_size
        self.test_n = int(self.N * test)
        self.M_original = M.copy()
        self.X = X.copy()
        self.M_num = np.array([[True] * i + [False] * (self.d_x - i) for i in range(self.d_x)])
        self.scaler = StandardScaler()
        self.scaler.fit(self.X)
        self.X = self.scaler.transform(self.X)
        self.X_test = self.X[:self.test_n]
        self.X = self.X[self.test_n:]
        self.X_test_torch = torch.tensor(self.X_test, dtype=torch.float32)
        self.X_torch = torch.tensor(self.X, dtype=torch.float32)
        self.N_train = self.X.shape[0]
        return

    def __len__(self):
        return len(self.X_torch)

    def __iter__(self):
        self.start_idx = 0
        self.indices = torch.arange(len(self.X_torch))
        indices = np.random.choice(self.d_x, size=self.N, replace=True)
        self.M = self.M_num[indices]
        for row in self.M:
            np.random.shuffle(row)
        assert self.M.shape == self.M_original.shape, f'M shape mismatch: M.shape={self.M.shape}, M_original.shape={self.M_original.shape}, {self.X.shape}, {self.M_num.shape}'
        row_check = (self.M & self.M_original).sum(axis=1) == self.M_original.sum(axis=1)
        self.Y = row_check.astype(int)
        if self.contain_y:
            self.M = np.hstack((self.M, np.zeros((self.M.shape[0], 1))))
        self.M_test = self.M[:self.test_n]
        self.Y_test = self.Y[:self.test_n]
        self.M = self.M[self.test_n:]
        self.Y = self.Y[self.test_n:]
        self.M_torch = torch.tensor(self.M, dtype=torch.bool)
        self.Y_torch = torch.tensor(self.Y, dtype=torch.float32).view(-1, 1)
        self.X_nan_torch = self.X_torch.masked_fill(self.M_torch, float('nan'))
        self.M_test_torch = torch.tensor(self.M_test, dtype=torch.bool)
        self.Y_test_torch = torch.tensor(self.Y_test, dtype=torch.float32).view(-1, 1)
        self.X_nan_test_torch = self.X_test_torch.masked_fill(self.M_test_torch, float('nan'))
        return self

    def __next__(self):
        if self.start_idx < self.N_train:
            end_idx = min(self.start_idx + self.batch_size, self.N_train)
            batch_indices = self.indices[self.start_idx:end_idx]
            self.start_idx = end_idx
            return (self.X_nan_torch[batch_indices], self.Y_torch[batch_indices])
        else:
            raise StopIteration

class Miss_proba_simple_model:

    def __init__(self, data, model='xgboost', epoch=300, **hyperparam):
        self.data = data
        self.model_type = model
        self.hyperparam = hyperparam
        self.imputer = utils.get_impute('iterative_ridge')
        self.epoch = epoch
        self.X_train, self.Y_train, self.X_test, self.Y_test = self.fetch_data(epoch=self.epoch)
        if model == 'xgboost':
            self.model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', **hyperparam)
        elif model == 'histgbdt':
            self.model = HistGradientBoostingClassifier(**hyperparam)
        elif model == 'logistic':
            self.model = LogisticRegression(**hyperparam)
        else:
            raise ValueError(f'Unsupported model type: {model}')

    def fetch_data(self, epoch=300):
        all_X, all_Y = ([], [])
        all_X_test, all_Y_test = ([], [])
        for i in range(epoch):
            for x_batch, y_batch in self.data:
                x_numpy = x_batch.numpy()
                y_numpy = y_batch.numpy()
                if i == epoch - 1:
                    all_X_test.append(x_numpy)
                    all_Y_test.append(y_numpy)
                else:
                    all_X.append(x_numpy)
                    all_Y.append(y_numpy)
        X_train = np.vstack(all_X)
        Y_train = np.vstack(all_Y).flatten()
        X_test = np.vstack(all_X_test)
        Y_test = np.vstack(all_Y_test).flatten()
        print(f'Fetched {X_train.shape[0]} training samples, {Y_train.mean() * 100:.2f}% positive.')
        print(f'Fetched {X_test.shape[0]} testing samples, {Y_test.mean() * 100:.2f}% positive.')
        return (X_train, Y_train, X_test, Y_test)

    def train(self):
        if self.model_type == 'logistic':
            train_mask = np.isnan(self.X_train).astype(int)
            x_train_imputed = self.imputer.fit_transform(self.X_train)
            self.X_train = np.hstack((x_train_imputed, train_mask))
            test_mask = np.isnan(self.X_test).astype(int)
            x_test_imputed = self.imputer.transform(self.X_test)
            self.X_test = np.hstack((x_test_imputed, test_mask))
        self.model.fit(self.X_train, self.Y_train)
        y_pred_test = self.model.predict_proba(self.X_test)[:, 1]
        try:
            self.loss = log_loss(self.Y_test, y_pred_test)
        except:
            self.loss = 0
        print(f'Training completed. Test loss (BCE): {self.loss:.4f}')

    def normalize_eval(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = self.data.scaler.transform(x)
        if self.model_type == 'logistic':
            mask = np.isnan(x).astype(int)
            x_imputed = self.imputer.transform(x)
            x = np.hstack((x_imputed, mask))
        proba = self.model.predict_proba(x)[:, 1]
        normalize_proba = np.clip(proba / np.quantile(proba, 0.9), 0, 1)
        return normalize_proba

    def eval(self, x):
        if isinstance(x, torch.Tensor):
            x = x.numpy()
        x = self.data.scaler.transform(x)
        if self.model_type == 'logistic':
            mask = np.isnan(x).astype(int)
            x_imputed = self.imputer.transform(x)
            x = np.hstack((x_imputed, mask))
        proba = self.model.predict_proba(x)[:, 1]
        proba = np.clip(proba, 0, 0.9)
        return proba / (1 - proba)

def predict_basemodel(fitted_basemodel, X_test, target='Mean', basemodel='Linear', alpha=0.1):
    if target != 'Quantiles' or basemodel != 'histgbdt':
        raise ValueError('The experiment base model is HistGBDT quantile regression')
    return {'y_inf': fitted_basemodel['q_low'].predict(X_test), 'y_sup': fitted_basemodel['q_high'].predict(X_test)}

def make_score_fns(reg, model_type='Quantiles', alpha=0.1):
    assert model_type == 'Quantiles'

    def score_fn(x, y):
        predictions = predict_basemodel(reg, x, 'Quantiles', 'histgbdt', alpha)
        return np.maximum(predictions['y_inf'] - y, y - predictions['y_sup'])

    def score_inv_fn(s, x):
        predictions = predict_basemodel(reg, x, 'Quantiles', 'histgbdt', alpha)
        return [predictions['y_inf'] - s, predictions['y_sup'] + s]
    return (score_fn, score_inv_fn, None)
