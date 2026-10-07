import os
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

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
