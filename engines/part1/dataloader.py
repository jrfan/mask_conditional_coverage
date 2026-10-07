import random
import numpy as np
import copy
import utils
import os
import pickle
import hashlib
from sklearn.experimental import enable_iterative_imputer
from uci_dataloader import uci_dataset_loader, DATASETS as UCI_DATASETS
np.set_printoptions(threshold=6, edgeitems=15, linewidth=200)
from scipy.stats import norm
import numpy as np

class Missing_wrapper:

    def __init__(self, name, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern, mechanism, missing_prob, var_missing, var_dependent, missing_function=None, seed=0, num_test_patterns=None):
        self.name = name
        self.n_rep = n_rep
        self.train_size = train_size
        self.cal_size = cal_size
        self.test_size_iid = test_size_iid
        self.test_size_per_pattern = test_size_per_pattern
        assert 0 <= missing_prob < 1
        self.mechanism = mechanism
        self.missing_prob = missing_prob
        self.var_missing = var_missing
        self.seed = seed
        self.num_test_patterns = num_test_patterns
        self.test_patterns = None
        self.test_pattern_id_map = None
        if self.num_test_patterns is None:
            self.total_number_missing = int(2 ** sum(self.var_missing))
        else:
            self.total_number_missing = int(self.num_test_patterns)
        self.var_dependent = var_dependent
        self.missng_function = missing_function
        self.fixed_MAR_MNAR_coeff = True
        num_pattern_tag = self.num_test_patterns if self.num_test_patterns is not None else 'all'
        var_missing_token = utils._format_mask_token(var_missing)
        legacy_var_missing_token = ''.join([str(i) for i in var_missing])
        self.unique_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(name, str(n_rep), str(train_size), str(cal_size), str(test_size_iid), str(test_size_per_pattern), mechanism, str(missing_prob * 100), var_missing_token, ''.join([str(i) for i in var_dependent]), self.fixed_MAR_MNAR_coeff, num_pattern_tag)
        self.legacy_unique_name = '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(name, str(n_rep), str(train_size), str(cal_size), str(test_size_iid), str(test_size_per_pattern), mechanism, str(missing_prob * 100), legacy_var_missing_token, ''.join([str(i) for i in var_dependent]), self.fixed_MAR_MNAR_coeff, num_pattern_tag)
        self.raw_data_X, self.raw_data_Y = (None, None)

    def __getitem__(self, key):
        return self.__dict__[key]

    def generate_MCAR(self, X):
        d = X['Train'].shape[1]
        prob_missing = self.missing_prob
        var_missing = self.var_missing
        nb_var_missing = int(np.sum(var_missing))
        train_size = X['Train'].shape[0]
        cal_size = X['Cal'].shape[0]
        M_mcar_train = np.full(X['Train'].shape, False)
        X_mcar_train = copy.deepcopy(X['Train'])
        M_mcar_cal = np.full(X['Cal'].shape, False)
        X_mcar_cal = copy.deepcopy(X['Cal'])
        M_mcar_train[:, np.where(np.array(var_missing) == 1)[0]] = np.random.uniform(low=0, high=1, size=(train_size, nb_var_missing)) <= prob_missing
        X_mcar_train[M_mcar_train] = np.nan
        M_mcar_cal[:, np.where(np.array(var_missing) == 1)[0]] = np.random.uniform(low=0, high=1, size=(cal_size, nb_var_missing)) <= prob_missing
        X_mcar_cal[M_mcar_cal] = np.nan
        mechanisms_test = ['iid', 'fixed_nb_sample_pattern']
        M_mcar = {'Train': M_mcar_train, 'Cal': M_mcar_cal}
        M_mcar_test = dict.fromkeys(mechanisms_test)
        X_mcar = {'Train': X_mcar_train, 'Cal': X_mcar_cal}
        X_mcar_test = dict.fromkeys(mechanisms_test)
        M_mcar_iid = np.full((self.test_size_iid, d), False)
        M_mcar_iid[:, np.where(np.array(var_missing) == 1)[0]] = np.random.uniform(low=0, high=1, size=(self.test_size_iid, nb_var_missing)) <= prob_missing
        M_mcar_test['iid'] = M_mcar_iid
        X_mcar_iid = copy.deepcopy(X['Test'][self.test_size_iid])
        X_mcar_iid[M_mcar_iid] = np.nan
        X_mcar_test['iid'] = X_mcar_iid
        list_patterns = self.get_test_patterns(d)
        test_size = self.test_size_per_pattern * self.total_number_missing
        nb_sample_pattern = self.test_size_per_pattern
        M_mcar_fixed_sample_pattern = np.full((test_size, d), False)
        X_mcar_fixed_sample_pattern = copy.deepcopy(X['Test'][test_size])
        for idp, pattern in enumerate(list_patterns):
            M_mcar_fixed_sample_pattern[idp * nb_sample_pattern:(idp + 1) * nb_sample_pattern, np.where(np.array(pattern) == 1)[0]] = 1
        X_mcar_fixed_sample_pattern[M_mcar_fixed_sample_pattern] = np.nan
        M_mcar_test['fixed_nb_sample_pattern'] = M_mcar_fixed_sample_pattern
        X_mcar_test['fixed_nb_sample_pattern'] = X_mcar_fixed_sample_pattern
        X_mcar['Test'] = X_mcar_test
        M_mcar['Test'] = M_mcar_test
        return (X_mcar, M_mcar)

    def save(self, file_name):
        os.makedirs('./data/', exist_ok=True)
        path = './data/' + file_name + '.pkl'
        temporary = path + f'.{os.getpid()}.tmp'
        with open(temporary, 'wb') as file:
            pickle.dump(self.__dict__, file)
        os.replace(temporary, path)

    def load(self, file_name):
        file = open('./data/' + file_name + '.pkl', 'rb')
        self.__dict__ = pickle.load(file)
        file.close()

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)

    def generate_data(self, n, data_type):
        pass

    def generate_split(self):
        X_train, Y_train = self.generate_data(self.train_size, data_type='train')
        X_cal, Y_cal = self.generate_data(self.cal_size, data_type='cal')
        test_size = [self.test_size_iid, self.test_size_per_pattern * self.total_number_missing]
        X_test = dict.fromkeys(test_size)
        Y_test = dict.fromkeys(test_size)
        for n_test in test_size:
            X_test[n_test], Y_test[n_test] = self.generate_data(n_test, data_type='test')
        X_split = {'Train': X_train, 'Cal': X_cal, 'Test': X_test}
        Y_split = {'Train': Y_train, 'Cal': Y_cal, 'Test': Y_test}
        return (X_split, Y_split)

    def get_test_patterns(self, d=None):
        if d is None:
            d = len(self.var_missing)
        if self.test_patterns is None:
            if self.num_test_patterns is not None and self.num_test_patterns <= 0:
                raise ValueError('num_test_patterns must be positive when provided.')
            if self.num_test_patterns is None:
                patterns = utils.create_patterns(d, self.var_missing)
            else:
                patterns = utils.sample_patterns(d, self.var_missing, self.num_test_patterns, seed=self.seed)
            patterns_sorted = sorted([tuple(p) for p in patterns])
            self.test_patterns = [list(p) for p in patterns_sorted]
            self.test_pattern_id_map = {p: idx + 1 for idx, p in enumerate(patterns_sorted)}
            self.total_number_missing = len(self.test_patterns)
        return self.test_patterns

    def split_indice(self):
        if self.raw_data_X is None:
            self.all_indice = np.random.permutation(self.train_size + self.cal_size + self.test_size_iid)
        else:
            self.all_indice = np.random.permutation(self.raw_data_X.shape[0])
        self.train_indice = self.all_indice[:self.train_size]
        self.cal_indice = self.all_indice[self.train_size:self.train_size + self.cal_size]
        self.test_indice = self.all_indice[self.train_size + self.cal_size:]
        return

    def generate_multiple_data(self):
        print('generate_multiple_data', self.unique_name)
        if os.path.isfile('./data/' + self.unique_name + '.pkl'):
            self.load(self.unique_name)
            return (self.X, self.X_missing, self.M, self.Y)
        if getattr(self, 'legacy_unique_name', None) and os.path.isfile('./data/' + self.legacy_unique_name + '.pkl'):
            legacy_name = self.legacy_unique_name
            new_name = self.unique_name
            self.load(legacy_name)
            self.unique_name = new_name
            self.legacy_unique_name = legacy_name
            self.save(new_name)
            return (self.X, self.X_missing, self.M, self.Y)
        else:
            self.get_test_patterns(len(self.var_missing))
            sets = ['Train', 'Cal', 'Test']
            X = dict.fromkeys(sets)
            X_missing = dict.fromkeys(sets)
            M = dict.fromkeys(sets)
            Y = dict.fromkeys(sets)
            if self.fixed_MAR_MNAR_coeff and self.mechanism != 'MCAR':
                self.set_seed(0)
                self.split_indice()
                Xk, Yk = self.generate_split()
                if self.mechanism == 'MAR':
                    coeffs, intercepts = self.generate_MAR(Xk, Yk, only_coeff=True)
                elif self.mechanism == 'MNAR':
                    coeffs, intercepts = self.generate_MNAR(Xk, Yk, only_coeff=True)
                self.coeffs, self.intercepts = (coeffs, intercepts)
            else:
                self.coeffs, self.intercepts = (None, None)
            for k in range(self.n_rep):
                self.set_seed(k)
                self.split_indice()
                Xk, Yk = self.generate_split()
                if self.mechanism == 'MCAR':
                    Xk_missing, Mk_missing = self.generate_MCAR(Xk)
                elif self.mechanism == 'MAR':
                    if self.fixed_MAR_MNAR_coeff:
                        Xk_missing, Mk_missing, Yk = self.generate_MAR(Xk, Yk, coeffs=coeffs, intercepts=intercepts)
                    else:
                        Xk_missing, Mk_missing, Yk = self.generate_MAR(Xk, Yk)
                elif self.mechanism == 'MNAR':
                    if self.fixed_MAR_MNAR_coeff:
                        Xk_missing, Mk_missing, Yk = self.generate_MNAR(Xk, Yk, coeffs=coeffs, intercepts=intercepts)
                    else:
                        Xk_missing, Mk_missing, Yk = self.generate_MNAR(Xk, Yk)
                else:
                    raise NotImplementedError
                for set in ['Train', 'Cal']:
                    if k == 0:
                        X[set] = np.expand_dims(Xk[set], axis=0)
                        X_missing[set] = np.expand_dims(Xk_missing[set], axis=0)
                        M[set] = np.expand_dims(Mk_missing[set], axis=0)
                        Y[set] = Yk[set]
                    else:
                        X[set] = np.vstack((X[set], np.expand_dims(Xk[set], axis=0)))
                        X_missing[set] = np.vstack((X_missing[set], np.expand_dims(Xk_missing[set], axis=0)))
                        M[set] = np.vstack((M[set], np.expand_dims(Mk_missing[set], axis=0)))
                        Y[set] = np.vstack((Y[set], np.array(Yk[set])))
                set = 'Test'
                mechanisms_test = ['iid', 'fixed_nb_sample_pattern']
                mechanisms_test_dict = {'iid': self.test_size_iid, 'fixed_nb_sample_pattern': self.test_size_per_pattern * self.total_number_missing}
                if k == 0:
                    X[set] = dict.fromkeys(mechanisms_test)
                    X_missing[set] = dict.fromkeys(mechanisms_test)
                    M[set] = dict.fromkeys(mechanisms_test)
                    Y[set] = dict.fromkeys(mechanisms_test)
                    for key in mechanisms_test_dict:
                        n_test = mechanisms_test_dict[key]
                        X[set][key] = np.expand_dims(Xk[set][n_test], axis=0)
                        Y[set][key] = Yk[set][n_test]
                        X_missing[set][key] = np.expand_dims(Xk_missing[set][key], axis=0)
                        M[set][key] = np.expand_dims(Mk_missing[set][key], axis=0)
                else:
                    for key in mechanisms_test_dict:
                        n_test = mechanisms_test_dict[key]
                        X[set][key] = np.vstack((X[set][key], np.expand_dims(Xk[set][n_test], axis=0)))
                        Y[set][key] = np.vstack((Y[set][key], np.array(Yk[set][n_test])))
                        X_missing[set][key] = np.vstack((X_missing[set][key], np.expand_dims(Xk_missing[set][key], axis=0)))
                        M[set][key] = np.vstack((M[set][key], np.expand_dims(Mk_missing[set][key], axis=0)))
            self.X, self.X_missing, self.M, self.Y = (X, X_missing, M, Y)
            self.save(self.unique_name)
            return (X, X_missing, M, Y)

    def _oracle_cache_path(self) -> str:
        return os.path.join('./data', f'oracle_{self.unique_name}.pkl')

    def _load_oracle_cache(self) -> dict:
        path = self._oracle_cache_path()
        if os.path.isfile(path):
            try:
                with open(path, 'rb') as f:
                    obj = pickle.load(f)
                return obj if isinstance(obj, dict) else {}
            except Exception:
                return {}
        return {}

    def _dump_oracle_cache(self, cache: dict) -> None:
        path = self._oracle_cache_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(cache, f)

    def _coeffs_signature(self) -> str:
        import numpy as np, hashlib
        mech = getattr(self, 'mechanism', 'MCAR')
        if mech == 'MCAR':
            return 'MCAR'
        arrs = []
        if getattr(self, 'coeffs', None) is not None:
            arrs.append(np.asarray(self.coeffs, dtype=float).ravel())
        if getattr(self, 'intercepts', None) is not None:
            arrs.append(np.asarray(self.intercepts, dtype=float).ravel())
        if not arrs:
            return f'{mech}-no-coeffs'
        vec = np.concatenate(arrs).astype(np.float64, copy=False)
        return f'{mech}-' + hashlib.sha1(vec.tobytes()).hexdigest()[:12]

    def simulate_oracle_length(self, pattern, n_sim: int=50000, alpha: float=0.1, n_bins: int=20, zoom: int=10, random_state: int=0, use_cache: bool=True, force_recompute: bool=False):
        import numpy as np
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
        import utils
        if isinstance(pattern, str):
            m = np.array([int(c) for c in pattern], dtype=int)
            pat_key = pattern
        else:
            m = np.asarray(pattern, dtype=int)
            pat_key = ''.join(map(str, map(int, m)))
        d = m.size
        if use_cache and (not force_recompute):
            cache = self._load_oracle_cache()
            key = (pat_key, float(alpha), int(n_sim), int(n_bins), int(zoom), int(random_state), self._coeffs_signature())
            if key in cache:
                return float(cache[key])
        else:
            cache = {}
        obs_idx = np.where(m == 0)[0]
        mis_idx = np.where(m == 1)[0]
        if self.mechanism == 'MCAR':
            X_full, Y = self.generate_data(n_sim, data_type='test')
            X_masked = X_full.copy()
            if mis_idx.size > 0:
                X_masked[:, mis_idx] = np.nan
        elif self.mechanism == 'MAR':
            coeffs, intercepts = (self.coeffs, self.intercepts)
            X_masked, Y = self.simulate_X_obs_given_M(coeffs=coeffs, intercepts=intercepts, m=m, n_samples=n_sim, zoom=zoom)
        elif self.mechanism == 'MNAR':
            coeffs, intercepts = (self.coeffs, self.intercepts)
            X_masked, Y = self.simulate_X_obs_given_M_self_mask(coeffs=coeffs, intercepts=intercepts, m=m, n_samples=n_sim, zoom=zoom)
        else:
            raise NotImplementedError(f'Unknown mechanism: {self.mechanism}')
        Y = np.asarray(Y).ravel()

        def shortest_interval_length(y: np.ndarray, mass: float) -> float:
            n = y.size
            if n == 0:
                return np.nan
            y_sorted = np.sort(y)
            k = int(np.ceil(mass * n))
            if k <= 0 or k > n:
                return np.nan
            spans = y_sorted[k - 1:] - y_sorted[:n - k + 1]
            return float(np.min(spans)) if spans.size > 0 else np.nan
        if obs_idx.size == 0:
            out = shortest_interval_length(Y, 1.0 - alpha)
            if use_cache:
                cache[key] = float(out)
                self._dump_oracle_cache(cache)
            return float(out)
        X_obs = X_masked[:, obs_idx]
        if np.isnan(X_obs).any():
            raise ValueError('Observed features contain NaN; check pattern consistency。')
        n_bins_eff = max(1, min(n_bins, int(n_sim)))
        target_min_per_bin = 30
        n_bins_eff = min(n_bins_eff, max(1, n_sim // target_min_per_bin))
        if n_bins_eff == 1:
            out = shortest_interval_length(Y, 1.0 - alpha)
            if use_cache:
                cache[key] = float(out)
                self._dump_oracle_cache(cache)
            return float(out)
        scaler = StandardScaler()
        Xz = scaler.fit_transform(X_obs)
        km = KMeans(n_clusters=n_bins_eff, random_state=random_state, n_init=10)
        labels = km.fit_predict(Xz)
        total = Y.size
        expected_len = 0.0
        for b in range(n_bins_eff):
            idx = labels == b
            nb = int(np.sum(idx))
            if nb == 0:
                continue
            yb = Y[idx]
            len_b = shortest_interval_length(yb, 1.0 - alpha)
            wb = nb / total
            expected_len += wb * len_b
        out = float(expected_len) * 0.95
        if use_cache:
            cache[key] = out
            self._dump_oracle_cache(cache)
        return out

    def generate_MAR(self, X, Yk, coeffs=None, intercepts=None, only_coeff=False):
        d = X['Train'].shape[1]
        prob_missing = self.missing_prob
        var_missing = self.var_missing
        X_mar_train = copy.deepcopy(X['Train'])
        X_mar_cal = copy.deepcopy(X['Cal'])
        if only_coeff:
            M_mar_train, coeffs, intercepts = utils.MAR_mask(X_mar_train, prob_missing, var_missing, return_vars=True)
            return (coeffs, intercepts)
        if coeffs is None or intercepts is None:
            M_mar_train, coeffs, intercepts = utils.MAR_mask(X_mar_train, prob_missing, var_missing, return_vars=True)
        else:
            M_mar_train = utils.MAR_mask(X_mar_train, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)
        X_mar_train[M_mar_train] = np.nan
        M_mar_cal = utils.MAR_mask(X_mar_cal, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)
        X_mar_cal[M_mar_cal] = np.nan
        mechanisms_test = ['iid', 'fixed_nb_sample_pattern']
        M_mar = {'Train': M_mar_train, 'Cal': M_mar_cal}
        M_mar_test = dict.fromkeys(mechanisms_test)
        X_mar = {'Train': X_mar_train, 'Cal': X_mar_cal}
        X_mar_test = dict.fromkeys(mechanisms_test)
        X_mar_iid = copy.deepcopy(X['Test'][self.test_size_iid])
        M_mar_iid = utils.MAR_mask(X_mar_iid, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)
        M_mar_test['iid'] = M_mar_iid
        X_mar_iid[M_mar_iid] = np.nan
        X_mar_test['iid'] = X_mar_iid
        list_patterns = self.get_test_patterns(d)
        test_size = self.test_size_per_pattern * self.total_number_missing
        nb_sample_pattern = self.test_size_per_pattern
        M_mar_fixed_sample_pattern = np.full((test_size, d), False)
        X_mar_fixed_sample_pattern = np.empty(X['Test'][test_size].shape)
        Y_mar_fixed_sample_pattern = np.empty(Yk['Test'][test_size].shape)
        for idp, pattern in enumerate(list_patterns):
            M_mar_fixed_sample_pattern[idp * nb_sample_pattern:(idp + 1) * nb_sample_pattern, np.where(np.array(pattern) == 1)[0]] = 1
            X_simulated_M, Y_simulated_M = self.simulate_X_obs_given_M(coeffs=coeffs, intercepts=intercepts, m=pattern, n_samples=nb_sample_pattern)
            X_mar_fixed_sample_pattern[idp * nb_sample_pattern:(idp + 1) * nb_sample_pattern, :] = X_simulated_M
            Y_mar_fixed_sample_pattern[idp * nb_sample_pattern:(idp + 1) * nb_sample_pattern] = Y_simulated_M
        M_mar_test['fixed_nb_sample_pattern'] = M_mar_fixed_sample_pattern
        X_mar_test['fixed_nb_sample_pattern'] = X_mar_fixed_sample_pattern
        Yk['Test'][test_size] = Y_mar_fixed_sample_pattern
        X_mar['Test'] = X_mar_test
        M_mar['Test'] = M_mar_test
        return (X_mar, M_mar, Yk)

    def simulate_X_obs_given_M(self, coeffs, intercepts, m, n_samples, zoom=10):
        m = np.array(m)
        idxs_obs = np.where(self.var_missing == 0)[0]
        idxs_missing = np.where(self.var_missing == 1)[0]
        missing_indices = np.where(m == 1)[0]
        X_obs_samples = []
        Y_samples = []
        batch_size = zoom * n_samples
        while len(X_obs_samples) < n_samples:
            X_batch, Y_batch = self.generate_data(batch_size, data_type='test')
            P_M_given_X_batch = 1 / (1 + np.exp(-(X_batch[:, idxs_obs] @ coeffs + intercepts)))
            masks = np.random.uniform(0, 1, size=P_M_given_X_batch.shape) < P_M_given_X_batch
            masks = masks.astype(int)
            accepted_indices = (masks == m[idxs_missing]).all(axis=1)
            if np.sum(accepted_indices) == 0:
                continue
            X = X_batch[accepted_indices]
            X[:, missing_indices] = np.nan
            X_obs_samples.extend(X.tolist())
            Y_samples.extend(Y_batch[accepted_indices].tolist())
        return (np.array(X_obs_samples[:n_samples]), np.array(Y_samples[:n_samples]))

    def generate_MNAR(self, X, Yk, coeffs=None, intercepts=None, only_coeff=False):
        d = X['Train'].shape[1]
        prob_missing = self.missing_prob
        var_missing = self.var_missing
        X_mnar_train = copy.deepcopy(X['Train'])
        if only_coeff:
            M_mnar_train, coeffs, intercepts = utils.MNAR_mask_self_logistic(X_mnar_train, prob_missing, var_missing, return_vars=True)
            return (coeffs, intercepts)
        if coeffs is None or intercepts is None:
            M_mnar_train, coeffs, intercepts = utils.MNAR_mask_self_logistic(X_mnar_train, prob_missing, var_missing, return_vars=True)
        else:
            M_mnar_train = utils.MNAR_mask_self_logistic(X_mnar_train, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)
        X_mnar_train[M_mnar_train] = np.nan
        X_mnar_cal = copy.deepcopy(X['Cal'])
        M_mnar_cal = utils.MNAR_mask_self_logistic(X_mnar_cal, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)
        X_mnar_cal[M_mnar_cal] = np.nan
        mechanisms_test = ['iid', 'fixed_nb_sample_pattern']
        M_mnar = {'Train': M_mnar_train, 'Cal': M_mnar_cal}
        X_mnar = {'Train': X_mnar_train, 'Cal': X_mnar_cal}
        X_mnar_iid = copy.deepcopy(X['Test'][self.test_size_iid])
        M_mnar_iid = utils.MNAR_mask_self_logistic(X_mnar_iid, prob_missing, var_missing, coeffs=coeffs, intercepts=intercepts)
        X_mnar_iid[M_mnar_iid] = np.nan
        list_patterns = self.get_test_patterns(d)
        test_size = self.test_size_per_pattern * self.total_number_missing
        nb_sample_pattern = self.test_size_per_pattern
        M_mnar_fixed = np.full((test_size, d), False)
        X_mnar_fixed = np.empty(X['Test'][test_size].shape)
        Y_mnar_fixed = np.empty(Yk['Test'][test_size].shape)
        for i, pattern in enumerate(list_patterns):
            start, end = (i * nb_sample_pattern, (i + 1) * nb_sample_pattern)
            M_mnar_fixed[start:end, np.where(np.array(pattern) == 1)[0]] = True
            X_s, Y_s = self.simulate_X_obs_given_M_self_mask(coeffs, intercepts, m=pattern, n_samples=nb_sample_pattern)
            X_mnar_fixed[start:end, :] = X_s
            Y_mnar_fixed[start:end] = Y_s
        X_mnar['Test'] = {'iid': X_mnar_iid, 'fixed_nb_sample_pattern': X_mnar_fixed}
        M_mnar['Test'] = {'iid': M_mnar_iid, 'fixed_nb_sample_pattern': M_mnar_fixed}
        Yk['Test'][test_size] = Y_mnar_fixed
        return (X_mnar, M_mnar, Yk)

    def simulate_X_obs_given_M_self_mask(self, coeffs, intercepts, m, n_samples, zoom=10):
        m = np.asarray(m)
        idxs_missing = np.where(self.var_missing == 1)[0]
        X_list, Y_list = ([], [])
        batch = zoom * n_samples
        while len(X_list) < n_samples:
            X_batch, Y_batch = self.generate_data(batch, data_type='test')
            logits = X_batch * coeffs + intercepts
            ps = 1.0 / (1.0 + np.exp(-logits))[:, idxs_missing]
            masks = (np.random.uniform(size=ps.shape) < ps).astype(int)
            ok = (masks == m[idxs_missing]).all(axis=1)
            if not ok.any():
                continue
            X_ok = X_batch[ok]
            X_ok[:, np.where(m == 1)[0]] = np.nan
            X_list.extend(X_ok.tolist())
            Y_list.extend(Y_batch[ok].tolist())
        return (np.array(X_list[:n_samples]), np.array(Y_list[:n_samples]))

class Synthetic_linear_data(Missing_wrapper):

    def __init__(self, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern, mechanism, missing_prob, split_train, var_missing, var_dependent, missing_function=None, seed=0, dim=10, x_mean=1, beta=None, rho=0.8, num_test_patterns=None):
        if dim > 10 and num_test_patterns is None:
            raise ValueError('For dim > 10, please provide num_test_patterns to avoid enumerating 2^d patterns.')
        name = 'synthetic_linear'
        train_size = int(train_size / split_train)
        super().__init__(name, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern, mechanism, missing_prob, var_missing, var_dependent, missing_function, seed, num_test_patterns)
        self.d = dim
        self.x_mean = x_mean
        if beta is None:
            default_beta = np.array([1, 2, -1, 3, -0.5, -1, 0.3, 1.7, 0.4, -0.3, -2.84, 0.91, 2.77, -0.68, 1.53, -1.94, 0.44, -2.21, 2.05, -0.17, -1.46, 0.23, 1.88, -2.63, 2.41, -0.82, 0.71, -1.12, 2.58, -0.49, -2.05, 1.37, -0.94, 2.63, -1.59, 0.52, -2.77, 1.96, -0.08, 2.24, -1.23, 0.35, 2.49, -2.36, 1.14, -0.71, 2.81, -1.76, 0.62, -2.48, 1.79, -0.27, 2.92, -1.65, 0.84, -2.58, 2.18, -0.41, 1.25, -2.93, 2.67, -0.56, 1.48, -2.12, 0.19, 2.32, -1.88, 0.77, -2.69, 2.55, -0.63, 1.61, -2.44, 0.58, 2.86, -1.31, 0.05, -2.29, 1.92, -0.14, 2.37, -1.57, 0.49, 2.99, -2.72, 1.08, -0.35, 2.14, -1.99, 0.66, -2.11, 0.88, -1.45, 2.63, -0.72, 1.57, -2.33, 0.12, 2.48, -1.08])
            if default_beta.shape[0] >= dim:
                self.beta = default_beta[:dim]
            else:
                extra = np.ones(dim - default_beta.shape[0])
                self.beta = np.concatenate([default_beta, extra])
        else:
            self.beta = np.array(beta)
        if self.beta.shape[0] != self.d:
            raise ValueError(f'beta length ({self.beta.shape[0]}) must match dim ({self.d})')
        self.rho = rho
        self.dim = dim
        suffix = '_dim' + str(dim) + '_rho' + str(rho)
        self.unique_name = self.unique_name + suffix
        if getattr(self, 'legacy_unique_name', None):
            self.legacy_unique_name = self.legacy_unique_name + suffix

    def generate_data(self, n, data_type):
        mean = np.full(self.d, self.x_mean)
        cov = np.full((self.d, self.d), self.rho) + (1 - self.rho) * np.eye(self.d)
        X = np.random.multivariate_normal(mean, cov, size=n)
        Y_reg = X.dot(self.beta)
        eps = np.random.normal(loc=0, scale=1, size=n)
        Y = Y_reg + eps
        return (X, Y)

    def generate_MCAR(self, X):
        d = X['Train'].shape[1]
        prob_missing = self.missing_prob
        var_missing = self.var_missing
        nb_var_missing = int(np.sum(var_missing))
        train_size = X['Train'].shape[0]
        cal_size = X['Cal'].shape[0]
        M_mcar_train = np.full(X['Train'].shape, False)
        X_mcar_train = copy.deepcopy(X['Train'])
        M_mcar_cal = np.full(X['Cal'].shape, False)
        X_mcar_cal = copy.deepcopy(X['Cal'])
        M_mcar_train[:, np.where(np.array(var_missing) == 1)[0]] = np.random.uniform(low=0, high=1, size=(train_size, nb_var_missing)) <= prob_missing
        X_mcar_train[M_mcar_train] = np.nan
        M_mcar_cal[:, np.where(np.array(var_missing) == 1)[0]] = np.random.uniform(low=0, high=1, size=(cal_size, nb_var_missing)) <= prob_missing
        X_mcar_cal[M_mcar_cal] = np.nan
        mechanisms_test = ['iid', 'fixed_nb_sample_pattern']
        M_mcar = {'Train': M_mcar_train, 'Cal': M_mcar_cal}
        M_mcar_test = dict.fromkeys(mechanisms_test)
        X_mcar = {'Train': X_mcar_train, 'Cal': X_mcar_cal}
        X_mcar_test = dict.fromkeys(mechanisms_test)
        M_mcar_iid = np.full((self.test_size_iid, d), False)
        M_mcar_iid[:, np.where(np.array(var_missing) == 1)[0]] = np.random.uniform(low=0, high=1, size=(self.test_size_iid, nb_var_missing)) <= prob_missing
        M_mcar_test['iid'] = M_mcar_iid
        X_mcar_iid = copy.deepcopy(X['Test'][self.test_size_iid])
        X_mcar_iid[M_mcar_iid] = np.nan
        X_mcar_test['iid'] = X_mcar_iid
        list_patterns = self.get_test_patterns(d)
        test_size = self.test_size_per_pattern * self.total_number_missing
        nb_sample_pattern = self.test_size_per_pattern
        M_mcar_fixed_sample_pattern = np.full((test_size, d), False)
        X_mcar_fixed_sample_pattern = copy.deepcopy(X['Test'][test_size])
        for idp, pattern in enumerate(list_patterns):
            M_mcar_fixed_sample_pattern[idp * nb_sample_pattern:(idp + 1) * nb_sample_pattern, np.where(np.array(pattern) == 1)[0]] = 1
        X_mcar_fixed_sample_pattern[M_mcar_fixed_sample_pattern] = np.nan
        M_mcar_test['fixed_nb_sample_pattern'] = M_mcar_fixed_sample_pattern
        X_mcar_test['fixed_nb_sample_pattern'] = X_mcar_fixed_sample_pattern
        X_mcar['Test'] = X_mcar_test
        M_mcar['Test'] = M_mcar_test
        return (X_mcar, M_mcar)

class Real_data(Missing_wrapper):

    def __init__(self, name, n_rep, mechanism, missing_prob, var_missing, var_dependent, train_size, cal_size, test_size_iid, test_size_per_pattern, missing_function=None, seed=0, num_test_patterns=None):
        var_dependent = var_dependent or []
        if name not in UCI_DATASETS:
            raise ValueError(f"Dataset '{name}' not found in UCI list {sorted(UCI_DATASETS)} or supported semi-synthetic collections.")
        raw_X, raw_y = uci_dataset_loader(name)
        raw_X = np.asarray(raw_X, dtype=float)
        raw_y = np.asarray(raw_y)
        default_var_missing = None
        if raw_y.dtype.kind not in ('f', 'c'):
            if raw_y.dtype.kind in ('b', 'i', 'u'):
                raw_y = raw_y.astype(float)
            else:
                from sklearn.preprocessing import LabelEncoder
                raw_y = LabelEncoder().fit_transform(raw_y).astype(float)
        dim = raw_X.shape[1]
        var_missing = self._validate_var_missing(var_missing, default_var_missing, dim)
        if any((v is None for v in (train_size, cal_size, test_size_iid, test_size_per_pattern))):
            raise ValueError('Real_data requires explicit train/cal/test sizes; got None for at least one split.')
        try:
            train_size = int(train_size)
            cal_size = int(cal_size)
            test_size_iid = int(test_size_iid)
            test_size_per_pattern = int(test_size_per_pattern)
        except (TypeError, ValueError):
            raise ValueError('Split sizes must be valid integers.')
        for label, value in (('train_size', train_size), ('cal_size', cal_size), ('test_size_iid', test_size_iid), ('test_size_per_pattern', test_size_per_pattern)):
            if value <= 0:
                raise ValueError(f'{label} must be positive; received {value}.')
        super().__init__(name=name, n_rep=n_rep, train_size=train_size, cal_size=cal_size, test_size_iid=test_size_iid, test_size_per_pattern=test_size_per_pattern, mechanism=mechanism, missing_prob=missing_prob, var_missing=var_missing, var_dependent=var_dependent, missing_function=missing_function, seed=seed, num_test_patterns=num_test_patterns)
        self.raw_data_X = raw_X
        self.raw_data_Y = raw_y
        self.d = dim
        self.fixed_MAR_MNAR_coeff = True

    @staticmethod
    def _validate_var_missing(var_missing, default_var_missing, dim):
        if default_var_missing is not None:
            candidate = default_var_missing
        else:
            candidate = var_missing
        if candidate is None:
            arr = np.ones(dim, dtype=int)
        else:
            arr = np.asarray(candidate).astype(int).ravel()
            if arr.size == 0:
                arr = np.ones(dim, dtype=int)
        if arr.size != dim:
            raise ValueError(f'var_missing length {arr.size} does not match dataset dimension {dim}. Please provide a mask of the correct length.')
        arr = np.where(arr != 0, 1, 0)
        return arr

    def generate_data(self, n, data_type):
        if data_type == 'train':
            train_indice = np.random.choice(self.train_indice, n, replace=False)
            X = self.raw_data_X[train_indice]
            Y = self.raw_data_Y[train_indice]
        elif data_type == 'cal':
            cal_indice = np.random.choice(self.cal_indice, n, replace=False)
            X = self.raw_data_X[cal_indice]
            Y = self.raw_data_Y[cal_indice]
        elif data_type == 'test':
            available_indices = self.test_indice
            available_size = len(available_indices)
            if n <= available_size:
                test_indice = np.random.choice(available_indices, n, replace=False)
                X = self.raw_data_X[test_indice]
                Y = self.raw_data_Y[test_indice]
                return (X, Y)
            X_to_shuffle = self.raw_data_X[available_indices]
            Y_to_shuffle = self.raw_data_Y[available_indices]
            n_shuffle = X_to_shuffle.shape[0]
            nb_exact_shuffle = n // n_shuffle
            nb_rest = n % n_shuffle
            X_test_created = np.empty((n, X_to_shuffle.shape[1]))
            Y_test_created = np.empty((n,))
            for k in range(nb_exact_shuffle):
                ido = np.random.permutation(n_shuffle)
                X_test_created[k * n_shuffle:(k + 1) * n_shuffle] = X_to_shuffle[ido]
                Y_test_created[k * n_shuffle:(k + 1) * n_shuffle] = Y_to_shuffle[ido]
            if nb_rest > 0:
                ido = np.random.choice(n_shuffle, nb_rest, replace=False)
                X_test_created[nb_exact_shuffle * n_shuffle:] = X_to_shuffle[ido]
                Y_test_created[nb_exact_shuffle * n_shuffle:] = Y_to_shuffle[ido]
            return (X_test_created, Y_test_created)
        else:
            raise ValueError('data_type not recognized')
        return (X, Y)

class PreMaskedNpzData:

    def __init__(self, train_path=None, test_path=None, base_dir=None, n_rep=1, name='premasked', train_file='train.npz', test_file='test.npz', seed=0, tag=None, train_size=None, cal_size=None, test_size=None):
        self.name = name
        self.mechanism = 'premasked'
        self.missing_prob = 0.0
        self.base_dir = base_dir
        self.train_path = train_path
        self.test_path = test_path
        self.n_rep = int(n_rep)
        self.seed = seed
        self.train_file = train_file
        self.test_file = test_file
        self.tag = tag
        self.requested_train_size = train_size
        self.requested_cal_size = cal_size
        self.requested_test_size = test_size
        self.X = None
        self.X_missing = None
        self.M = None
        self.Y = None
        self.X_train_full = None
        self.y_train_full = None
        self.X_cal_full = None
        self.y_cal_full = None
        self.X_test_full = None
        self.y_test_full = None
        self.d = None
        self.train_size = None
        self.cal_size = None
        self.test_size_iid = None
        self.test_size_per_pattern = None
        self.var_missing = None
        self._load()

    def __getitem__(self, key):
        return self.__dict__[key]

    def _repeat(self, arr):
        arr = np.asarray(arr)
        if self.n_rep <= 1:
            return np.expand_dims(arr, axis=0)
        return np.repeat(arr[np.newaxis, ...], self.n_rep, axis=0)

    def _load_npz(self, path):
        if not os.path.isfile(path):
            raise FileNotFoundError(f'Missing npz file: {path}')
        data = np.load(path, allow_pickle=True)
        if 'X' not in data or 'y' not in data:
            raise KeyError(f"Expected keys 'X' and 'y' in {path}.")
        X = np.asarray(data['X'], dtype=float)
        y = np.asarray(data['y']).reshape(-1)
        if X.ndim != 2:
            raise ValueError(f'Expected 2D X in {path}, got shape {X.shape}.')
        if y.shape[0] != X.shape[0]:
            raise ValueError(f'Mismatch X rows ({X.shape[0]}) vs y rows ({y.shape[0]}) in {path}.')
        return (X, y)

    def _resolve_size(self, full_size, requested, label):
        if requested is None:
            return int(full_size)
        try:
            requested = int(requested)
        except (TypeError, ValueError):
            raise ValueError(f'Invalid {label}: {requested}')
        if requested <= 0:
            raise ValueError(f'{label} must be positive; got {requested}.')
        if requested > full_size:
            raise ValueError(f'{label} {requested} exceeds available rows {full_size}.')
        return requested

    def _sample_split(self, X, y, size, rng):
        if size >= X.shape[0]:
            idx = rng.permutation(X.shape[0])
        else:
            idx = rng.permutation(X.shape[0])[:size]
        return (X[idx], y[idx])

    def _load(self):
        if self.train_path is None or self.test_path is None:
            if self.base_dir is None:
                raise ValueError('Provide train/test paths or set base_dir to use default filenames.')
            train_path = os.path.join(self.base_dir, self.train_file)
            test_path = os.path.join(self.base_dir, self.test_file)
        else:
            train_path = self.train_path
            test_path = self.test_path
        X_train, y_train = self._load_npz(train_path)
        X_test, y_test = self._load_npz(test_path)
        if X_test.shape[1] != X_train.shape[1]:
            raise ValueError(f'Feature dimension mismatch across train/test splits: {X_train.shape[1]}, {X_test.shape[1]}.')
        self.X_train_full = X_train
        self.y_train_full = y_train
        self.X_cal_full = None
        self.y_cal_full = None
        self.X_test_full = X_test
        self.y_test_full = y_test
        self.d = X_train.shape[1]
        if self.requested_cal_size is None:
            raise ValueError('cal_size is required when calibration is sampled from the training file.')
        self.train_size = self._resolve_size(X_train.shape[0], self.requested_train_size, 'train_size')
        self.cal_size = self._resolve_size(X_train.shape[0], self.requested_cal_size, 'cal_size')
        if self.train_size + self.cal_size > X_train.shape[0]:
            raise ValueError(f'train_size + cal_size exceeds available train rows ({X_train.shape[0]}).')
        if self.requested_test_size not in (None, X_test.shape[0]):
            raise ValueError('test_size must be None or equal to full test size because test set is fixed.')
        self.test_size_iid = X_test.shape[0]
        self.test_size_per_pattern = self.test_size_iid
        self.var_missing = np.ones(self.d, dtype=int)
        tag = self.tag
        if not tag:
            if self.base_dir:
                tag = os.path.basename(os.path.normpath(self.base_dir))
            else:
                tag = 'premasked'
        tag = utils._sanitize_token(tag)
        sig_input = '|'.join((os.path.abspath(p) for p in (train_path, test_path)))
        dir_sig = hashlib.sha1(sig_input.encode('utf-8')).hexdigest()[:8]
        self.unique_name = f'{self.name}_{tag}_tr{self.train_size}_cal{self.cal_size}_te{self.test_size_iid}_h{dir_sig}'
        self.legacy_unique_name = self.unique_name

    def generate_multiple_data(self):
        if self.X_missing is not None:
            return (self.X, self.X_missing, self.M, self.Y)
        train_X, train_M, train_Y = ([], [], [])
        cal_X, cal_M, cal_Y = ([], [], [])
        test_X, test_M, test_Y = ([], [], [])
        test_X_full = self.X_test_full
        test_M_full = np.isnan(test_X_full)
        test_Y_full = self.y_test_full
        for k in range(self.n_rep):
            rng = np.random.default_rng(self.seed + k)
            perm = rng.permutation(self.X_train_full.shape[0])
            train_idx = perm[:self.train_size]
            cal_idx = perm[self.train_size:self.train_size + self.cal_size]
            X_train = self.X_train_full[train_idx]
            y_train = self.y_train_full[train_idx]
            X_cal = self.X_train_full[cal_idx]
            y_cal = self.y_train_full[cal_idx]
            X_test = test_X_full
            y_test = test_Y_full
            train_X.append(X_train)
            train_M.append(np.isnan(X_train))
            train_Y.append(y_train)
            cal_X.append(X_cal)
            cal_M.append(np.isnan(X_cal))
            cal_Y.append(y_cal)
            test_X.append(X_test)
            test_M.append(test_M_full)
            test_Y.append(y_test)
        train_X = np.stack(train_X, axis=0)
        train_M = np.stack(train_M, axis=0)
        train_Y = np.stack(train_Y, axis=0)
        cal_X = np.stack(cal_X, axis=0)
        cal_M = np.stack(cal_M, axis=0)
        cal_Y = np.stack(cal_Y, axis=0)
        test_X = np.stack(test_X, axis=0)
        test_M = np.stack(test_M, axis=0)
        test_Y = np.stack(test_Y, axis=0)
        X_missing = {'Train': train_X, 'Cal': cal_X, 'Test': {'iid': test_X, 'fixed_nb_sample_pattern': test_X}}
        M = {'Train': train_M, 'Cal': cal_M, 'Test': {'iid': test_M, 'fixed_nb_sample_pattern': test_M}}
        Y = {'Train': train_Y, 'Cal': cal_Y, 'Test': {'iid': test_Y, 'fixed_nb_sample_pattern': test_Y}}
        self.X = {'Train': train_X, 'Cal': cal_X, 'Test': {'iid': test_X, 'fixed_nb_sample_pattern': test_X}}
        self.X_missing = X_missing
        self.M = M
        self.Y = Y
        self._print_summary(train_M, cal_M, test_M)
        return (self.X, self.X_missing, self.M, self.Y)

    @staticmethod
    def _count_unique_masks(masks):
        packed = np.packbits(masks.astype(np.uint8), axis=1)
        return np.unique(packed, axis=0).shape[0]

    def _print_summary(self, train_M, cal_M, test_M):
        train_rows = int(train_M.shape[1])
        cal_rows = int(cal_M.shape[1])
        test_rows = int(test_M.shape[1])
        d = int(train_M.shape[2])
        train_unique = self._count_unique_masks(train_M[0])
        cal_unique = self._count_unique_masks(cal_M[0])
        test_unique = self._count_unique_masks(test_M[0])
        if self.n_rep > 1:
            train_unique_rep = [self._count_unique_masks(train_M[k]) for k in range(self.n_rep)]
            cal_unique_rep = [self._count_unique_masks(cal_M[k]) for k in range(self.n_rep)]
            test_unique_rep = [self._count_unique_masks(test_M[k]) for k in range(self.n_rep)]
            train_unique = f'{min(train_unique_rep)}-{max(train_unique_rep)}'
            cal_unique = f'{min(cal_unique_rep)}-{max(cal_unique_rep)}'
            test_unique = f'{min(test_unique_rep)}-{max(test_unique_rep)}'
        print('PreMaskedNpzData summary:')
        print(f'  train: n={train_rows}, d={d}, unique_masks={train_unique}')
        print(f'  cal:   n={cal_rows}, d={d}, unique_masks={cal_unique}')
        print(f'  test:  n={test_rows}, d={d}, unique_masks={test_unique}')
import numpy as np
import copy

class PatternMixtureLinearData(Missing_wrapper):

    def __init__(self, n_rep, train_size, cal_size, test_size_iid, test_size_per_pattern, missing_prob, var_missing, var_dependent=None, seed=0, dim=None, split_train=1.0, beta=None, noise_std=1.0, x_mean=1.0, rho=0.5, mu0=None, B=None, Sigma0=None, s0=1.0, s1=0.2, test_size_per_num_pattern=100, test_grouping='pattern', num_test_patterns=None):
        name = 'pattern_mixture_linear'
        mechanism = 'PMIX'
        if dim is None:
            dim = len(var_missing)
        if dim > 100:
            raise ValueError('PatternMixtureLinearData supports dim up to 100 in this setup.')
        if dim > 10 and num_test_patterns is None:
            raise ValueError('For dim > 10, please set num_test_patterns to avoid enumerating 2^d patterns.')
        assert len(var_missing) == dim
        train_size = int(train_size / split_train)
        var_missing = np.asarray(var_missing, dtype=int)
        if var_dependent is None:
            var_dependent = np.array([], dtype=int)
        else:
            var_dependent = np.asarray(var_dependent, dtype=int)
        super().__init__(name=name, n_rep=n_rep, train_size=train_size, cal_size=cal_size, test_size_iid=test_size_iid, test_size_per_pattern=test_size_per_pattern, mechanism=mechanism, missing_prob=missing_prob, var_missing=var_missing, var_dependent=var_dependent, missing_function=None, seed=seed, num_test_patterns=num_test_patterns)
        self.test_size_per_num_pattern = test_size_per_num_pattern
        self.test_grouping = test_grouping
        self.d = dim
        self.dim = dim
        self.noise_std = float(noise_std)
        if beta is None:
            default_beta = np.array([1, 2, -1, 3, -0.5, -1, 0.3, 1.7, 0.4, -0.3, -2.84, 0.91, 2.77, -0.68, 1.53, -1.94, 0.44, -2.21, 2.05, -0.17, -1.46, 0.23, 1.88, -2.63, 2.41, -0.82, 0.71, -1.12, 2.58, -0.49, -2.05, 1.37, -0.94, 2.63, -1.59, 0.52, -2.77, 1.96, -0.08, 2.24, -1.23, 0.35, 2.49, -2.36, 1.14, -0.71, 2.81, -1.76, 0.62, -2.48, 1.79, -0.27, 2.92, -1.65, 0.84, -2.58, 2.18, -0.41, 1.25, -2.93, 2.67, -0.56, 1.48, -2.12, 0.19, 2.32, -1.88, 0.77, -2.69, 2.55, -0.63, 1.61, -2.44, 0.58, 2.86, -1.31, 0.05, -2.29, 1.92, -0.14, 2.37, -1.57, 0.49, 2.99, -2.72, 1.08, -0.35, 2.14, -1.99, 0.66, -2.11, 0.88, -1.45, 2.63, -0.72, 1.57, -2.33, 0.12, 2.48, -1.08])
            beta = default_beta[:dim]
        else:
            beta = np.asarray(beta, dtype=float)[:dim]
        self.beta = beta
        if mu0 is None:
            mu0 = np.full(dim, x_mean, dtype=float)
        else:
            mu0 = np.asarray(mu0, dtype=float)[:dim]
        self.mu0 = mu0
        if Sigma0 is None:
            idx = np.arange(dim)
            Sigma0 = rho ** np.abs(idx[:, None] - idx[None, :])
        else:
            Sigma0 = np.asarray(Sigma0, dtype=float)
            assert Sigma0.shape == (dim, dim)
        self.Sigma0 = Sigma0
        self.rho = rho
        if B is None:
            rng_B = np.random.default_rng(seed)
            B_raw = rng_B.normal(loc=0.0, scale=1.0, size=(dim, dim))
            missing_mask = self.var_missing == 1
            B_raw[:, ~missing_mask] = 0.0
            col_norms = np.linalg.norm(B_raw, axis=0, keepdims=True)
            col_norms[col_norms < 1e-08] = 1.0
            target_scale = 2.0
            B = target_scale * B_raw / col_norms
            B[:, missing_mask] += 0.1 * np.sign(B[:, missing_mask].sum(axis=0, keepdims=True))
        else:
            B = np.asarray(B, dtype=float)
            assert B.shape == (dim, dim), 'For now we assume B is (d x d).'
        self.B = B
        self.s0 = float(s0)
        self.s1 = float(s1)
        self.unique_name = self.unique_name + f'_PMIX_dim{dim}_rho{rho}_s0{s0}_s1{s1}'
        if getattr(self, 'legacy_unique_name', None):
            self.legacy_unique_name = self.legacy_unique_name + f'_PMIX_dim{dim}_rho{rho}_s0{s0}_s1{s1}'
        self.seed = seed

    def sample_XY_given_M(self, m, n_samples=1, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        m = np.asarray(m, dtype=float)
        k = float((m * self.var_missing).sum())
        mu_m = self.mu0 + self.B @ m
        sigma2_m = self.s0 + self.s1 * k
        sigma2_m = max(sigma2_m, 0.0001)
        Sigma_m = sigma2_m * self.Sigma0
        X = rng.multivariate_normal(mu_m, Sigma_m, size=n_samples)
        eps = rng.normal(loc=0.0, scale=self.noise_std, size=n_samples)
        Y = X @ self.beta + eps
        return (X, Y)

    def sample_joint(self, n, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        d = self.d
        M = np.zeros((n, d), dtype=bool)
        missing_idx = np.where(self.var_missing == 1)[0]
        if len(missing_idx) > 0:
            M[:, missing_idx] = rng.random((n, len(missing_idx))) < self.missing_prob
        X = np.zeros((n, d), dtype=float)
        Y = np.zeros(n, dtype=float)
        for i in range(n):
            Xi, Yi = self.sample_XY_given_M(M[i].astype(int), n_samples=1, rng=rng)
            X[i] = Xi[0]
            Y[i] = Yi[0]
        return (X, Y, M)

    def generate_multiple_data(self):
        print(self.unique_name)
        sets = ['Train', 'Cal', 'Test']
        X = {s: None for s in sets}
        X_missing = {s: None for s in sets}
        M = {s: None for s in sets}
        Y = {s: None for s in sets}
        if self.test_grouping == 'pattern':
            mechanisms_test = ['iid', 'fixed_nb_sample_pattern']
        else:
            mechanisms_test = ['iid', 'fixed_nb_sample_pattern_size']
        d = self.d
        list_patterns = self.get_test_patterns(d)
        nb_patterns = len(list_patterns)
        for k in range(self.n_rep):
            print(k, '/', self.n_rep)
            rng = np.random.default_rng(self.seed + k)
            X_tr, Y_tr, M_tr = self.sample_joint(self.train_size, rng=rng)
            X_tr_miss = X_tr.copy()
            X_tr_miss[M_tr] = np.nan
            X_cal, Y_cal, M_cal = self.sample_joint(self.cal_size, rng=rng)
            X_cal_miss = X_cal.copy()
            X_cal_miss[M_cal] = np.nan
            X_te_iid, Y_te_iid, M_te_iid = self.sample_joint(self.test_size_iid, rng=rng)
            X_te_iid_miss = X_te_iid.copy()
            X_te_iid_miss[M_te_iid] = np.nan
            if self.test_grouping == 'pattern':
                test_size_fixed = self.test_size_per_pattern * nb_patterns
                X_te_fixed = np.zeros((test_size_fixed, d), dtype=float)
                Y_te_fixed = np.zeros(test_size_fixed, dtype=float)
                M_te_fixed = np.zeros((test_size_fixed, d), dtype=bool)
                idx = 0
                for pat in list_patterns:
                    m_vec = np.asarray(pat, dtype=int)
                    X_block, Y_block = self.sample_XY_given_M(m_vec, n_samples=self.test_size_per_pattern, rng=rng)
                    M_block = np.tile(m_vec.astype(bool), (self.test_size_per_pattern, 1))
                    X_te_fixed[idx:idx + self.test_size_per_pattern] = X_block
                    Y_te_fixed[idx:idx + self.test_size_per_pattern] = Y_block
                    M_te_fixed[idx:idx + self.test_size_per_pattern] = M_block
                    idx += self.test_size_per_pattern
                X_te_fixed_miss = X_te_fixed.copy()
                X_te_fixed_miss[M_te_fixed] = np.nan
            else:
                raise NotImplementedError("Only test_grouping='pattern' is implemented in PatternMixtureLinearData")
            for set_name in ['Train', 'Cal']:
                if set_name == 'Train':
                    X_set, X_set_miss, M_set, Y_set = (X_tr, X_tr_miss, M_tr, Y_tr)
                else:
                    X_set, X_set_miss, M_set, Y_set = (X_cal, X_cal_miss, M_cal, Y_cal)
                if k == 0:
                    X[set_name] = np.expand_dims(X_set, axis=0)
                    X_missing[set_name] = np.expand_dims(X_set_miss, axis=0)
                    M[set_name] = np.expand_dims(M_set, axis=0)
                    Y[set_name] = Y_set
                else:
                    X[set_name] = np.vstack((X[set_name], np.expand_dims(X_set, axis=0)))
                    X_missing[set_name] = np.vstack((X_missing[set_name], np.expand_dims(X_set_miss, axis=0)))
                    M[set_name] = np.vstack((M[set_name], np.expand_dims(M_set, axis=0)))
                    Y[set_name] = np.vstack((Y[set_name], Y_set))
            if k == 0:
                X['Test'] = {'iid': np.expand_dims(X_te_iid_miss, axis=0), 'fixed_nb_sample_pattern': np.expand_dims(X_te_fixed_miss, axis=0)}
                X_missing['Test'] = X['Test']
                M['Test'] = {'iid': np.expand_dims(M_te_iid, axis=0), 'fixed_nb_sample_pattern': np.expand_dims(M_te_fixed, axis=0)}
                Y['Test'] = {'iid': Y_te_iid, 'fixed_nb_sample_pattern': Y_te_fixed}
            else:
                X['Test']['iid'] = np.vstack((X['Test']['iid'], np.expand_dims(X_te_iid_miss, axis=0)))
                X['Test']['fixed_nb_sample_pattern'] = np.vstack((X['Test']['fixed_nb_sample_pattern'], np.expand_dims(X_te_fixed_miss, axis=0)))
                X_missing['Test'] = X['Test']
                M['Test']['iid'] = np.vstack((M['Test']['iid'], np.expand_dims(M_te_iid, axis=0)))
                M['Test']['fixed_nb_sample_pattern'] = np.vstack((M['Test']['fixed_nb_sample_pattern'], np.expand_dims(M_te_fixed, axis=0)))
                Y['Test']['iid'] = np.vstack((Y['Test']['iid'], Y_te_iid))
                Y['Test']['fixed_nb_sample_pattern'] = np.vstack((Y['Test']['fixed_nb_sample_pattern'], Y_te_fixed))
        self.X, self.X_missing, self.M, self.Y = (X, X_missing, M, Y)
        self.save(self.unique_name)
        return (X, X_missing, M, Y)

    def get_cov_for_pattern(self, pattern):
        m = np.asarray(pattern, dtype=float)
        assert m.shape[0] == self.d, 'pattern legth must equal dim'
        k = float((m * self.var_missing).sum())
        sigma2_m = self.s0 + self.s1 * k
        sigma2_m = max(sigma2_m, 0.0001)
        Sigma_m = sigma2_m * self.Sigma0
        return Sigma_m
