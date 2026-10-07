import ast
from pathlib import Path
import numpy as np
from scipy.stats import norm

def generator_beta(dim):
    source = Path(__file__).parent / 'engines/part1/dataloader.py'
    module = ast.parse(source.read_text())
    vectors = []
    for cls in module.body:
        if isinstance(cls, ast.ClassDef) and cls.name in ('Synthetic_linear_data', 'PatternMixtureLinearData'):
            values = [n for n in ast.walk(cls) if isinstance(n, ast.Assign) and any((isinstance(t, ast.Name) and t.id == 'default_beta' for t in n.targets))]
            assert len(values) == 1
            vectors.append(np.asarray(ast.literal_eval(values[0].value.args[0]), dtype=float))
    assert len(vectors) == 2 and 0 < dim <= len(vectors[0])
    assert np.array_equal(*vectors)
    return vectors[0][:dim]

def oracle_lengths(masks, args):
    assert args['dataset_name'] == 'synthetic'
    d, mechanism, rho = (args['dim'], args['mechanism'], args['phi'])
    args = dict(args)
    if isinstance(args['var_missing'], str):
        args['var_missing'] = [int(x) for x in args['var_missing'].split(',')]
    assert mechanism in ('MCAR', 'MAR', 'PMIX')
    beta = generator_beta(d)
    if mechanism == 'PMIX':
        index = np.arange(d)
        sigma0 = rho ** np.abs(index[:, None] - index[None, :])
    else:
        sigma0 = np.full((d, d), rho) + (1 - rho) * np.eye(d)
    result = {}
    for key in masks:
        missing = np.array([c == '1' for c in key])
        assert len(missing) == d
        if mechanism == 'MAR':
            assert not np.any(missing & ~np.asarray(args['var_missing'], dtype=bool))
        sigma = sigma0 * (1 + 0.2 * missing.sum()) if mechanism == 'PMIX' else sigma0
        cov = sigma[np.ix_(missing, missing)]
        cross = sigma[np.ix_(missing, ~missing)]
        conditional = cov - cross @ np.linalg.pinv(sigma[np.ix_(~missing, ~missing)]) @ cross.T
        variance = 1 + beta[missing] @ conditional @ beta[missing]
        assert variance > 0
        result[key] = float(2 * norm.ppf(1 - args['alpha'] / 2) * np.sqrt(variance))
    return result
