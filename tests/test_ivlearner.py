import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import statsmodels.api as sm
from xgboost import XGBRegressor, XGBClassifier
import warnings

from causalml.inference.iv import BaseDRIVLearner, XGBDRIVClassifier
from causalml.metrics import ape, get_cumgain, auuc_score
from causalml.tuner import OptunaSearch

from .const import RANDOM_SEED, N_SAMPLE, ERROR_THRESHOLD, CONTROL_NAME, CONVERSION


def test_drivlearner():
    np.random.seed(RANDOM_SEED)
    n = 1000
    p = 8
    sigma = 1.0

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = (
        np.sin(np.pi * X[:, 0] * X[:, 1])
        + 2 * (X[:, 2] - 0.5) ** 2
        + X[:, 3]
        + 0.5 * X[:, 4]
    )
    assignment = (np.random.uniform(size=n) > 0.5).astype(int)
    eta = 0.1
    e_raw = np.maximum(
        np.repeat(eta, n),
        np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)),
    )
    e = e_raw.copy()
    e[assignment == 0] = 0
    tau = (X[:, 0] + X[:, 1]) / 2
    X_obs = X[:, [i for i in range(8) if i != 1]]

    w = np.random.binomial(1, e, size=n)
    treatment = w
    y = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)

    learner = BaseDRIVLearner(
        learner=XGBRegressor(), treatment_effect_learner=LinearRegression()
    )

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(
        X=X,
        assignment=assignment,
        treatment=treatment,
        y=y,
        p=(np.ones(n) * 1e-6, e_raw),
    )
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X,
        assignment=assignment,
        treatment=treatment,
        y=y,
        p=(np.ones(n) * 1e-6, e_raw),
        return_ci=True,
        n_bootstraps=10,
    )

    auuc_metrics = pd.DataFrame(
        {
            "cate_p": cate_p.flatten(),
            "W": treatment,
            "y": y,
            "treatment_effect_col": tau,
        }
    )

    cumgain = get_cumgain(
        auuc_metrics, outcome_col="y", treatment_col="W", treatment_effect_col="tau"
    )

    # Check if the cumulative gain when using the model's prediction is
    # higher than it would be under random targeting
    assert cumgain["cate_p"].sum() > cumgain["Random"].sum()


def test_xgbdrivclassifier():
    n = 10000
    p = 8
    sigma = 1.0

    X = np.random.uniform(size=n * p).reshape((n, -1))
    b = np.sin(np.pi * X[:, 0] * X[:, 1]) + 2 * (X[:, 2] - 0.5) ** 2 + X[:, 3] + 0.5 * X[:, 4]
    assignment = (np.random.uniform(size=10000) > 0.5).astype(int)
    eta = 0.1
    e = np.maximum(np.repeat(eta, n), np.minimum(np.sin(np.pi * X[:, 0] * X[:, 1]), np.repeat(1 - eta, n)))
    e[assignment == 0] = 0
    tau = (X[:, 0] + X[:, 1]) / 2
    X_obs = X[:, [i for i in range(8) if i != 1]]

    w = np.random.binomial(1, e, size=n)
    treatment = w
    y_numeric = b + (w - 0.5) * tau + sigma * np.random.normal(size=n)
    y = np.zeros_like(y_numeric)
    y[y_numeric < np.median(y_numeric)] = 1

    x_names = ['col_{}'.format(i) for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=x_names)
    df['treatment_group_key'] = treatment
    df['treatment_group_key'] = df['treatment_group_key'].map({1: 'treatment', 0: 'control'})
    df['assignment'] = assignment
    df['pZ'] = 0.5
    df['d1'] = 0.7
    df['d0'] = 0.2
    df['y'] = y
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=0)

    class Wrapper(XGBDRIVClassifier):
        def __init__(self, x_cols, treatment_col, y_col, *args, **kargs):
            control_outcome_learner_param_dict = {'booster': 'gbtree',
                                                  'n_estimators': 1000,
                                                  'n_jobs': 1,
                                                  'objective': "binary:logistic",
                                                  'use_label_encoder': False,
                                                  'verbosity': 0,
                                                  'max_depth': 3,
                                                  'reg_lambda': 5,
                                                  'reg_alpha': 2,
                                                  'gamma': 1,
                                                  'min_child_weight': 5,
                                                  'base_score': 0.5,
                                                  'colsample_bytree': 0.6,
                                                  'colsample_bylevel': 0.6,
                                                  'colsample_bynode': 0.6,
                                                  'subsample': 0.6,
                                                  'learning_rate': 0.5,
                                                  'random_state': 2
                                                  }
            treatment_outcome_learner_param_dict = {'booster': 'gbtree',
                                                    'n_estimators': 1000,
                                                    'n_jobs': 1,
                                                    'objective': "binary:logistic",
                                                    'use_label_encoder': False,
                                                    'verbosity': 0,
                                                    'max_depth': 3,
                                                    'reg_lambda': 5,
                                                    'reg_alpha': 2,
                                                    'gamma': 1,
                                                    'min_child_weight': 5,
                                                    'base_score': 0.5,
                                                    'colsample_bytree': 0.6,
                                                    'colsample_bylevel': 0.6,
                                                    'colsample_bynode': 0.6,
                                                    'subsample': 0.6,
                                                    'learning_rate': 0.5,
                                                    'random_state': 2
                                                    }

            super().__init__(control_outcome_learner=XGBClassifier(**control_outcome_learner_param_dict),
                             treatment_outcome_learner=XGBClassifier(**treatment_outcome_learner_param_dict),
                             treatment_effect_learner=XGBRegressor(*args, **kargs),
                             control_name='control',
                             control_outcome_learner_eval_metric='auc',
                             treatment_outcome_learner_eval_metric='auc',
                             effect_learner_eval_metric='rmse',
                             control_outcome_learner_early_stopping_rounds=5,
                             treatment_outcome_learner_early_stopping_rounds=5,
                             effect_learner_early_stopping_rounds=5
                             )
            self.x_cols = x_cols
            self.treatment_col = treatment_col
            self.y_col = y_col

        def score(self, X, y=None):
            auuc = auuc_score(X[[self.y_col, self.treatment_col]].assign(
                pred=self.predict(X[self.x_cols].values).max(axis=1),
                if_treat=(X[self.treatment_col] != self.control_name).astype(int).values),
                outcome_col=self.y_col,
                treatment_col='if_treat',
                normalize=True).loc['pred']
            return auuc

        def fit(self, X, treatment=None, y=None, **kwargs):
            super().fit(X[self.x_cols].values, treatment=X[self.treatment_col].values, y=X[self.y_col].values, **kwargs)

    op = OptunaSearch(n_startup_trials=1, n_warmup_steps=1, n_trials=5, optuna_njobs=1)

    tuning_param_dict = {'x_cols': x_names,
                         'y_col': 'y',
                         'treatment_col': 'treatment_group_key',
                         'booster': 'gbtree',
                         'n_estimators': 1000,
                         'n_jobs': 1,
                         'random_state': 2,
                         'verbosity': 0,
                         'objective': "reg:squarederror",
                         'max_depth': ('int', {'low': 2, 'high': 6}),
                         'reg_lambda': ('int', {'low': 1, 'high': 20}),
                         'reg_alpha': ('int', {'low': 1, 'high': 20}),
                         'gamma': ('int', {'low': 0, 'high': 3}),
                         'min_child_weight': ('int', {'low': 1, 'high': 30}),
                         'base_score': ('discrete_uniform', {'low': 0.5, 'high': 0.9, 'q': 0.1}),
                         'colsample_bytree': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bylevel': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'colsample_bynode': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'subsample': ('discrete_uniform', {'low': 0.7, 'high': 1, 'q': 0.05}),
                         'learning_rate': ('discrete_uniform', {'low': 0.07, 'high': 1.2, 'q': 0.01})
                         }

    fit_params = {'assignment': train_data['assignment'].values,
                  'p': (train_data['d0'].values, train_data['d1'].values),
                  'pZ': train_data['pZ'].values,
                  'val_X': val_data[x_names].values,
                  'val_assignment': val_data['assignment'].values,
                  'val_treatment': val_data['treatment_group_key'].values,
                  'val_y': val_data['y'].values,
                  'val_p': (val_data['d0'].values, val_data['d1'].values),
                  'val_pZ': val_data['pZ'].values,
                  'verbose': False}

    op.search(Wrapper, tuning_param_dict, train_data, val_data, **fit_params)

    train_param = op.get_params()
    print(train_param)
    a = Wrapper(**train_param[0])
    a.fit(train_data, **fit_params)
    print(a.score(val_data))
