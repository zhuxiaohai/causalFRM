import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestRegressor

from causalml.dataset import synthetic_data
from causalml.inference.meta import (
    BaseSLearner,
    BaseSRegressor,
    BaseSClassifier,
    LRSRegressor,
)
from causalml.inference.meta import (
    BaseTLearner,
    BaseTRegressor,
    BaseTClassifier,
    XGBTRegressor,
    MLPTRegressor,
)
from causalml.inference.meta import BaseXLearner, BaseXClassifier, BaseXRegressor
from causalml.inference.meta import (
    BaseRLearner,
    BaseRClassifier,
    BaseRRegressor,
    XGBRRegressor,
    XGBRClassifier,
)
from causalml.inference.meta import TMLELearner
from causalml.inference.meta import BaseDRLearner
from causalml.metrics import ape, get_cumgain, auuc_score
from causalml.tuner import OptunaSearch

from .const import RANDOM_SEED, N_SAMPLE, ERROR_THRESHOLD, CONTROL_NAME, CONVERSION


def test_synthetic_data():
    y, X, treatment, tau, b, e = synthetic_data(mode=1, n=N_SAMPLE, p=8, sigma=0.1)

    assert (
        y.shape[0] == X.shape[0]
        and y.shape[0] == treatment.shape[0]
        and y.shape[0] == tau.shape[0]
        and y.shape[0] == b.shape[0]
        and y.shape[0] == e.shape[0]
    )

    y, X, treatment, tau, b, e = synthetic_data(mode=2, n=N_SAMPLE, p=8, sigma=0.1)

    assert (
        y.shape[0] == X.shape[0]
        and y.shape[0] == treatment.shape[0]
        and y.shape[0] == tau.shape[0]
        and y.shape[0] == b.shape[0]
        and y.shape[0] == e.shape[0]
    )

    y, X, treatment, tau, b, e = synthetic_data(mode=3, n=N_SAMPLE, p=8, sigma=0.1)

    assert (
        y.shape[0] == X.shape[0]
        and y.shape[0] == treatment.shape[0]
        and y.shape[0] == tau.shape[0]
        and y.shape[0] == b.shape[0]
        and y.shape[0] == e.shape[0]
    )

    y, X, treatment, tau, b, e = synthetic_data(mode=4, n=N_SAMPLE, p=8, sigma=0.1)

    assert (
        y.shape[0] == X.shape[0]
        and y.shape[0] == treatment.shape[0]
        and y.shape[0] == tau.shape[0]
        and y.shape[0] == b.shape[0]
        and y.shape[0] == e.shape[0]
    )


def test_BaseSLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseSLearner(learner=LinearRegression())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, return_ci=True)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseSRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseSRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
    )
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
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


def test_LRSRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = LRSRegressor()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseTLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseTLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
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

    # test of using control_learner and treatment_learner
    learner = BaseTLearner(
        learner=XGBRegressor(),
        control_learner=RandomForestRegressor(),
        treatment_learner=RandomForestRegressor(),
    )
    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseTRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseTRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
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


def test_MLPTRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = MLPTRegressor()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
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


def test_XGBTRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = XGBTRegressor()

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
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


def test_BaseXLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseXLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
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

    # basic test of using outcome_learner and effect_learner
    learner = BaseXLearner(
        learner=XGBRegressor(),
        control_outcome_learner=RandomForestRegressor(),
        treatment_outcome_learner=RandomForestRegressor(),
        control_effect_learner=RandomForestRegressor(),
        treatment_effect_learner=RandomForestRegressor(),
    )
    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseXRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseXRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
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


def test_BaseXLearner_without_p(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseXLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
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


def test_BaseXRegressor_without_p(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseXRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
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


def test_BaseRLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseRLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
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

    # basic test of using outcome_learner and effect_learner
    learner = BaseRLearner(
        learner=XGBRegressor(),
        outcome_learner=RandomForestRegressor(),
        effect_learner=RandomForestRegressor(),
    )
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert (
        ape(tau.mean(), ate_p) < ERROR_THRESHOLD * 5
    )  # might need to look into higher ape


def test_BaseRRegressor(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseRRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
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


def test_BaseRLearner_without_p(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseRLearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
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


def test_BaseRRegressor_without_p(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseRRegressor(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, return_ci=True, n_bootstraps=10
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


def test_TMLELearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = TMLELearner(learner=XGBRegressor())

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, p=e, treatment=treatment, y=y)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD


def test_BaseSClassifier(generate_classification_data):

    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    uplift_model = BaseSClassifier(learner=XGBClassifier())

    uplift_model.fit(
        X=df_train[x_names].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(
        X=df_test[x_names].values, treatment=df_test["treatment_group_key"].values
    )

    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    cumgain = get_cumgain(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
    )

    # Check if the cumulative gain when using the model's prediction is
    # higher than it would be under random targeting
    assert cumgain["tau_pred"].sum() > cumgain["Random"].sum()


def test_BaseTClassifier(generate_classification_data):

    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    uplift_model = BaseTClassifier(learner=LogisticRegression())

    uplift_model.fit(
        X=df_train[x_names].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(
        X=df_test[x_names].values, treatment=df_test["treatment_group_key"].values
    )

    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    cumgain = get_cumgain(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
    )

    # Check if the cumulative gain when using the model's prediction is
    # higher than it would be under random targeting
    assert cumgain["tau_pred"].sum() > cumgain["Random"].sum()


def test_BaseXClassifier(generate_classification_data):

    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    propensity_model = LogisticRegression()
    propensity_model.fit(X=df[x_names].values, y=df["treatment_group_key"].values)
    df["propensity_score"] = propensity_model.predict_proba(df[x_names].values)[:, 1]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    # specify all 4 learners
    uplift_model = BaseXClassifier(
        control_outcome_learner=XGBClassifier(),
        control_effect_learner=XGBRegressor(),
        treatment_outcome_learner=XGBClassifier(),
        treatment_effect_learner=XGBRegressor(),
    )

    uplift_model.fit(
        X=df_train[x_names].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(
        X=df_test[x_names].values, p=df_test["propensity_score"].values
    )

    # specify 2 learners
    uplift_model = BaseXClassifier(
        outcome_learner=XGBClassifier(), effect_learner=XGBRegressor()
    )

    uplift_model.fit(
        X=df_train[x_names].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(
        X=df_test[x_names].values, p=df_test["propensity_score"].values
    )

    # calculate metrics
    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    cumgain = get_cumgain(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
    )

    # Check if the cumulative gain when using the model's prediction is
    # higher than it would be under random targeting
    assert cumgain["tau_pred"].sum() > cumgain["Random"].sum()


def test_BaseRClassifier(generate_classification_data):

    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )

    propensity_model = LogisticRegression()
    propensity_model.fit(X=df[x_names].values, y=df["treatment_group_key"].values)
    df["propensity_score"] = propensity_model.predict_proba(df[x_names].values)[:, 1]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    uplift_model = BaseRClassifier(
        outcome_learner=XGBClassifier(), effect_learner=XGBRegressor()
    )

    uplift_model.fit(
        X=df_train[x_names].values,
        p=df_train["propensity_score"].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
    )

    tau_pred = uplift_model.predict(X=df_test[x_names].values)

    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    cumgain = get_cumgain(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
    )

    # Check if the cumulative gain when using the model's prediction is
    # higher than it would be under random targeting
    assert cumgain["tau_pred"].sum() > cumgain["Random"].sum()


def test_BaseRClassifier_with_sample_weights(generate_classification_data):

    np.random.seed(RANDOM_SEED)

    df, x_names = generate_classification_data()

    df["treatment_group_key"] = np.where(
        df["treatment_group_key"] == CONTROL_NAME, 0, 1
    )
    df["sample_weights"] = np.random.randint(low=1, high=3, size=df.shape[0])

    propensity_model = LogisticRegression()
    propensity_model.fit(X=df[x_names].values, y=df["treatment_group_key"].values)
    df["propensity_score"] = propensity_model.predict_proba(df[x_names].values)[:, 1]

    df_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_SEED)

    uplift_model = BaseRClassifier(
        outcome_learner=XGBClassifier(), effect_learner=XGBRegressor()
    )

    uplift_model.fit(
        X=df_train[x_names].values,
        p=df_train["propensity_score"].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
        sample_weight=df_train["sample_weights"],
    )

    tau_pred = uplift_model.predict(X=df_test[x_names].values)

    auuc_metrics = pd.DataFrame(
        {
            "tau_pred": tau_pred.flatten(),
            "W": df_test["treatment_group_key"].values,
            CONVERSION: df_test[CONVERSION].values,
            "treatment_effect_col": df_test["treatment_effect"].values,
        }
    )

    cumgain = get_cumgain(
        auuc_metrics,
        outcome_col=CONVERSION,
        treatment_col="W",
        treatment_effect_col="treatment_effect_col",
    )

    # Check if the cumulative gain when using the model's prediction is
    # higher than it would be under random targeting
    assert cumgain["tau_pred"].sum() > cumgain["Random"].sum()

    # Check if XGBRRegressor successfully produces treatment effect estimation
    # when sample_weight is passed
    uplift_model = XGBRRegressor()
    uplift_model.fit(
        X=df_train[x_names].values,
        p=df_train["propensity_score"].values,
        treatment=df_train["treatment_group_key"].values,
        y=df_train[CONVERSION].values,
        sample_weight=df_train["sample_weights"],
    )
    tau_pred = uplift_model.predict(X=df_test[x_names].values)
    assert len(tau_pred) == len(df_test["sample_weights"].values)


def test_pandas_input(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()
    # convert to pandas types
    y = pd.Series(y)
    X = pd.DataFrame(X)
    treatment = pd.Series(treatment)

    try:
        learner = BaseSLearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(
            X=X, treatment=treatment, y=y, return_ci=True
        )
    except AttributeError:
        assert False
    try:
        learner = BaseTLearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y)
    except AttributeError:
        assert False
    try:
        learner = BaseXLearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    except AttributeError:
        assert False
    try:
        learner = BaseRLearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    except AttributeError:
        assert False
    try:
        learner = TMLELearner(learner=LinearRegression())
        ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    except AttributeError:
        assert False


def test_BaseDRLearner(generate_regression_data):
    y, X, treatment, tau, b, e = generate_regression_data()

    learner = BaseDRLearner(
        learner=XGBRegressor(), treatment_effect_learner=LinearRegression()
    )

    # check the accuracy of the ATE estimation
    ate_p, lb, ub = learner.estimate_ate(X=X, treatment=treatment, y=y, p=e)
    assert (ate_p >= lb) and (ate_p <= ub)
    assert ape(tau.mean(), ate_p) < ERROR_THRESHOLD

    # check the accuracy of the CATE estimation with the bootstrap CI
    cate_p, _, _ = learner.fit_predict(
        X=X, treatment=treatment, y=y, p=e, return_ci=True, n_bootstraps=10
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


def test_XGBRClassifier(generate_classification_data_two_treatments):
    df, x_names = generate_classification_data_two_treatments()
    train_data, val_data = train_test_split(df, test_size=0.2, random_state=0)
    train_data['p'] = 0.5
    val_data['p'] = 0.5

    class Wrapper(XGBRClassifier):
        def __init__(self, x_cols, treatment_col, y_col, *args, **kargs):
            outcome_learner_param_dict = {'booster': 'gbtree',
                                          'objective': "binary:logistic",
                                          'n_estimators': 1000,
                                          'n_jobs': -1,
                                          'random_state': 5,
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
                                          'learning_rate': 0.5
                                          }
            super().__init__(outcome_learner=XGBClassifier(**outcome_learner_param_dict),
                             effect_learner=XGBRegressor(*args, **kargs),
                             control_name='control',
                             outcome_learner_eval_metric='auc',
                             effect_learner_eval_metric='rmse',
                             outcome_learner_early_stopping_rounds=5,
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
            super().fit(X[self.x_cols].values, X[self.treatment_col].values, X[self.y_col].values, **kwargs)

    op = OptunaSearch(n_startup_trials=1, n_warmup_steps=1, n_trials=5, optuna_njobs=-1)

    tuning_param_dict = {'x_cols': x_names,
                         'y_col': 'conversion',
                         'treatment_col': 'treatment_group_key',
                         'booster': 'gbtree',
                         'n_estimators': 1000,
                         'n_jobs': -1,
                         'random_state': 5,
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

    fit_params = {'p': train_data['p'].values,
                  'val_X': val_data[x_names].values,
                  'val_treatment': val_data['treatment_group_key'].values,
                  'val_y': val_data['conversion'].values,
                  'val_p': val_data['p'].values,
                  'verbose': False}

    op.search(Wrapper, tuning_param_dict, train_data, val_data, **fit_params)

    train_param = op.get_params()
    print(train_param)
    a = Wrapper(**train_param[0])
    a.fit(train_data, **fit_params)
    print(a.score(val_data))
