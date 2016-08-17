import numpy as np
import sklearn.metrics as metrics
import sklearn.cross_validation as cv
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble

import models
import data_config
import xgboost


def cv_test(clf, df_rep, df_price, k):
    _X, _y = df_rep.align(df_price, join="inner", axis=0)

    X = _X.as_matrix()
    y = _y.next_day_up_down.as_matrix().astype(int)

    results = []

    for train_idx, test_idx in cv.ShuffleSplit(y.shape[0], k):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        preds_proba = clf.predict_proba(X_test)[:, 1]
        y_test_mean = y_test.mean()

        results.append((
            metrics.accuracy_score(y_test, preds),
            metrics.roc_auc_score(y_test, preds),
            metrics.matthews_corrcoef(y_test, preds),
        ))

    return results


def oos_test(clf, df_rep_is, df_rep_oos, df_price_is, df_price_oos):
    clf = ensemble.RandomForestClassifier()
    #clf = xgboost.XGBClassifier(n_estimators=30)

    X_train = df_rep_is.as_matrix()
    y_train = df_price_is.next_day_up_down.as_matrix().astype(int)
    X_test = df_rep_oos.as_matrix()
    y_test = df_price_oos.next_day_up_down.as_matrix().astype(int)

    clf.fit(X_train, y_train)

    preds = clf.predict(X_test)
    preds_proba = clf.predict_proba(X_test)[:, 1]

    acc = metrics.accuracy_score(y_test, preds)
    roc_auc = metrics.roc_auc_score(y_test, preds)
    matthews = metrics.matthews_corrcoef(y_test, preds)

    # buy_sell = np.where(preds, np.ones(preds.shape[0]), -1*np.ones(preds.shape[0]))
    # realized_returns_long_short = (buy_sell * df_price_oos.log_return).cumsum()
    # p_buy_sell = (preds_proba - .5) * 2
    # realized_returns_long_short_soft = (p_buy_sell * df_price_oos.log_return).cumsum()
    realized_returns_hard = (preds * df_price_oos.next_day_log_return).cumsum()
    realized_returns_soft = (preds_proba * df_price_oos.next_day_log_return).cumsum()
    return [acc, roc_auc, matthews], [realized_returns_hard, realized_returns_soft]
    #return [acc, ll, f1], [realized_returns_long_short, realized_returns_long_short_soft, realized_returns_hard, realized_returns_soft]
