import numpy as np
import sklearn.metrics as metrics
import sklearn.cross_validation as cv
import sklearn.linear_model as lm

import models
import data_config
import xgboost


def cv_test(df_rep, df_price):
    joined = df_rep.join(df_price, how="inner").ix[data_config.date_begin:].dropna()

    X = joined.drop(df_price.columns, axis=1).as_matrix()
    y = joined.next_day_up_down.as_matrix().astype(int)

    results = []

    for train_idx, test_idx in cv.ShuffleSplit(y.shape[0], 10):

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        clf = lm.LogisticRegression()
        clf.fit(X_train, y_train)
        preds = clf.predict(X_test)
        y_test_mean = y_test.mean()

        results.append((
            metrics.accuracy_score(y_test, np.ones(y_test.shape[0]) * y_train.mean() > .5),
            metrics.accuracy_score(y_test, preds),
            metrics.f1_score(y_test, preds)
        ))

    return results


def oos_test(df_rep_oos, clf, df_price):
    pass
