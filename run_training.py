import datetime as dt

import numpy as np
import sklearn.decomposition as decomp
import sklearn.linear_model as lm
import sklearn.ensemble as ensemble
import pandas as pd
import xgboost

import models
import training
import data_config
import random_returns
import plots


def pre():
    models.create_price()
    models.create_unigrams()
    models.create_user_stats()
    models.create_sentiment()
    models.Ratios.create().save()
    #models.LDA.create().save()


def main():

    class name:
        """ algebra of LaTeX names of representations and transformations """
        def __init__(self, n, is_compound=False):
            self.n = n
            self.is_compound = is_compound
        def bracket(self):
            if not self.is_compound:
                return self
            return name("$(" + self.n.strip("$") + ")$", is_compound=self.is_compound)
        def __mul__(self, other):
            return name(self.n.rstrip("$") + "\cdot" + other.n.lstrip("$"), is_compound=True)
        def __or__(self, other):
            return name(self.bracket().n.rstrip("$") + " | " + other.bracket().n.lstrip("$"), is_compound=False)
        def __call__(self, *params):
            return name(self.n % params, is_compound=self.is_compound)
        def __hash__(self):
            return hash(self.n)
        def __eq__(self, other):
            return self.n == other.n
        def __ne__(self, other):
            return not self.__eq__(other)
        def __str__(self):
            return self.n
        def __repr__(self):
            return "name({})".format(self.n)

    class r:
        bow = name("$\mathrm{Bow}$")
        bow_norm = name("$\mathrm{BowNorm}$")
        sentiment = name("$\mathrm{Sentiment}$")
        ratios = name("$\mathrm{Ratios}$")
        svd = name("$\mathrm{SVD}(%s)$")
        lda = name("$\mathrm{LDA}(%s)$")
        diff = name("$\mathrm{Diff}$")
        ewm = name("$\mathrm{Ewm}(%s)$")
        user_stats = name("$\mathrm{UserStats}$")
        day = name("$\mathrm{Day}$")
        returns = name("$\mathrm{Returns}_t$")
        speed = name("$\mathrm{Speed}$")

    class c:
        lr = name("$\mathrm{LR}$")
        xgb = name("$\mathrm{XGB}$")

    class pair:
        def __init__(self, clf, rep):
            self.clf = clf
            self.rep = rep
            self.str = "$\\langle {}, {} \\rangle$".format(self.clf.n.strip("$"), self.rep.n.strip("$"))
        def __str__(self):
            return self.str
        def __repr__(self):
            return str(self)
        def __float__(self):
            raise ValueError
        def __hash__(self):
            return hash((self.clf, self.rep))
        def __eq__(self, other):
            return (self.clf, self.rep) == (other.clf, other.rep)
        def __ne__(self, other):
            return not self.__eq__(other)

    df_price = models.load_price()
    df_ug, widx = models.load_unigrams()

    reps = {}

    reps[r.bow] = df_ug
    reps[r.bow_norm] = np.log1p(df_ug).div(np.log1p(df_ug).apply(np.linalg.norm, axis=1), axis=0)
    reps[r.bow_norm * r.diff] = reps[r.bow_norm].diff().dropna()
    reps[r.bow_norm * r.svd(32)] = models.sklearn_transform_in_sample(
        decomp.TruncatedSVD(n_components=32),
        reps[r.bow_norm]
    )
    reps[r.bow_norm * r.diff * r.svd(32)] = models.sklearn_transform_in_sample(
        decomp.TruncatedSVD(n_components=32),
        reps[r.bow_norm * r.diff]
    )
    reps[r.bow_norm * r.svd(32) * r.ewm(55)] = reps[r.bow_norm * r.svd(32)].ewm(55).mean()
    reps[r.bow_norm * r.diff * r.svd(32) * r.ewm(120)] = reps[r.bow_norm * r.diff * r.svd(32)].ewm(120).mean()

    reps[r.ratios] = models.Ratios.load().get()
    reps[r.ratios * r.diff] = reps[r.ratios].diff().dropna()
    reps[r.ratios * r.svd(32)] = models.sklearn_transform_in_sample(
        decomp.TruncatedSVD(n_components=32),
        reps[r.ratios]
    )
    reps[r.ratios * r.diff * r.svd(32)] = models.sklearn_transform_in_sample(
        decomp.TruncatedSVD(n_components=32),
        reps[r.ratios * r.diff]
    )
    reps[r.ratios * r.svd(32) * r.ewm(30)] = reps[r.ratios * r.svd(32)].ewm(30).mean()

    reps[r.sentiment] = models.load_sentiment().fillna(0)
    reps[r.sentiment * r.diff] = reps[r.sentiment].diff()
    reps[r.sentiment * r.ewm(90)] = reps[r.sentiment].ewm(90).mean()
    reps[r.sentiment | (r.sentiment * r.ewm(90))] = util.join(
        reps[r.sentiment],
        reps[r.sentiment * r.ewm(90)]
    )

    reps[r.lda(50)] = df_lda #models.load_lda()
    reps[r.lda(50) * r.ewm(90)] = reps[r.lda(50)].ewm(90).mean()
    reps[(r.lda(50) * r.ewm(90)) | r.lda(50)] = util.join(
        reps[r.lda(50)],
        reps[r.lda(50) * r.ewm(90)]
    )

    df_user_stats = np.log1p(models.load_user_stats())
    reps[r.user_stats] = df_user_stats - df_user_stats.shift(56)

    df_time = pd.DataFrame(dict(time=np.arange(df_price.shape[0])), index=df_price.index)

    reps[(r.lda(50) * r.ewm(90)) | (r.ratios * r.svd(32)) | (r.sentiment * r.ewm(90)) | r.day] = util.join(
        reps[r.lda(50) * r.ewm(90)],
        reps[r.ratios * r.svd(32)],
        reps[r.sentiment * r.ewm(90)],
        df_time
    )

    reps[(r.lda(50) * r.ewm(90)) | (r.ratios * r.svd(32)) | (r.sentiment * r.ewm(90)) | r.user_stats | r.day] = util.join(
        reps[r.lda(50) * r.ewm(90)],
        reps[r.ratios * r.svd(32)],
        reps[r.sentiment * r.ewm(90)],
        reps[r.user_stats],
        df_time
    )

    new_rep = {}
    new_rep[(r.lda(50) * r.ewm(90)) | (r.ratios * r.svd(32)) | (r.sentiment * r.ewm(90)) | r.user_stats | r.returns | r.day] = util.join(
        reps[r.lda(50) * r.ewm(90)],
        reps[r.ratios * r.svd(32)],
        reps[r.sentiment * r.ewm(90)],
        reps[r.user_stats],
        df_price[["log_return", "up_down"]],
        df_time
    )

    # df_ratios_speed = reps[r.ratios * r.svd(32)]\
    #                   .diff()\
    #                   .dropna()\
    #                   .apply(np.linalg.norm, axis=1)\
    #                   .to_frame("ratio_speed")
    # df_svd_speed = reps[r.bow_norm * r.svd(32)]\
    #                .diff()\
    #                .dropna()\
    #                .apply(np.linalg.norm, axis=1)\
    #                .to_frame("svd_speed")

    in_sample = slice(data_config.date_begin, data_config.date_is_end)
    in_sample_recent = slice(data_config.date_turning_point, data_config.date_is_end)
    out_of_sample = slice(data_config.date_oos_begin, data_config.date_oos_end - dt.timedelta(days=1))

    df_price_is, df_price_oos = df_price.ix[in_sample], df_price.ix[out_of_sample]

    clfs = {
        c.lr: lambda: lm.LogisticRegression(),
        c.xgb: lambda: xgboost.XGBClassifier(reg_lambda=2, max_depth=3, subsample=.5),
    }

    results_cv = []
    for rep_name, rep in sorted(new_rep.items(), key=lambda x:x[0]):
        for clf_name, clf_f in clfs.items():
            if rep_name == r.bow_norm or rep_name == r.bow_norm * r.diff or rep_name == r.bow and clf_name == c.xgb:
                continue
            print rep_name.n, clf_name.n
            acc, roc_auc, matthews = zip(*training.cv_test(clf_f(), rep.ix[in_sample], df_price_is, k=100))
            df_results = pd.DataFrame(
                dict(acc=acc, roc_auc=roc_auc, matthews=matthews)
            )
            df_results["rep"] = rep_name
            df_results["clf"] = clf_name
            results_cv.append(df_results)

    results_cv_all = pd.concat(results_cv)

    results_cv_all["rep_clf_name"] = [pair(a, b) for a,b in zip(results_cv_all.clf, results_cv_all.rep)]

    rank_by = "acc"
    top_3_cv = results_cv_all\
               .query("rep != 'user_stats'")\
               .groupby("rep_clf_name")\
               .quantile(.1)\
               .sort_values(rank_by, ascending=False)\
               .head(3)

    bottom_3_cv = results_cv_all\
               .query("rep != 'user_stats'")\
               .groupby("rep_clf_name")\
               .quantile(.1)\
               .sort_values(rank_by)\
               .head(3)

    plots.plot_best_worst_cv(results_cv_all, top_3_cv, bottom_3_cv, rank_by)

    df_soft_returns = pd.DataFrame(index=df_price_oos.index)
    df_hard_returns = pd.DataFrame(index=df_price_oos.index)

    results_oos = []

    for pair_clf_rep in top_3_cv.index.values:
        rep, clf_f = reps[pair_clf_rep.rep], clfs[pair_clf_rep.clf]
        pair_name = str(pair_clf_rep )
        [acc, roc_auc, matthews], [realized_returns_hard, realized_returns_soft] = training.oos_test(
            clf_f(),
            rep.ix[in_sample],
            rep.ix[out_of_sample],
            df_price_is,
            df_price_oos
        )

        results_oos.append([pair_name, acc, roc_auc, matthews])

        df_soft_returns[pair_name] = realized_returns_soft
        df_hard_returns[pair_name] = realized_returns_hard

    results_oos_all = pd.DataFrame(results_oos, columns=["rep", "acc", "roc_auc", "matthews"])

    oos_sharpe = (df_hard_returns.diff().mean() / df_hard_returns.diff().std()).to_frame("sharpe")
    oos_log_return = df_hard_returns.ix[-1].to_frame("log_return")

    df_random_classifier = random_returns.random_returns_stats(df_price_oos, n=1000)
    df_buy_hold = pd.DataFrame(
        dict(
            log_return=[df_price_oos.log_return.sum()],
            sharpe=[df_price_oos.log_return.mean()/df_price_oos.log_return.std()])
    )
    df_oos = results_oos_all.join(oos_sharpe.join(oos_log_return), on="rep")

    plots.plot_returns(df_price_oos, df_hard_returns)
    random_returns.plot_random_returns(df_price_oos, random_returns.random_returns_stats(df_price_oos))
