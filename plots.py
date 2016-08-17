import pandas as pd
import numpy as np
import pylab as plt
import matplotlib.dates as dates
import seaborn as sns
import scipy
import sklearn.decomposition as decomp
import sklearn.preprocessing as pre

import data_config
import models

sns.set_style("white")


def plot_2d_per_time():
    fig_style = dict(cmap="viridis", marker='o')
    df, _ = models.load_unigrams()
    df_ratios = models.load_ratios()

    ug_normalized = (np.log1p(df).T / np.log1p(df).sum(axis=1)).T
    ug_smoothed = ug_normalized.rolling(data_config.unigram_rm_smoothing).mean()

    d_svd_unsmooth = pre.scale(
        decomp.TruncatedSVD(n_components=2).fit_transform(
            pre.StandardScaler().fit_transform(
                ug_normalized.ix[data_config.date_begin:].as_matrix()
            )
        )
    )

    d_svd_smooth = pre.scale(
        decomp.TruncatedSVD(n_components=2).fit_transform(
            pre.StandardScaler().fit_transform(
                ug_smoothed.ix[data_config.date_begin:].as_matrix()
            )
        )
    )

    ratios_smoothed = df_ratios.ewm(com=21).mean()
    d_ratios_ica = pre.scale(
        decomp.FastICA(n_components=2).fit_transform(
            pre.scale(ratios_smoothed.ix[data_config.date_begin:].as_matrix())
        )
    )
    d_ratios_ica_restricted = pre.scale(
        decomp.FastICA(n_components=2).fit_transform(
            pre.StandardScaler().fit_transform(
                ratios_smoothed.ix[data_config.date_turning_point:].as_matrix()
            )
        )
    )

    d_ratios_ica_rotated = d_ratios_ica.dot(
        scipy.linalg.orthogonal_procrustes(d_ratios_ica, d_svd_smooth)[0]
    )
    d_ratios_ica_restricted_rotated = d_ratios_ica_restricted.dot(
        scipy.linalg.orthogonal_procrustes(
            d_ratios_ica_restricted,
            d_svd_smooth[-d_ratios_ica_restricted.shape[0]:]
        )[0]
    )

    fig, ax = plt.subplots(ncols=3)
    idx = [d.toordinal() for d in df.ix[data_config.date_begin:].index.date]
    scatter_svd_unsmooth = ax[0].scatter(
        *d_svd_unsmooth.T, c=idx, s=15, **fig_style
    )
    ax[1].scatter(
        *d_svd_smooth.T, c=idx, s=15, **fig_style
    )
    scatter_ica_restricted = ax[2].scatter(
        *d_ratios_ica_restricted_rotated.T,
        c=idx[-d_ratios_ica_restricted.shape[0]:], s=15, vmin=min(idx), vmax=max(idx), **fig_style
    )

    for a in ax:
        a.set_frame_on(False)
        a.get_xaxis().set_visible(False)
        a.get_yaxis().set_visible(False)

    cb = fig.colorbar(
        scatter_svd_unsmooth,
        orientation='vertical',
        ticks=dates.YearLocator(),
        format=dates.DateFormatter('%Y')
    )
    cb.outline.set_visible(False)
    fig.savefig("thesis/plots/unigram_decomp.png")


def plot_sentiment_corr():
    df_sentiment = models.load_sentiment()
    idx = [d.toordinal() for d in df_sentiment.ix[data_config.date_begin:].index.date]
    smoothed = df_sentiment.ewm(data_config.unigram_ewm_smoothing).mean()

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        scatter_sentiment = ax.scatter(
            *smoothed.ix[data_config.date_begin:].as_matrix().T,
            c=idx,
            cmap="viridis"
        )
        cb = fig.colorbar(
            scatter_sentiment,
            orientation='vertical',
            ticks=dates.YearLocator(),
            format=dates.DateFormatter('%Y')
        )
        cb.outline.set_visible(False)
        fig.savefig("thesis/plots/sentiment_corr.png")


def plot_sentiment_rolling_corr():
    df_sentiment = models.load_sentiment()
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        df_sentiment.pattern_polarity.ewm(84).corr(df_sentiment.vader_compound).ix[data_config.date_begin:].plot(ax=ax)
        fig.savefig("thesis/plots/sentiment_rolling_corr.png")


def plot_sentiment_ts():
    df_sentiment = models.load_sentiment()
    smoothed_sentiment = df_sentiment.ewm(data_config.unigram_ewm_smoothing).mean()
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        smoothed_sentiment.ix[data_config.date_begin:].plot(ax=ax, colormap="viridis")
        fig.savefig("thesis/plots/sentiment_ts.png")


def plot_sentiment_price():
    df_sentiment = models.load_sentiment()
    df_price = models.load_price()
    smoothed_sentiment = df_sentiment.ewm(data_config.unigram_ewm_smoothing).mean()
    smoothed_price = df_price.ewm(data_config.unigram_ewm_smoothing).mean()
    fig, ax = plt.subplots()
    smoothed_price[["log_return"]].ix[data_config.date_begin:].plot(ax=ax)
    smoothed_sentiment.ix[data_config.date_begin:].plot(ax=ax)
    fig.savefig("thesis/plots/sentiment_ts.png")


def plot_price_users():
    df_price = models.load_price()
    df_user_stats = models.load_user_stats()
    joined = df_user_stats.join(df_price, how="inner")
    smoothed = joined[["price", "num_posts"]].rolling(21).mean()
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        smoothed.plot(secondary_y=["num_posts"], ax=ax)
        fig.savefig("thesis/plots/users_price.png")


def slopegraph(df, left_col, right_col):
    yticks_list = []
    fig, ax1 = plt.subplots()
    fig.patch.set_facecolor('white')
    ax2 = ax1.twinx()
    ax1.set_axis_bgcolor("white")
    ax2.set_axis_bgcolor("white")
    ax1.grid(b=False)
    ax2.grid(b=False)
    for name, row in df.iterrows():
        ax1.plot([0, 1], [-row[left_col], -row[right_col]], c="black")
        yticks_list.append((-row[left_col], -row[right_col], name))
    y_ticks_l_pos, y_ticks_r_pos, y_ticks_names = zip(*yticks_list)

    l_names = ["{} - {:.2f}".format(n, d) for d, n in zip(y_ticks_l_pos, y_ticks_names)]
    r_names = ["{:.2f} - {}".format(d, n) for d, n in zip(y_ticks_r_pos, y_ticks_names)]

    ax1.set_yticks(y_ticks_l_pos)
    ax1.set_yticklabels(l_names)
    ax2.set_yticks(y_ticks_r_pos)
    ax2.set_yticklabels(r_names)
    ax1.set_xlim((0, 1))
    ax2.set_xlim((0, 1))
    ax2.set_ylim(ax1.get_ylim())
    ax2.tick_params(labelbottom="off")
    ax1.tick_params(labelbottom="off")
    plt.show()
    return fig


def plot_best_worst_cv(results_cv_all, top_3_cv, bottom_3_cv, col):
    best_worst = pd.concat([
        top_3_cv.join(results_cv_all.set_index(["rep_clf_name"]), lsuffix="_min"),
        bottom_3_cv.join(results_cv_all.set_index(["rep_clf_name"]), lsuffix="_min")
    ]).reset_index().sort_values("{}_min".format(col), ascending=False)
    #best_worst["rep_clf_name"] = best_worst.rep + " (" + best_worst.clf + ")"

    with sns.axes_style("whitegrid"):
        #plt.xticks(rotation=70)
        fig, ax = plt.subplots()
        sns.violinplot("rep_clf_name", col, data=best_worst, ax=ax, cmap="viridis")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=70, horizontalalignment='right')
        plt.tight_layout()
        fig.savefig("thesis/plots/cv_dists.png")


def plot_returns(df_price_oos, df_hard_returns):
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        np.exp(df_price_oos[["log_return"]].cumsum()).plot(ax=ax, color="black", ls="--")
        np.exp(df_hard_returns).plot(ax=ax)
        fig.savefig("thesis/plots/strat_returns.png")

def plot_sentiment_correlation_ewm():
    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        ax.plot(
            [
                df_sentiment.pattern_polarity.ewm(i).mean().ix[in_sample].corr(
                    df_price.next_day_up_down.astype(float)
                ).max()
                for i in range(100)
            ],
            label="pattern_polarity"
        )
        ax.plot(
            [
                df_sentiment.vader_compound.ewm(i).mean().ix[in_sample].corr(
                    df_price.next_day_up_down.astype(float)
                ).max()
                for i in range(100)
            ],
            label="vader_compound"
        )
        ax.legend()
        plt.xlabel(r.ewm("k"))
        plt.ylabel("Correlation")
        fig.savefig("thesis/plots/sentiment_correlation_ewm.png")
