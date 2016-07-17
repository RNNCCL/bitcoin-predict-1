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
