import numpy as np
import sklearn.metrics as metrics
import matplotlib.pylab as plt
import pandas as pd
import seaborn as sns


def random_returns_stats(df_price_oos, n=1000):
    returns_length = df_price_oos.shape[0]
    rand = np.random.random((n, returns_length))
    rand_yn = rand > .5
    y_true = df_price_oos.up_down.as_matrix()
    percentiles = [50, 90, 95]

    accuracies = (rand_yn == np.repeat([y_true], n, axis=0)).mean(axis=1)
    accuracy_quantiles = np.percentile(accuracies, percentiles)

    f1s = [metrics.f1_score(y_true, rand_yn[i]) for i in range(n)]
    f1_quantiles = np.percentile(f1s, percentiles)

    nlls = [-metrics.log_loss(y_true, rand[i]) for i in range(n)]
    nll_quantiles = np.percentile(nlls, percentiles)

    rand_returns = rand_yn * df_price_oos.log_return.as_matrix()
    return_quantiles = np.percentile(rand_returns.sum(axis=1), percentiles)

    sharpes = rand_returns.mean(axis=1) / rand_returns.std(axis=1)
    sharpe_quantiles = np.percentile(sharpes, percentiles)

    return pd.DataFrame(
        dict(
            quantile=percentiles,
            accuracy=accuracy_quantiles,
            f1=f1_quantiles,
            nll=nll_quantiles,
            log_return=return_quantiles,
            sharpe=sharpe_quantiles,
        )
    )

def plot_random_returns(df_price_oos, df_hard_returns, n=1000):
    returns_length = df_price_oos.shape[0]
    rand = np.random.random((n, returns_length))
    rand_yn = rand > .5
    rand_returns = rand_yn * df_price_oos.next_day_log_return.as_matrix()
    returns = np.exp(np.cumsum(rand_returns, axis=1))
    x = np.arange(rand_returns.shape[1])

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        np.exp(df_price_oos.log_return.cumsum()).plot(ax=ax, color="black")
        xlim = ax.get_xlim()
        x = np.linspace(xlim[0], xlim[1], rand_returns.shape[1])

        ps = []
        labels = []
        #p1 =
        #p2 = Rectangle((0, 0), 1, 1, fc="red")
        #legend([p1, p2], [a1_label, a2_label])

        for alpha, q in [(.10, 1), (.20, 5), (.30, 10), (.40, 20), (.50, 35)]:
            upper = np.percentile(returns, 100 - q, axis=0)
            lower = np.percentile(returns, q, axis=0)
            ps.append(plt.Rectangle((0, 0), 1, 1, fc="blue", alpha=alpha))
            labels.append("{} - {}".format(float(q) / 100, (100-float(q)) / 100))
            ax.fill_between(x, upper, lower, alpha=alpha)
        np.exp(df_hard_returns[df_hard_returns.tail(1).idxmax(axis=1)]).plot(ax=ax, color="red")
        l1 = ax.legend(ps, labels, loc=2)
        l2 = ax.legend(["Bitcoin", df_hard_returns.tail(1).idxmax(axis=1)[0]], loc=3)
        ax.add_artist(l1)
        fig.savefig("thesis/plots/strat_random_returns.png")
