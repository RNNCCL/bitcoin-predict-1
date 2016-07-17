import datetime as dt
import os
import glob
import cPickle as pickle
import collections as coll
import itertools as it
import logging

import numpy as np
import scipy.sparse as sparse
import sklearn.decomposition as decomp
import pandas as pd
import pattern.en as pattern
import nltk.sentiment.vader as vader
import nltk.corpus as nltk_corpus
import toolz
import requests

import data_config
import corpus


def create_price():
    req = requests.get(
        "https://api.coindesk.com/v1/bpi/historical/close.json",
        params=dict(
            start=data_config.date_first_comment.strftime("%Y-%m-%d"),
            end=(data_config.date_oos_end + pd.datetools.timedelta(days=1)).strftime("%Y-%m-%d")
        )
    ).json()
    dates, prices = zip(*sorted(req["bpi"].items(), key=lambda x:x[0]))
    df_price = pd.DataFrame(dict(price=prices), index=pd.DatetimeIndex(dates))
    df_price.to_csv("data-price/price.csv")


def load_price():
    df = pd.read_csv(
        "data-price/price.csv",
        names=["date", "price"],
        index_col="date",
        parse_dates=True,
        skiprows=1
    )
    df["log_price"] = np.log(df.price)
    df["log_return"] = np.log(df.price).diff()
    df["up_down"] = df.log_return > 0
    df["next_day_up_down"] = df.up_down.shift(1)
    return df


def create_unigrams():
    w_2_idx = coll.defaultdict(it.count(0).next)
    max_widx = 0
    for day, day_messages in corpus.get_day_preprocessed_comments():
        day_str = day.strftime("%Y-%m-%d")
        logging.info("{} {}".format(day, max_widx))
        day_cnt = np.zeros(800000)
        for message in day_messages:
            sent_tokens = corpus.process(message)
            for sent in sent_tokens:
                for token in sent:
                    widx = w_2_idx[token]
                    max_widx = max(max_widx, widx)
                    day_cnt[widx] += 1
        with open(open("models-cache/unigram_{}.pkl".format(day_str), "w")) as o:
            pickle.dump(sparse.coo_matrix(day_cnt), o, pickle.HIGHEST_PROTOCOL)
    with open("models-cache/unigram.widx.pkl", "w") as o:
        pickle.dump(dict(w_2_idx), o)


def load_unigrams():
    def get_date_from_file_name(f):
        return dt.datetime.strptime(
            os.path.splitext(os.path.split(f)[-1])[0].split("_")[1],
            "%Y-%m-%d"
        )
    w_2_idx = pickle.load(open("models-cache/unigram.widx.pkl"))
    files = sorted(glob.glob("models-cache/unigram_*.pkl"))
    dates = [get_date_from_file_name(f) for f in files]
    X_full = sparse.vstack([pickle.load(open(f)) for f in files]).tocsr()
    gt_thresh = np.array((X_full.sum(axis=0) > data_config.unigram_filter_threshold)).nonzero()[1]
    X = X_full[:, gt_thresh].todense()
    df = pd.DataFrame(X, index=pd.DatetimeIndex(dates))
    return df, w_2_idx


def create_sentiment():
    sa = vader.SentimentIntensityAnalyzer()

    comments_with_sentiment = [
        toolz.merge(
            dict(
                time=pd.Timestamp.fromtimestamp(float(d["created_utc"])),
                ups=d["ups"]
            ),
            toolz.keymap(
                lambda x: "vader_" + x,
                sa.polarity_scores(d["body"])
            ),
            dict(zip(
                ["pattern_polarity", "pattern_subjectivity"],
                pattern.sentiment(d["body"])
            ))
        )
        for d in corpus.load_json(p=data_config.sentiment_sample_comment_prob)
    ]

    df = pd.DataFrame(comments_with_sentiment).set_index("time")
    df_by_day = df[["pattern_polarity", "vader_compound"]].resample("d").mean()
    df_by_day.to_csv("data-sentiment/sentiment.csv")


def load_sentiment():
    return pd.read_csv("data-sentiment/sentiment.csv", index_col="time", parse_dates=True)


def transform_sentiment(df_sentiment):
    dropped_sentiment = df_sentiment.ix[data_config.date_begin:]
    scaled_sentiment = (dropped_sentiment - dropped_sentiment.mean()) / dropped_sentiment.std()
    return scaled_sentiment.ewm(com=data_config.sentiment_ewm_smoothing).mean()


def create_ratios():
    df_unigrams, w2idx = load_unigrams()

    def get_date(f):
        return dt.datetime.strptime(
            os.path.splitext(os.path.split(f)[-1])[0].split("_")[1],
            "%Y-%m-%d"
        )

    files = sorted(glob.glob("models-cache/unigram_*.pkl"))
    dates = [get_date(d) for d in files]

    wn = nltk_corpus.wordnet
    pairs = set()
    for i in wn.all_synsets():
        for j in i.lemmas():
            if not j.name() in w2idx:
                continue
            for ant in j.antonyms():
                if not ant.name() in w2idx:
                    continue
                pairs.add(tuple(sorted([j.name(), ant.name()])))

    data = sparse.vstack([pickle.load(open(x)) for x in files]).tocsc()
    df = pd.DataFrame(
        [[f, s, data[:, w2idx[f]].sum(), data[:, w2idx[s]].sum()] for f,s in pairs],
        columns=["first", "second", "count_first", "count_second"]
    )
    df["min_count"] = df[["count_first", "count_second"]].min(axis=1)

    def get_ratio_ts(f, s):
        add_one_f = data[:, w2idx[f]].todense() + 1
        add_one_s = data[:, w2idx[s]].todense() + 1
        return add_one_f / (add_one_f + add_one_s)

    data_ratios = np.hstack([
        get_ratio_ts(i.first, i.second)
        for i
        in df.query("min_count> 1000").itertuples()
    ])

    df_ratios = pd.DataFrame(data_ratios, index=pd.DatetimeIndex(dates))
    df_ratios.to_csv("data-sentiment/ratios.csv")


def load_ratios():
    return pd.read_csv("data-sentiment/ratios.csv", index_col=0, parse_dates=True)


def transform_ratios(df_ratios):
    ratios_smooth = df_ratios.ewm(com=data_config.sentiment_ewm_smoothing).mean().ix[data_config.date_begin:]
    icad = decomp.FastICA().fit_transform(ratios_smooth.as_matrix())
    return pd.DataFrame(icad, index=ratios_smooth.index)


# class LDA(object):
#     @classmethod
#     def create(cls):
#         pass
#     def save(self):
#         pass
#     def get(self, include_is=True, include_oos=False):
#         pass

def create_lda():
    df_unigrams, w2idx = load_unigrams()
    lda = decomp.LatentDirichletAllocation()
    pickle.dump(lda, open("models-cache/lda.pickle", "w"), pickle.HIGHEST_PROTOCOL)


def load_lda():
    return pickle.load(open("models-cache/lda.pickle"))


def load_lda_df():
    lda = load_lda()
    df_unigrams, _ = load_unigrams()
    ldad = lda.transform(df_unigrams)
    return pd.DataFrame(ldad, index=df_unigrams.index)


def create_user_stats():
    authors_seen = set()

    day_num_new = coll.Counter()
    day_num_authors = coll.defaultdict(set)
    day_num_posts = coll.Counter()

    for line in corpus.load_json():
        date = dt.datetime.utcfromtimestamp(float(line["created_utc"])).date()
        author = line["author"]
        if author not in authors_seen:
            day_num_new[date] += 1
            authors_seen.add(author)
        day_num_authors[date].add(author)
        day_num_posts[date] += 1

    def dict_to_df(d, col_name):
        dates, items = zip(*sorted(d.viewitems(), key=lambda x:x[0]))
        return pd.DataFrame(
            {col_name: items},
            index = pd.DatetimeIndex(dates, freq="D")
        )

    df_user_stats = dict_to_df(day_num_new, "num_new")\
        .join(dict_to_df({k: len(v) for (k, v) in day_num_authors.viewitems()}, "num_authors"))\
        .join(dict_to_df(day_num_posts, "num_posts"))
    df_user_stats["comments_per_author"] = df_user_stats.num_posts / df_user_stats.num_authors
    df_user_stats.to_csv("data-sentiment/user_stats.csv")


def load_user_stats():
    return pd.read_csv("data-sentiment/user_stats.csv", index_col=0, parse_dates=True)
