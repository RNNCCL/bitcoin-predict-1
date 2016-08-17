import datetime as dt
import os
import glob
import cPickle as pickle
import collections as coll
import itertools as it
import logging
import string

import numpy as np
import scipy.sparse as sparse
import sklearn.decomposition as decomp
import pandas as pd
import pattern.en as pattern
import nltk.sentiment.vader as vader
import nltk.corpus as nltk_corpus
import toolz
import requests
import unicodecsv as csv
from nltk.corpus import stopwords

import data_config
import corpus
import util


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
    df["next_day_log_return"] = df["log_return"].shift(-1)
    df["up_down"] = df.log_return > 0
    df["next_day_up_down"] = df.up_down.shift(-1)
    return df


def create_unigrams():
    w_2_idx = coll.defaultdict(it.count(0).next)
    max_widx = 0
    for day, day_messages in corpus.get_day_preprocessed_comments(include_is=True, include_oos=True):
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
        with open("models-cache/unigram_{}.pkl".format(day_str), "w") as o:
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
    gt_thresh = np.array(X_full.sum(axis=0) > data_config.unigram_filter_threshold)[0]

    sw = set(stopwords.words("english"))
    idx_2_word = {idx: w for w, idx in w_2_idx.items()}
    is_not_stopword = [idx_2_word.get(i) not in sw for i in range(gt_thresh.shape[0])]

    do_include = np.logical_and(gt_thresh, is_not_stopword)

    idx_include_words = do_include.nonzero()[0]

    w_include = [(w, idx) for w, idx in w_2_idx.viewitems() if do_include[idx]]
    w_2_idx_include = {
        w: idx
        for idx, (w, _)
        in enumerate(sorted(w_include, key=lambda x: x[1]))
    }

    X = X_full[:, idx_include_words].todense()
    df = pd.DataFrame(X, index=pd.DatetimeIndex(dates))
    return df, w_2_idx_include


class TokenFilter(object):
    def __init__(self):
        self.to_filter = set(stopwords.words("english")) | set(string.punctuation)

    def should_filter(self, token):
        return token in self.to_filter



def create_lda_sample(p=.02):
    _, w_2_idx_include = load_unigrams()
    #lda_w_2_idx = coll.defaultdict(it.count(0).next)
    coo_dict = coll.defaultdict(int)
    comments = corpus.get_processed_full_comments(
        p=p,
        include_is=True,
        include_oos=False,
        filter_deleted=True
    )
    token_filter = TokenFilter()
    for doc_idx, sent_tokens in enumerate(comments):
        for sent in sent_tokens:
            for token in sent:
                if token not in w_2_idx_include:
                    continue
                if token_filter.should_filter(token):
                    continue
                word_idx = w_2_idx_include[token]
                coo_dict[doc_idx, word_idx] += 1
    return w_2_idx_include, util.d_to_sparse(coo_dict, shape=(doc_idx+1, max(w_2_idx_include.values())+1))


def get_lda_df():
    lda = decomp.LatentDirichletAllocation(
        n_topics=data_config.lda_num_topics,
        doc_topic_prior=1./200,
        topic_word_prior=1./200,
    )
    print "getting sample 1"
    lda_w_2_idx, lda_sample = models.create_lda_sample(p=.1)
    print "partial fit 1"
    lda.partial_fit(lda_sample)
    print "getting sample 2"
    _, lda_sample = models.create_lda_sample(p=.1)
    print "partial fit 2"
    lda.partial_fit(lda_sample)
    lda_mean = {}
    lda_mean_mean = {}
    #lda_max = {}
    print "labeling corpus"
    for day, comments in corpus.get_day_preprocessed_comments(p=.25, include_oos=True):
        print day
        coo_dict = coll.defaultdict(int)
        for doc_idx, sent_tokens in enumerate(comments):
            for sent in sent_tokens:
                for token in sent:
                    if token not in lda_w_2_idx:
                        continue
                    word_idx = lda_w_2_idx[token]
                    coo_dict[doc_idx, word_idx] += 1
        lda_day = lda.transform(util.d_to_sparse(coo_dict, shape=(doc_idx+1, lda.components_.shape[1])))
        lda_day_mean = (lda_day.T / lda_day.sum(axis=1)).T
        lda_mean[day] = lda_day.mean(axis=0)
        lda_mean_mean[day] = lda_day_mean.mean(axis=0)
        #lda_max[day] = lda_day.max(axis=0)

    lda_idx_2_w = {idx: w for w, idx in lda_w_2_idx.viewitems()}

    days, lda_reps = zip(*sorted(lda_mean.items(), key=lambda x:x[0]))
    df_lda_mean = pd.DataFrame(map(tuple, lda_reps), index=days)

    days, lda_reps = zip(*sorted(lda_mean_mean.items(), key=lambda x:x[0]))
    df_lda_mean_mean = pd.DataFrame(map(tuple, lda_reps), index=days)

    days, lda_reps = zip(*sorted(lda_max.items(), key=lambda x:x[0]))
    df_lda_max = pd.DataFrame(map(tuple, lda_reps), index=days)


    return lda, lda_w_2_idx, pd.DataFrame(map(tuple, lda_reps), index=days)

    # lda_idx_2_w = {idx: w for w, idx in lda_w_2_idx.viewitems()}

def print_topics():
    for idx, topic in enumerate(lda.components_):
        if not np.isclose(0, df_lda_mean[idx].var()):
            print "{}:".format(idx), u" ".join([lda_idx_2_w[word_index] for word_index in np.argsort(-topic)[:10]])


def create_sentiment():
    sa = vader.SentimentIntensityAnalyzer()
    import unicodecsv as csv

    comments_with_sentiment = (
        toolz.merge(
            dict(
                id=d["id"],
                time=dt.datetime.utcfromtimestamp(float(d["created_utc"])).strftime("%Y-%m-%d %H:%M:%S"),
                ups=d["ups"],
                contr=d["controversiality"]
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
        for d in corpus.load_json(
                p=None,  # just do all
                include_is=True,
                include_oos=True,
                #filter_deleted=False
        )
    )

    with open("data-sentiment/sentiment.csv", "w") as o:
        c = comments_with_sentiment.next()
        writer = csv.DictWriter(o, c.keys())
        writer.writeheader()
        writer.writerow(c)
        for c in comments_with_sentiment:
            writer.writerow(c)


    # df = pd.DataFrame(comments_with_sentiment).set_index("time")
    # df.to_csv("data-sentiment/sentiment.csv")
    # # df_by_day = df[["pattern_polarity", "vader_compound"]].resample("d").mean()
    # # df_by_day.to_csv("data-sentiment/sentiment.csv")


def load_sentiment():
    return pd.read_csv(
        "data-sentiment/sentiment.csv",
        index_col="time",
        usecols=["time", "pattern_polarity", "vader_compound"],
        parse_dates=True
    ).resample("d").mean()


def transform_sentiment(df_sentiment):
    dropped_sentiment = df_sentiment.ix[data_config.date_begin:]
    scaled_sentiment = (dropped_sentiment - dropped_sentiment.mean()) / dropped_sentiment.std()
    return scaled_sentiment.ewm(com=data_config.sentiment_ewm_smoothing).mean()


def load_raw_sentiment():
    df_sentiment_raw = pd.read_csv(
        "data-sentiment/sentiment.csv",
        index_col="time",
        parse_dates=True
    )
    df_sentiment_raw["sentiment_mean"] = df_sentiment_raw[["pattern_polarity", "vader_compound"]].mean(axis=1)
    return df_sentiment_raw


class Ratios(object):
    def __init__(self, antonym_list):
        sw = set(stopwords.words("english"))
        self.antonym_list = [
            (f, s) for f, s in antonym_list
            if f not in sw and s not in sw
        ]

    @staticmethod
    def get_ratio_ts(df_unigram, w2idx, f, s):
        add_one_f = df_unigram[w2idx[f]] + 1
        add_one_s = df_unigram[w2idx[s]] + 1
        return (add_one_f / (add_one_f + add_one_s)).as_matrix()

    @classmethod
    def create(cls):
        _df_unigrams, w2idx = load_unigrams()
        df_unigrams = _df_unigrams.ix[:data_config.date_is_end]

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

        df = pd.DataFrame(
            [
                [f, s, df_unigrams[w2idx[f]].sum(), df_unigrams[w2idx[s]].sum()]
                for f, s in pairs
            ],
            columns=["first", "second", "count_first", "count_second"]
        )
        df["min_count"] = df[["count_first", "count_second"]].min(axis=1)

        antonym_list = [[i.first, i.second] for i in df.query("min_count> 1000").itertuples()]
        return cls(antonym_list)

    def get(self):
        df_unigrams, w2idx = load_unigrams()

        data_ratios = np.vstack([
            self.get_ratio_ts(df_unigrams, w2idx, f, s)
            for f, s in self.antonym_list
        ]).T
        return pd.DataFrame(data_ratios, index=df_unigrams.index)

    @classmethod
    def load(cls):
        with open("models-cache/antonym_list.csv") as f:
            antonym_list = list(csv.reader(f))
        return cls(antonym_list)

    def save(self):
        with open("models-cache/antonym_list.csv", "w") as o:
            csv.writer(o).writerows(self.antonym_list)

"""
def RatiosICA():
    def __init__(self, ica):
        self.ica = ica

    @classmethod
    def create(cls):
        df_ratios = Ratios.get()
        ratios_smooth = df_ratios.ewm(com=data_config.sentiment_ewm_smoothing).mean()
        ica = decomp.FastICA()
        ica.fit(ratios_smooth)
        return cls(ica)

    def save(self):
        pickle.dump(
            self.ica,
            open("models-cache/ratios_ica.pickle", "w"),
            pickle.HIGHEST_PROTOCOL
        )

    @classmethod
    def load(cls):
        return cls(pickle.load(open("models-cache/ratios_ica.pickle")))

    def get(self, include_is=True, include_oos=False):
        df_ratios = Ratios.get(include_is=include_is, include_oos=include_oos)
        ratios_smooth = df_ratios.ewm(com=data_config.sentiment_ewm_smoothing).mean()
        return self.ica.transform(ratios_smooth)
"""

class LDA(object):
    def __init__(self, lda):
        self.lda = lda

    @classmethod
    def create(cls):
        df_unigrams = load_unigrams()[0].ix[:data_config.date_is_end]
        lda = decomp.LatentDirichletAllocation()
        lda.fit(df_unigrams)
        return cls(lda)

    @classmethod
    def load(cls):
        return cls(pickle.load(open("models-cache/lda.pickle")))

    def save(self):
        pickle.dump(self.lda, open("models-cache/lda.pickle", "w"), pickle.HIGHEST_PROTOCOL)

    def get(self):
        df_unigrams, _ = load_unigrams()
        ldad = self.lda.transform(df_unigrams)
        return pd.DataFrame(ldad, index=df_unigrams.index)


def create_user_stats():
    authors_seen = set()

    day_num_new = coll.Counter()
    day_num_authors = coll.defaultdict(set)
    day_num_posts = coll.Counter()

    for line in corpus.load_json(include_oos=True):
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
    return pd.read_csv("data-sentiment/user_stats.csv", index_col=0, parse_dates=True)\
             .resample("d")\
             .sum()\
             .fillna(0)


def create_opinionfinder_corpus():
    import codecs
    with codecs.open("opinionfinder-input/input.txt", "w", encoding="utf-8") as o:
        for doc in corpus.get_normalized_texts(include_oos=True):
            print >> o, u" ".join(doc.strip().splitlines())


def sklearn_transform_in_sample(transformer, df):
    transformer.fit(df.ix[:data_config.date_is_end].as_matrix())
    transformed = transformer.transform(df.as_matrix())
    return pd.DataFrame(transformed, index=df.index)
