import models
import corpus
import sklearn_util


def get_most_positive_negative():
    df_sentiment_raw = models.load_raw_sentiment()
    largest = df_sentiment_raw.nlargest(5, "sentiment_mean")
    smallest = df_sentiment_raw.nsmallest(5, "sentiment_mean")

    comment_ids = set(largest.id) | set(smallest.id)
    comments = [
        c for c
        in corpus.load_json(include_is=True, include_oos=True, filter_deleted=False)
        if c["id"] in comment_ids
    ]

    df_comments = pd.concat([largest, smallest]).set_index("id").join(
        pd.DataFrame(comments).drop(["ups"], axis=1).set_index("id"), rsuffix="comment"
    )[["created_utc", "body", "sentiment_mean"]]
    df_comments["date"] = df_comments.created_utc.apply(lambda d: dt.datetime.utcfromtimestamp(float(d)).strftime("%Y-%m-%d"))





    # df_sentiment_raw["sentiment_sign"] = np.sign(df_sentiment_raw.sentiment_mean)
    # df_sentiment_raw["up_sign"] = np.sign(df_sentiment_raw.ups)
    # df_sentiment_raw["disagree"] = df_sentiment_raw["sentiment_sign"] != df_sentiment_raw["up_sign"]


def get_topics():
    df_ug, widx = models.load_unigrams()
    idx_2_word = {idx: w for w, idx in widx.viewitems()}
    lda = decomp.LatentDirichletAllocation(batch_size=10, n_topics=data_config.lda_num_topics)
    df_ug_lda = models.sklearn_transform_in_sample(lda, df_ug)

    topic_sum = df_ug_lda.sum()>50

    def print_topics():
        for idx, topic in enumerate(lda.components_):
            if topic_sum[idx]:
                print "{}:".format(idx), u" ".join([idx_2_word[word_index] for word_index in np.argsort(-topic)[:20]])


def print_topics():
    ewm_lda = 90
    n_words = 50
    lda_trans = df_lda_mean_mean.ewm(ewm_lda).mean().ix[in_sample]
    lda_corrs = lda_trans.corrwith(df_price.log_return.shift(1))
    max_topics = lda_corrs.abs().nlargest(3)
    for topic in max_topics.index:
        topic_words = [
            lda_idx_2_w[word_index]
            for word_index
            in np.argsort(-lda.components_[topic])[:n_words]
        ]
        print "{} ({:.2f}):".format(topic, lda_corrs[topic]), u" ".join(topic_words)

    with sns.axes_style("whitegrid"):
        fig, ax = plt.subplots()
        sklearn_util\
            .PandasTransform(pre.StandardScaler)\
            .fit_transform(
                lda_trans[max_topics.index]
            ).plot(ax=ax)
        fig.savefig("thesis/plots/lda_plot.png")

def output_oos_results():
    print results_oos_all.set_index("rep").join(oos_sharpe).join(oos_log_return).to_latex()
