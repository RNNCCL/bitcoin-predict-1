
import models
import training


def main():
    df_price = models.load_price()
    df_user_stats = models.load_user_stats()
    df_sentiment = models.transform_sentiment(models.load_sentiment())
    df_ug = models.load_unigrams()
    df_ratios = models.load_ratios()
    df_ratios_transformed = models.transform_ratios(df_ratios)
    df_lda = models.load_lda_df()

    results = []
    results.append(training.cv_test(df_sentiment, df_price))
    results.append(training.cv_test(df_ug, df_price))
    results.append(training.cv_test(df_ratios, df_price))
    results.append(training.cv_test(df_ratios_transformed, df_price))
    results.append(training.cv_test(df_lda, df_price))
