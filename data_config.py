import datetime as dt

date_first_comment = dt.date(2010, 9, 11)
# date mt gox: hacked https://bitcoinhelp.net/know/more/price-chart-history
date_begin = dt.date(2011, 6, 19)
# seems discourse took a turning point here, little bit after $1000 peak
date_turning_point = dt.date(2013, 12, 8)
date_oos_begin = dt.date(2016, 3, 1)
date_oos_end = dt.date(2016, 6, 1)

unigram_filter_threshold = 50

unigram_rm_smoothing = unigram_ewm_smoothing = 21

sentiment_sample_comment_prob = 1./50
sentiment_ewm_smoothing = 14
