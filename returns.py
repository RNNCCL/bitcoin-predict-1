
import pandas as pd

def get_returns(yes_no, df_price, date_from, date_to):

    up_down = df_price.ix[date_from - pd.datetools.timedelta(days=1):date_to]

    df_price.log_return

    date_from


def test_get_returns():
    log_returns = df_price.log_return.ix[300:320]
    yes_no = np.random.random(size=(20,)) > .5
    realized_returns = (yes_no * log_returns).cumsum()
    realized_returns.plot()
