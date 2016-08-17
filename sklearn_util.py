import sklearn.base as base
import numpy as np
import pandas as pd


class PandasTransform(base.TransformerMixin):
    def __init__(self, base_transformer):
        self.base_transformer = base_transformer()

    def fit(self, X, y=None):
        self.base_transformer.fit(X, y=y)
        return self

    def transform(self, X, y=None):
        transformed = self.base_transformer.transform(X)
        if hasattr(self.base_transformer, "get_support"):
            cols = X.columns[self.base_transformer.get_support()]
        elif hasattr(self.base_transformer, "statistics_"):
            cols = X.columns[np.flatnonzero(np.logical_not(np.isnan(self.base_transformer.statistics_)))]
        else:
            cols = X.columns.copy()
        return pd.DataFrame(transformed, columns=cols, index=X.index)
