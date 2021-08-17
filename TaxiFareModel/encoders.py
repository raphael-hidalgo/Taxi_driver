from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
from TaxiFareModel.utils import haversine_vectorized


class TimeFeaturesEncoder(BaseEstimator, TransformerMixin):

    def __init__(self, time_column, time_zone_name='America/Los_Angeles'):
        self.time_column = time_column
        self.time_zone_name = time_zone_name

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X.index = pd.to_datetime(X[self.time_column])
        X.index = X.index.tz_convert(self.time_zone_name)
        X["hour"] = X.index.hour
        return X[[
                'hour',
            ]].reset_index(drop=True)


class DistanceTransformer(BaseEstimator, TransformerMixin):

    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        assert isinstance(X, pd.DataFrame)

        X['distance'] = haversine_vectorized(X)
        return X[[
            'distance',
        ]]
