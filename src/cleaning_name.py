from sklearn.base import BaseEstimator, TransformerMixin
from constants import titles
import pandas as pd


class CleaningName(TransformerMixin, BaseEstimator):
    def __init__(self, name_column: str):
        self.name_column = name_column

    def fit(self, x: pd.DataFrame, y=None):
        return self

    def fit_transform(self, x: pd.DataFrame, y=None) -> pd.DataFrame:
        def _extract_title(text):
            start = text.find(",")
            end = text.find(".")
            return text[start + 2:end]

        x["title"] = x[self.name_column].apply(_extract_title)

        x["title"] = x["title"].map(titles)
        x = x.drop(self.name_column, axis=1)
        return x

    def transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        return self.fit_transform(X)
