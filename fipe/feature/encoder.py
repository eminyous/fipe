from copy import deepcopy

import numpy as np
import pandas as pd

from ..typing import FeatureType


class FeatureEncoder:
    """
    Encoder class for encoding data features.

    This class is used to encode the features of a dataset.
    """

    columns: list[str]
    X: pd.DataFrame

    types: dict[str, FeatureType]
    categories: dict[str, list[str]]
    inverse_categories: dict[str, str]

    _tol: float
    _scale: float

    def __init__(self, X: pd.DataFrame, **kwargs) -> None:
        self._tol = kwargs.get("tol", 1e-4)
        self._scale = kwargs.get("scale", 1e4)

        self.X = deepcopy(X)

        self.types = {}
        self.categories = {}
        self.inverse_categories = {}
        self._parse()

    @property
    def n_features(self) -> int:
        return len(self.types)

    @property
    def binary(self) -> set[str]:
        return {f for f, t in self.types.items() if t == FeatureType.BIN}

    @property
    def categorical(self) -> set[str]:
        return {f for f, t in self.types.items() if t == FeatureType.CAT}

    @property
    def continuous(self) -> set[str]:
        return {f for f, t in self.types.items() if t == FeatureType.CON}

    def _parse(self) -> None:
        self._clean_data()

        self.columns = list(self.X.columns)
        self._parse_binary_features()
        self._parse_continuous_features()
        self._parse_categorical_features()
        self._save_columns()

    def _clean_data(self) -> None:
        # Drop missing values
        self.X = self.X.dropna()
        # Drop columns with only one unique value
        X = self.X.to_numpy()
        contains_unique = np.apply_along_axis(
            lambda x: len(np.unique(x)) > 1,
            0,
            X,
        )
        self.X = self.X.loc[:, contains_unique]

    def _parse_binary_features(self) -> None:
        # For each column in the data
        # if the number of unique values is 2
        # then the feature is binary.
        # Replace the values with 0 and 1.
        for c in self.columns:
            NUM_UNIQUE_VALUES = 2
            if self.X[c].nunique() == NUM_UNIQUE_VALUES:
                self.types[c] = FeatureType.BIN
                x = pd.get_dummies(self.X[c], drop_first=True)
                self.X = self.X.drop(columns=c)
                self.X[c] = x.iloc[:, 0]

    def _parse_continuous_features(self) -> None:
        # For each column in the data
        # if the column has been identified as binary
        # then skip it. Otherwise, check if the column
        # has category dtype. If it does, skip it.
        # Otherwise, try to convert the column to numeric.
        # If the conversion is successful, then the column
        # is numerical. Convert the column to numeric
        # and store the upper and lower bounds.
        for c in self.columns:
            if c in self.types:
                continue

            if self.X[c].dtype == "category":
                continue

            x = pd.to_numeric(self.X[c], errors="coerce")
            if x.notna().all():
                self.types[c] = FeatureType.CON
                # Rescale the numerical values
                # to the range [0, scale]
                # for numerical stability.
                if np.diff(x).min() < self._tol:
                    x = (x - x.min()) / (x.max() - x.min())
                    x = np.round(x * self._scale)
                self.X[c] = x

    def _parse_categorical_features(self) -> None:
        # For each column in the data
        # if the column has been identified as binary
        # or numerical, then skip it. Otherwise, the column
        # is categorical. Store the categories and
        # the inverse categories. Replace the column
        # with the encoded columns.
        for c in self.columns:
            if c in self.types:
                continue

            self.types[c] = FeatureType.CAT
            x = pd.get_dummies(self.X[c], prefix=c)
            self.categories[c] = list(x.columns)
            for v in self.categories[c]:
                self.inverse_categories[v] = c
            # Drop the original column
            self.X = self.X.drop(columns=c)
            # Add the encoded columns
            self.X = pd.concat([self.X, x], axis=1)

    def _save_columns(self) -> None:
        self.columns = list(self.X.columns)
