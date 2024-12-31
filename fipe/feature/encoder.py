from copy import deepcopy

import numpy as np
import numpy.typing as npt
import pandas as pd

from ..typing import Categories, FeatureType


class FeatureEncoder:
    """
    Encoder class for encoding data features.

    This class is used to encode the features of a dataset.
    """

    DEFAULT_TOL = 1e-4
    DEFAULT_SCALE = 1e4
    NUM_BINARY_VALUES = 2

    columns: list[str]
    X: pd.DataFrame

    types: dict[str, FeatureType]
    categories: dict[str, Categories]
    inverse_categories: dict[str, str]

    _tol: float
    _scale: float

    def __init__(
        self,
        X: pd.DataFrame,
        *,
        tol: float = 1e-4,
        scale: float = 1e4,
    ) -> None:
        self._tol = tol
        self._scale = scale

        self.X = deepcopy(X)

        self.types = {}
        self.categories = {}
        self.inverse_categories = {}
        self._encode()

    @property
    def features(self) -> set[str]:
        return set(self.types.keys())

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

    def _encode(self) -> None:
        self._clean_data()

        self.columns = list(self.X.columns)
        self._encode_binary_features()
        self._encode_continuous_features()
        self._encode_categorical_features()
        self._save_columns()

    def _clean_data(self) -> None:
        # Drop missing values
        self.X = self.X.dropna()
        # Drop columns with only one unique value
        X = self.X.to_numpy()

        def nunique(x: npt.NDArray) -> int:
            return len(np.unique(x))

        num = np.apply_along_axis(nunique, 0, X)
        self.X = self.X.loc[:, num > 1]

    def _encode_binary_features(self) -> None:
        # For each column in the data
        # if the number of unique values is 2
        # then the feature is binary.
        # Replace the values with 0 and 1.
        for column in self.columns:
            if self.X[column].nunique() == self.NUM_BINARY_VALUES:
                self.types[column] = FeatureType.BIN
                x = pd.get_dummies(self.X[column], drop_first=True)
                self.X[column] = x.iloc[:, 0]

    def _encode_continuous_features(self) -> None:
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

    def _encode_categorical_features(self) -> None:
        # For each column in the data
        # if the column has been identified as binary
        # or numerical, then skip it. Otherwise, the column
        # is categorical. Store the categories and
        # the inverse categories. Replace the column
        # with the encoded columns.
        for column in self.columns:
            if column in self.types:
                continue

            self.types[column] = FeatureType.CAT
            x = pd.get_dummies(self.X[column], prefix=column)
            self.categories[column] = list(x.columns)
            for v in self.categories[column]:
                self.inverse_categories[v] = column
            # Drop the original column
            self.X = self.X.drop(columns=column)
            # Add the encoded columns
            self.X = pd.concat([self.X, x], axis=1)

    def _save_columns(self) -> None:
        self.columns = list(self.X.columns)
