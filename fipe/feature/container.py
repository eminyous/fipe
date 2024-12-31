from abc import ABCMeta
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from ..typing import Categories, FeatureType, MNumber, Transformable
from .encoder import FeatureEncoder


@dataclass
class FeatureContainer:
    """
    Abstract class for feature containers.

    This class is a wrapper around the FeatureEncoder class
    and provide a transform method to transform a sample
    to a numpy array.
    """

    __metaclass__ = ABCMeta

    encoder: FeatureEncoder

    @property
    def columns(self) -> list[str]:
        return self.encoder.columns

    @property
    def binary(self) -> set[str]:
        return self.encoder.binary

    @property
    def continuous(self) -> set[str]:
        return self.encoder.continuous

    @property
    def categorical(self) -> set[str]:
        return self.encoder.categorical

    @property
    def types(self) -> Mapping[str, FeatureType]:
        return self.encoder.types

    @property
    def features(self) -> set[str]:
        return self.encoder.features

    @property
    def n_features(self) -> int:
        return self.encoder.n_features

    @property
    def categories(self) -> Mapping[str, Categories]:
        return self.encoder.categories

    @property
    def inverse_categories(self) -> Mapping[str, str]:
        return self.encoder.inverse_categories

    def transform(self, X: Transformable) -> MNumber:
        if isinstance(X, pd.Series):
            X = [X]
        data = pd.concat(X, axis=1).T if isinstance(X, list) else X.T
        return data[self.columns].to_numpy()
