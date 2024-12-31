from abc import ABCMeta
from collections.abc import Mapping

import pandas as pd

from ..typing import MNumber
from .encoder import FeatureEncoder
from .types import Categories, FeatureType

Transformable = pd.Series | list[pd.Series] | pd.DataFrame


class FeatureContainer:
    """
    Abstract class for feature containers.

    This class is a wrapper around the FeatureEncoder class
    and provide a transform method to transform a sample
    to a numpy array.
    """

    __metaclass__ = ABCMeta

    _encoder: FeatureEncoder

    def __init__(self, encoder: FeatureEncoder) -> None:
        self._encoder = encoder

    @property
    def columns(self) -> list[str]:
        return self._encoder.columns

    @property
    def binary(self) -> set[str]:
        return self._encoder.binary

    @property
    def continuous(self) -> set[str]:
        return self._encoder.continuous

    @property
    def categorical(self) -> set[str]:
        return self._encoder.categorical

    @property
    def types(self) -> Mapping[str, FeatureType]:
        return self._encoder.types

    @property
    def features(self) -> set[str]:
        return self._encoder.features

    @property
    def n_features(self) -> int:
        return self._encoder.n_features

    @property
    def categories(self) -> Mapping[str, Categories]:
        return self._encoder.categories

    @property
    def inverse_categories(self) -> Mapping[str, str]:
        return self._encoder.inverse_categories

    def transform(self, X: Transformable) -> MNumber:
        if isinstance(X, pd.Series):
            X = [X]
        data = pd.concat(X, axis=1).T if isinstance(X, list) else X.T
        return data[self.columns].to_numpy()
