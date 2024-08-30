from abc import ABCMeta

import numpy as np
import pandas as pd

from ..typing import Sample
from .encoder import FeatureEncoder


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
    def n_features(self) -> int:
        return self._encoder.n_features

    @property
    def categories(self) -> dict[str, list[str]]:
        return self._encoder.categories

    @property
    def inverse_categories(self) -> dict[str, str]:
        return self._encoder.inverse_categories

    def transform(self, X: Sample | list[Sample]) -> np.ndarray:
        if not isinstance(X, list):
            X = [X]
        return pd.DataFrame(X, columns=self.columns).to_numpy()
