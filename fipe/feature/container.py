from abc import ABCMeta

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

    def __init__(self, encoder: FeatureEncoder):
        self._encoder = encoder

    @property
    def columns(self):
        return self._encoder.columns

    @property
    def binary(self):
        return self._encoder.binary

    @property
    def continuous(self):
        return self._encoder.continuous

    @property
    def categorical(self):
        return self._encoder.categorical

    @property
    def n_features(self):
        return self._encoder.n_features

    @property
    def categories(self):
        return self._encoder.categories

    @property
    def inverse_categories(self):
        return self._encoder.inverse_categories

    def transform(self, X: Sample | list[Sample]):
        if not isinstance(X, list):
            X = [X]

        df = pd.DataFrame(X, columns=self.columns)
        return df.values
