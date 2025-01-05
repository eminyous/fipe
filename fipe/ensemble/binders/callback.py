from abc import ABC, abstractmethod

from ...typing import Prob


class BinderCallback(ABC):
    @abstractmethod
    def predict_leaf(self, e: int, index: int) -> Prob:
        raise NotImplementedError
