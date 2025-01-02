from abc import ABC, ABCMeta, abstractmethod
from typing import Any, Generic, TypeVar

import gurobipy as gp
import numpy as np

from .typing import MNumber, Number, SNumber

VT = TypeVar("VT", bound=Number | MNumber | SNumber)


class MIP(gp.Model):
    """
    Mixed-Integer Programming (MIP) model.

    This class is a wrapper around the Gurobi Model
    class and allows to add attributes to the model.
    """

    __metaclass__ = ABCMeta

    def __init__(self, name: str = "", env: gp.Env | None = None) -> None:
        gp.Model.__init__(self, name=name, env=env)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        return object.__setattr__(self, name, value)


class BaseVar(ABC, Generic[VT]):
    """
    Base class for variables.

    This class is an abstract class that defines the
    interface for variables in a MIP model.
    When subclassing this class, the following methods:
        - build: add the variable to the MIP model,
        - apply: apply a function to the variable,
    must be implemented.
    """

    __metaclass__ = ABCMeta

    name: str

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def build(self, mip: MIP) -> None:
        raise NotImplementedError

    @abstractmethod
    def _apply(self, prop_name: str) -> VT:
        raise NotImplementedError

    @staticmethod
    def _apply_prop(var: gp.Var, prop_name: str) -> Number:
        return np.float64(getattr(var, prop_name))

    @staticmethod
    def _apply_m_prop(mvar: gp.MVar, prop_name: str) -> MNumber:
        return np.array(getattr(mvar, prop_name))

    @property
    def X(self) -> VT:
        return self._apply(prop_name=gp.GRB.Attr.X)

    @property
    def Xn(self) -> VT:
        return self._apply(prop_name=gp.GRB.Attr.Xn)
