from abc import ABC, ABCMeta, abstractmethod
from typing import Any, NoReturn

import gurobipy as gp


class MIP(gp.Model):
    """
    Mixed-Integer Programming (MIP) model.

    This class is a wrapper around the Gurobi Model
    class and allows to add attributes to the model.
    """

    __metaclass__ = ABCMeta

    def __init__(self, name: str = "", env: gp.Env | None = None) -> None:
        gp.Model.__init__(self, name, env)

    def __setattr__(self, name: str, value: Any) -> None:  # noqa: ANN401
        return object.__setattr__(self, name, value)


class BaseVar(ABC):
    """
    Base class for variables.

    This class is an abstract class that defines the
    interface for variables in a MIP model.
    When subclassing this class, the following methods and properties:
        - build: add the variable to the MIP model,
        - X: get the value of the variable,
        - Xn: get the value of the variable in the current node,
    must be implemented.
    """

    __metaclass__ = ABCMeta

    name: str
    msg = "Subclasses must implement the {name} {method}"

    def __init__(self, name: str = "") -> None:
        self.name = name

    @abstractmethod
    def build(self, mip: MIP) -> None:
        msg = self.msg.format(name="build", method="method")
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def X(self) -> NoReturn:
        msg = self.msg.format(name="X", method="property")
        raise NotImplementedError(msg)

    @property
    @abstractmethod
    def Xn(self) -> NoReturn:
        msg = self.msg.format(name="Xn", method="property")
        raise NotImplementedError(msg)
