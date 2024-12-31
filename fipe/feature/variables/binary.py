import gurobipy as gp
import numpy as np

from ...mip import MIP, BaseVar
from ...typing import Number


class BinaryVar(BaseVar[Number]):
    _var: gp.Var

    def __init__(self, name: str = "") -> None:
        BaseVar.__init__(self, name)

    def build(self, mip: MIP) -> None:
        self._add_var(mip)

    @property
    def var(self) -> gp.Var:
        return self._var

    def _apply(self, prop_name: str) -> Number:
        value = self._apply_prop(var=self.var, prop_name=prop_name)
        return np.round(value)

    def _add_var(self, mip: MIP) -> None:
        self._var = mip.addVar(vtype=gp.GRB.BINARY, name=self.name)
