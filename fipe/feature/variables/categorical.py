from collections.abc import Iterable

import gurobipy as gp
import pandas as pd

from ...mip import MIP, BaseVar
from ...typing import Categories, SNumber


class CategoricalVar(BaseVar[SNumber], gp.tupledict[str, gp.Var]):
    categories: Categories

    _logic_constr: gp.Constr

    def __init__(
        self,
        categories: Iterable[str],
        name: str = "",
    ) -> None:
        BaseVar.__init__(self, name=name)
        gp.tupledict.__init__(self)
        self.categories = set(categories)

    def build(self, mip: MIP) -> None:
        self._add_vars(mip)
        self._add_logic_constr(mip)

    def _apply(self, prop_name: str) -> SNumber:
        values = {
            cat: self._apply_prop(var, prop_name=prop_name)
            for cat, var in self.items()
        }
        return pd.Series(values)

    def _add_vars(self, mip: MIP) -> None:
        for cat in self.categories:
            self._add_var(mip, cat=cat)

    def _add_var(self, mip: MIP, *, cat: str) -> None:
        self[cat] = mip.addVar(
            vtype=gp.GRB.BINARY,
            name=f"{self.name}_{cat}",
        )

    def _add_logic_constr(self, mip: MIP) -> None:
        self._logic_constr = mip.addConstr(
            gp.quicksum(self) == 1.0,
            name=f"{self.name}_logic",
        )
