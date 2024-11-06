from collections.abc import Callable, Generator
from copy import deepcopy
from itertools import chain

import gurobipy as gp
import numpy as np

from ..mip import MIP, BaseVar
from ..typing import Sample, numeric


class BinaryVar(BaseVar):
    var: gp.Var

    def __init__(self, name: str = "") -> None:
        BaseVar.__init__(self, name)

    def build(self, mip: MIP) -> None:
        self._add_var(mip)

    @property
    def X(self) -> numeric:
        return self.var.X

    @property
    def Xn(self) -> numeric:
        return self.var.Xn

    def _add_var(self, mip: MIP) -> None:
        self.var = mip.addVar(vtype=gp.GRB.BINARY, name=self.name)


class ContinuousVar(BaseVar, gp.tupledict[int, gp.Var]):
    levels: list[numeric]

    _logic_constrs: gp.tupledict[int, gp.Constr]

    def __init__(self, levels: list[numeric], name: str = "") -> None:
        BaseVar.__init__(self, name)
        gp.tupledict.__init__(self)
        self.levels = deepcopy(levels)
        self._logic_constrs = gp.tupledict()

    def build(self, mip: MIP) -> None:
        self._add_vars(mip)
        self._add_logic_constrs(mip)

    @property
    def X(self) -> numeric:
        n = len(self.levels)
        mu = [self[j].X for j in range(n)]
        return self._compute_value(mu)

    @property
    def Xn(self) -> numeric:
        n = len(self.levels)
        mu = [self[j].Xn for j in range(n)]
        return self._compute_value(mu)

    def __getitem__(self, j: int) -> gp.Var:
        return gp.tupledict.__getitem__(self, j)

    def __setitem__(self, j: int, var: gp.Var) -> None:
        gp.tupledict.__setitem__(self, j, var)

    def _add_vars(self, mip: MIP) -> None:
        n = len(self.levels)
        for j in range(n):
            self[j] = mip.addVar(
                vtype=gp.GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"{self.name}_{j}",
            )

    def _add_logic_constrs(self, mip: MIP) -> None:
        n = len(self.levels)
        for j in range(1, n):
            self._logic_constrs[j] = mip.addConstr(
                (self[j - 1] >= self[j]),
                name=f"{self.name}_logic_{j}",
            )

    def _compute_value(self, mu: list[numeric]) -> numeric:
        n = len(self.levels)
        if n == 0:
            return 0.0

        nu = np.array(mu)
        nu = np.floor(nu + 0.5)

        if np.isclose(nu[0], 0.0):
            return self.levels[0] - 1

        if np.isclose(nu[-1], 1.0):
            return self.levels[-1] + 1

        levels = np.array(self.levels)
        dl = np.diff(levels)
        x = levels[0]
        for j in range(1, n):
            if np.isclose(nu[j], 0.0):
                x += dl[j - 1] * 0.5
                break
            x += dl[j - 1]
        return x


class CategoricalVar(BaseVar, gp.tupledict[str, gp.Var]):
    categories: list[str]

    _logic_constr: gp.Constr

    def __init__(self, categories: list[str], name: str = "") -> None:
        BaseVar.__init__(self, name)
        gp.tupledict.__init__(self)
        self.categories = categories

    def build(self, mip: MIP) -> None:
        self._add_vars(mip)
        self._add_logic_constr(mip)

    @property
    def X(self) -> dict[str, numeric]:
        return {cat: self[cat].X for cat in self.categories}

    @property
    def Xn(self) -> dict[str, numeric]:
        return {cat: self[cat].Xn for cat in self.categories}

    def __setitem__(self, cat: str, var: gp.Var) -> None:
        gp.tupledict.__setitem__(self, cat, var)

    def __getitem__(self, cat: str) -> gp.Var:
        return gp.tupledict.__getitem__(self, cat)

    def _add_vars(self, mip: MIP) -> None:
        for cat in self.categories:
            self[cat] = mip.addVar(
                vtype=gp.GRB.BINARY,
                name=f"{self.name}_{cat}",
            )

    def _add_logic_constr(self, mip: MIP) -> None:
        self._logic_constr = mip.addConstr(
            gp.quicksum(self[cat] for cat in self.categories) == 1.0,
            name=f"{self.name}_logic",
        )


class FeatureVars(BaseVar):
    levels: dict[str, list[numeric]]
    categories: dict[str, list[str]]
    binary: dict[str, BinaryVar]
    continuous: dict[str, ContinuousVar]
    categorical: dict[str, CategoricalVar]

    def __init__(self, name: str = "") -> None:
        BaseVar.__init__(self, name=name)
        self.levels = {}
        self.categories = {}
        self.binary = {}
        self.continuous = {}
        self.categorical = {}

    def build(self, mip: MIP) -> None:
        for var in self.values():
            var.build(mip)

    def add_binary(self, feature: str) -> None:
        var = BinaryVar(feature)
        self.binary[feature] = var

    def add_continuous(self, feature: str, levels: list[numeric]) -> None:
        self.levels[feature] = levels
        var = ContinuousVar(levels, feature)
        self.continuous[feature] = var

    def add_categorical(self, feature: str, categories: list[str]) -> None:
        self.categories[feature] = categories
        var = CategoricalVar(categories, feature)
        self.categorical[feature] = var

    def values(
        self,
    ) -> Generator[BinaryVar | ContinuousVar | CategoricalVar, None, None]:
        yield from chain(
            self.binary.values(),
            self.continuous.values(),
            self.categorical.values(),
        )

    def items(
        self,
    ) -> Generator[
        tuple[str, BinaryVar | ContinuousVar | CategoricalVar], None, None
    ]:
        yield from chain(
            self.binary.items(),
            self.continuous.items(),
            self.categorical.items(),
        )

    @property
    def X(self) -> Sample:
        def func(var: BaseVar) -> numeric | dict[str, numeric]:
            return var.X

        return self.apply(func)

    @property
    def Xn(self) -> Sample:
        def func(var: BaseVar) -> numeric | dict[str, numeric]:
            return var.Xn

        return self.apply(func)

    def apply(
        self,
        func: Callable[[BaseVar], numeric | dict[str, numeric]],
    ) -> dict[str, numeric | dict[str, numeric]]:
        v = {}
        for f, var in self.items():
            if isinstance(var, CategoricalVar):
                v.update(func(var))
            else:
                v[f] = func(var)
        return v
