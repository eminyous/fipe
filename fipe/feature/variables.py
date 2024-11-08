from collections.abc import Callable

import gurobipy as gp
import numpy as np
from numpy.typing import ArrayLike, NDArray

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
    levels: NDArray[np.float64]

    _logic_constrs: gp.tupledict[int, gp.Constr]

    def __init__(self, levels: ArrayLike, name: str = "") -> None:
        BaseVar.__init__(self, name)
        gp.tupledict.__init__(self)
        self.levels = np.asarray(levels)
        self._logic_constrs = gp.tupledict()

    def build(self, mip: MIP) -> None:
        self._add_vars(mip)
        self._add_logic_constrs(mip)

    @property
    def X(self) -> numeric:
        def X(var: gp.Var) -> numeric:
            return var.X

        return self.apply(func=X)

    @property
    def Xn(self) -> numeric:
        def Xn(var: gp.Var) -> numeric:
            return var.Xn

        return self.apply(func=Xn)

    def apply(self, func: Callable[[gp.Var], numeric]) -> numeric:
        n = len(self.levels)
        mu = [func(self[j]) for j in range(n)]
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
        return float(x)

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
        def X(var: gp.Var) -> numeric:
            return np.round(var.X)

        return self.apply(func=X)

    @property
    def Xn(self) -> dict[str, numeric]:
        def Xn(var: gp.Var) -> numeric:
            return np.round(var.Xn)

        return self.apply(func=Xn)

    def apply(self, func: Callable[[gp.Var], numeric]) -> dict[str, numeric]:
        return {cat: func(var) for cat, var in self.items()}

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


FeatureVar = BinaryVar | ContinuousVar | CategoricalVar
FeatureVarValue = numeric | dict[str, numeric]


class FeatureVars(BaseVar, dict[str, FeatureVar]):
    def __init__(self, name: str = "") -> None:
        BaseVar.__init__(self, name=name)
        dict.__init__(self)

    def build(self, mip: MIP) -> None:
        for var in self.values():
            var.build(mip)

    def add_binary(self, feature: str) -> None:
        self[feature] = BinaryVar(name=feature)

    def add_continuous(self, feature: str, levels: list[numeric]) -> None:
        self[feature] = ContinuousVar(levels=levels, name=feature)

    def add_categorical(self, feature: str, categories: list[str]) -> None:
        self[feature] = CategoricalVar(categories=categories, name=feature)

    def __setitem__(self, feature: str, var: FeatureVar) -> None:
        dict.__setitem__(self, feature, var)

    def __getitem__(self, feature: str) -> FeatureVar:
        return dict.__getitem__(self, feature)

    @property
    def X(self) -> Sample:
        def X(var: FeatureVar) -> FeatureVarValue:
            return var.X

        return self.apply(X)

    @property
    def Xn(self) -> Sample:
        def Xn(var: FeatureVar) -> FeatureVarValue:
            return var.Xn

        return self.apply(Xn)

    def apply(
        self,
        func: Callable[[FeatureVar], FeatureVarValue],
    ) -> dict[str, numeric]:
        v = {}
        for f, var in self.items():
            m = func(var)
            if isinstance(m, dict):
                v.update(m)
            else:
                v[f] = m
        return v
