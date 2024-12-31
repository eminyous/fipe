import gurobipy as gp
import numpy as np
import numpy.typing as npt

from ...mip import MIP, BaseVar
from ...typing import Number


class ContinuousVar(BaseVar[Number]):
    levels: npt.NDArray[Number]

    _vars: gp.MVar
    _logic_constrs: gp.tupledict[int, gp.MConstr]
    _floor: bool

    INFINITY = 1.0
    DEFAULT_VALUE: np.float64 = np.float64(0.0)

    def __init__(
        self,
        levels: npt.ArrayLike,
        name: str = "",
        *,
        floor: bool = True,
    ) -> None:
        BaseVar.__init__(self, name=name)
        self.levels = np.asarray(levels, dtype=np.float64)
        self._logic_constrs = gp.tupledict()
        self._floor = floor

    def build(self, mip: MIP) -> None:
        self._add_vars(mip)
        self._add_logic_constrs(mip)

    def __getitem__(self, key: int) -> gp.MVar:
        return self._vars[key]

    def _apply(self, prop_name: str) -> Number:
        n = self.levels.size
        if n == 0:
            return self.DEFAULT_VALUE

        mu = self._apply_m_prop(mvar=self._vars, prop_name=prop_name)

        if self._floor:
            mu = np.floor(mu + 0.5)

        if np.isclose(mu[0], 0.0):
            return self.levels[0] - self.INFINITY

        if np.isclose(mu[-1], 1.0):
            return self.levels[-1] + self.INFINITY

        levels = np.array(self.levels)
        dl = np.diff(levels)
        x = levels[0]
        for j in range(1, n):
            if np.isclose(mu[j], 0.0):
                x += dl[j - 1] * 0.5
                break
            x += dl[j - 1]
        return x

    def _add_vars(self, mip: MIP) -> None:
        n = self.levels.size
        self._vars = mip.addMVar(
            shape=n,
            lb=0.0,
            ub=1.0,
            vtype=gp.GRB.CONTINUOUS,
            name=self.name,
        )

    def _add_logic_constrs(self, mip: MIP) -> None:
        n = self.levels.size
        for j in range(1, n):
            self._logic_constrs[j] = mip.addConstr(
                (self._vars[j - 1] >= self._vars[j]),
                name=f"{self.name}_logic_{j}",
            )
