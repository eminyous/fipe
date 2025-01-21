from dataclasses import dataclass
from enum import Enum


class PrunerSolver(Enum):
    GUROBI = "gurobi"
    SCIP = "scip"

    @staticmethod
    def values() -> list[str]:
        return [solver.value for solver in PrunerSolver]


@dataclass
class Environment:
    _pruner_solver: PrunerSolver = PrunerSolver.SCIP

    @property
    def pruner_solver(self) -> PrunerSolver:
        return self._pruner_solver

    @pruner_solver.setter
    def pruner_solver(self, value: str) -> None:
        if value not in PrunerSolver.values():
            msg = f"The pruner solver must be one of {PrunerSolver.values()}."
            raise ValueError(msg)
        self._pruner_solver = PrunerSolver(value)


ENV = Environment()
