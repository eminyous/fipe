from collections.abc import Callable

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from ...feature.variables import (
    BinaryVar,
    CategoricalVar,
    ContinuousVar,
    FeatureVar,
    FeatureVars,
)
from ...mip import MIP, BaseVar
from ...typing import Node
from ..tree import Tree


class FlowVars(BaseVar, gp.tupledict[Node, gp.Var]):
    tree: Tree

    _depth_vars: gp.tupledict[Node, gp.Var]
    _root_constr: gp.Constr
    _flow_constrs: gp.tupledict[Node, gp.Constr]
    _left_constrs: gp.tupledict[tuple[int, Node], gp.Constr]
    _right_constrs: gp.tupledict[tuple[int, Node], gp.Constr]

    def __init__(self, tree: Tree, name: str = "") -> None:
        BaseVar.__init__(self, name=name)
        gp.tupledict.__init__(self)
        self.tree = tree

        self._depth_vars = gp.tupledict()
        self._flow_constrs = gp.tupledict()
        self._left_constrs = gp.tupledict()
        self._right_constrs = gp.tupledict()

    def build(self, mip: MIP) -> None:
        self._add_flow_vars(mip=mip)
        self._add_depth_vars(mip=mip)
        self._add_root_constr(mip=mip)
        self._add_flow_constrs(mip=mip)
        self._add_branch_constrs(mip=mip)

    def add_branch_rule(
        self,
        mip: MIP,
        var: gp.Var,
        node: Node,
        name: str = "branch",
    ) -> tuple[gp.Constr, gp.Constr]:
        left = self.tree.left[node]
        right = self.tree.right[node]

        left_constr = mip.addConstr(
            self[left] <= 1 - var,
            name=f"{name}_left_{node}",
        )
        right_constr = mip.addConstr(
            self[right] <= var,
            name=f"{name}_right_{node}",
        )
        return left_constr, right_constr

    def add_feature_constrs(
        self,
        mip: MIP,
        feature_vars: FeatureVars,
    ) -> None:
        for feature, var in feature_vars.items():
            self._add_feature_var(mip=mip, feature=feature, var=var)

    @property
    def value(self) -> gp.LinExpr:
        return gp.quicksum(
            self.tree.value[node] * self[node] for node in self.tree.leaves
        )

    @property
    def X(self) -> float:
        def X(var: gp.Var) -> float:
            return var.X

        return self.apply(func=X)

    @property
    def Xn(self) -> float:
        def Xn(var: gp.Var) -> float:
            return var.Xn

        return self.apply(func=Xn)

    def apply(self, func: Callable[[gp.Var], float]) -> float:
        return sum(
            self.tree.value[node] * func(self[node])
            for node in self.tree.leaves
        )

    @property
    def flow(self) -> dict[Node, float]:
        return {node: self[node].Xn for node in self.tree.nodes}

    def __setitem__(self, node: Node, var: gp.Var) -> None:
        gp.tupledict.__setitem__(self, node, var)

    def __getitem__(self, node: Node) -> gp.Var:
        return gp.tupledict.__getitem__(self, node)

    def _add_flow_vars(self, mip: MIP) -> None:
        for node in self.tree:
            self[node] = mip.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"{self.name}_flow_{node}",
            )

    def _add_depth_vars(self, mip: MIP) -> None:
        for depth in range(self.tree.max_depth):
            self._depth_vars[depth] = mip.addVar(
                vtype=GRB.BINARY,
                name=f"{self.name}_depth_{depth}",
            )

    def _add_root_constr(self, mip: MIP) -> None:
        self._root_constr = mip.addConstr(
            self[self.tree.root_id] == 1.0,
            name=f"{self.name}_root",
        )

    def _add_flow_constrs(self, mip: MIP) -> None:
        for node in self.tree.internal_nodes:
            left = self.tree.left[node]
            right = self.tree.right[node]
            self._flow_constrs[node] = mip.addConstr(
                self[node] == self[left] + self[right],
                name=f"{self.name}_flow_{node}",
            )

    def _add_branch_constrs(self, mip: MIP) -> None:
        for depth in range(self.tree.max_depth):
            var = self._depth_vars[depth]
            for node in self.tree.nodes_at_depth(depth):
                left, right = self.add_branch_rule(
                    mip=mip,
                    var=var,
                    node=node,
                    name=f"{self.name}_depth_{depth}",
                )
                self._left_constrs[depth, node] = left
                self._right_constrs[depth, node] = right

    def _add_feature_var(
        self,
        mip: MIP,
        feature: str,
        var: FeatureVar,
    ) -> None:
        for node in self.tree.nodes_split_on(feature):
            if isinstance(var, BinaryVar):
                self.add_branch_rule(
                    mip=mip,
                    var=var.var,
                    node=node,
                    name=f"{self.name}_binary_{feature}",
                )
            elif isinstance(var, ContinuousVar):
                th = self.tree.threshold[node]
                j = np.where(var.levels == th)[0][0]
                self.add_branch_rule(
                    mip=mip,
                    var=var[j],
                    node=node,
                    name=f"{self.name}_continuous_{feature}",
                )
            elif isinstance(var, CategoricalVar):
                cat = self.tree.categories[node]
                self.add_branch_rule(
                    mip=mip,
                    var=var[cat],
                    node=node,
                    name=f"{self.name}_categorical_{feature}_{cat}",
                )
