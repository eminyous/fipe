from itertools import chain

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from ..feature.variables import (
    BinaryVar,
    CategoricalVar,
    ContinuousVar,
    FeatureVars,
)
from ..mip import MIP, BaseVar
from ..tree.tree import Tree
from ..typing import Node


class FlowVars(BaseVar, gp.tupledict[Node, gp.Var]):
    """
    This class represents the flow variables of a tree.
    Each node in the tree has a flow variable associated with it.

    Parameters:
    -----------
    tree: Tree
        The tree for which the flow variables are created.
    name: str
        The name of the flow variables.
    """

    tree: Tree

    _depth_vars: gp.tupledict[int, gp.Var]
    _root_constr: gp.Constr
    _flow_constrs: gp.tupledict[Node, gp.Constr]
    _left_constrs: gp.tupledict[tuple[int, Node], gp.Constr]
    _right_constrs: gp.tupledict[tuple[int, Node], gp.Constr]

    def __init__(self, tree: Tree, name: str = ""):
        BaseVar.__init__(self, name)
        gp.tupledict.__init__(self)
        self.tree = tree
        self._depth_vars = gp.tupledict()
        self._flow_constrs = gp.tupledict()
        self._left_constrs = gp.tupledict()
        self._right_constrs = gp.tupledict()

    def build(self, mip: MIP):
        """
        Build the flow variables and constraints.

        Parameters:
        -----------
        mip: MIP
            The Gurobi model to which the variables
            and constraints are added.
        """
        self._add_flow_vars(mip=mip)
        self._add_depth_vars(mip=mip)
        self._add_root_constr(mip=mip)
        self._add_flow_constrs(mip=mip)
        self._add_branch_constrs(mip=mip)

    def add_branch_rule(
        self, mip: MIP, var: gp.Var, node: Node, name: str = "branch"
    ):
        """
        Add a branching rule to flow variable of the tree
        based on the `var` variable at the given `node`.

        This means that if `var` is 1, then the flow variable
        at the left child of `node` is also 1, and the flow
        variable at the right child of `node` is 0.
        However, if `var` is 0, then the flow variable at the
        left child of `node` is 0, and the flow variable at the
        right child of `node` is 1.

        Parameters:
        -----------
        mip: MIP
            The Gurobi model to which the constraints are added.
        var: gp.Var
            The variable based on which the branching rule is applied.
        node: Node
            The node in the tree to which the branching rule is applied.
        name: str
            The name of the branching rule.

        Returns:
        --------
        left_constr: gp.Constr
            The constraint that the flow variable at the left child
            of `node` is 1 if `var` is 1.
        right_constr: gp.Constr
            The constraint that the flow variable at the right child
            of `node` is 1 if `var` is 0.
        """
        left = self.tree.left[node]
        right = self.tree.right[node]
        left_constr = mip.addConstr(
            self[left] <= 1 - var, name=f"{name}_left_{node}"
        )
        right_constr = mip.addConstr(
            self[right] <= var, name=f"{name}_right_{node}"
        )
        return left_constr, right_constr

    def add_feature_constrs(self, mip: MIP, feature_vars: FeatureVars):
        for feature, var in chain(
            feature_vars.binary.items(),
            feature_vars.continuous.items(),
            feature_vars.categorical.items(),
        ):
            self._add_feature_var(mip, feature, var)

    @property
    def value(self) -> dict[int, gp.LinExpr]:
        value = {}
        for c in range(self.tree.n_classes):
            value[c] = gp.quicksum(
                self.tree.value[n][c] * self[n] for n in self.tree.leaves
            )
        return value

    @property
    def X(self):
        v = np.zeros(self.tree.n_classes)
        for n in self.tree.leaves:
            v += self.tree.value[n] * np.round(self[n].X)
        return v

    @property
    def Xn(self):
        v = np.zeros(self.tree.n_classes)
        for n in self.tree.leaves:
            v += self.tree.value[n] * np.round(self[n].Xn)
        return v

    @property
    def flow(self):
        flow = {}
        for node in self.tree:
            flow[node] = np.round(self[node].Xn)
        return flow

    def __setitem__(self, node: Node, var: gp.Var):
        gp.tupledict.__setitem__(self, node, var)

    def __getitem__(self, node: Node) -> gp.Var:
        return gp.tupledict.__getitem__(self, node)

    def _add_flow_vars(self, mip: MIP):
        for node in self.tree:
            self[node] = mip.addVar(
                vtype=GRB.CONTINUOUS,
                lb=0.0,
                ub=1.0,
                name=f"{self.name}_flow_{node}",
            )

    def _add_depth_vars(self, mip: MIP):
        for depth in range(self.tree.max_depth):
            self._depth_vars[depth] = mip.addVar(
                vtype=GRB.BINARY, name=f"{self.name}_depth_{depth}"
            )

    def _add_root_constr(self, mip: MIP):
        self._root_constr = mip.addConstr(
            self[self.tree.root] == 1.0, name=f"{self.name}_root"
        )

    def _add_flow_constrs(self, mip: MIP):
        for node in self.tree.internal_nodes:
            left = self.tree.left[node]
            right = self.tree.right[node]
            self._flow_constrs[node] = mip.addConstr(
                self[node] == self[left] + self[right],
                name=f"{self.name}_flow_{node}",
            )

    def _add_branch_constrs(self, mip: MIP):
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
        var: BinaryVar | ContinuousVar | CategoricalVar,
    ):
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
                j = var.levels.index(th)
                self.add_branch_rule(
                    mip=mip,
                    var=var[j],
                    node=node,
                    name=f"{self.name}_continuous_{feature}_{j}",
                )
            elif isinstance(var, CategoricalVar):
                cat = self.tree.category[node]
                self.add_branch_rule(
                    mip=mip,
                    var=var[cat],
                    node=node,
                    name=f"{self.name}_categorical_{cat}",
                )
