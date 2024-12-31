from abc import ABCMeta
from dataclasses import dataclass

import gurobipy as gp
import numpy as np

from ..feature import FeatureVar, FeatureVars
from ..mip import MIP, BaseVar
from ..tree import Tree, TreeContainer
from ..typing import LeafValue, MNumber


class FlowVars(BaseVar[LeafValue], TreeContainer):
    __metaclass__ = ABCMeta

    FLOW_VAR_FMT = "{name}_flow"
    BRANCH_VAR_FMT = "{name}_branch"

    ROOT_CONSTR_FMT = "{name}_root"
    FLOW_CONSTR_FMT = "{name}_flow_{node}"
    BRANCH_CONSTR_FMT = "{name}_branch_{depth}"

    CHILDREN = ("left", "right")
    BRANCH_FMT = "{name}_{{}}_{node}"
    FEATURE_FMT = "{name}_{feature}"

    @dataclass
    class Branch:
        """
        Branch constraint.

        - left: left constraint:
            * flow[left] <= 1 - var
        - right: right constraint:
            * flow[right] <= var
        """

        left: gp.MConstr
        right: gp.MConstr

    # Variables:
    # Flow variables: each node has a flow variable
    _flow_vars: gp.MVar
    # Branch variables: each depth has a branch variable
    _branch_vars: gp.MVar

    # Constraints:
    # Root constraint:
    #  * flow[root] == 1
    _root_constr: gp.MConstr

    # Flow constraints:
    #  * flow[node] == flow[left] + flow[right]
    _flow_constrs: gp.tupledict[int, gp.MConstr]

    # Flow branchs:
    #  * flow[left] <= 1 - branch[depth]
    #  * flow[right] <= branch[depth]
    _flow_branchs: gp.tupledict[tuple[int, int], Branch]

    # Feature branchs:
    #   * flow[node] <= 1 - var
    #   * flow[node] <= var
    _feature_branchs: gp.tupledict[tuple[str, int], Branch]

    def __init__(
        self,
        tree: Tree,
        name: str = "",
    ) -> None:
        TreeContainer.__init__(self, tree=tree)
        BaseVar.__init__(self, name=name)
        self._flow_constrs = gp.tupledict()
        self._flow_branchs = gp.tupledict()
        self._feature_branchs = gp.tupledict()

    @property
    def value(self) -> gp.MLinExpr:
        node_value = self.node_value
        return np.sum([self[node] * node_value[node] for node in self.leaves])

    @property
    def flow(self) -> MNumber:
        return np.asarray(self._flow_vars.Xn)

    # Public methods:
    # --------------
    #  * build (override): BaseVar
    #  * add_feature_vars
    #  * __getitem__ (override): object
    def build(self, mip: MIP) -> None:
        self._add_flow_vars(mip=mip)
        self._add_branch_vars(mip=mip)
        self._add_root_constr(mip=mip)
        self._add_flow_constrs(mip=mip)
        self._add_flow_branchs(mip=mip)

    def add_feature_vars(self, mip: MIP, feature_vars: FeatureVars) -> None:
        for feature, var in feature_vars.items():
            self._add_feature_branchs(mip=mip, feature=feature, var=var)

    def __getitem__(self, key: int) -> gp.MVar:
        return self._flow_vars[key]

    # Protected methods:
    # ------------------
    #  * _apply (override): BaseVar
    def _apply(self, prop_name: str) -> LeafValue:
        flow = self._apply_m_prop(mvar=self._flow_vars, prop_name=prop_name)
        node_value = self.node_value
        values = [flow[node] * node_value[node] for node in self.leaves]
        return np.sum(values, axis=0)

    # Private methods:
    # ----------------
    #  * _add_flow_vars
    #  * _add_branch_vars
    #  * _add_root_constr
    #  * _add_flow_constrs
    #  * _add_flow_constr_at_node
    #  * _create_branch
    #  * _add_flow_branchs
    #  * _add_flow_branchs_at_depth
    #  * _add_flow_branch_at_node
    #  * _add_feature_branchs
    #  * _add_feature_branch_at_node

    def _add_flow_vars(self, mip: MIP) -> None:
        name = self.FLOW_VAR_FMT.format(name=self.name)
        self._flow_vars = mip.addMVar(
            shape=self.n_nodes,
            vtype=gp.GRB.CONTINUOUS,
            lb=0.0,
            ub=1.0,
            name=name,
        )

    def _add_branch_vars(self, mip: MIP) -> None:
        name = self.BRANCH_VAR_FMT.format(name=self.name)
        self._branch_vars = mip.addMVar(
            shape=self.max_depth,
            vtype=gp.GRB.BINARY,
            name=name,
        )

    def _add_root_constr(self, mip: MIP) -> None:
        expr = self[self.root_id] == 1.0
        name = self.ROOT_CONSTR_FMT.format(name=self.name)
        self._root_constr = mip.addConstr(expr, name=name)

    def _add_flow_constrs(self, mip: MIP) -> None:
        for node in self.internal_nodes:
            self._add_flow_constr_at_node(mip=mip, node=node)

    def _add_flow_constr_at_node(self, mip: MIP, node: int) -> None:
        left = self.left[node]
        right = self.right[node]
        expr = self[node] == self[left] + self[right]
        name = self.FLOW_CONSTR_FMT.format(name=self.name, node=node)
        constr = mip.addConstr(expr, name=name)
        self._flow_constrs[node] = constr

    def _create_branch(
        self,
        mip: MIP,
        var: gp.Var | gp.MVar,
        node: int,
        name: str = "",
    ) -> Branch:
        fmt = self.BRANCH_FMT.format(name=name, node=node)
        lname, rname = map(fmt.format, self.CHILDREN)

        left = self.left[node]
        right = self.right[node]

        lexpr = self[left] <= 1 - var
        rexpr = self[right] <= var

        lconstr = mip.addConstr(lexpr, name=lname)
        rconstr = mip.addConstr(rexpr, name=rname)

        return self.Branch(left=lconstr, right=rconstr)

    def _add_flow_branchs(self, mip: MIP) -> None:
        for depth in range(self.max_depth):
            var = self._branch_vars[depth]
            self._add_flow_branchs_at_depth(mip=mip, var=var, depth=depth)

    def _add_flow_branchs_at_depth(
        self,
        mip: MIP,
        var: gp.MVar,
        depth: int,
    ) -> None:
        for node in self.nodes_at_depth(depth):
            self._add_flow_branch_at_node(
                mip=mip,
                var=var,
                node=node,
                depth=depth,
            )

    def _add_flow_branch_at_node(
        self,
        mip: MIP,
        var: gp.MVar,
        node: int,
        depth: int,
    ) -> None:
        name = self.BRANCH_CONSTR_FMT.format(name=self.name, depth=depth)
        branch = self._create_branch(mip=mip, var=var, node=node, name=name)
        self._flow_branchs[depth, node] = branch

    def _add_feature_branchs(
        self,
        mip: MIP,
        feature: str,
        var: FeatureVar,
    ) -> None:
        for node in self.nodes_split_on(feature):
            self._add_feature_branch_at_node(
                mip=mip,
                feature=feature,
                fvar=var,
                node=node,
            )

    def _add_feature_branch_at_node(
        self,
        mip: MIP,
        feature: str,
        fvar: FeatureVar,
        node: int,
    ) -> None:
        level = self.threshold.get(node)
        cat = self.category.get(node)
        name = self.FEATURE_FMT.format(name=self.name, feature=feature)
        var = FeatureVars.fetch(fvar, level=level, category=cat)
        rule = self._create_branch(mip=mip, var=var, node=node, name=name)
        self._feature_branchs[feature, node] = rule
