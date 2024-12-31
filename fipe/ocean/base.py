import gurobipy as gp
import numpy.typing as npt

from ..ensemble import Ensemble, EnsembleContainer
from ..feature import FeatureContainer, FeatureEncoder, FeatureVars
from ..flow import FlowVars
from ..mip import MIP
from ..parsers import LevelParser
from ..typing import LeafValue, MNumber


class BaseOCEAN(
    MIP,
    EnsembleContainer,
    FeatureContainer,
    LevelParser,
):
    FEATURE_VARS_NAME = "feature_vars"
    FLOW_VAR_FMT = "tree_{t}"

    _feature_vars: FeatureVars
    _flow_vars: dict[int, FlowVars[LeafValue]]

    def __init__(
        self,
        encoder: FeatureEncoder,
        ensemble: Ensemble,
        weights: npt.ArrayLike,
        **kwargs,
    ) -> None:
        name = kwargs.get("name", "OCEAN")
        env = kwargs.get("env")
        MIP.__init__(self, name=name, env=env)
        EnsembleContainer.__init__(self, ensemble=ensemble, weights=weights)
        FeatureContainer.__init__(self, encoder=encoder)
        LevelParser.__init__(self, **kwargs)
        self.parse_levels(ensembles=[ensemble], encoder=encoder)
        self._add_feature_vars()
        self._add_flow_vars()

    def build(self) -> None:
        self._build_feature_vars()
        self._build_flow_vars()
        self._build_feature_constrs()

    def _build_feature_vars(self) -> None:
        self._feature_vars.build(mip=self)

    def _build_flow_vars(self) -> None:
        for flow_vars in self._flow_vars.values():
            flow_vars.build(mip=self)

    def _build_feature_constrs(self) -> None:
        for flow_vars in self._flow_vars.values():
            flow_vars.add_feature_vars(
                mip=self,
                feature_vars=self._feature_vars,
            )

    def _add_feature_vars(self) -> None:
        self._feature_vars = FeatureVars(name=self.FEATURE_VARS_NAME)
        for feature in self.features:
            self._add_feature_var(feature)

    def _add_feature_var(self, feature: str) -> None:
        vtype = self.types[feature].value
        levels = self.levels.get(feature)
        categories = self.categories.get(feature)
        self._feature_vars.add_var(
            feature=feature,
            vtype=vtype,
            levels=levels,
            categories=categories,
        )

    def _add_flow_vars(self) -> None:
        self._flow_vars = {}
        for t, tree in enumerate(self.ensemble):
            name = self.FLOW_VAR_FMT.format(t=t)
            self._flow_vars[t] = FlowVars(tree=tree, name=name)

    def function(self, class_: int) -> gp.LinExpr:
        return self.weighted_function(class_=class_, weights=self._weights)

    def weighted_function(
        self,
        class_: int,
        weights: MNumber,
    ) -> gp.LinExpr:
        if self.ensemble.m_valued:
            return self._function_m_valued(class_=class_, weights=weights)
        return self._function_valued(class_=class_, weights=weights)

    def _function_valued(
        self,
        class_: int,
        weights: MNumber,
    ) -> gp.LinExpr:
        return gp.quicksum(
            weights[t] * self._flow_vars[t].value[class_]
            for t in range(self.n_estimators)
        )

    def _function_m_valued(
        self,
        class_: int,
        weights: MNumber,
    ) -> gp.LinExpr:
        if self.n_classes == Ensemble.NUM_BINARY_CLASSES:
            return self._function_binary(class_=class_, weights=weights)
        return self._function_multi(class_=class_, weights=weights)

    def _function_binary(
        self,
        class_: int,
        weights: MNumber,
    ) -> gp.LinExpr:
        sign = 2 * class_ - 1
        return gp.quicksum(
            sign * weights[t] * self._flow_vars[t].value
            for t in range(self.n_estimators)
        )

    def _function_multi(
        self,
        class_: int,
        weights: MNumber,
    ) -> gp.LinExpr:
        return gp.quicksum(
            weights[t] * self._flow_vars[t * self.n_classes + class_].value
            for t in range(self.n_estimators)
        )
