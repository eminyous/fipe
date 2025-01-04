from functools import partial

import gurobipy as gp
import numpy.typing as npt

from ..ensemble import EnsembleContainer
from ..feature import FeatureContainer, FeatureEncoder, FeatureVars
from ..flow import FlowVars
from ..mip import MIP
from ..typing import BaseEnsemble, MNumber
from .parsers import LevelParser


class BaseOCEAN(
    MIP,
    EnsembleContainer,
    FeatureContainer,
):
    DEFAULT_TOL = 1e-4
    FEATURE_VARS_NAME = "feature_vars"
    FLOW_VAR_FMT = "tree_{t}"

    _feature_vars: FeatureVars
    _flow_vars: dict[int, FlowVars]

    _level_parser: LevelParser
    _levels: dict[str, MNumber]

    def __init__(
        self,
        base: BaseEnsemble,
        encoder: FeatureEncoder,
        weights: npt.ArrayLike,
        *,
        name: str = "OCEAN",
        env: gp.Env | None = None,
        tol: float = DEFAULT_TOL,
    ) -> None:
        MIP.__init__(self, name=name, env=env)
        EnsembleContainer.__init__(
            self,
            ensemble=(base, encoder),
            weights=weights,
        )
        FeatureContainer.__init__(self, encoder=encoder)
        self._parse_levels(encoder=encoder, tol=tol)
        self._add_feature_vars()
        self._add_flow_vars()

    @property
    def levels(self) -> dict[str, MNumber]:
        return self._levels

    def build(self) -> None:
        self._build_feature_vars()
        self._build_flow_vars()
        self._build_feature_constrs()

    def function(self, class_: int) -> gp.LinExpr:
        weights = self._weights
        wf = partial(self.weighted_function, weights=weights)
        return wf(class_=class_)

    def weighted_function(
        self,
        class_: int,
        weights: MNumber,
    ) -> gp.LinExpr:
        return gp.quicksum(
            weights[t] * self._flow_function(t=t, class_=class_)
            for t in range(self.n_estimators)
        )

    def _flow_function(self, t: int, class_: int) -> gp.MLinExpr:
        if self._flow_vars[t].value.ndim == 0:
            n_classes = self.n_classes
            if self.is_binary:
                return (2 * class_ - 1) * self._flow_vars[t].value
            return self._flow_vars[t * n_classes + class_].value
        return self._flow_vars[t].value[class_]

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

    def _parse_levels(
        self,
        encoder: FeatureEncoder,
        tol: float,
    ) -> None:
        self._level_parser = LevelParser(tol=tol)
        self._levels = self._level_parser.parse_levels(
            self.ensemble,
            encoder=encoder,
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
