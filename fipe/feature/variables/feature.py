from collections.abc import Mapping

import gurobipy as gp
import pandas as pd

from ...mip import MIP, BaseVar
from ...typing import Categories, FeatureType, MNumber, Number, SNumber
from .binary import BinaryVar
from .categorical import CategoricalVar
from .continuous import ContinuousVar

FeatureVar = BinaryVar | ContinuousVar | CategoricalVar
Args = MNumber | Categories


class FeatureVars(BaseVar[SNumber], dict[str, FeatureVar]):
    CLASSES: Mapping[str, type[FeatureVar]] = {
        FeatureType.BIN.value: BinaryVar,
        FeatureType.CAT.value: CategoricalVar,
        FeatureType.CON.value: ContinuousVar,
    }

    def __init__(self, name: str = "") -> None:
        BaseVar.__init__(self, name=name)
        dict.__init__(self)

    def build(self, mip: MIP) -> None:
        for var in self.values():
            var.build(mip)

    def add_var(
        self,
        feature: str,
        vtype: str,
        *,
        levels: MNumber | None = None,
        categories: Categories | None = None,
    ) -> None:
        self[feature] = self._add_var(
            vtype=vtype,
            name=feature,
            levels=levels,
            categories=categories,
        )

    def _apply(self, prop_name: str) -> SNumber:
        series = pd.Series()
        for name, var in self.items():
            value = getattr(var, prop_name)
            if isinstance(value, pd.Series):
                series = pd.concat([series, value])
            else:
                series[name] = value
        return series

    @staticmethod
    def _build_args(**kwargs: Args) -> dict[str, Args]:
        return {
            key: value for key, value in kwargs.items() if value is not None
        }

    @staticmethod
    def _add_var(
        vtype: str,
        *,
        name: str = "",
        levels: MNumber | None = None,
        categories: Categories | None = None,
    ) -> FeatureVar:
        cls = FeatureVars.CLASSES.get(vtype)
        if cls is None:
            msg = f"Invalid variable type: {vtype}"
            raise ValueError(msg)
        args = FeatureVars._build_args(
            levels=levels,
            categories=categories,
        )
        if vtype == FeatureType.CON.value:
            args["floor"] = True

        return cls(name=name, **args)

    @staticmethod
    def fetch(
        feature_var: FeatureVar,
        *,
        level: Number | None = None,
        category: str | None = None,
    ) -> gp.Var | gp.MVar:
        if isinstance(feature_var, BinaryVar):
            return FeatureVars.fetch_bin(feature_var=feature_var)
        if isinstance(feature_var, ContinuousVar):
            return FeatureVars.fetch_con(
                feature_var=feature_var,
                level=level,
            )
        if isinstance(feature_var, CategoricalVar):
            return FeatureVars.fetch_cat(
                feature_var=feature_var,
                category=category,
            )
        msg = f"Unknown feature var type: {type(feature_var)}"
        raise ValueError(msg)

    @staticmethod
    def fetch_bin(feature_var: BinaryVar) -> gp.Var:
        return feature_var.var

    @staticmethod
    def fetch_con(
        feature_var: ContinuousVar,
        level: Number | None,
    ) -> gp.MVar:
        if level is None:
            msg = "Level must be provided for continuous variable."
            raise ValueError(msg)
        j = int(feature_var.levels.searchsorted(level))
        return feature_var[j]

    @staticmethod
    def fetch_cat(
        feature_var: CategoricalVar,
        category: str | None,
    ) -> gp.Var:
        if category is None:
            msg = "Category must be provided for categorical variable."
            raise ValueError(msg)
        return feature_var[category]
