from collections.abc import Mapping
from typing import override

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

    @override
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

    @override
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
        match vtype:
            case FeatureType.BIN.value:
                return BinaryVar(name=name)
            case FeatureType.CAT.value:
                if categories is None:
                    msg = (
                        "Categories must be provided for categorical variable."
                    )
                    raise ValueError(msg)
                return CategoricalVar(name=name, categories=categories)
            case FeatureType.CON.value:
                if levels is None:
                    msg = "Levels must be provided for continuous variable."
                    raise ValueError(msg)
                return ContinuousVar(name=name, levels=levels, floor=True)
            case _:
                msg = f"Invalid variable type: {vtype}"
                raise ValueError(msg)

    @staticmethod
    def fetch(
        var: FeatureVar,
        *,
        level: Number | None = None,
        category: str | None = None,
    ) -> gp.Var | gp.MVar:
        if isinstance(var, BinaryVar):
            return FeatureVars._fetch_bin(var=var)
        if isinstance(var, ContinuousVar):
            return FeatureVars._fetch_con(var=var, level=level)
        if isinstance(var, CategoricalVar):
            return FeatureVars._fetch_cat(var=var, category=category)
        msg = f"Unknown feature var type: {type(var)}"
        raise ValueError(msg)

    @staticmethod
    def _fetch_bin(var: BinaryVar) -> gp.Var:
        return var.var

    @staticmethod
    def _fetch_con(var: ContinuousVar, level: Number | None) -> gp.MVar:
        if level is None:
            msg = "Level must be provided for continuous variable."
            raise ValueError(msg)
        j = int(var.levels.searchsorted(level))
        return var[j]

    @staticmethod
    def _fetch_cat(var: CategoricalVar, category: str | None) -> gp.Var:
        if category is None:
            msg = "Category must be provided for categorical variable."
            raise ValueError(msg)
        return var[category]
