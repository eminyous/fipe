from ..ensemble import Ensemble, EnsembleContainer
from ..feature import FeatureContainer, FeatureEncoder, FeatureVars
from ..flow.variables import FlowVars
from ..mip import MIP
from ..parsers import LevelParser


class BaseOCEAN(MIP, EnsembleContainer, FeatureContainer, LevelParser):
    _feature_vars: FeatureVars
    _flow_vars: dict[int, FlowVars]

    def __init__(
        self, encoder: FeatureEncoder, ensemble: Ensemble, weights, **kwargs
    ):
        MIP.__init__(
            self, name=kwargs.get("name", ""), env=kwargs.get("env", None)
        )
        EnsembleContainer.__init__(self, ensemble=ensemble, weights=weights)
        FeatureContainer.__init__(self, encoder=encoder)
        LevelParser.__init__(self, **kwargs)
        self.parse_levels([ensemble], encoder)
        self._feature_vars = FeatureVars()
        self._flow_vars = {}

    def build(self):
        self._build_features()
        self._build_ensemble()

    def _build_ensemble(self):
        for t, tree in enumerate(self._ensemble):
            self._flow_vars[t] = FlowVars(tree=tree, name=f"tree_{t}")
            self._flow_vars[t].build(mip=self)
            self._flow_vars[t].add_feature_constrs(
                mip=self, feature_vars=self._feature_vars
            )

    def _build_features(self):
        var = self._feature_vars
        for f in self.binary:
            var.add_binary(f)
        for f in self.continuous:
            levels = self.levels[f]
            var.add_continuous(f, levels)
        for f in self.categorical:
            categories = self.categories[f]
            var.add_categorical(f, categories)
        var.build(mip=self)
