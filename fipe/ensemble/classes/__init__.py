from . import ab, gb, lgbm, rf, xgb

CLASSES = ab.CLASSES | gb.CLASSES | lgbm.CLASSES | rf.CLASSES | xgb.CLASSES


__all__ = ["CLASSES"]
