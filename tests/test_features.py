import unittest
from typing import ClassVar

from utils import DATASETS_PATH, load

from fipe import FeatureEncoder


class TestFeatures(unittest.TestCase):
    dataset: ClassVar[str] = "Breast-Cancer-Wisconsin"
    true_features: ClassVar[set] = {
        "Clump-T",
        "Uniformity-Size",
        "Uniformity-Shape",
        "Adhesion",
        "Cell-Size",
        "Bare-Nuclei",
        "Bland-Chromatin",
        "Normal-Nucleoli",
        "Mitoses",
    }

    def test_fit(self) -> None:
        data, _, _ = load(DATASETS_PATH / self.dataset)
        encoder = FeatureEncoder(data)
        assert encoder.continuous == self.true_features
        assert encoder.categorical == set()
        assert encoder.binary == set()


if __name__ == "__main__":
    unittest.main()
