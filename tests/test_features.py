import unittest

from utils import load

from fipe import FeatureEncoder


class TestFeatures(unittest.TestCase):
    dataset = "Breast-Cancer-Wisconsin"
    true_features = set(
        [
            "Clump-T",
            "Uniformity-Size",
            "Uniformity-Shape",
            "Adhesion",
            "Cell-Size",
            "Bare-Nuclei",
            "Bland-Chromatin",
            "Normal-Nucleoli",
            "Mitoses",
        ]
    )

    def test_fit(self):
        data, _, _ = load(self.dataset)
        encoder = FeatureEncoder(data)
        self.assertEqual(encoder.continuous, self.true_features)
        self.assertEqual(encoder.categorical, set())
        self.assertEqual(encoder.binary, set())


if __name__ == "__main__":
    unittest.main()
