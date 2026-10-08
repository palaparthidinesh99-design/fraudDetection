import unittest
import numpy as np
from src.samplers import get_sampler, SAMPLER_REGISTRY


class TestSamplers(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        self.X = np.random.randn(200, 5)
        self.y = np.zeros(200, dtype=int)
        self.y[:20] = 1  # 20 minority instances, 180 majority instances
        self.amounts = np.random.uniform(5.0, 500.0, size=200)

    def test_all_10_samplers(self):
        for name in SAMPLER_REGISTRY:
            with self.subTest(sampler=name):
                sampler = get_sampler(name, random_state=42)
                X_res, y_res = sampler.fit_resample(self.X, self.y, amounts=self.amounts)
                self.assertGreater(len(X_res), len(self.X))
                self.assertEqual(len(X_res), len(y_res))
                # Minority count should be increased
                self.assertGreater(np.sum(y_res == 1), 20)


if __name__ == "__main__":
    unittest.main()
