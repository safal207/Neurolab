from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from neurolab.experiment import LabConfig, load_predictor, read_splits, run_experiment
from neurolab.lab_demo import DemoPredictor, result_html
from test_reproducible_lab import fixture


class LabDemoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temp.name)
        cls.path = cls.root / "fixture.csv"
        fixture().to_csv(cls.path, index=False)
        cls.config = LabConfig(dim=8, epochs=2, max_train=0, iterations=(1, 5), batch_size=32)
        cls.bundle = cls.root / "bundle"
        run_experiment(cls.path, cls.bundle, cls.config)
        cls.predictor = DemoPredictor(cls.bundle)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_demo_matches_each_recorded_heldout_prediction(self):
        splits, _ = read_splits(self.path, self.config)
        table = self.predictor.predict(splits["test"]["text"])
        with np.load(self.bundle / "test_predictions.npz", allow_pickle=False) as saved:
            for method in ("Ridge", "Neurolab K=1", "Neurolab K=5"):
                observed = table.loc[table["method"] == method, ["V", "A", "D"]].to_numpy()
                np.testing.assert_allclose(observed, saved[method], atol=1e-6, rtol=1e-6)

    def test_blank_unrepresented_and_long_text_do_not_get_a_prediction(self):
        for text in ("", "   ", "zxqvzxqvzxqv", "calm " * 401):
            with self.subTest(text=text[:20]), self.assertRaises(ValueError):
                self.predictor.predict([text])
        with self.assertRaises(ValueError):
            self.predictor.predict([])

    def test_repeated_calls_and_batch_order_preserve_all_methods(self):
        texts = ["calm hopeful trainword", "hopeful group2 trainword"]
        first = self.predictor.predict(texts)
        self.predictor.predict(["calm group5 trainword"])
        reversed_table = self.predictor.predict(texts[::-1])
        for text in texts:
            np.testing.assert_allclose(
                first.loc[first.text == text, ["V", "A", "D"]],
                reversed_table.loc[reversed_table.text == text, ["V", "A", "D"]], atol=1e-6)

    def test_user_text_is_escaped_and_coverage_is_not_called_confidence(self):
        table = self.predictor.predict(["calm <script>alert(1)</script>"])
        rendered = result_html(table, self.predictor)
        self.assertNotIn("<script>", rendered)
        self.assertIn("&lt;script&gt;", rendered)
        self.assertIn("а не уверенность", rendered)
        self.assertIn("а не ошибка введённого текста", rendered)

    def test_mislabeled_checkpoint_is_rejected(self):
        path = self.bundle / "model_k1.pt"
        original = path.read_bytes()
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            checkpoint["K"] = 5
            torch.save(checkpoint, path)
            with self.assertRaises(ValueError):
                load_predictor(self.bundle, k=1)
        finally:
            path.write_bytes(original)


if __name__ == "__main__":
    unittest.main()
