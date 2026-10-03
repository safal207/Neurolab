import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from neurolab.data import EmoBankDataset, create_dataloaders
from neurolab.experiment import LabConfig, fit_features, read_splits, run_experiment, predict_texts
from neurolab.models import TinyRecursiveModelTRMv6
from neurolab.training.trainer import LIMINALTrainer


def fixture():
    rows = []
    for split, count in (("train", 64), ("dev", 16), ("test", 16)):
        for i in range(count):
            rows.append({"id": f"{split}-{i}", "split": split,
                         "text": f"calm hopeful group{i % 12} item{i} {split}word",
                         "V": 2.5 + (i % 5) / 5, "A": 2.6 + (i % 4) / 5,
                         "D": 2.7 + (i % 3) / 5})
    return pd.DataFrame(rows)


class ReproducibleLabTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.path = self.root / "fixture.csv"
        fixture().to_csv(self.path, index=False)

    def tearDown(self):
        self.temp.cleanup()

    def test_labels_use_model_scale_and_official_splits(self):
        data = fixture()
        data.loc[0, ["V", "A", "D"]] = [1, 3, 5]
        data.to_csv(self.path, index=False)
        train = EmoBankDataset(self.path, split="train")
        self.assertEqual(len(train), 64)
        torch.testing.assert_close(train[0][1], torch.tensor([-1., 0., 1.]))
        loaders = create_dataloaders(self.path)
        self.assertTrue((loaders[1].dataset.df["split"] == "dev").all())

    def test_missing_labels_are_rejected(self):
        fixture().drop(columns="V").to_csv(self.path, index=False)
        with self.assertRaises(ValueError):
            EmoBankDataset(self.path)

    def test_literal_na_text_is_preserved(self):
        data = fixture()
        data.loc[0, "text"] = "NA"
        data.to_csv(self.path, index=False)
        splits, _ = read_splits(self.path, LabConfig(max_train=0))
        self.assertEqual(splits["train"].iloc[0]["text"], "NA")

    def test_independent_predictions_ignore_batch_order_and_previous_calls(self):
        torch.manual_seed(42)
        model = TinyRecursiveModelTRMv6(dim=8).eval()
        x = torch.randn(5, 8)
        with torch.no_grad():
            first = model(x, torch.zeros_like(x), K=3)[2]
            model(torch.randn_like(x), torch.zeros_like(x), K=3)
            reordered = model(x.flip(0), torch.zeros_like(x), K=3)[2].flip(0)
            singleton = torch.cat([model(row[None], torch.zeros(1, 8), K=3)[2] for row in x])
        torch.testing.assert_close(first, reordered, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(first, singleton, atol=1e-6, rtol=1e-6)
        self.assertFalse(model.soul.mem)

    def test_stream_memory_requires_singleton_and_can_reset(self):
        model = TinyRecursiveModelTRMv6(dim=8, use_memory=True)
        with self.assertRaises(ValueError):
            model(torch.zeros(2, 8), torch.zeros(2, 8))
        model(torch.zeros(1, 8), torch.zeros(1, 8), K=1)
        self.assertTrue(model.soul.mem)
        model.reset_memory()
        self.assertFalse(model.soul.mem)

    def test_legacy_trainer_never_passes_target_as_affect(self):
        class Spy(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(3))
                self.affects = []
            def forward(self, x, initial, a=None):
                self.affects.append(a)
                return initial, [0.0], self.weight.expand(len(x), 3)
        spy = Spy()
        trainer = LIMINALTrainer(spy, torch.optim.SGD(spy.parameters(), lr=0.01),
                                 device="cpu", log_dir=self.root / "logs")
        batches = [(["one", "two"], torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]))]
        embedder = lambda texts: torch.ones(len(texts), 8)
        trainer.train_epoch(batches, embedder)
        trainer.validate(batches, embedder)
        self.assertEqual(spy.affects, [None, None])

    def test_vocabulary_is_fitted_on_train_only(self):
        config = LabConfig(dim=8, epochs=1, max_train=0)
        splits, _ = read_splits(self.path, config)
        features, _ = fit_features(splits, config)
        vocabulary = features[0].vocabulary_
        self.assertIn("trainword", vocabulary)
        self.assertNotIn("devword", vocabulary)
        self.assertNotIn("testword", vocabulary)

    def test_reloaded_predictions_and_selection_do_not_depend_on_test_labels(self):
        config = LabConfig(dim=8, epochs=2, max_train=0, iterations=(1,), batch_size=32)
        first_dir = self.root / "first"
        second_dir = self.root / "second"
        first = run_experiment(self.path, first_dir, config)
        before = predict_texts(first_dir, ["calm hopeful trainword"])
        data = fixture()
        data.loc[data["split"] == "test", ["V", "A", "D"]] += 0.3
        data.to_csv(self.path, index=False)
        second = run_experiment(self.path, second_dir, config)
        after = predict_texts(second_dir, ["calm hopeful trainword"])
        np.testing.assert_allclose(before[["V", "A", "D"]], after[["V", "A", "D"]], atol=1e-6)
        self.assertEqual(first["selection_on_dev_only"], second["selection_on_dev_only"])
        self.assertEqual(first["selected_K"], second["selected_K"])
        self.assertTrue((before[["V", "A", "D"]] >= 1).all().all())
        self.assertTrue((before[["V", "A", "D"]] <= 5).all().all())


if __name__ == "__main__":
    unittest.main()
