#!/usr/bin/env python
"""Eval unit tests (no GPU). Run from repo root: python -m eval.test_eval -v"""

import json
import tempfile
import unittest
from pathlib import Path

from eval.manifest import EVAL_DIR, load_manifest, sample_entries
from eval.metrics.postprocess import postprocess_prediction, postprocess_reference
from eval.metrics.registry import compute_metrics
from eval.score import join_records, score_dataset
from eval.scorers import bcnb as bcnb_scorer

try:
    from sklearn.metrics import f1_score, recall_score
except ImportError:
    f1_score = None
    recall_score = None


class EvalTests(unittest.TestCase):
    _entry = {
        "name": "tiny_mcq",
        "path": str(EVAL_DIR / "fixtures/tiny_mcq.json"),
        "scorer": "bcnb",
    }

    def test_mcq_postprocess(self):
        self.assertEqual(postprocess_prediction("<answer>B</answer>", "mcq"), "B")
        self.assertEqual(postprocess_prediction("...\n\n(C) is correct", "mcq"), "C")
        self.assertEqual(postprocess_reference("c", "mcq"), "C")

    def test_sampling(self):
        m = {"seed": 42, "sample_ratio": 0.5}
        a = sample_entries(self._entry, m)
        b = sample_entries(self._entry, m)
        self.assertEqual([s["id"] for s in a], [s["id"] for s in b])
        self.assertEqual(len(a), 1)

        full = sample_entries({**self._entry, "sample_ratio": 1.0}, {"seed": 42, "sample_ratio": 0.5})
        self.assertEqual(len(full), 3)

    def test_score_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "score.json"
            result = score_dataset(
                str(EVAL_DIR / "fixtures/tiny_manifest.yaml"),
                "tiny_mcq",
                str(EVAL_DIR / "fixtures/tiny_pred.json"),
                gt_samples=json.loads((EVAL_DIR / "fixtures/tiny_mcq.json").read_text()),
                output_json=str(out),
            )
        self.assertAlmostEqual(result["scores"]["acc"], 2 / 3)

    def test_bcnb_scorer_by_task(self):
        gt = [
            {"id": "1", "ground_truth": "A", "task": "T1"},
            {"id": "2", "ground_truth": "B", "task": "T1"},
            {"id": "3", "ground_truth": "A", "task": "T2"},
        ]
        pred = [
            {"id": "1", "status": "success", "prediction": "A"},
            {"id": "2", "status": "success", "prediction": "B"},
            {"id": "3", "status": "success", "prediction": "B"},
        ]
        result = bcnb_scorer.score(gt, pred)
        self.assertAlmostEqual(result["scores"]["acc"], 2 / 3)
        self.assertEqual(result["counts"]["n_correct"], 2)
        self.assertAlmostEqual(result["by_task"]["T1"]["scores"]["acc"], 1.0)
        self.assertAlmostEqual(result["by_task"]["T2"]["scores"]["acc"], 0.0)

    def test_manifest_load(self):
        m = load_manifest(EVAL_DIR / "fixtures/tiny_manifest.yaml")
        self.assertEqual(m["datasets"][0]["name"], "tiny_mcq")
        self.assertTrue(Path(m["datasets"][0]["path"]).is_file())

    def test_join_records_missing_pred(self):
        gt = [{"id": "x", "ground_truth": "A"}]
        pred = []
        records = join_records(gt, pred)
        self.assertEqual(records[0]["status"], "missing_prediction")

    def test_plot_load_series(self):
        from eval.plot_curves import load_series_from_run_dir

        run_dir = EVAL_DIR / "fixtures/plot_runs"
        series = load_series_from_run_dir("fixture", run_dir, "tiny_mcq")
        self.assertEqual(len(series.points), 3)
        self.assertEqual(series.points[0].label, "100")
        self.assertAlmostEqual(series.points[-1].values["acc"], 1.0)

    def test_plot_writes_png(self):
        import matplotlib

        matplotlib.use("Agg")
        from eval.plot_curves import load_series_from_run_dir, plot_dataset

        run_dir = EVAL_DIR / "fixtures/plot_runs"
        series = load_series_from_run_dir("fixture", run_dir, "tiny_mcq")
        out = EVAL_DIR / "fixtures" / "_test_plot.png"
        try:
            plot_dataset("tiny_mcq", [series], out, metrics=("acc",))
            self.assertTrue(out.is_file())
            self.assertGreater(out.stat().st_size, 0)
        finally:
            if out.is_file():
                out.unlink()


if __name__ == "__main__":
    unittest.main()
