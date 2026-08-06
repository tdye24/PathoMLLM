#!/usr/bin/env python
"""Eval unit tests (no GPU). Run from repo root: python -m eval.test_eval -v"""

import json
import tempfile
import unittest
from pathlib import Path

from eval.manifest import EVAL_DIR, load_manifest, load_run_config, sample_entries
from eval.metrics.postprocess import postprocess_prediction, postprocess_reference
from eval.metrics.registry import compute_metrics
from eval.run_eval import _inference_argv
from eval.score import join_records, score_dataset
from eval.scorers import bcnb as bcnb_scorer
from eval.scorers import chaoyang as chaoyang_scorer
from eval.scorers import bbox_seg as bbox_seg_scorer

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
        # Letter + class name / CJK prefixes (Unicode \\b used to fail these)
        self.assertEqual(postprocess_prediction("A. Normal", "mcq"), "A")
        self.assertEqual(postprocess_prediction("C. Adenocarcinoma", "mcq"), "C")
        self.assertEqual(postprocess_prediction("选项A. Serrated", "mcq"), "A")
        self.assertEqual(postprocess_prediction("答案是B. Adenoma", "mcq"), "B")
        self.assertEqual(postprocess_prediction("a. normal", "mcq"), "A")
        self.assertEqual(postprocess_prediction("选C", "mcq"), "C")
        self.assertEqual(postprocess_prediction("The correct option is: **D**", "mcq"), "D")

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

    def test_chaoyang_scorer_metrics(self):
        gt = [
            {"id": "1", "ground_truth": "A"},
            {"id": "2", "ground_truth": "B"},
            {"id": "3", "ground_truth": "C"},
            {"id": "4", "ground_truth": "D"},
        ]
        pred = [
            {"id": "1", "status": "success", "prediction": "A"},
            {"id": "2", "status": "success", "prediction": "B"},
            {"id": "3", "status": "success", "prediction": "A"},
            {"id": "4", "status": "success", "prediction": "D"},
        ]
        result = chaoyang_scorer.score(gt, pred)
        self.assertEqual(set(result["scores"]), {"acc", "bacc", "f1"})
        self.assertAlmostEqual(result["scores"]["acc"], 0.75)
        self.assertEqual(result["counts"]["n_correct"], 3)

    def test_manifest_load(self):
        m = load_manifest(EVAL_DIR / "fixtures/tiny_manifest.yaml")
        self.assertEqual(m["datasets"][0]["name"], "tiny_mcq")
        self.assertTrue(Path(m["datasets"][0]["path"]).is_file())

    def test_inference_argv_with_adapter(self):
        run_cfg = {
            "model_id": "/models/base",
            "checkpoint_arg": "--adapter_dir",
            "extra_args": {"attn_implementation": "sdpa"},
        }
        argv = _inference_argv(run_cfg, "/ckpts/checkpoint-100", "out.json", enable_thinking=False)
        self.assertEqual(
            argv,
            [
                "--model_id",
                "/models/base",
                "--output_json",
                "out.json",
                "--adapter_dir",
                "/ckpts/checkpoint-100",
                "--attn_implementation",
                "sdpa",
            ],
        )

    def test_inference_argv_merged_no_adapter(self):
        run_cfg = {
            "model_id": "/models/Qwen3.5-9B-v1",
            "checkpoint_arg": None,
            "extra_args": {"attn_implementation": "sdpa"},
        }
        argv = _inference_argv(run_cfg, "Qwen3.5-9B-v1", "out.json", enable_thinking=False)
        self.assertNotIn("--adapter_dir", argv)
        self.assertEqual(
            argv,
            [
                "--model_id",
                "/models/Qwen3.5-9B-v1",
                "--output_json",
                "out.json",
                "--attn_implementation",
                "sdpa",
            ],
        )

    def test_load_run_config_merged(self):
        cfg = load_run_config(EVAL_DIR / "manifests/run_merged.yaml")
        self.assertIsNone(cfg["checkpoint_arg"])
        self.assertTrue(cfg["model_id"])
        self.assertTrue(cfg["checkpoints"])

    def test_join_records_missing_pred(self):
        gt = [{"id": "x", "ground_truth": "A"}]
        pred = []
        records = join_records(gt, pred)
        self.assertEqual(records[0]["status"], "missing_prediction")

    def test_bbox_seg_scorer(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "square.png"
            Image.new("RGB", (1000, 1000)).save(image_path)
            gt = [
                {"id": "exact", "images": [str(image_path)], "ground_truth": [10, 20, 110, 220]},
                {
                    "id": "partial",
                    "images": [str(image_path)],
                    "ground_truth": {"bbox": [0, 0, 100, 100]},
                },
                {
                    "id": "missing",
                    "images": [str(image_path)],
                    "ground_truth": "<bbox>[0, 0, 10, 10]</bbox>",
                },
            ]
            pred = [
                {"id": "exact", "prediction": "<bbox>[10,20,110,220]</bbox>"},
                {"id": "partial", "prediction": "The lesion is at [50, 0, 150, 100]."},
            ]
            result = bbox_seg_scorer.score(gt, pred)
        self.assertAlmostEqual(result["details"][0]["scores"]["iou"], 1.0)
        self.assertAlmostEqual(result["details"][1]["scores"]["iou"], 1 / 3)
        self.assertAlmostEqual(result["details"][1]["scores"]["dice"], 0.5)
        self.assertEqual(result["details"][2]["scores"]["iou"], 0.0)
        self.assertEqual(result["counts"]["n_missing_or_invalid"], 1)

    def test_bbox_seg_restores_qwen_norm1000_to_image_pixels(self):
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            image_path = Path(tmp) / "rect.png"
            Image.new("RGB", (2000, 1000)).save(image_path)
            result = bbox_seg_scorer.score(
                [{"id": "scaled", "images": [str(image_path)], "ground_truth": [[0, 0, 1000, 1000]]}],
                [{"id": "scaled", "prediction": "[[0, 0, 500, 1000]]"}],
            )

        self.assertEqual(result["details"][0]["prediction_boxes"], [[0.0, 0.0, 1000.0, 1000.0]])
        self.assertAlmostEqual(result["scores"]["iou"], 1.0)
        self.assertAlmostEqual(result["scores"]["dice"], 1.0)

    def test_bbox_seg_empty_ground_truth(self):
        gt = [
            {"id": "true_negative", "ground_truth": "[]"},
            {"id": "false_positive", "ground_truth": []},
        ]
        pred = [
            {"id": "true_negative", "prediction": "[]"},
            {"id": "false_positive", "prediction": "[[0, 0, 100, 100]]"},
        ]
        # Add an image only where a norm1000 prediction needs conversion.
        with tempfile.TemporaryDirectory() as tmp:
            from PIL import Image

            image_path = Path(tmp) / "square.png"
            Image.new("RGB", (1000, 1000)).save(image_path)
            gt[1]["images"] = [str(image_path)]
            result = bbox_seg_scorer.score(gt, pred)

        self.assertEqual(result["details"][0]["scores"]["iou"], 1.0)
        self.assertEqual(result["details"][1]["scores"]["iou"], 0.0)
        self.assertEqual(result["counts"]["n_empty_ground_truth"], 2)

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
