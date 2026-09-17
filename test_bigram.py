import json
import tempfile
import unittest
from pathlib import Path

import torch

from bigram import BigramModel, check_split, generate, make_pairs, next_probabilities, run_lesson
from tokenizer import CharacterTokenizer


class BigramTests(unittest.TestCase):
    def test_scores_are_trainable_and_have_expected_shape(self):
        model = BigramModel(3)
        self.assertEqual(tuple(model(torch.tensor([0, 1])).shape), (2, 3))
        self.assertTrue(model.scores.weight.requires_grad)

    def test_probability_distribution(self):
        tokenizer = CharacterTokenizer.from_text("cat")
        probabilities = next_probabilities(BigramModel(3), tokenizer, "c")
        self.assertAlmostEqual(sum(probabilities.values()), 1, places=6)
        self.assertTrue(all(0 <= p <= 1 for p in probabilities.values()))

    def test_no_cross_document_pairs(self):
        tokenizer = CharacterTokenizer.from_text("catdog")
        inputs, targets = make_pairs(tokenizer, ["cat", "dog"])
        self.assertEqual(tokenizer.decode(inputs.tolist()), "cado")
        self.assertEqual(tokenizer.decode(targets.tolist()), "atog")

    def test_duplicate_split_rejected(self):
        with self.assertRaises(ValueError):
            check_split(["cat dog"], [" CAT   dog "])

    def test_generation_is_bounded_and_repeatable(self):
        tokenizer = CharacterTokenizer.from_text("cat")
        model = BigramModel(3)
        a = generate(model, tokenizer, "c", count=5)
        self.assertEqual(len(a), 6)
        self.assertEqual(a, generate(model, tokenizer, "c", count=5))
        for prefix, count in (("", 5), ("c", -1), ("c", 257)):
            with self.assertRaises(ValueError):
                generate(model, tokenizer, prefix, count)

    def test_local_run_learns_and_reload_matches(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "run"
            result = run_lesson(path, steps=100)
            self.assertTrue(result["validation_loss_improved"])
            self.assertTrue(result["checkpoint_reload_exact"])
            self.assertGreater(result["after"]["after_c"]["a"], result["before"]["after_c"]["a"])
            self.assertEqual(json.loads((path / "metrics.json").read_text())["completed_steps"], 100)
            with self.assertRaises(FileExistsError):
                run_lesson(path, steps=1)

    def test_invalid_training_limits_rejected_before_creating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "invalid"
            for steps, seconds in ((0, 10), (501, 10), (5, 0), (5, float("nan")), (5, 121)):
                with self.assertRaises(ValueError):
                    run_lesson(path, steps, seconds)
                self.assertFalse(path.exists())

    def test_training_deadline_stops_without_overclaiming(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = run_lesson(Path(tmp) / "run", steps=500, max_seconds=1e-12)
            self.assertEqual(result["completed_steps"], 0)
            self.assertEqual(result["stop_reason"], "time_limit")
            self.assertFalse(result["validation_loss_improved"])


if __name__ == "__main__":
    unittest.main()
