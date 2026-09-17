import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tokenizer import CharacterTokenizer
from transformer import Config, TinyTransformer
from train_transformer import corpus, train, export, load_checkpoint


class TransformerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(7)

    def test_causal_logits_and_attention_mask(self):
        model = TinyTransformer(Config(10)).eval()
        ids = torch.tensor([[1, 2, 3, 4, 5, 6]])
        altered = torch.tensor([[1, 2, 3, 9, 9, 9]])
        with torch.no_grad():
            logits, attention = model(ids, inspect=True)
            changed = model(altered)
        self.assertTrue(torch.allclose(logits[:, :3], changed[:, :3], atol=1e-6, rtol=1e-5))
        self.assertEqual(float(attention.triu(1).sum()), 0.0)
        self.assertTrue(torch.allclose(attention.sum(-1), torch.ones_like(attention.sum(-1)), atol=1e-6))

    def test_context_and_bad_ids(self):
        model = TinyTransformer(Config(5, context_length=4))
        for ids in (torch.tensor([[]], dtype=torch.long), torch.zeros((1, 5), dtype=torch.long),
                    torch.tensor([[5]]), torch.tensor([[-1]]), torch.tensor([[1.0]])):
            with self.assertRaises(ValueError):
                model(ids)

    def test_generation_bounds_unknown_and_seed(self):
        tokenizer = CharacterTokenizer.from_text("abc")
        model = TinyTransformer(Config(3, context_length=4))
        self.assertEqual(model.generate(tokenizer, "abcabc", count=8), model.generate(tokenizer, "abcabc", count=8))
        for prompt, count, temp in (("", 4, 1), ("d", 4, 1), ("a", 257, 1), ("a", 2, 0), ("a", 2, float("nan"))):
            with self.assertRaises(ValueError):
                model.generate(tokenizer, prompt, count, temp)

    def test_data_disjoint_and_train_vocabulary_covers_validation(self):
        training, validation, _ = corpus()
        self.assertFalse(set(training) & set(validation))
        CharacterTokenizer.from_text("".join(training)).encode("".join(validation))

    def test_resume_matches_uninterrupted_training_exactly(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            train(base / "all", steps=6, seed=21)
            train(base / "part", steps=3, seed=21)
            metrics = train(base / "resumed", steps=3, resume=base / "part/last-valid.pt")
            whole, _, _ = load_checkpoint(base / "all/last-valid.pt")
            resumed, _, _ = load_checkpoint(base / "resumed/last-valid.pt")
            self.assertTrue(all(torch.equal(value, resumed.state_dict()[name]) for name, value in whole.state_dict().items()))
            self.assertEqual(metrics["steps"], 6)
            self.assertTrue(metrics["tests"]["checkpoint_reload_exact"])

    def test_short_context_export_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            train(base / "run", steps=1, context_length=4)
            export(base / "run/last-valid.pt", base / "run/metrics.json", base / "public")
            self.assertTrue((base / "public/model.json").exists())
            with self.assertRaises(FileExistsError):
                train(base / "run", steps=1)

    def test_deadline_and_invalid_limits(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            metrics = train(base / "time", steps=500, max_seconds=1e-12)
            self.assertEqual(metrics["session_steps"], 0)
            self.assertEqual(metrics["stop_reason"], "time_limit")
            for steps, seconds in ((0, 1), (501, 1), (1, 0), (1, float("nan"))):
                with self.assertRaises(ValueError):
                    train(base / "invalid", steps=steps, max_seconds=seconds)

    def test_interruption_preserves_loadable_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with patch("torch.optim.AdamW.step", side_effect=KeyboardInterrupt("simulated interrupt")):
                with self.assertRaises(KeyboardInterrupt):
                    train(base / "interrupted", steps=5)
            _, _, state = load_checkpoint(base / "interrupted/last-valid.pt")
            self.assertEqual(state["step"], 0)
            self.assertTrue((base / "interrupted/failure.json").exists())
            result = train(base / "recovered", steps=1, resume=base / "interrupted/last-valid.pt")
            self.assertEqual(result["steps"], 1)


if __name__ == "__main__":
    unittest.main()
