"""Lesson 2: train a tiny next-character probability table from random weights.

This is a bigram baseline, NOT a transformer or a conversational LLM. It sees
only the current character, not the whole prefix. All example text is original
synthetic data. Validation tests only these toy patterns, not general language.
"""

import argparse
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from tokenizer import CharacterTokenizer

TRAIN_TEXTS = ("cat dog cat", "dog cat dog", "cat cat dog", "dog dog cat")
VALIDATION_TEXTS = ("cat cat dog dog", "dog cat cat dog")


class BigramModel(nn.Module):
    def __init__(self, vocabulary_size: int):
        super().__init__()
        if vocabulary_size < 2:
            raise ValueError("Need at least two possible characters.")
        # One row per current character, one score per possible next character.
        # PyTorch initializes these learnable scores randomly, not from a model.
        self.scores = nn.Embedding(vocabulary_size, vocabulary_size)

    def forward(self, token_ids):
        return self.scores(token_ids)


def make_pairs(tokenizer, texts):
    if not texts:
        raise ValueError("Training examples cannot be empty.")
    inputs, targets = [], []
    for text in texts:
        x, y = tokenizer.training_pair(text)
        inputs.extend(x)
        targets.extend(y)
    # Do not create transitions across separate document boundaries.
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def check_split(train_texts, validation_texts):
    normalized = lambda texts: {" ".join(t.casefold().split()) for t in texts}
    if normalized(train_texts) & normalized(validation_texts):
        raise ValueError("Identical normalized examples occur in both data splits.")


@torch.no_grad()
def evaluate(model, pairs):
    model.eval()
    return float(F.cross_entropy(model(pairs[0]), pairs[1]))


@torch.no_grad()
def next_probabilities(model, tokenizer, character):
    if len(character) != 1:
        raise ValueError("This baseline takes exactly one current character.")
    scores = model(torch.tensor(tokenizer.encode(character), dtype=torch.long))[0]
    probabilities = scores.softmax(dim=-1)
    return {c: float(probabilities[i]) for i, c in enumerate(tokenizer.characters)}


@torch.no_grad()
def generate(model, tokenizer, prefix="c", count=40, seed=11):
    if not prefix:
        raise ValueError("Provide a nonempty prefix.")
    if type(count) is not int or not 0 <= count <= 256:
        raise ValueError("Output length must be an integer from 0 through 256.")
    ids = tokenizer.encode(prefix)
    generator = torch.Generator().manual_seed(seed)
    model.eval()
    for _ in range(count):
        probabilities = model(torch.tensor([ids[-1]])).softmax(dim=-1)[0]
        ids.append(int(torch.multinomial(probabilities, 1, generator=generator)))
    return tokenizer.decode(ids)


def save_checkpoint(path, model, optimizer, tokenizer, step):
    # Only load locally created checkpoints. No downloaded pickle/model files.
    temporary = path.with_suffix(".tmp")
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "characters": list(tokenizer.characters), "step": step,
                "rng": torch.get_rng_state(), "version": 1}, temporary)
    temporary.replace(path)


def run_lesson(output_dir, steps=200, max_seconds=15, seed=7):
    if type(steps) is not int or not 1 <= steps <= 500:
        raise ValueError("Training is limited to 1–500 steps.")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 120:
        raise ValueError("Training time limit must be above 0 and at most 120 seconds.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)  # Never silently overwrite a run.
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    check_split(TRAIN_TEXTS, VALIDATION_TEXTS)
    tokenizer = CharacterTokenizer.from_text("".join(TRAIN_TEXTS))
    train_pairs = make_pairs(tokenizer, TRAIN_TEXTS)
    validation_pairs = make_pairs(tokenizer, VALIDATION_TEXTS)
    model = BigramModel(len(tokenizer.characters))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.07, weight_decay=0.01)
    checkpoint = output_dir / "last-valid.pt"
    save_checkpoint(checkpoint, model, optimizer, tokenizer, 0)
    before = {"train_loss": evaluate(model, train_pairs),
              "validation_loss": evaluate(model, validation_pairs),
              "after_c": next_probabilities(model, tokenizer, "c"),
              "sample": generate(model, tokenizer)}
    history = [{"step": 0, "train_loss": before["train_loss"],
                "validation_loss": before["validation_loss"]}]
    completed = 0
    stop_reason = "step_limit"
    started = time.monotonic()
    try:
        for step in range(1, steps + 1):
            if time.monotonic() - started >= max_seconds:
                stop_reason = "time_limit"
                break
            model.train()
            optimizer.zero_grad()
            # Compare the probability distribution with the actual next letter.
            loss = F.cross_entropy(model(train_pairs[0]), train_pairs[1])
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; last valid checkpoint preserved.")
            loss.backward()  # Calculate how changing each score changes the error.
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradient; last valid checkpoint preserved.")
            optimizer.step()  # Adjust learnable scores; validation is never used here.
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Invalid weights; last valid checkpoint preserved.")
            completed = step
            if step % 25 == 0 or step == steps:
                save_checkpoint(checkpoint, model, optimizer, tokenizer, completed)
                history.append({"step": completed, "train_loss": evaluate(model, train_pairs),
                                "validation_loss": evaluate(model, validation_pairs)})
    except BaseException as exc:
        (output_dir / "failure.json").write_text(json.dumps({
            "error": type(exc).__name__, "completed_steps": completed,
            "message": str(exc), "checkpoint": "last-valid.pt"}, indent=2))
        raise
    elapsed = time.monotonic() - started
    save_checkpoint(checkpoint, model, optimizer, tokenizer, completed)
    after = {"train_loss": evaluate(model, train_pairs),
             "validation_loss": evaluate(model, validation_pairs),
             "after_c": next_probabilities(model, tokenizer, "c"),
             "sample": generate(model, tokenizer)}
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = BigramModel(len(saved["characters"]))
    restored.load_state_dict(saved["model"])
    model.eval()
    restored.eval()
    with torch.no_grad():
        reload_matches = torch.equal(model(train_pairs[0]), restored(train_pairs[0]))
    result = {"stage": "bigram baseline, not a transformer", "device": "cpu",
              "torch_version": str(torch.__version__), "seed": seed,
              "parameters": sum(p.numel() for p in model.parameters()),
              "vocabulary": list(tokenizer.characters), "completed_steps": completed,
              "elapsed_training_seconds": elapsed, "stop_reason": stop_reason,
              "data": {"source": "original synthetic cat/dog strings",
                       "training": list(TRAIN_TEXTS), "validation": list(VALIDATION_TEXTS),
                       "limits": "Distinct strings, shared toy patterns; not broad language generalization."},
              "before": before, "after": after, "history": history,
              "checkpoint_reload_exact": reload_matches,
              "validation_loss_improved": after["validation_loss"] < before["validation_loss"]}
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--max-seconds", type=float, default=15)
    parser.add_argument("--output", type=Path,
                        default=Path("runs") / datetime.now(timezone.utc).strftime("bigram-%Y%m%dT%H%M%S%fZ"))
    args = parser.parse_args()
    result = run_lesson(args.output, args.steps, args.max_seconds)
    print(json.dumps(result, indent=2))
    print("Saved local results:", args.output.resolve())
