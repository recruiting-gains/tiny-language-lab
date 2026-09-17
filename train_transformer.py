"""Bounded, resumable CPU experiment on original synthetic nature sentences."""
import argparse
import hashlib
import itertools
import json
import math
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from tokenizer import CharacterTokenizer
from transformer import Config, TinyTransformer


def corpus():
    adjectives = ["small", "quiet", "young", "bright"]
    animals = ["fox", "bird", "cat", "deer", "owl", "dog"]
    verbs = ["waits", "rests", "walks", "sleeps"]
    places = ["river", "garden", "forest", "hill"]
    documents = [f"the {a} {n} {v} near the {p}.\n"
                 for a, n, v, p in itertools.product(adjectives, animals, verbs, places)]
    documents += [f"a {n} sees the {p}.\n" for n, p in itertools.product(animals, places)]
    train, validation = [], []
    for document in sorted(documents):
        bucket = int(hashlib.sha256(document.encode()).hexdigest()[:8], 16) % 5
        (validation if bucket == 0 else train).append(document)
    assert not set(train) & set(validation)
    digest = hashlib.sha256(json.dumps([train, validation]).encode()).hexdigest()
    return train, validation, digest


def tensors(tokenizer, documents):
    # Newlines mark document boundaries. No window can cross between data splits.
    return torch.tensor(tokenizer.encode("".join(documents)), dtype=torch.long)


def batch(data, context, size=16, generator=None):
    starts = torch.randint(len(data) - context, (size,), generator=generator)
    x = torch.stack([data[i:i + context] for i in starts])
    y = torch.stack([data[i + 1:i + context + 1] for i in starts])
    return x, y


@torch.no_grad()
def evaluate(model, train, validation):
    model.eval()
    # Fixed evaluation windows and private RNG; evaluation never changes training RNG.
    result = {}
    for name, data in (("train_loss", train), ("validation_loss", validation)):
        rng = torch.Generator().manual_seed(123)
        losses = []
        for _ in range(5):
            x, y = batch(data, model.config.context_length, generator=rng)
            losses.append(float(F.cross_entropy(model(x).reshape(-1, model.config.vocab_size), y.reshape(-1))))
        result[name] = sum(losses) / len(losses)
    return result


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def save(path, model, optimizer, tokenizer, step, seed, digest, history, before):
    state = {"format_version": 1, "config": model.config_dict(),
             "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "vocabulary": list(tokenizer.characters), "step": step, "seed": seed,
             "corpus_sha256": digest, "rng_state": torch.get_rng_state(),
             "history": history, "before": before}
    temporary = path.with_suffix(".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("format_version") != 1:
        raise ValueError("Unsupported local checkpoint format.")
    model = TinyTransformer(Config(**state["config"]))
    model.load_state_dict(state["model"])
    tokenizer = CharacterTokenizer(tuple(state["vocabulary"]))
    if len(tokenizer.characters) != model.config.vocab_size:
        raise ValueError("Checkpoint vocabulary does not match model.")
    return model, tokenizer, state


def train(output, steps=500, max_seconds=120, seed=17, resume=None, context_length=48):
    if type(steps) is not int or not 1 <= steps <= 500:
        raise ValueError("Each session is limited to 1–500 updates.")
    if not math.isfinite(max_seconds) or not 0 < max_seconds <= 120:
        raise ValueError("Training loop is limited to 120 seconds per session.")
    output = Path(output)
    if output.exists():
        raise FileExistsError("Use a new output folder, including when resuming.")
    torch.set_num_threads(2)
    train_docs, val_docs, digest = corpus()
    torch.manual_seed(seed)
    if resume:
        model, tokenizer, state = load_checkpoint(resume)
        if state["corpus_sha256"] != digest:
            raise ValueError("Resume corpus differs from the saved experiment.")
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        optimizer.load_state_dict(state["optimizer"])
        seed = state["seed"]
        completed, history, before = state["step"], state["history"], state["before"]
        torch.set_rng_state(state["rng_state"])
    else:
        tokenizer = CharacterTokenizer.from_text("".join(train_docs))
        model = TinyTransformer(Config(len(tokenizer.characters), context_length=context_length))
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        completed, history, before = 0, [], None
    training, validation = tensors(tokenizer, train_docs), tensors(tokenizer, val_docs)
    if before is None:
        before = {**evaluate(model, training, validation), "sample": model.generate(tokenizer)}
        history.append({"step": 0, **{k: before[k] for k in ("train_loss", "validation_loss")}})
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = output / "last-valid.pt"
    save(checkpoint, model, optimizer, tokenizer, completed, seed, digest, history, before)
    started = time.monotonic()
    initial_step = completed
    stop_reason = "step_limit"
    try:
        for _ in range(steps):
            if time.monotonic() - started >= max_seconds:
                stop_reason = "time_limit"
                break
            model.train()
            x, y = batch(training, model.config.context_length)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, model.config.vocab_size), y.reshape(-1))
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite loss; previous checkpoint remains valid.")
            loss.backward()
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite gradients; previous checkpoint remains valid.")
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise FloatingPointError("Nonfinite weights; previous checkpoint remains valid.")
            completed += 1
            if completed % 50 == 0:
                measured = evaluate(model, training, validation)
                history.append({"step": completed, **measured})
                save(checkpoint, model, optimizer, tokenizer, completed, seed, digest, history, before)
                print(json.dumps(history[-1]), flush=True)
    except BaseException as exc:
        atomic_json(output / "failure.json", {"type": type(exc).__name__, "message": str(exc),
                                               "last_completed_step": completed,
                                               "recovery": "Resume last-valid.pt into a new output directory."})
        raise
    elapsed = time.monotonic() - started
    after = {**evaluate(model, training, validation), "sample": model.generate(tokenizer)}
    if history[-1]["step"] != completed:
        history.append({"step": completed, **{k: after[k] for k in ("train_loss", "validation_loss")}})
    save(checkpoint, model, optimizer, tokenizer, completed, seed, digest, history, before)
    restored, _, _ = load_checkpoint(checkpoint)
    model.eval()
    restored.eval()
    test_ids = torch.tensor([tokenizer.encode("the quiet fox")[-model.config.context_length:]])
    with torch.no_grad():
        reload_exact = torch.equal(model(test_ids), restored(test_ids))
        modified = test_ids.clone()
        prefix_length = min(5, max(1, test_ids.shape[1] // 2))
        modified[:, prefix_length:] = 0
        causal = torch.allclose(model(test_ids)[:, :prefix_length], model(modified)[:, :prefix_length], atol=1e-6, rtol=1e-5)
    metrics = {"parameters": sum(p.numel() for p in model.parameters()), "config": model.config_dict(),
               "seed": seed, "steps": completed, "session_steps": completed - initial_step,
               "device": "cpu", "training_seconds": elapsed, "stop_reason": stop_reason,
               "before": before, "after": after, "history": history,
               "tests": {"checkpoint_reload_exact": reload_exact, "causal_prefix_unchanged": causal,
                         "held_out_loss_improved": after["validation_loss"] < before["validation_loss"]},
               "dataset": {"description": "Original synthetic nature sentences, split before window sampling.",
                           "train_documents": len(train_docs), "validation_documents": len(val_docs),
                           "sha256": digest,
                           "limitations": "Shared grammar and vocabulary. Not evidence of broad language understanding."}}
    atomic_json(output / "metrics.json", metrics)
    return metrics


def export(checkpoint, metrics_path, destination):
    model, tokenizer, state = load_checkpoint(checkpoint)
    model.eval()
    parity = []
    with torch.no_grad():
        for prompt in ["the ", "the quiet fox", "a cat sees the river."]:
            logits = model(torch.tensor([tokenizer.encode(prompt)[-model.config.context_length:]]))[0, -1].tolist()
            parity.append({"prompt": prompt, "logits": logits})
    artifact = {"version": 1, "config": model.config_dict(), "vocabulary": list(tokenizer.characters),
                "weights": {name: value.tolist() for name, value in model.state_dict().items()}, "parity": parity}
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    atomic_json(destination / "model.json", artifact)
    atomic_json(destination / "metrics.json", json.loads(Path(metrics_path).read_text()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--max-seconds", type=float, default=120)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--export", type=Path)
    args = parser.parse_args()
    result = train(args.output, args.steps, args.max_seconds, resume=args.resume)
    if args.export:
        export(args.output / "last-valid.pt", args.output / "metrics.json", args.export)
    print(json.dumps(result, indent=2))
