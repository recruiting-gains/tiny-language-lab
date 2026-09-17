# Tiny Language Lab — model card

## Intended use

An educational demonstration of character tokenization, next-token prediction, causal attention, gradient-based training, sampling, and checkpoint recovery. The included model generates short synthetic nature sentences and can run entirely in a browser worker after its weights load.

It is unsuitable for answering questions, giving advice, following general instructions, or making decisions about people. No broad language, factuality, reasoning, fairness, or safety benchmark has been run.

## Model

| Property | Exported model |
| --- | --- |
| Architecture | Causal decoder transformer, trained from random initialization |
| Trainable parameters | 28,480 |
| Vocabulary | 27 individual characters |
| Context | 48 characters |
| Embedding width | 32 |
| Decoder blocks | 2 |
| Attention heads per block | 4, each with width 8 |
| Feed-forward width | 128, with ReLU |
| Normalization | Pre-layer normalization, epsilon 0.00001 |
| Positional representation | Learned embeddings |
| Training framework | CPU PyTorch 2.14.0 |
| Browser inference | Original dependency-free JavaScript, Float64 arithmetic |

Each block applies normalized causal self-attention with a residual addition, then a normalized feed-forward network with another residual addition. A final normalization and linear projection produce next-character logits. Attention cannot access future positions. Browser generation keeps the latest 48 characters and restarts positional indices at zero when cropping.

## Training data and procedure

`train_transformer.py` constructs 408 original synthetic documents from a small grammar: combinations of four adjectives, six animals, four verbs, and four places, plus short animal/place sentences. Example forms are `the quiet fox rests near the river.` and `a cat sees the garden.`, each followed by a newline.

A deterministic SHA-256 split assigns whole documents to **316 training documents** and **92 validation documents** before next-character windows are sampled. Exact documents do not overlap, and windows never cross between the two splits. Both splits share their grammar and vocabulary; validation therefore measures performance on this narrow distribution. There is no independent test set.

The corpus fingerprint stored with the model is:

```text
abe7f4f04ab6b0d4c198b67f4f738fa2798a1eac4e676a1464f818ffd28f11e3
```

The included run uses seed 17, AdamW with learning rate 0.003, batches of 16 sequences, gradient clipping at norm 1.0, and 500 CPU updates. Weights begin randomly initialized. No pretrained model, downloaded text corpus, private messages, or user prompts are training inputs.

## Evaluation

Metrics are next-character cross-entropy in natural-log units; lower values indicate better prediction of this corpus. They are not accuracy percentages. Each reported loss averages five fixed batches of sampled windows, using an evaluation random generator separate from training.

| Measurement | Before training | After 500 updates |
| --- | ---: | ---: |
| Training loss | 3.28889270 | 0.22508767 |
| Validation loss | 3.28959427 | 0.23227010 |

The recorded training loop took **1.843343 seconds** on the machine used for this run. That interval includes periodic evaluations and checkpoint writes inside the loop and excludes setup, initial/final evaluation, sample generation, final reload checks, and export. It is not an end-to-end runtime or a hardware-independent performance claim.

One recorded after-training continuation at temperature 0.8, seed 31, and prompt `the ` was:

```text
the hill.
the small dog rests near the hill.
the quiet fox sleeps near the gardi the
```

The incomplete and malformed ending illustrates the model's limitations. See [the full recorded metrics](public/metrics.json) for the before sample and loss history.

## Verification and reproducibility

The local verification passed 32 Python tests and 8 Node tests. These include causal prefix invariance, zero attention to future positions, normalized probabilities, input bounds, checkpoint reload, simulated interruption recovery, exact split-session resume, worker behavior, and the optional HTTP API.

The exported model includes three independently computed PyTorch logit examples. JavaScript inference matched them with a maximum absolute difference of **0.0000017596**, below the **0.0001** acceptance tolerance. This verifies numerical agreement for those examples, not general model quality. Separate tests cover context cropping and the future-position mask.

Exact resume was verified by comparing uninterrupted six-update training with three updates plus a three-update resume. Reproducibility across different hardware, dependency versions, or Python/browser sampling algorithms is not asserted. The browser and Python use different random sampling algorithms; a matching seed need not produce matching text across them.

The browser model and metrics are included as JSON. Full training checkpoints, optimizer states, and local run folders are excluded from version control and can be regenerated using the [README instructions](README.md). HTTP tests skip without their locally trained checkpoint; the browser parity test skips if `public/model.json` is absent.

## Limits and data handling

The model has a tiny vocabulary and a 48-character context. It cannot generalize from its toy grammar to unrestricted language, reliably finish sentences, or understand the truth of generated text. Lower validation loss here provides no evidence of general intelligence.

The browser worker rejects empty prompts, unsupported characters, prompts longer than 512 characters, outputs requested beyond 96 new characters, and temperatures outside 0.2–1.5. The interface currently requests 48 new characters. Requests run one at a time; Stop terminates the worker.

Browser inference does not send prompt text to a model service. The optional loopback Python API receives text only when called explicitly and does not log request prompts. These implementation properties do not control extensions, browser diagnostics, or infrastructure outside this repository.

The attention display shows mixing weights from head zero of the final block. It is a view of one internal calculation, not an explanation of intent, consciousness, or human-like reasoning.

## Attribution

The educational reference is Elliot Arledge's [freeCodeCamp course on building a language model from scratch](https://www.freecodecamp.org/news/how-to-build-a-large-language-model-from-scratch-using-python/). This project uses an original compact architecture implementation, synthetic corpus, training workflow, and browser interface. It imports no pretrained weights or course repository and claims no endorsement by the course creators.
