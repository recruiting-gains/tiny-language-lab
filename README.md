# Tiny Language Lab

[Open the live lab](https://tiny-language-lab.pages.dev/) · [GitHub source](https://github.com/recruiting-gains/tiny-language-lab)

A working, inspectable language model trained from random weights on original synthetic sentences. Its 28,480-parameter causal transformer runs directly in a browser, where you can generate text, inspect character tokens, compare next-character probabilities, and view attention weights.

The model learns a narrow grammar about animals and places. Its imperfect continuations make the mechanics visible; it has no general conversational ability. See the [model card](MODEL_CARD.md) for the architecture, measurements, and limits.

## Run the browser demo

From the project directory, with Python 3 installed:

```sh
python3 -m http.server --directory public 8847 --bind 127.0.0.1
```

Open [localhost:8847](http://localhost:8847). A fresh checkout includes the exported weights and measured results in `public/`, so no package installation or retraining is needed for this route. Use an HTTP server instead of opening `index.html` as a file: module workers need a served origin.

The browser downloads the model once, then performs prediction and sampling in a JavaScript worker on your device. Generation prompts are not sent to an external model API. The optional Python backend described below is a separate way to exercise the same architecture; the frontend always uses its browser worker.

Try `the quiet fox` or `a cat sees the `. Generation adds 48 characters in the interface. Temperature changes the sampling distribution; the interface uses seed 7 so repeating the same prompt and settings gives the same browser result. Stop terminates the worker and reloads it.

## Train locally

The training path was verified with Python 3.12 and CPU PyTorch 2.14.0. Install its packages in a project-local environment:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python train_transformer.py --output runs/transformer-v1 --steps 500 --max-seconds 120 --export public
```

These commands use macOS/Linux executable paths; on Windows, use `.venv\Scripts\python.exe`. Training downloads no pretrained weights or third-party corpus. `requirements.txt` pins PyTorch; `requirements.lock.txt` records the dependencies observed in the verified macOS ARM64 environment and is not a cross-platform lock with package hashes.

Each training session accepts 1–500 updates and a loop budget up to 120 seconds. The clock is checked between updates. Setup, initial and final evaluation, samples, final checkpoint checks, and export are outside that budget. Use a new output directory every time: training refuses to overwrite an existing run. `--export public` explicitly replaces the browser's `model.json` and `metrics.json` with that run's export.

The checked-in model completed 500 CPU updates. Held-out next-character cross-entropy fell from **3.28959 to 0.23227**. The recorded **1.84 seconds** measures that training loop, including its periodic evaluation and saves; it does not include the complete command or installation. Timings depend on the machine. These results concern held-out combinations of the same toy grammar, not general-language performance. Exact measurements and example continuations are in [public/metrics.json](public/metrics.json).

## Resume a checkpoint

```sh
.venv/bin/python train_transformer.py --resume runs/transformer-v1/last-valid.pt --output runs/transformer-resumed --steps 100 --max-seconds 120
```

`--steps` means additional updates for that session. Resume restores model weights, optimizer state, the training random-number state, vocabulary, and prior history, and checks that the corpus has not changed. A test verifies that a 3-update run plus a 3-update resume matches an uninterrupted 6-update run exactly in the tested environment.

The initial state, every 50th update, and a normal completion are saved atomically as `last-valid.pt`. If interrupted, resume from that most recent saved checkpoint into a new directory; unsaved updates may need to be repeated. Failure information is recorded separately as `failure.json`. Training runs and Python checkpoints are excluded from version control; the browser JSON export is included.

## Optional local Python API

This route requires the virtual environment and a locally trained checkpoint. After the training command above:

```sh
.venv/bin/python server.py --checkpoint runs/transformer-v1/last-valid.pt --port 8847
```

The server binds to loopback, serves only `public/`, and exposes two endpoints:

```sh
curl http://127.0.0.1:8847/api/health
curl http://127.0.0.1:8847/api/generate \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"the quiet fox","maxTokens":48,"temperature":0.8,"seed":7}'
```

The generation response contains the full prompt and continuation. Request bodies are capped at 4,096 bytes; the API permits one generation at a time. It accepts same-origin browser requests and does not expose a public inference service. Browser and Python probabilities closely agree, but their seeded sampling algorithms differ, so equal numeric seeds do not imply identical text across runtimes. Stop either local server with Ctrl-C before starting another on the same port.

## Verify the implementation

```sh
.venv/bin/python -m unittest -v
node --test tests/browser-math.test.mjs
```

The verified run passed **32 Python tests and 8 Node tests**. Python covers tokenizer behavior, the bigram baseline, causal masking, training bounds, checkpoint recovery, exact resume, and the real local HTTP API. HTTP tests skip if `runs/transformer-v1/last-valid.pt` has not been trained locally; a fresh checkout does not include that checkpoint. The remaining Python tests create their own temporary fixtures.

Node tests need no installed packages. They cover input validation, stable probabilities, causal attention, context cropping, seeded sampling, the worker message contract, and PyTorch-to-JavaScript logit parity against the shipped export. The measured maximum logit difference was **0.00000176**, below the 0.0001 tolerance. The parity test skips if the exported model file is removed. Node 25.8.1 was used for verification.

PyTorch may report that its optional NumPy bridge is unavailable. This project does not use that bridge; the recorded training and tests passed without NumPy.

## Explore the code

- `tokenizer.py` assigns a vocabulary ID to each character; `bigram.py` provides the earlier one-character-context baseline.
- `transformer.py` implements the causal decoder; `train_transformer.py` creates the corpus, trains, evaluates, checkpoints, resumes, and exports it.
- `public/inference.js` implements the exported model's forward pass without a machine-learning runtime; `public/model-worker.js` handles browser inference.
- `public/index.html`, `public/style.css`, and `public/app.js` provide the interface; `server.py` provides the optional local API.

The worker and HTTP API accept 1–512-character prompts, 1–96 requested new characters, and temperatures from 0.2 to 1.5. Unknown characters are rejected. Only the latest 48 characters influence the next prediction, with learned positions restarted at zero after cropping. The attention view shows one head of the final layer, not a human-readable explanation of the model's reasoning.

## Hosting

The `public/` directory is deployed to Cloudflare Pages. The site performs real inference in the visitor's browser; it is not a hosted Python service or a prerecorded text animation. No API key is required. Security headers restrict scripts and model requests to the same origin.

To publish an approved update with an already authorized Cloudflare account:

```sh
npm ci --ignore-scripts
npm run deploy
```

The Pages project already exists. Deploying uses the configured `public/` output only; local checkpoints, environment files, and private preparation notes are not published.

## Course reference

This original implementation was inspired by Elliot Arledge's freeCodeCamp course, [How to Build a Large Language Model from Scratch Using Python](https://www.freecodecamp.org/news/how-to-build-a-large-language-model-from-scratch-using-python/) ([course video](https://www.youtube.com/watch?v=UU1WVnMk4E8)). The model, synthetic corpus, browser inference, and interface here are project-specific implementations; no course repository, dataset, or pretrained model was imported. Course attribution does not imply endorsement.
