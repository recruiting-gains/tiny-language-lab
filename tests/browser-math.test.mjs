import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {prepareModel, encode, decode, forward, softmax, validateGenerationOptions, seededRandom, sample} from '../public/inference.js';

function fixture() {
  const config = {vocab_size: 5, context_length: 4, embed_dim: 8, num_heads: 2, num_layers: 2};
  const weights = {};
  let index = 0;
  const vector = (size, value) => Array.from({length: size}, () => value ?? Math.sin(++index * 0.31) * 0.15);
  const matrix = (rows, columns) => Array.from({length: rows}, () => vector(columns));
  weights['token_embedding.weight'] = matrix(5, 8);
  weights['position_embedding.weight'] = matrix(4, 8);
  for (let block = 0; block < 2; block += 1) {
    for (const norm of ['ln1', 'ln2']) {
      weights[`blocks.${block}.${norm}.weight`] = vector(8, 1);
      weights[`blocks.${block}.${norm}.bias`] = vector(8, 0);
    }
    weights[`blocks.${block}.qkv.weight`] = matrix(24, 8);
    weights[`blocks.${block}.proj.weight`] = matrix(8, 8);
    weights[`blocks.${block}.fc1.weight`] = matrix(32, 8);
    weights[`blocks.${block}.fc1.bias`] = vector(32);
    weights[`blocks.${block}.fc2.weight`] = matrix(8, 32);
    weights[`blocks.${block}.fc2.bias`] = vector(8);
  }
  weights['ln_final.weight'] = vector(8, 1);
  weights['ln_final.bias'] = vector(8, 0);
  weights['lm_head.weight'] = matrix(5, 8);
  return {version: 1, config, vocabulary: ['a', 'b', 'c', ' ', '\n'], weights};
}

test('character encoding round trips and rejects unsupported, empty, and oversized prompts', () => {
  const vocabulary = fixture().vocabulary;
  const prompt = 'a b\nc';
  assert.equal(decode(encode(prompt, vocabulary), vocabulary), prompt);
  assert.deepEqual(encode('a'.repeat(512), vocabulary), Array(512).fill(0));
  assert.throws(() => encode('', vocabulary), /at least one/);
  assert.throws(() => encode('a'.repeat(513), vocabulary), /512/);
  assert.throws(() => encode('A', vocabulary), /not learned/);
  assert.throws(() => encode(null, vocabulary), /text prompt/);
  assert.throws(() => decode([-1], vocabulary), /outside/);
  assert.throws(() => decode([1.5], vocabulary), /outside/);
});

test('softmax is stable, normalized, and respects temperature', () => {
  const probabilities = softmax([10000, 10001, 10002]);
  assert.ok(Math.abs(probabilities.reduce((a, b) => a + b, 0) - 1) < 1e-12);
  assert.ok(probabilities.every((value) => value >= 0 && value <= 1));
  assert.ok(softmax([0, 1], 0.2)[1] > softmax([0, 1], 1.5)[1]);
  assert.throws(() => softmax([]), /nonempty/);
  assert.throws(() => softmax([NaN]), /non-finite/);
  assert.throws(() => softmax([1], 0), /positive/);
});

test('attention normalizes each causal row and assigns exactly zero to future positions', () => {
  const model = prepareModel(fixture());
  const full = forward(model, [0, 1, 2, 3]);
  assert.equal(full.logits.length, 5);
  assert.ok(full.logits.every(Number.isFinite));
  for (let row = 0; row < 4; row += 1) {
    assert.ok(Math.abs(full.attention[row].reduce((a, b) => a + b, 0) - 1) < 1e-12);
    for (let column = row + 1; column < 4; column += 1) assert.equal(full.attention[row][column], 0);
  }
  const changedFuture = forward(model, [0, 1, 4, 4]);
  assert.deepEqual(full.attention.slice(0, 2), changedFuture.attention.slice(0, 2));
});

test('cropping restarts learned positions at zero and invalid token IDs reject', () => {
  const model = prepareModel(fixture());
  assert.deepEqual(forward(model, [4, 0, 1, 2, 3]), forward(model, [0, 1, 2, 3]));
  assert.throws(() => forward(model, []), /at least one/);
  assert.throws(() => forward(model, [5]), /outside/);
  assert.throws(() => forward(model, [-1]), /outside/);
  assert.throws(() => forward(model, [0.5]), /outside/);
  assert.throws(() => forward(model, Array(609).fill(0)), /generation limit/);
});

test('model loading validates dimensions and rejects non-finite weights', () => {
  const badShape = fixture();
  badShape.weights['blocks.0.qkv.weight'][0].pop();
  assert.throws(() => prepareModel(badShape), /invalid shape/);
  const badValue = fixture();
  badValue.weights['token_embedding.weight'][0][0] = Infinity;
  assert.throws(() => prepareModel(badValue), /invalid number/);
  const badHeads = fixture();
  badHeads.config.num_heads = 3;
  assert.throws(() => prepareModel(badHeads), /divisible/);
});

test('generation settings are bounded and seeded sampling repeats', () => {
  assert.deepEqual(validateGenerationOptions(), {maxTokens: 64, temperature: 0.8, seed: 7});
  assert.throws(() => validateGenerationOptions({maxTokens: 97}), /96/);
  assert.throws(() => validateGenerationOptions({maxTokens: 0}), /1 and 96/);
  assert.throws(() => validateGenerationOptions({temperature: 0.1}), /0.2 and 1.5/);
  assert.throws(() => validateGenerationOptions({temperature: 1.6}), /0.2 and 1.5/);
  assert.throws(() => validateGenerationOptions({seed: -1}), /seed/);
  const first = seededRandom(7);
  const second = seededRandom(7);
  assert.deepEqual(Array.from({length: 64}, () => sample([0.2, 0.3, 0.5], first)), Array.from({length: 64}, () => sample([0.2, 0.3, 0.5], second)));
});

test('worker loads locally, streams accumulated text, rejects overlapping work, and bounds output', async () => {
  const previousSelf = globalThis.self;
  const previousFetch = globalThis.fetch;
  const messages = [];
  let pending;
  globalThis.self = {postMessage(message) {
    messages.push(message);
    if (pending?.predicate(message)) {
      clearTimeout(pending.timeout);
      const resolve = pending.resolve;
      pending = undefined;
      resolve(message);
    }
  }};
  globalThis.fetch = async (url) => {
    assert.equal(url.pathname, new URL('../public/model.json', import.meta.url).pathname);
    return {ok: true, json: async () => fixture()};
  };
  function request(message, responseType) {
    return new Promise((resolve, reject) => {
      pending = {
        predicate: (response) => response.type === responseType,
        resolve,
        timeout: setTimeout(() => reject(new Error(`Worker did not return ${responseType}.`)), 2000),
      };
      self.onmessage({data: message});
    });
  }
  // Let the handler's finally callback clear its busy state after each reply.
  const settle = () => new Promise((resolve) => setTimeout(resolve, 0));
  try {
    await import('../public/model-worker.js');
    const ready = await request({type: 'load'}, 'ready');
    assert.deepEqual(ready.vocabulary, fixture().vocabulary);
    await settle();
    const completion = request({type: 'generate', prompt: 'ab', maxTokens: 3, temperature: 0.8, seed: 7}, 'done');
    self.onmessage({data: {type: 'inspect', prompt: 'ab'}});
    const done = await completion;
    const streamed = messages.filter((message) => message.type === 'token');
    assert.deepEqual(streamed.map((message) => message.text.length), [3, 4, 5]);
    assert.equal(done.text, streamed.at(-1).text);
    assert.ok(done.text.startsWith('ab'));
    assert.ok(done.elapsedMs >= 0);
    assert.ok(messages.some((message) => message.type === 'error' && /already running/.test(message.message)));
    await settle();
    const tooMany = await request({type: 'generate', prompt: 'a', maxTokens: 97}, 'error');
    assert.match(tooMany.message, /96/);
    await settle();
    const inspection = await request({type: 'inspect', prompt: 'abc a'}, 'inspection');
    assert.equal(inspection.tokens.length, 5);
    assert.deepEqual(inspection.context, ['b', 'c', ' ', 'a']);
    assert.equal(inspection.attention.length, 4);
    assert.equal(inspection.probabilities.length, 5);
    assert.ok(Math.abs(inspection.probabilities.reduce((sum, entry) => sum + entry.probability, 0) - 1) < 1e-12);
    await settle();
  } finally {
    if (pending) clearTimeout(pending.timeout);
    if (previousSelf === undefined) delete globalThis.self;
    else globalThis.self = previousSelf;
    globalThis.fetch = previousFetch;
  }
});

test('browser forward logits match independent PyTorch export within 1e-4', async (context) => {
  let artifact;
  try {
    artifact = JSON.parse(await readFile(new URL('../public/model.json', import.meta.url), 'utf8'));
  } catch (error) {
    if (error.code === 'ENOENT') return context.skip('Train/export public/model.json to run cross-runtime parity.');
    throw error;
  }
  assert.ok(Array.isArray(artifact.parity) && artifact.parity.length >= 2, 'Export at least two independent PyTorch parity examples.');
  const model = prepareModel(artifact);
  let maximumError = 0;
  for (const example of artifact.parity) {
    const result = forward(model, encode(example.prompt, model.vocabulary));
    assert.equal(result.logits.length, example.logits.length);
    for (let id = 0; id < result.logits.length; id += 1) maximumError = Math.max(maximumError, Math.abs(result.logits[id] - example.logits[id]));
  }
  assert.ok(maximumError < 1e-4, `Maximum browser/PyTorch logit error ${maximumError} exceeds 1e-4.`);
  context.diagnostic(`Maximum browser/PyTorch logit error: ${maximumError}`);
});
