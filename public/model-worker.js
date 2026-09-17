import {prepareModel, encode, decode, forward, softmax, validateGenerationOptions, seededRandom, sample} from './inference.js';

let model;
let busy = false;

async function handleMessage(message) {
  if (!message || typeof message !== 'object') throw new Error('Send a valid model request.');
  if (message.type === 'load') {
    if (!model) {
      const response = await fetch(new URL('./model.json', import.meta.url));
      if (!response.ok) throw new Error(`The model could not be loaded (HTTP ${response.status}). Reload the page to try again.`);
      model = prepareModel(await response.json());
    }
    self.postMessage({type: 'ready', config: model.config, vocabulary: model.vocabulary});
    return;
  }
  if (!model) throw new Error('The model is still loading. Try again when it is ready.');
  if (message.type !== 'generate' && message.type !== 'inspect') throw new Error('That model request is not supported.');
  const ids = encode(message.prompt, model.vocabulary);

  if (message.type === 'inspect') {
    const result = forward(model, ids);
    const probabilities = softmax(result.logits)
      .map((probability, id) => ({character: model.vocabulary[id], probability}))
      .sort((a, b) => b.probability - a.probability)
      .slice(0, 8);
    self.postMessage({
      type: 'inspection',
      tokens: ids.map((id) => ({character: model.vocabulary[id], id})),
      probabilities,
      attention: result.attention,
      context: result.contextIds.map((id) => model.vocabulary[id]),
    });
    return;
  }

  const options = validateGenerationOptions(message);
  const random = seededRandom(options.seed);
  const startedAt = performance.now();
  let generatedText = message.prompt;
  for (let index = 0; index < options.maxTokens; index += 1) {
    const result = forward(model, ids);
    const nextId = sample(softmax(result.logits, options.temperature), random);
    ids.push(nextId);
    generatedText += model.vocabulary[nextId];
    self.postMessage({type: 'token', text: generatedText});
    // Yield every token so this worker remains responsive. The UI can terminate
    // and recreate the worker to cancel a request immediately.
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
  self.postMessage({type: 'done', text: decode(ids, model.vocabulary), elapsedMs: performance.now() - startedAt});
}

self.onmessage = (event) => {
  if (busy) {
    self.postMessage({type: 'error', message: 'A request is already running. Wait for it to finish or press Stop.'});
    return;
  }
  busy = true;
  handleMessage(event.data)
    .catch((error) => self.postMessage({type: 'error', message: error instanceof Error ? error.message : 'The model request failed.'}))
    .finally(() => { busy = false; });
};
