// Original, dependency-free inference for this lab's small causal transformer.
// Float64 arithmetic makes the implementation readable and closely matches the
// exported PyTorch float32 model without introducing a runtime dependency.
const MAX_PROMPT_CHARACTERS = 512;
const LAYER_NORM_EPSILON = 1e-5;

function assert(condition, message) {
  if (!condition) throw new Error(message);
}

export function validatePrompt(text) {
  assert(typeof text === 'string', 'Enter a text prompt.');
  const characters = Array.from(text);
  assert(characters.length > 0, 'Enter at least one character.');
  assert(characters.length <= MAX_PROMPT_CHARACTERS, 'Keep the prompt to 512 characters or fewer.');
  return characters;
}

function validateVocabulary(vocabulary) {
  assert(Array.isArray(vocabulary) && vocabulary.length > 0, 'The model vocabulary is missing.');
  assert(vocabulary.every((character) => typeof character === 'string' && Array.from(character).length === 1), 'Vocabulary entries must be individual characters.');
  assert(new Set(vocabulary).size === vocabulary.length, 'The model vocabulary contains duplicate characters.');
}

export function encode(text, vocabulary) {
  const characters = validatePrompt(text);
  validateVocabulary(vocabulary);
  const ids = new Map(vocabulary.map((character, id) => [character, id]));
  return characters.map((character) => {
    assert(ids.has(character), `The model has not learned the character ${JSON.stringify(character)}. Use a character shown in the vocabulary.`);
    return ids.get(character);
  });
}

export function decode(ids, vocabulary) {
  validateVocabulary(vocabulary);
  assert(Array.isArray(ids) || ArrayBuffer.isView(ids), 'Token IDs must be an array.');
  return Array.from(ids, (id) => {
    assert(Number.isInteger(id) && id >= 0 && id < vocabulary.length, 'A token ID is outside the vocabulary.');
    return vocabulary[id];
  }).join('');
}

function tensor(weights, name, shape) {
  const source = weights[name];
  assert(Array.isArray(source) && source.length === shape[0], `The model tensor ${name} has an invalid shape.`);
  const flattened = [];
  if (shape.length === 2) {
    for (const row of source) {
      assert(Array.isArray(row) && row.length === shape[1], `The model tensor ${name} has an invalid shape.`);
      for (const value of row) flattened.push(value);
    }
  } else {
    flattened.push(...source);
  }
  assert(flattened.every((value) => typeof value === 'number' && Number.isFinite(value)), `The model tensor ${name} contains an invalid number.`);
  return Float64Array.from(flattened);
}

export function prepareModel(artifact) {
  assert(artifact && artifact.version === 1, 'This model format is not supported.');
  const config = artifact.config;
  assert(config && typeof config === 'object', 'The model configuration is missing.');
  const limits = {vocab_size: 512, context_length: 512, embed_dim: 512, num_heads: 16, num_layers: 12};
  for (const [key, maximum] of Object.entries(limits)) {
    assert(Number.isInteger(config[key]) && config[key] > 0 && config[key] <= maximum, `The model setting ${key} is invalid.`);
  }
  assert(config.embed_dim % config.num_heads === 0, 'Embedding width must be divisible by the number of attention heads.');
  validateVocabulary(artifact.vocabulary);
  assert(artifact.vocabulary.length === config.vocab_size, 'Vocabulary size does not match the model.');
  assert(artifact.weights && typeof artifact.weights === 'object', 'The model weights are missing.');

  const {vocab_size: V, context_length: T, embed_dim: D, num_layers: L} = config;
  const weights = artifact.weights;
  const blocks = [];
  for (let index = 0; index < L; index += 1) {
    const prefix = `blocks.${index}`;
    blocks.push({
      ln1Weight: tensor(weights, `${prefix}.ln1.weight`, [D]),
      ln1Bias: tensor(weights, `${prefix}.ln1.bias`, [D]),
      qkvWeight: tensor(weights, `${prefix}.qkv.weight`, [3 * D, D]),
      projWeight: tensor(weights, `${prefix}.proj.weight`, [D, D]),
      ln2Weight: tensor(weights, `${prefix}.ln2.weight`, [D]),
      ln2Bias: tensor(weights, `${prefix}.ln2.bias`, [D]),
      fc1Weight: tensor(weights, `${prefix}.fc1.weight`, [4 * D, D]),
      fc1Bias: tensor(weights, `${prefix}.fc1.bias`, [4 * D]),
      fc2Weight: tensor(weights, `${prefix}.fc2.weight`, [D, 4 * D]),
      fc2Bias: tensor(weights, `${prefix}.fc2.bias`, [D]),
    });
  }
  return {
    config: {...config},
    vocabulary: [...artifact.vocabulary],
    tokenEmbedding: tensor(weights, 'token_embedding.weight', [V, D]),
    positionEmbedding: tensor(weights, 'position_embedding.weight', [T, D]),
    blocks,
    lnFinalWeight: tensor(weights, 'ln_final.weight', [D]),
    lnFinalBias: tensor(weights, 'ln_final.bias', [D]),
    lmHeadWeight: tensor(weights, 'lm_head.weight', [V, D]),
  };
}

export function softmax(logits, temperature = 1) {
  assert((Array.isArray(logits) || ArrayBuffer.isView(logits)) && logits.length > 0, 'Logits must be a nonempty numeric array.');
  assert(Number.isFinite(temperature) && temperature > 0, 'Temperature must be a positive number.');
  let maximum = -Infinity;
  for (const value of logits) {
    assert(Number.isFinite(value), 'The model produced a non-finite prediction.');
    maximum = Math.max(maximum, value);
  }
  const probabilities = Array.from(logits, (value) => Math.exp((value - maximum) / temperature));
  const total = probabilities.reduce((sum, value) => sum + value, 0);
  return probabilities.map((value) => value / total);
}

function layerNorm(input, rows, width, scale, bias) {
  const output = new Float64Array(input.length);
  for (let row = 0; row < rows; row += 1) {
    const offset = row * width;
    let mean = 0;
    for (let column = 0; column < width; column += 1) mean += input[offset + column];
    mean /= width;
    let variance = 0;
    for (let column = 0; column < width; column += 1) variance += (input[offset + column] - mean) ** 2;
    variance /= width;
    const inverseStd = 1 / Math.sqrt(variance + LAYER_NORM_EPSILON);
    for (let column = 0; column < width; column += 1) {
      output[offset + column] = (input[offset + column] - mean) * inverseStd * scale[column] + bias[column];
    }
  }
  return output;
}

function linear(input, rows, inputWidth, outputWidth, weights, bias) {
  const output = new Float64Array(rows * outputWidth);
  for (let row = 0; row < rows; row += 1) {
    const inputOffset = row * inputWidth;
    for (let column = 0; column < outputWidth; column += 1) {
      let value = bias ? bias[column] : 0;
      const weightOffset = column * inputWidth;
      for (let index = 0; index < inputWidth; index += 1) value += input[inputOffset + index] * weights[weightOffset + index];
      output[row * outputWidth + column] = value;
    }
  }
  return output;
}

function attention(qkv, length, width, heads) {
  const headWidth = width / heads;
  const scale = 1 / Math.sqrt(headWidth);
  const output = new Float64Array(length * width);
  const headZero = Array.from({length}, () => Array(length).fill(0));
  for (let head = 0; head < heads; head += 1) {
    const headOffset = head * headWidth;
    for (let query = 0; query < length; query += 1) {
      const queryOffset = query * 3 * width + headOffset;
      const scores = new Float64Array(query + 1);
      for (let key = 0; key <= query; key += 1) {
        const keyOffset = key * 3 * width + width + headOffset;
        let score = 0;
        for (let dimension = 0; dimension < headWidth; dimension += 1) score += qkv[queryOffset + dimension] * qkv[keyOffset + dimension];
        scores[key] = score * scale;
      }
      const probabilities = softmax(scores);
      for (let key = 0; key <= query; key += 1) {
        if (head === 0) headZero[query][key] = probabilities[key];
        const valueOffset = key * 3 * width + 2 * width + headOffset;
        for (let dimension = 0; dimension < headWidth; dimension += 1) {
          output[query * width + headOffset + dimension] += probabilities[key] * qkv[valueOffset + dimension];
        }
      }
    }
  }
  return {output, headZero};
}

export function forward(model, tokenIds) {
  assert(model && model.config && Array.isArray(model.blocks), 'Load the model before running inference.');
  assert(Array.isArray(tokenIds) || ArrayBuffer.isView(tokenIds), 'Token IDs must be an array.');
  assert(tokenIds.length > 0, 'Enter at least one character.');
  assert(tokenIds.length <= MAX_PROMPT_CHARACTERS + 96, 'The input exceeds this lab’s generation limit.');
  for (const id of tokenIds) assert(Number.isInteger(id) && id >= 0 && id < model.config.vocab_size, 'A token ID is outside the vocabulary.');
  const contextIds = Array.from(tokenIds).slice(-model.config.context_length);
  const length = contextIds.length;
  const width = model.config.embed_dim;
  let state = new Float64Array(length * width);
  for (let row = 0; row < length; row += 1) {
    for (let column = 0; column < width; column += 1) {
      state[row * width + column] = model.tokenEmbedding[contextIds[row] * width + column] + model.positionEmbedding[row * width + column];
    }
  }
  let finalAttention;
  for (const block of model.blocks) {
    const normalized = layerNorm(state, length, width, block.ln1Weight, block.ln1Bias);
    const qkv = linear(normalized, length, width, 3 * width, block.qkvWeight);
    const attended = attention(qkv, length, width, model.config.num_heads);
    const projected = linear(attended.output, length, width, width, block.projWeight);
    for (let index = 0; index < state.length; index += 1) state[index] += projected[index];
    const normalizedAgain = layerNorm(state, length, width, block.ln2Weight, block.ln2Bias);
    const hidden = linear(normalizedAgain, length, width, 4 * width, block.fc1Weight, block.fc1Bias);
    for (let index = 0; index < hidden.length; index += 1) hidden[index] = Math.max(0, hidden[index]);
    const feedForward = linear(hidden, length, 4 * width, width, block.fc2Weight, block.fc2Bias);
    for (let index = 0; index < state.length; index += 1) state[index] += feedForward[index];
    finalAttention = attended.headZero;
  }
  state = layerNorm(state, length, width, model.lnFinalWeight, model.lnFinalBias);
  const lastState = state.subarray((length - 1) * width);
  const logits = Array.from(linear(lastState, 1, width, model.config.vocab_size, model.lmHeadWeight));
  assert(logits.every(Number.isFinite), 'The model produced a non-finite prediction.');
  return {logits, attention: finalAttention, contextIds};
}

export function validateGenerationOptions({maxTokens = 64, temperature = 0.8, seed = 7} = {}) {
  assert(Number.isInteger(maxTokens) && maxTokens >= 1 && maxTokens <= 96, 'Choose between 1 and 96 new characters.');
  assert(Number.isFinite(temperature) && temperature >= 0.2 && temperature <= 1.5, 'Choose a temperature between 0.2 and 1.5.');
  assert(Number.isInteger(seed) && seed >= 0 && seed <= 0xffffffff, 'The seed must be an integer between 0 and 4294967295.');
  return {maxTokens, temperature, seed};
}

export function seededRandom(seed) {
  // Mulberry32: repeatable sampling for a chosen seed, not cryptography.
  let state = seed >>> 0;
  return () => {
    state = (state + 0x6d2b79f5) | 0;
    let value = Math.imul(state ^ (state >>> 15), 1 | state);
    value ^= value + Math.imul(value ^ (value >>> 7), 61 | value);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

export function sample(probabilities, random) {
  let remaining = random();
  for (let index = 0; index < probabilities.length; index += 1) {
    remaining -= probabilities[index];
    if (remaining < 0) return index;
  }
  return probabilities.length - 1;
}
