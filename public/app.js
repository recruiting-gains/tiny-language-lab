const $ = (id) => document.getElementById(id);
const state = { worker: null, ready: false, busy: false, inspecting: false, config: {}, prompt: '', generated: '', start: 0, timeout: null, loadedOnce: false };
const formatNumber = (value) => Number.isFinite(Number(value)) && value !== null && value !== '' ? Number(value).toLocaleString('en-US') : 'Not recorded';
const configValue = (config, ...keys) => keys.map((key) => config?.[key]).find((value) => value !== undefined && value !== null);
const charLabel = (character) => character === ' ' ? 'space' : character === '\n' ? 'newline' : character === '\t' ? 'tab' : character;
const charDisplay = (character) => character === ' ' ? '␣' : character === '\n' ? '↵' : character === '\t' ? '⇥' : character;

function setStatus(label, mode = '') {
  $('model-status-label').textContent = label;
  $('model-status').className = `status-chip${mode ? ` is-${mode}` : ''}`;
}

function updateControls() {
  $('generate-button').disabled = !state.ready || state.busy || state.inspecting;
  $('inspect-button').disabled = !state.ready || state.busy || state.inspecting;
  $('prompt').disabled = state.busy;
  $('temperature').disabled = state.busy;
  $('stop-button').hidden = !state.busy;
  $('generate-button').innerHTML = state.busy ? 'Generating…' : 'Generate <span aria-hidden="true">↗</span>';
}

function updateModelFacts(config) {
  const layers = configValue(config, 'n_layer', 'n_layers', 'num_layers', 'layers');
  const heads = configValue(config, 'n_head', 'n_heads', 'num_heads', 'heads');
  const context = configValue(config, 'block_size', 'context_length', 'context_size', 'max_seq_len');
  if (layers && heads) $('architecture').textContent = `${layers} layers · ${heads} heads`;
  if (context) {
    $('context-window').textContent = `${context} characters`;
    $('inspection-prompt').maxLength = Number(context);
  }
}

function finishGeneration(message, isError = false) {
  clearTimeout(state.timeout);
  state.busy = false;
  $('generation-output').classList.remove('is-generating');
  $('generation-status').textContent = message;
  $('generation-status').classList.toggle('is-error', isError);
  setStatus(state.ready ? 'Ready · runs locally' : 'Model unavailable', state.ready ? 'ready' : 'error');
  updateControls();
}

function showModelError(message) {
  clearTimeout(state.timeout);
  state.ready = false;
  state.inspecting = false;
  finishGeneration(message, true);
  $('inspection-status').textContent = 'Model unavailable; inspection could not run.';
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.className = 'text-button';
  retry.textContent = 'Reload model';
  retry.addEventListener('click', loadModel, { once: true });
  $('generation-status').append(' ', retry);
}

function renderGeneration() {
  const output = $('generation-output');
  output.replaceChildren();
  const prompt = document.createElement('span');
  prompt.className = 'prompt-text';
  prompt.textContent = state.prompt;
  const generated = document.createElement('span');
  generated.className = 'continuation-text';
  generated.textContent = state.generated;
  output.append(prompt, generated);
  output.classList.remove('is-empty');
}

function loadModel() {
  state.worker?.terminate();
  clearTimeout(state.timeout);
  state.ready = false;
  state.busy = false;
  state.inspecting = false;
  setStatus('Loading model');
  $('generation-status').classList.remove('is-error');
  $('generation-status').textContent = 'Loading the trained weights into this browser…';
  $('inspection-status').textContent = 'Waiting for the model.';
  updateControls();
  try {
    state.worker = new Worker('/model-worker.js', { type: 'module' });
    state.worker.addEventListener('error', () => showModelError('The model could not load. Check your connection and try reloading.'));
    state.worker.addEventListener('message', ({ data }) => {
      if (data.type === 'ready') {
        clearTimeout(state.timeout);
        state.ready = true;
        state.config = data.config || {};
        updateModelFacts(state.config);
        setStatus('Ready · runs locally', 'ready');
        $('generation-status').textContent = 'Ready. Generate a continuation using the trained model.';
        updateControls();
        inspectSequence();
        state.loadedOnce = true;
      } else if (data.type === 'token' && state.busy) {
        const accumulated = typeof data.text === 'string' ? data.text : '';
        state.generated = accumulated.startsWith(state.prompt) ? accumulated.slice(state.prompt.length) : accumulated;
        renderGeneration();
      } else if (data.type === 'done' && state.busy) {
        if (typeof data.text === 'string') {
          state.generated = data.text.startsWith(state.prompt) ? data.text.slice(state.prompt.length) : data.text;
          renderGeneration();
        }
        const elapsed = Number.isFinite(data.elapsedMs) ? data.elapsedMs : performance.now() - state.start;
        $('generation-timing').textContent = `${(elapsed / 1000).toFixed(2)}s · LOCAL`;
        finishGeneration(`Generated ${Array.from(state.generated).length} characters at temperature ${Number($('temperature').value).toFixed(2)}. Same settings use the same seed (7).`);
      } else if (data.type === 'inspection') {
        clearTimeout(state.timeout);
        state.inspecting = false;
        renderInspection(data);
        updateControls();
      } else if (data.type === 'error') {
        if (!state.ready) return showModelError(data.message || 'The model could not load.');
        if (state.inspecting) {
          state.inspecting = false;
          clearTimeout(state.timeout);
          $('inspection-status').textContent = data.message || 'Inspection failed. Try a different sequence.';
          updateControls();
        } else {
          finishGeneration(data.message || 'Generation failed. Try a different prompt.', true);
        }
      }
    });
    state.worker.postMessage({ type: 'load' });
    state.timeout = setTimeout(() => {
      state.worker?.terminate();
      showModelError('Loading took too long. Check your connection, then reload the model.');
    }, 45000);
  } catch (error) {
    showModelError(`The browser could not start the model. ${error.message || ''}`);
  }
}

function generate(event) {
  event.preventDefault();
  if (!state.ready || state.busy || state.inspecting) return;
  const prompt = $('prompt').value;
  if (!prompt.length) {
    $('generation-status').textContent = 'Enter at least one character to give the model a starting point.';
    $('generation-status').classList.add('is-error');
    $('prompt').focus();
    return;
  }
  state.prompt = prompt;
  state.generated = '';
  state.busy = true;
  state.start = performance.now();
  $('generation-timing').textContent = 'COMPUTING LOCALLY';
  $('generation-status').textContent = 'Predicting and sampling one character at a time…';
  $('generation-status').classList.remove('is-error');
  renderGeneration();
  $('generation-output').classList.add('is-generating');
  setStatus('Generating locally', 'running');
  updateControls();
  state.worker.postMessage({ type: 'generate', prompt, maxTokens: 48, temperature: Number($('temperature').value), seed: 7 });
  state.timeout = setTimeout(() => {
    state.worker?.terminate();
    showModelError('Generation exceeded 60 seconds. The worker was stopped; reload to try again.');
  }, 60000);
}

function inspectSequence(event) {
  event?.preventDefault();
  if (!state.ready || state.busy || state.inspecting) return;
  const prompt = $('inspection-prompt').value;
  if (!prompt.length) {
    $('inspection-status').textContent = 'Enter at least one character.';
    $('inspection-prompt').focus();
    return;
  }
  state.inspecting = true;
  $('inspection-status').textContent = 'Computing tokens, probabilities, and attention…';
  updateControls();
  state.worker.postMessage({ type: 'inspect', prompt });
  state.timeout = setTimeout(() => {
    state.worker?.terminate();
    showModelError('Inspection took too long. Reload the model to try again.');
  }, 30000);
}

function renderInspection(data) {
  const tokenList = $('token-list');
  tokenList.replaceChildren();
  for (const token of data.tokens || []) {
    const item = document.createElement('div');
    item.className = `token${token.character === ' ' ? ' is-space' : ''}`;
    item.setAttribute('aria-label', `${charLabel(token.character)}: token ID ${token.id}`);
    const character = document.createElement('span');
    character.className = 'token-character';
    character.textContent = charDisplay(token.character);
    const id = document.createElement('span');
    id.className = 'token-id';
    id.textContent = token.id;
    item.append(character, id);
    tokenList.append(item);
  }
  const probabilities = $('probabilities');
  probabilities.replaceChildren();
  const entries = [...(data.probabilities || [])].sort((a, b) => b.probability - a.probability).slice(0, 6);
  for (const entry of entries) {
    const row = document.createElement('div');
    row.className = 'probability-row';
    const character = document.createElement('span');
    character.className = 'probability-character';
    character.textContent = charDisplay(entry.character);
    character.setAttribute('aria-label', charLabel(entry.character));
    const track = document.createElement('div');
    track.className = 'probability-track';
    track.setAttribute('aria-hidden', 'true');
    const fill = document.createElement('div');
    fill.className = 'probability-fill';
    fill.style.width = `${Math.max(0, Math.min(1, entry.probability)) * 100}%`;
    track.append(fill);
    const value = document.createElement('span');
    value.className = 'probability-value';
    value.textContent = `${(entry.probability * 100).toFixed(1)}%`;
    row.append(character, track, value);
    probabilities.append(row);
  }
  renderAttention(data.attention || [], data.context || []);
  $('inspection-status').textContent = `Showing ${data.tokens?.length || 0} character tokens. ␣ marks a space.`;
}

function renderAttention(attention, context) {
  const container = $('attention-container');
  container.replaceChildren();
  const chars = Array.isArray(context) ? context : Array.from(context);
  const length = Math.min(chars.length, attention.length);
  if (!length) {
    container.textContent = 'No attention values were returned.';
    return;
  }
  const table = document.createElement('table');
  table.className = 'attention-table';
  const caption = document.createElement('caption');
  caption.className = 'sr-only';
  caption.textContent = 'Final-layer, head-1 causal attention. Rows are querying positions; columns are positions read. Values are attention weights from zero to one. Future positions are masked.';
  table.append(caption);
  const thead = document.createElement('thead');
  const header = document.createElement('tr');
  const empty = document.createElement('th');
  empty.setAttribute('aria-label', 'Query position / key position');
  header.append(empty);
  for (let j = 0; j < length; j++) {
    const th = document.createElement('th');
    th.scope = 'col';
    th.textContent = charDisplay(chars[j]);
    th.setAttribute('aria-label', `Position ${j + 1}: ${charLabel(chars[j])}`);
    header.append(th);
  }
  thead.append(header);
  table.append(thead);
  const tbody = document.createElement('tbody');
  for (let i = 0; i < length; i++) {
    const row = document.createElement('tr');
    const label = document.createElement('th');
    label.scope = 'row';
    label.textContent = charDisplay(chars[i]);
    label.setAttribute('aria-label', `Position ${i + 1}: ${charLabel(chars[i])}`);
    row.append(label);
    for (let j = 0; j < length; j++) {
      const cell = document.createElement('td');
      const weight = Number(attention[i]?.[j]);
      if (j > i) {
        cell.className = 'masked';
        cell.setAttribute('aria-label', 'Future position, masked');
        cell.title = 'Future position: masked';
      } else if (Number.isFinite(weight)) {
        const alpha = Math.max(0, Math.min(1, weight));
        cell.style.backgroundColor = `rgb(${Math.round(237 - 188 * alpha)} ${Math.round(241 - 151 * alpha)} ${Math.round(250 - 39 * alpha)})`;
        cell.style.color = alpha > 0.52 ? '#ffffff' : '#263d74';
        cell.setAttribute('aria-label', weight.toFixed(4));
        cell.title = `Position ${i + 1} reads position ${j + 1}: ${weight.toFixed(4)}`;
        const label = document.createElement('span');
        label.setAttribute('aria-hidden', 'true');
        label.textContent = weight.toFixed(1).replace(/^0/, '');
        cell.append(label);
      } else {
        cell.setAttribute('aria-label', 'Value unavailable');
        cell.textContent = '—';
      }
      row.append(cell);
    }
    tbody.append(row);
  }
  table.append(tbody);
  container.append(table);
  if (length > 12) {
    container.tabIndex = 0;
    container.setAttribute('aria-label', 'Attention table. Scroll horizontally to see all positions.');
  } else {
    container.removeAttribute('tabindex');
    container.removeAttribute('aria-label');
  }
}

function textSample(value) {
  if (typeof value === 'string') return value;
  if (value && typeof value.text === 'string') return value.text;
  return 'No sample was recorded.';
}

function formatTime(seconds) {
  if (!Number.isFinite(seconds)) return 'Not recorded';
  return seconds < 60 ? `${seconds.toFixed(1)} seconds` : `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`;
}

async function loadMetrics() {
  try {
    const response = await fetch('/metrics.json', { cache: 'no-cache' });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const metrics = await response.json();
    $('parameter-count').textContent = formatNumber(metrics.parameters);
    if (!state.ready) updateModelFacts(metrics.config || {});
    $('before-loss').textContent = Number.isFinite(metrics.before?.validation_loss) ? metrics.before.validation_loss.toFixed(3) : '—';
    $('after-loss').textContent = Number.isFinite(metrics.after?.validation_loss) ? metrics.after.validation_loss.toFixed(3) : '—';
    $('sample-before').textContent = textSample(metrics.before?.sample);
    $('sample-after').textContent = textSample(metrics.after?.sample);
    $('training-steps').textContent = formatNumber(metrics.steps);
    $('training-time').textContent = formatTime(metrics.training_seconds);
    $('training-device').textContent = metrics.device || 'Not recorded';
    $('training-seed').textContent = metrics.seed === undefined ? 'Not recorded' : String(metrics.seed);
    $('dataset-description').textContent = metrics.dataset?.description || 'No dataset description was recorded.';
    if (metrics.dataset?.train_documents !== undefined && metrics.dataset?.validation_documents !== undefined) {
      $('dataset-counts').textContent = `${formatNumber(metrics.dataset.train_documents)} training documents · ${formatNumber(metrics.dataset.validation_documents)} held-out validation documents.`;
    }
    const limitations = metrics.dataset?.limitations;
    $('dataset-limitations').textContent = Array.isArray(limitations) ? limitations.join(' ') : limitations || '';
    renderLossChart(metrics.history || []);
    $('metrics-status').textContent = '';
  } catch (error) {
    $('metrics-status').textContent = `The training record could not be loaded (${error.message}). Measurements and samples are unavailable.`;
    $('metrics-status').classList.add('is-error');
    $('sample-before').textContent = 'Recorded sample unavailable.';
    $('sample-after').textContent = 'Recorded sample unavailable.';
    $('loss-chart').textContent = 'Recorded loss measurements unavailable.';
    $('dataset-description').textContent = 'The dataset record could not be loaded.';
  }
}

function renderLossChart(history) {
  const container = $('loss-chart');
  const points = history.filter((entry) => Number.isFinite(entry.step) && (Number.isFinite(entry.train_loss) || Number.isFinite(entry.validation_loss)));
  if (points.length < 2) {
    container.textContent = 'At least two recorded measurements are needed to draw the loss curve.';
    return;
  }
  const ns = 'http://www.w3.org/2000/svg';
  const create = (name, attrs, value) => {
    const element = document.createElementNS(ns, name);
    for (const [key, val] of Object.entries(attrs || {})) element.setAttribute(key, val);
    if (value !== undefined) element.textContent = value;
    return element;
  };
  const width = 500, height = 225;
  const margin = { top: 14, right: 16, bottom: 36, left: 32 };
  const values = points.flatMap((entry) => [entry.train_loss, entry.validation_loss]).filter(Number.isFinite);
  const yMin = 0;
  const yMax = Math.ceil(Math.max(...values) * 1.12 * 2) / 2;
  const xMin = Math.min(...points.map((entry) => entry.step));
  const xMax = Math.max(...points.map((entry) => entry.step));
  const x = (step) => margin.left + ((step - xMin) / (xMax - xMin || 1)) * (width - margin.left - margin.right);
  const y = (loss) => height - margin.bottom - ((loss - yMin) / (yMax - yMin || 1)) * (height - margin.top - margin.bottom);
  const svg = create('svg', { viewBox: `0 0 ${width} ${height}`, role: 'img', 'aria-labelledby': 'loss-chart-title loss-chart-description' });
  svg.append(create('title', { id: 'loss-chart-title' }, 'Training and validation loss by training step'));
  svg.append(create('desc', { id: 'loss-chart-description' }, `${points.length} recorded measurements from step ${xMin} to ${xMax}. First training loss ${points[0].train_loss}, last training loss ${points.at(-1).train_loss}. First validation loss ${points[0].validation_loss}, last validation loss ${points.at(-1).validation_loss}. Lower loss means better next-character prediction on this dataset.`));
  for (let i = 0; i <= 4; i++) {
    const value = yMax * i / 4;
    svg.append(create('line', { x1: margin.left, y1: y(value), x2: width - margin.right, y2: y(value), class: 'chart-grid' }));
    svg.append(create('text', { x: margin.left - 9, y: y(value) + 3, 'text-anchor': 'end' }, value.toFixed(1)));
  }
  for (let i = 0; i <= 4; i++) {
    const step = xMin + (xMax - xMin) * i / 4;
    svg.append(create('text', { x: x(step), y: height - 19, 'text-anchor': 'middle' }, Math.round(step).toLocaleString('en-US')));
  }
  svg.append(create('text', { x: width - margin.right, y: height - 2, 'text-anchor': 'end' }, 'training steps'));
  for (const [key, className] of [['train_loss', 'train-path'], ['validation_loss', 'validation-path']]) {
    const series = points.filter((entry) => Number.isFinite(entry[key]));
    const d = series.map((entry, i) => `${i ? 'L' : 'M'} ${x(entry.step).toFixed(2)} ${y(entry[key]).toFixed(2)}`).join(' ');
    svg.append(create('path', { d, class: className }));
  }
  container.replaceChildren(svg);
}

$('generation-form').addEventListener('submit', generate);
$('inspection-form').addEventListener('submit', inspectSequence);
$('temperature').addEventListener('input', () => { $('temperature-value').textContent = Number($('temperature').value).toFixed(2); });
$('prompt').addEventListener('input', () => {
  const length = Array.from($('prompt').value).length;
  $('prompt-count').textContent = `${length} character${length === 1 ? '' : 's'}`;
});
$('stop-button').addEventListener('click', () => {
  state.worker?.terminate();
  finishGeneration('Generation stopped. Reloading the model for the next attempt.');
  $('generation-timing').textContent = 'STOPPED';
  loadModel();
});
loadMetrics();
loadModel();
