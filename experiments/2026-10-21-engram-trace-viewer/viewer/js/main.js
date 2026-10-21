// Wiring: load a trace, drive the scene and the panel from one position, keep the URL shareable.

import { indexTrace, loadTrace, TraceError } from './trace.js';
import { EngramScene } from './scene.js';
import { renderHeader, renderPosition } from './panel.js';

const $ = (id) => document.getElementById(id);
const MAX_WORLD = 4096;
const state = { trace: null, index: null, t: 0, layer: 0, world: 8, playing: false, source: null };
let loads = 0;

// an integer in [min, max], or the fallback when the value is not an integer at all
function boundedInt(value, fallback, min, max) {
  if (value == null || value === '') return fallback;
  const n = Number(value);
  return Number.isInteger(n) ? Math.min(max, Math.max(min, n)) : fallback;
}

const scene = new EngramScene($('stage'));
scene.onStepEnd = () => {
  if (!state.playing) return;
  if (state.t + 1 < state.trace.tokens.length) go(state.t + 1, true);
  else setPlaying(false);
};

// a view of a bundled or remote trace can be shared; a local file cannot, so its URL drops trace
function syncUrl() {
  const url = new URL(location.href);
  if (typeof state.source === 'string') {
    url.searchParams.set('trace', state.source);
    url.searchParams.set('t', state.t);
    url.searchParams.set('layer', state.layer);
  } else {
    for (const key of ['trace', 't', 'layer']) url.searchParams.delete(key);
  }
  url.searchParams.set('world', state.world);
  history.replaceState(null, '', url);
}

function go(t, animate) {
  if (!state.trace) return;
  state.t = boundedInt(t, state.t, 0, state.trace.tokens.length - 1);
  $('scrub').value = state.t;
  $('where').textContent = `${state.t} / ${state.trace.tokens.length - 1}`;
  scene.show(state.t, animate);
  renderPosition($('position'), state.trace, state.index, state.t, state.layer, state.world);
  syncUrl();
}

function setPlaying(playing) {
  state.playing = playing && state.trace != null;
  $('play').classList.toggle('playing', state.playing);
  $('play').setAttribute('aria-label', state.playing ? 'Pause' : 'Play');
  if (state.playing) go(state.t + 1 < state.trace.tokens.length ? state.t + 1 : 0, true);
}

// one cell per position, brighter the more of its rows an earlier position already read
function drawHeat() {
  const canvas = $('heat');
  const lookups = state.index[state.layer].lookups;
  canvas.width = lookups.length;
  const ctx = canvas.getContext('2d');
  lookups.forEach((perPos, t) => {
    const share = perPos.filter((x) => x.firstSeen < t).length / perPos.length;
    ctx.fillStyle = `rgba(251, 211, 50, ${0.08 + 0.92 * share})`;
    ctx.fillRect(t, 0, 1, canvas.height);
  });
}

function setView() {
  if (!state.trace) return;
  setPlaying(false);
  scene.setView(state.layer, state.world);
  drawHeat();
  go(state.t, false);
}

function showError(error) {
  $('error').textContent = error instanceof TraceError ? error.message : `Could not open the trace: ${error.message}`;
  $('error').hidden = false;
}

function selectInCatalog(source, name) {
  const catalog = $('catalog');
  catalog.querySelector('option[data-local]')?.remove();
  if (typeof source === 'string' && [...catalog.options].some((o) => o.value === source)) {
    catalog.value = source;
    return;
  }
  const option = new Option(typeof source === 'string' ? source : `${name} (local file)`, '');
  option.dataset.local = '';
  catalog.add(option);
  catalog.value = '';
}

// Opens a trace and commits it only if it is still the latest request and its view builds.
async function open(source, name, view = {}) {
  const ticket = ++loads;
  try {
    const trace = await loadTrace(source);
    if (ticket !== loads) return;
    const index = indexTrace(trace);
    setPlaying(false);
    Object.assign(state, {
      trace,
      index,
      source: typeof source === 'string' ? source : null,
      name,
      layer: boundedInt(view.layer, 0, 0, trace.layers.length - 1),
      t: boundedInt(view.t, 0, 0, trace.tokens.length - 1),
    });
    $('error').hidden = true;
    $('layer').replaceChildren(...trace.layers.map((layer, l) => new Option(`Layer ${layer.layer_id}`, l)));
    $('layer').value = state.layer;
    $('scrub').max = trace.tokens.length - 1;
    selectInCatalog(source, name);
    renderHeader($('header'), trace);
    scene.setTrace(trace, index);
    setView();
    document.title = `${trace.title || name} · Engram trace viewer`;
  } catch (error) {
    if (ticket !== loads) return;
    showError(error);
    if (state.trace) selectInCatalog(state.source ?? {}, state.name);
  }
}

$('play').addEventListener('click', () => setPlaying(!state.playing));
$('prev').addEventListener('click', () => { setPlaying(false); go(state.t - 1, true); });
$('next').addEventListener('click', () => { setPlaying(false); go(state.t + 1, true); });
$('scrub').addEventListener('input', (e) => { setPlaying(false); go(Number(e.target.value), false); });
$('speed').addEventListener('change', (e) => scene.setSpeed(Number(e.target.value)));
$('layer').addEventListener('change', (e) => { state.layer = Number(e.target.value); setView(); });
$('world').addEventListener('change', (e) => {
  const value = e.target.value.trim();
  state.world = boundedInt(value === '' ? null : Math.round(Number(value)), state.world, 1, MAX_WORLD);
  e.target.value = state.world;
  setView();
});
$('open-file').addEventListener('click', () => $('file').click());
$('file').addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (file) open(file, file.name);
  e.target.value = '';
});
$('catalog').addEventListener('change', (e) => e.target.value && open(e.target.value, e.target.value));

// text fields keep every key; buttons, links and summaries keep Space, which activates them
const TEXT_FIELDS = new Set(['INPUT', 'SELECT', 'TEXTAREA']);
const ACTIVATED_BY_SPACE = new Set(['BUTTON', 'SUMMARY', 'A']);
document.addEventListener('keydown', (e) => {
  if (!state.trace || TEXT_FIELDS.has(e.target.tagName)) return;
  if (e.key === ' ' && !ACTIVATED_BY_SPACE.has(e.target.tagName)) { e.preventDefault(); setPlaying(!state.playing); }
  if (e.key === 'ArrowRight') { setPlaying(false); go(state.t + 1, true); }
  if (e.key === 'ArrowLeft') { setPlaying(false); go(state.t - 1, true); }
});
for (const type of ['dragover', 'drop']) {
  window.addEventListener(type, (e) => {
    e.preventDefault();
    document.body.classList.toggle('dragging', type === 'dragover');
    const file = type === 'drop' ? e.dataTransfer.files[0] : null;
    if (file) open(file, file.name);
  });
}
window.addEventListener('dragleave', () => document.body.classList.remove('dragging'));

async function boot() {
  const params = new URLSearchParams(location.search);
  state.world = boundedInt(params.get('world'), 8, 1, MAX_WORLD);
  $('world').value = state.world;
  let catalog = [];
  try {
    catalog = await (await fetch('traces/index.json')).json();
  } catch {
    catalog = [];
  }
  $('catalog').replaceChildren(...catalog.map((entry) => new Option(entry.title, entry.file)));
  if (loads > 0) return;  // the user opened a file while the catalog was loading
  const first = params.get('trace') ?? catalog[0]?.file;
  if (first) await open(first, first, { t: params.get('t'), layer: params.get('layer') });
  else showError(new TraceError('No trace given. Open a file, drop one here, or pass ?trace=<url>.'));
}

boot();
