// The ladder view. For each position of a text, the block from which a model's top-1 next-token
// prediction stops changing ("settles"), and how many blocks run above it, compared between two
// variants; and, for one position, what every block would predict.

import { loadTrace, TraceError, tokenLabel } from './trace.js';

const $ = (id) => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const params = new URLSearchParams(location.search);
const state = { trace: null, t: 0, a: 'engram', b: 'baseline', playing: false, timer: null };
const STEP_MS = 1400;
// chart geometry: one column per position
const COL = 26, LEFT = 46, TOP = 12, HEIGHT = 200, BOTTOM = 70;
// colour by what a variant does where Engram would be added
const COLORS = { added: '#3bd16f', skipped: '#9aa5b1', absent: '#2f9bff' };

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'style') for (const [name, value] of Object.entries(v)) node.style.setProperty(name, value);
    else node.setAttribute(k, v);
  }
  for (const child of children.flat(Infinity)) if (child != null) node.append(child instanceof Node ? child : String(child));
  return node;
}

function svg(tag, attrs = {}, ...children) {
  const node = document.createElementNS(SVG, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  for (const child of children) node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  return node;
}

const word = (id) => (state.trace.lens.vocab[id] ?? `#${id}`).replace(/^ /, '·').replace(/\n/g, '↵');
const pct = (p) => `${(100 * p).toFixed(p >= 0.1 ? 0 : 1)} %`;
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / Math.max(xs.length, 1);

// The block from which a variant's top-1 prediction equals its final one and stays equal.
export function settledBlock(blocks) {
  const final = blocks[blocks.length - 1].top[0][0];
  let k = blocks.length - 1;
  while (k > 0 && blocks[k - 1].top[0][0] === final) k -= 1;
  return k;
}

function checkLens(trace, name) {
  const lens = trace.lens;
  if (!lens) throw new TraceError(`${name}: this trace has no "lens" field; the ladder view needs one (the table view opens it)`);
  if (!Number.isInteger(lens.n_layer) || lens.n_layer < 1) throw new TraceError(`${name}: lens.n_layer must be a positive integer`);
  const names = (lens.variants ?? []).map((v) => v.name);
  if (!names.length || new Set(names).size !== names.length) throw new TraceError(`${name}: lens.variants must name each variant once`);
  for (const variant of lens.variants) {
    const values = lens.values?.[variant.name];
    if (!Array.isArray(values) || values.length !== trace.tokens.length) {
      throw new TraceError(`${name}: lens.values.${variant.name} has ${values?.length} positions, tokens has ${trace.tokens.length}`);
    }
    values.forEach((blocks, t) => {
      if (!Array.isArray(blocks) || blocks.length !== lens.n_layer) {
        throw new TraceError(`${name}: lens.values.${variant.name}[${t}] has ${blocks?.length} blocks, expected ${lens.n_layer}`);
      }
      blocks.forEach((cell, b) => {
        const where = `${name}: lens.values.${variant.name}[${t}][${b}]`;
        const [id, p] = cell?.top?.[0] ?? [];
        if (!Number.isInteger(id) || !(String(id) in (lens.vocab ?? {}))) throw new TraceError(`${where} has no top-1 token with text in lens.vocab`);
        if (![p, cell.p_final].every((x) => Number.isFinite(x) && x >= 0 && x <= 1)) {
          throw new TraceError(`${where} needs a top-1 probability and p_final in [0, 1]`);
        }
      });
    });
  }
}

const variant = (name) => state.trace.lens.variants.find((v) => v.name === name) ?? { name };
const label = (name) => variant(name).label ?? name;
const color = (name) => COLORS[variant(name).engram] ?? '#ccc';
// highest number of blocks above the settled one, never zero so it can divide
const span = () => Math.max(1, state.trace.lens.n_layer - 1);
// positions that have a next token in the text; the last one has none
const scored = () => state.trace.tokens.length - 1;
function settled(name) {
  return state.trace.lens.values[name].map(settledBlock);
}

function buildPickers() {
  for (const id of ['a', 'b']) {
    $(id).replaceChildren(...state.trace.lens.variants.map((v) => new Option(label(v.name), v.name)));
    $(id).value = state[id];
  }
}

// The text, each token shaded by how many blocks sooner variant a settles than variant b.
function renderText() {
  const { trace } = state;
  const sa = settled(state.a), sb = settled(state.b);
  const prompt = trace.generation?.prompt_tokens ?? trace.tokens.length;
  $('text').replaceChildren(...trace.tokens.map((token, t) => {
    if (t >= scored()) return el('span', { class: `tok last${t < prompt ? ' prompt' : ''}` }, tokenLabel(token));
    const gain = sb[t] - sa[t];
    return el('span', {
      class: `tok${t < prompt ? ' prompt' : ''}`, 'data-t': t,
      style: { '--gain': String(Math.max(0, gain) / span()), '--loss': String(Math.max(0, -gain) / span()) },
      title: `settles at block ${sa[t]} vs ${sb[t]}`,
    }, tokenLabel(token));
  }));
}

// Spare blocks (blocks above the settled one) per position, one bar per variant.
function renderChart() {
  const { trace } = state;
  const n = scored();
  const blocks = trace.lens.n_layer;
  const W = COL, H = HEIGHT, left = LEFT, top = TOP;
  const width = left + n * W + 10;
  const y = (spare) => top + H - (spare / span()) * H;
  const chart = svg('svg', { width, height: top + H + BOTTOM, class: 'chart' });
  for (let s = 0; s <= blocks - 1; s += 1) {
    chart.append(svg('line', { x1: left, x2: width - 10, y1: y(s), y2: y(s), class: s % 5 === 0 ? 'grid major' : 'grid' }));
    if (s % 5 === 0) chart.append(svg('text', { x: left - 8, y: y(s) + 4, class: 'axis', 'text-anchor': 'end' }, s));
  }
  chart.append(svg('text', { x: 12, y: top + H / 2, class: 'axis', transform: `rotate(-90 12 ${top + H / 2})`, 'text-anchor': 'middle' },
    'blocks above the settled one'));
  const sa = settled(state.a), sb = settled(state.b);
  for (let t = 0; t < n; t += 1) {
    const x = left + t * W;
    const group = svg('g', { class: 'col', 'data-t': t });
    group.append(svg('rect', { x, y: top, width: W, height: H + BOTTOM - 4, class: 'hit' }));
    [[state.a, sa[t], 3], [state.b, sb[t], 13]].forEach(([name, s, dx]) => {
      const spare = blocks - 1 - s;
      group.append(svg('rect', { x: x + dx, y: y(spare), width: 9, height: Math.max(1, top + H - y(spare)),
        fill: color(name), class: 'bar' }));
    });
    const text = tokenLabel(trace.tokens[t]).slice(0, 10);
    group.append(svg('text', { x: x + W / 2, y: top + H + 10, class: 'tick', transform: `rotate(-60 ${x + W / 2} ${top + H + 10})`,
      'text-anchor': 'end' }, text));
    chart.append(group);
  }
  chart.append(svg('rect', { id: 'cursor', x: left, y: top - 4, width: W, height: H + 8, class: 'cursor' }));
  chart.onclick = (e) => {
    const t = e.target.closest('.col')?.dataset.t;
    if (t != null) { setPlaying(false); go(Number(t)); }
  };
  $('chart').replaceChildren(chart);
  $('chart-legend').replaceChildren(
    ...[state.a, state.b].map((name) => el('span', { class: 'key' }, el('span', { class: 'swatch', style: { background: color(name) } }), label(name))));
}

// Counts over a set of positions: where variant a settles sooner, later, or at the same block as b.
export function compare(sa, sb, positions, blocks) {
  const sooner = positions.filter((t) => sa[t] < sb[t]).length;
  const later = positions.filter((t) => sa[t] > sb[t]).length;
  const above = (s) => mean(positions.map((t) => blocks - 1 - s[t]));
  return { n: positions.length, sooner, later, same: positions.length - sooner - later, aboveA: above(sa), aboveB: above(sb) };
}

function renderSummary() {
  const { trace } = state;
  const { n_layer: blocks, values } = trace.lens;
  const sa = settled(state.a), sb = settled(state.b);
  const all = [...Array(scored()).keys()];
  // both variants end on the token that actually comes next, so they settle on the same, right answer
  const finalTop = (name, t) => values[name][t][blocks - 1].top[0][0];
  const right = all.filter((t) => finalTop(state.a, t) === trace.tokens[t + 1].id && finalTop(state.b, t) === trace.tokens[t + 1].id);
  const row = (title, c) => el('div', { class: 'stat' }, el('span', { class: 'muted' }, `${title} (${c.n}): `),
    ...(c.n === 0 ? ['no such positions'] : [`${label(state.a)} settles sooner at `, el('b', {}, `${c.sooner}`), ', later at ', el('b', {}, `${c.later}`), ', same at ',
    el('b', {}, `${c.same}`), '; blocks above the settled one, on average ', el('b', {}, c.aboveA.toFixed(2)), ' against ',
    el('b', {}, c.aboveB.toFixed(2))]));
  $('summary').replaceChildren(
    row('all positions', compare(sa, sb, all, blocks)),
    row('both end on the actual next token', compare(sa, sb, right, blocks)),
  );
}

// One position: what every block predicts, for the two variants side by side.
function renderLadder(animate) {
  const { trace, t } = state;
  const lens = trace.lens;
  const next = trace.tokens[t + 1];
  const prompt = trace.generation?.prompt_tokens ?? trace.tokens.length;
  $('focus').replaceChildren(
    el('span', { class: 'muted' }, `position ${t} · ${t < prompt ? 'prompt' : `generated by ${label(trace.generation?.by ?? '')}`}`),
    el('div', { class: 'focus-text' }, el('span', { class: 'focus-tok' }, tokenLabel(trace.tokens[t])), ' → ',
      el('span', { class: 'focus-next' }, tokenLabel(next))),
  );
  const columns = [state.a, state.b].map((name) => {
    const blocks = lens.values[name][t];
    const s = settledBlock(blocks);
    const final = blocks[lens.n_layer - 1].top[0];
    const engram = variant(name).engram;
    const cells = [];
    for (let block = lens.n_layer - 1; block >= 0; block -= 1) {
      const cell = blocks[block];
      cells.push(el('div', {
        class: `rung${block >= s ? ' settled' : ''}${block === s ? ' settles' : ''}`,
        style: { '--p': String(cell.p_final), '--delay': `${animate ? block * 70 : 0}ms` },
      },
      el('span', { class: 'rung-block' }, block),
      el('span', { class: 'rung-word' }, word(cell.top[0][0])),
      el('span', { class: 'rung-p' }, pct(cell.top[0][1])),
      el('span', { class: 'rung-bar' })));
      if ((lens.engram_before_block ?? []).includes(block)) {
        const layer = trace.layers.findIndex((l) => l.layer_id === block);
        // one gate per hc copy; the band fills by their mean
        const copies = engram === 'added' ? trace.gates?.[t]?.[layer] : null;
        const gate = copies?.length ? mean(copies) : null;
        const text = engram === 'added' ? `Engram added${gate == null ? '' : `, gate ${copies.map((g) => g.toFixed(2)).join(' / ')}`}`
          : engram === 'skipped' ? 'Engram skipped' : 'no Engram';
        cells.push(el('div', { class: `rung-engram${engram === 'added' ? '' : ' off'}`, style: { '--g': String(gate ?? 1) } }, text));
      }
    }
    return el('div', { class: 'ladder-col' },
      el('div', { class: 'ladder-head-col', style: { color: color(name) } }, label(name)),
      el('div', { class: 'muted small' }, 'final ', el('b', {}, word(final[0])), ` ${pct(final[1])} · settles at block ${s}`),
      el('div', { class: `rungs${animate ? ' climb' : ''}` }, cells));
  });
  $('ladder').replaceChildren(...columns);
}

function go(t, animate = true) {
  if (!state.trace) return;
  state.t = Math.max(0, Math.min(t, scored() - 1));
  $('where').textContent = `${state.t} / ${scored() - 1}`;
  $('scrub').value = state.t;
  document.querySelectorAll('.tok.current').forEach((n) => n.classList.remove('current'));
  document.querySelector(`.tok[data-t="${state.t}"]`)?.classList.add('current');
  const cursor = document.getElementById('cursor');
  if (cursor) cursor.setAttribute('x', LEFT + state.t * COL);
  const scroller = $('chart');
  const target = LEFT + state.t * COL - scroller.clientWidth / 2;
  scroller.scrollTo({ left: Math.max(0, target), behavior: 'smooth' });
  renderLadder(animate);
  const url = new URL(location.href);
  url.searchParams.set('t', state.t);
  url.searchParams.set('a', state.a);
  url.searchParams.set('b', state.b);
  history.replaceState(null, '', url);
}

function setPlaying(playing) {
  state.playing = playing && state.trace != null;
  $('play').classList.toggle('playing', state.playing);
  clearInterval(state.timer);
  if (state.playing) {
    if (state.t >= scored() - 1) go(0);
    else go(state.t + 1);
    state.timer = setInterval(() => (state.t >= scored() - 1 ? setPlaying(false) : go(state.t + 1)), STEP_MS);
  }
}

function render() {
  renderText();
  renderChart();
  renderSummary();
  go(state.t, false);
}

async function open(source) {
  try {
    const trace = await loadTrace(source);
    checkLens(trace, String(source));
    if (trace.tokens.length < 2) throw new TraceError(`${source}: the ladder needs at least two tokens`);
    state.trace = trace;
    const names = trace.lens.variants.map((v) => v.name);
    state.a = names.includes(params.get('a')) ? params.get('a') : names[0];
    state.b = names.includes(params.get('b')) ? params.get('b') : names[names.length - 1];
    const t = Number(params.get('t'));
    state.t = Number.isInteger(t) ? t : 0;
    $('title').textContent = trace.title || 'Untitled trace';
    $('scrub').max = scored() - 1;
    $('error').hidden = true;
    const tableView = new URL('index.html', location.href);
    tableView.searchParams.set('trace', source);
    $('table-link').href = tableView.href;
    buildPickers();
    render();
  } catch (error) {
    $('error').textContent = error instanceof TraceError ? error.message : `Could not open the trace: ${error.message}`;
    $('error').hidden = false;
  }
}

$('a').addEventListener('change', (e) => { state.a = e.target.value; render(); });
$('b').addEventListener('change', (e) => { state.b = e.target.value; render(); });
$('text').addEventListener('click', (e) => {
  const t = e.target.dataset?.t;
  if (t != null) { setPlaying(false); go(Number(t)); }
});
$('play').addEventListener('click', () => setPlaying(!state.playing));
$('prev').addEventListener('click', () => { setPlaying(false); go(state.t - 1); });
$('next').addEventListener('click', () => { setPlaying(false); go(state.t + 1); });
$('scrub').addEventListener('input', (e) => { setPlaying(false); go(Number(e.target.value), false); });
document.addEventListener('keydown', (e) => {
  if (!state.trace || ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === ' ' && !['BUTTON', 'SUMMARY', 'A'].includes(e.target.tagName)) { e.preventDefault(); setPlaying(!state.playing); }
  if (e.key === 'ArrowRight') { setPlaying(false); go(state.t + 1); }
  if (e.key === 'ArrowLeft') { setPlaying(false); go(state.t - 1); }
});

open(params.get('trace') ?? 'traces/nanogpt-hastings.json');
