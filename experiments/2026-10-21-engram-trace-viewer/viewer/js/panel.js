// The side panel: what the current position reads, and where each row lives. DOM only.

import { ngramColor, rankColor } from './colors.js';
import { ngramPositions, ownerOf, tokenLabel } from './trace.js';

const fmt = (x) => x.toLocaleString('en-US');

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'style') for (const [name, value] of Object.entries(v)) node.style.setProperty(name, value);
    else if (k === 'class') node.className = v;
    else node.setAttribute(k, v);
  }
  for (const child of children.flat(Infinity)) {
    if (child != null) node.append(child instanceof Node ? child : String(child));
  }
  return node;
}

function chip(text, color) {
  return el('span', { class: 'chip', style: { '--chip': color } }, text);
}

function ngramText(trace, t, n) {
  return ngramPositions(t, n)
    .map((p) => (p < 0 ? '∅' : tokenLabel(trace.tokens[p])))
    .join(' ');
}

export function renderHeader(root, trace) {
  root.replaceChildren();
  const { source = {}, provenance = {} } = trace;
  const revision = source.revision ? source.revision.slice(0, 12) : 'unpinned';
  const link = source.model
    ? el('a', { href: `https://huggingface.co/${encodeURI(source.model)}/tree/${encodeURIComponent(source.revision ?? 'main')}`, target: '_blank', rel: 'noopener' },
        `${source.model}@${revision}`)
    : 'unknown source';
  // whether a field is in the file, and the producer's own sentence on how it was obtained
  const fields = [
    ['hash rows', true, provenance.hash_rows],
    ['gates', trace.gates != null, provenance.gates],
  ];
  root.append(
    el('h2', {}, trace.title || 'Untitled trace'),
    el('p', { class: 'muted' }, link, ` · ${trace.tokens.length} tokens · ${trace.layers.length} Engram layers`),
    el('div', { class: 'provenance' },
      fields.map(([name, present]) => el('span', { class: present ? 'badge on' : 'badge off' }, `${name}: ${present ? 'present' : 'absent'}`))),
    el('dl', { class: 'provenance-notes' },
      fields.filter(([, present, note]) => present && note).map(([name, , note]) => [el('dt', {}, name), el('dd', {}, note)])),
  );
}

export function renderPosition(root, trace, index, t, layerIndex, worldSize) {
  const layer = trace.layers[layerIndex];
  const lookups = index[layerIndex].lookups[t];
  const token = trace.tokens[t];
  root.replaceChildren();

  root.append(
    el('div', { class: 'position' },
      el('span', { class: 'muted' }, `token ${t + 1} of ${trace.tokens.length} · position ${t}`),
      el('div', { class: 'token' }, tokenLabel(token)),
      el('span', { class: 'muted' }, `token id ${fmt(token.id)}`,
        token.compressed_id != null ? ` · compressed id ${fmt(token.compressed_id)}` : '')),
  );

  const ngrams = el('div', { class: 'ngrams' });
  for (let n = 2; n <= trace.model.max_ngram_size; n += 1) {
    ngrams.append(el('div', { class: 'ngram' },
      el('span', { class: 'dot', style: { background: ngramColor(n) } }),
      el('b', {}, `${n}-gram`),
      el('span', { class: 'ngram-text' }, ngramText(trace, t, n))));
  }
  root.append(el('h3', {}, 'N-grams ending here'), ngrams);

  if (trace.gates) {
    const values = trace.gates[t][layerIndex];
    root.append(el('h3', {}, 'Gate per hc copy, 0 to 1'),
      el('div', { class: 'gates' }, values.map((g, k) => el('div', { class: 'gate' },
        el('span', {}, `copy ${k}`),
        el('span', { class: 'gate-bar', style: { '--g': String(g) } }),
        el('span', { class: 'num' }, g.toFixed(3))))));
  }

  const share = (100 * lookups.length) / layer.num_rows;
  root.append(el('h3', {}, `Rows read: ${lookups.length} of ${fmt(layer.num_rows)} `,
    el('span', { class: 'muted' }, `(${share.toPrecision(2)} %)`)));

  const table = el('table', { class: 'rows' },
    el('thead', {}, el('tr', {}, ['n', 'head', 'row', 'owner', 'first read'].map((h) => el('th', {}, h)))));
  const body = el('tbody');
  lookups.forEach(({ row, column, firstSeen, collision }) => {
    const { n, head } = layer.columns[column];
    const owner = ownerOf(row, layer.num_rows, worldSize);
    const reused = firstSeen < t;
    body.append(el('tr', { class: reused ? 'reused' : '' },
      el('td', {}, el('span', { class: 'dot', style: { background: ngramColor(n) } }), n),
      el('td', {}, head),
      el('td', { class: 'num' }, fmt(row)),
      el('td', {}, chip(`rank ${owner}`, rankColor(owner, worldSize))),
      el('td', {}, reused ? `#${firstSeen}` : 'here',
        collision ? el('span', { class: 'flag', title: 'a different n-gram in this trace reads the same row' }, ' shared') : '')));
  });
  table.append(body);
  root.append(table);

  const owners = new Map();
  for (const { row } of lookups) {
    const owner = ownerOf(row, layer.num_rows, worldSize);
    owners.set(owner, (owners.get(owner) ?? 0) + 1);
  }
  root.append(el('h3', {}, `Where the ${lookups.length} rows live, at world size ${worldSize}`),
    el('div', { class: 'owners' }, [...owners.entries()].sort((a, b) => a[0] - b[0]).map(([owner, count]) =>
      chip(`rank ${owner} × ${count}`, rankColor(owner, worldSize)))),
    el('p', { class: 'note' },
      'Owner = row ÷ ⌈rows ÷ world size⌉, the contiguous split of the reference ParallelEngramEmbedding ',
      'and of the DeepEP Engram buffer. It is computed from the row id, not measured.'));

  const stats = index[layerIndex];
  const collisions = stats.collisions == null
    ? 'Collisions cannot be told apart from reuse here, because the trace lacks compressed ids or model.pad_compressed_id.'
    : `${fmt(stats.collisions)} lookups read a row that a different n-gram also reads (a hash collision).`;
  root.append(el('h3', {}, 'Whole trace, this layer'),
    el('p', { class: 'note' },
      `${fmt(stats.reused)} of ${fmt(stats.total)} lookups read a row that an earlier position already read. `, collisions));
  if (!trace.gates) {
    root.append(el('p', { class: 'note muted' },
      'Gate values are absent. They need the hidden state entering the Engram layer, so only a run of the full model produces them.'));
  }
}
