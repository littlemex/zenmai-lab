// Loading, checking and indexing an engram-trace v1 file. No DOM and no three.js here,
// so the same module can back a different renderer.

export class TraceError extends Error {}

export async function loadTrace(source) {
  if (typeof File !== 'undefined' && source instanceof File) return parseTrace(await source.text(), source.name);
  const response = await fetch(source);
  if (!response.ok) throw new TraceError(`${source}: HTTP ${response.status}`);
  return parseTrace(await response.text(), String(source));
}

export function parseTrace(text, name = 'trace') {
  let trace;
  try {
    trace = JSON.parse(text);
  } catch (error) {
    throw new TraceError(`${name} is not JSON: ${error.message}`);
  }
  checkTrace(trace, name);
  return trace;
}

const isCount = (x) => Number.isSafeInteger(x) && x > 0;
const isIndex = (x) => Number.isSafeInteger(x) && x >= 0;

// Checks every property the viewer relies on, including the cross-field rules a JSON Schema
// cannot state: the columns are (n, head) for n = 2..max_ngram_size and head = 0..n_heads-1 in
// that order, they tile the table, every row lies in its column, and gates, when present, have
// one value in [0, 1] per position, layer and hc copy.
export function checkTrace(trace, name = 'trace') {
  const fail = (message) => {
    throw new TraceError(`${name}: ${message}`);
  };
  if (trace?.format !== 'engram-trace') fail('format is not "engram-trace"');
  if (trace.version !== 1) fail(`version ${trace.version} is not supported (this viewer reads 1)`);
  const { layers, tokens, hash_rows: rows, model, gates } = trace;
  if (!isCount(model?.n_heads) || !Number.isSafeInteger(model?.max_ngram_size) || model.max_ngram_size < 2) {
    fail('model.max_ngram_size (at least 2) and model.n_heads (at least 1) are required');
  }
  if (!Array.isArray(layers) || layers.length === 0) fail('layers is empty');
  if (!Array.isArray(tokens) || tokens.length === 0) fail('tokens is empty');
  tokens.forEach((token, t) => {
    if (!isIndex(token?.id) || typeof token.text !== 'string') fail(`tokens[${t}] needs an integer id and a text`);
    if (token.display != null && typeof token.display !== 'string') fail(`tokens[${t}].display must be a string`);
  });
  if (!Array.isArray(rows) || rows.length !== tokens.length) {
    fail(`hash_rows has ${rows?.length} positions, tokens has ${tokens.length}`);
  }

  // column c is (n, head) = (2 + floor(c / n_heads), c mod n_heads)
  const columnCount = (model.max_ngram_size - 1) * model.n_heads;
  if (!Number.isSafeInteger(columnCount)) fail('(max_ngram_size - 1) × n_heads is too large');
  const expectedAt = (c) => ({ n: 2 + Math.floor(c / model.n_heads), head: c % model.n_heads });
  layers.forEach((layer, l) => {
    if (!Number.isSafeInteger(layer?.layer_id) || !isCount(layer.num_rows) || !Array.isArray(layer.columns)) {
      fail(`layers[${l}] needs layer_id, num_rows and columns`);
    }
    if (layer.columns.length !== columnCount) {
      fail(`layer ${layer.layer_id} has ${layer.columns.length} columns, expected ${columnCount} ((max_ngram_size - 1) × n_heads)`);
    }
    let end = 0;
    layer.columns.forEach((column, c) => {
      if (column === null || typeof column !== 'object') fail(`layer ${layer.layer_id} column ${c} is not an object`);
      const expected = expectedAt(c);
      if (column.n !== expected.n || column.head !== expected.head) {
        fail(`layer ${layer.layer_id} column ${c} is (n=${column.n}, head=${column.head}), expected (n=${expected.n}, head=${expected.head})`);
      }
      if (!isCount(column.size) || column.offset !== end) {
        fail(`layer ${layer.layer_id} column ${c} must start at ${end} and have a positive size`);
      }
      end += column.size;
    });
    if (end !== layer.num_rows) fail(`layer ${layer.layer_id} columns cover ${end} of ${layer.num_rows} rows`);
  });
  rows.forEach((perLayer, t) => {
    if (!Array.isArray(perLayer) || perLayer.length !== layers.length) {
      fail(`hash_rows[${t}] has ${perLayer?.length} layers, expected ${layers.length}`);
    }
    perLayer.forEach((ids, l) => {
      if (!Array.isArray(ids) || ids.length !== columnCount) {
        fail(`hash_rows[${t}][${l}] has ${ids?.length} rows, expected ${columnCount}`);
      }
      ids.forEach((row, c) => {
        const { offset, size } = layers[l].columns[c];
        if (!isIndex(row) || row < offset || row >= offset + size) {
          fail(`hash_rows[${t}][${l}][${c}] = ${row} is outside column ${c} [${offset}, ${offset + size})`);
        }
      });
    });
  });
  if (gates != null) {
    if (!isCount(model.hc_mult)) fail('gates need model.hc_mult');
    if (!Array.isArray(gates) || gates.length !== tokens.length) fail(`gates has ${gates?.length} positions, tokens has ${tokens.length}`);
    gates.forEach((perLayer, t) => {
      if (!Array.isArray(perLayer) || perLayer.length !== layers.length) fail(`gates[${t}] has ${perLayer?.length} layers, expected ${layers.length}`);
      perLayer.forEach((values, l) => {
        if (!Array.isArray(values) || values.length !== model.hc_mult || !values.every((g) => Number.isFinite(g) && g >= 0 && g <= 1)) {
          fail(`gates[${t}][${l}] must hold ${model.hc_mult} values in [0, 1]`);
        }
      });
    });
  }
}

// The positions of the n tokens an n-gram ending at t covers; negative positions are the
// padding the model uses before the start of the sequence.
export function ngramPositions(t, n) {
  return Array.from({ length: n }, (_, i) => t - n + 1 + i);
}

// The identity the hash sees: compressed ids, with the model's pad id before the start. Without
// them two different raw ids can be the same n-gram to the model, so no identity is claimed.
function ngramKey(trace, t, n) {
  const pad = trace.model.pad_compressed_id;
  const ids = ngramPositions(t, n).map((p) => (p < 0 ? pad : trace.tokens[p].compressed_id));
  return ids.every((id) => Number.isSafeInteger(id)) ? ids.join(',') : null;
}

// For every lookup: the first position that read the same row, and whether that row is read for
// a different n-gram somewhere in the trace (a hash collision); collision is null when the trace
// lacks the compressed ids needed to tell.
export function indexTrace(trace) {
  return trace.layers.map((layer, l) => {
    const firstSeen = new Map();
    const keysPerRow = new Map();
    const lookups = trace.hash_rows.map((perLayer, t) =>
      perLayer[l].map((row, c) => {
        const key = ngramKey(trace, t, layer.columns[c].n);
        if (!keysPerRow.has(row)) keysPerRow.set(row, new Set());
        keysPerRow.get(row).add(key);
        if (!firstSeen.has(row)) firstSeen.set(row, t);
        return { row, column: c, firstSeen: firstSeen.get(row) };
      }),
    );
    let reused = 0;
    let collisions = 0;
    lookups.forEach((perPos, t) => {
      for (const lookup of perPos) {
        const keys = keysPerRow.get(lookup.row);
        lookup.collision = keys.has(null) ? null : keys.size > 1;
        if (lookup.firstSeen < t) reused += 1;
        if (lookup.collision) collisions += 1;
      }
    });
    const known = !lookups.some((perPos) => perPos.some((x) => x.collision === null));
    return { lookups, total: lookups.length * layer.columns.length, reused, collisions: known ? collisions : null };
  });
}

// Which rank holds a row when the table is split into equal contiguous blocks, as the model's
// ParallelEngramEmbedding and DeepEP's Engram buffer both split it.
export function ownerOf(row, numRows, worldSize) {
  return Math.floor(row / Math.ceil(numRows / worldSize));
}

export function tokenLabel(token) {
  return (token.display ?? token.text).replace(/^ /, '·').replace(/\n/g, '↵');
}
