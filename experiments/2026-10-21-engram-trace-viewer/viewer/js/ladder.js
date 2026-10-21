// The ladder view: what happens at one position of a text, step by step, for the model with Engram
// and a model to compare it with. What the model has read, what the Engram lookup fetches and how
// much of it is added, what each block would pick as the next word with how much probability, and
// the word written at the end. Beside that, every position of the text: the block from which each
// model's top choice stops changing ("settles").

import { loadTrace, TraceError } from './trace.js';

const $ = (id) => document.getElementById(id);
const SVG = 'http://www.w3.org/2000/svg';
const params = new URLSearchParams(location.search);
const state = { trace: null, t: 0, a: null, b: null, block: 0, lang: 'en', timer: null };
const STEP_MS = 650;
const COLORS = { added: 'var(--engram)', skipped: 'var(--skipped)', absent: 'var(--baseline)' };
// chart geometry: one column per position; the block axis is a separate, unscrolled SVG
const COL = 24, LEFT = 6, AXIS = 40, TOP = 14, HEIGHT = 190, BOTTOM = 64;

const TEXT = {
  en: {
    title: 'Engram ladder',
    compareWith: 'Compare with',
    tableView: 'row ids view',
    lang: '日本語',
    headline: (n, sooner, later, b) => `With Engram, the top choice locks in at an earlier block than ${b} at <b>${sooner}</b> of ${n} positions, and at a later one at ${later}.`,
    intro: 'A language model writes one token at a time; a token is a word or a piece of one ("Sax" and "-on" make "Saxon"). Inside, the text passes through 12 blocks, from block 0 at the bottom to block 11 at the top, and after every block we can read which token the model would pick next. Engram adds a lookup before block 1: using the last two or three tokens as a key, it fetches numbers the model stored for that pattern in a large table and mixes them in. It does not fetch the next word itself. Click a token in the text to see what happened there.',
    textLegend: '<span class="swatch" style="background:var(--engram)"></span>with Engram the top choice locked in at an earlier block · <span class="swatch" style="background:var(--error)"></span>underline: at a later block · grey: the opening we gave · white: what the model with Engram wrote after it, always taking its top choice',
    chartTitle: 'At which block did the top choice lock in?',
    chartNote: 'One column per token. Each dot is the block from which that model\'s top choice no longer changes, found by looking at all blocks afterwards; lower is earlier. The dashed line is where the Engram lookup is added.',
    engramLine: 'Engram added',
    blockAxis: 'block',
    moreTitle: 'Details',
    more1: 'Each block\'s guess is read by passing its state through the model\'s last layer (a method called LogitLens). An answer is fixed at the first block from which the top choice equals the top choice of the last block and stays the same; the blocks above still run and still change how sure the model is. A word shown with a leading "-" continues the piece before it ("Sax" and "-on" make "Saxon"); "@-@" is how this dataset writes a hyphen.',
    more2: 'This page shows one text and one training seed, and the white part was written by the model with Engram, which favours it; a model this small also tends to repeat itself. Over the whole validation set and three seeds, 14.5% of positions are fixed at block 1 with Engram against 4.8% without it; see the experiment\'s README.',
    countsAll: (c, b) => `All positions (${c.n}), against ${b}: sooner at ${c.sooner}, later at ${c.later}, same at ${c.same}.`,
    countsRight: (c) => `Where both end on the token that actually comes next (${c.n}): sooner at ${c.sooner}, later at ${c.later}, same at ${c.same}.`,
    where: (t, n, kind) => `token ${t + 1} of ${n} · ${kind}`,
    prompt: 'in the prompt',
    generated: 'written by the model with Engram',
    prevPos: 'Previous token', nextPos: 'Next token', block: 'Block',
    step1: 'Step 1. What it has read',
    question: 'What comes after this?',
    actualNext: 'The token that actually comes next:',
    writtenNext: 'The token the model with Engram wrote next:',
    step2: 'Step 2. Engram looks it up',
    lookupAdded: (rows, total) => ['Using ', ` as keys, the model with Engram reads ${rows} rows from its table of ${total} rows, and adds what it finds before block 1.`],
    gate: 'How strongly it is mixed in (the model decides; this is not a word probability):',
    effect: (k) => `What the lookup changes, at block ${k} of the same model:`,
    effectOff: 'without the lookup',
    effectOn: 'with the lookup',
    effectRow: (w, p, next, q) => `top choice <b>${w}</b> (${p}) · the actual next token ${next}: ${q}`,
    lookupAbsent: () => 'This model was trained without a table. It has to work the same thing out with its blocks.',
    lookupSkipped: () => 'The same trained model with only the lookup turned off.',
    step3: 'Step 3. Block by block: which token would it pick?',
    step3Note: 'Move the slider or press play to go up the blocks. The bars are the five tokens each model finds most likely next at that block, and how sure it is of each; "other tokens" is everything else.',
    play: 'Play', pause: 'Pause',
    blockLabel: (k) => `block ${k}`,
    others: 'other tokens',
    nextLine: (w, p) => `actual next ${w}: ${p}`,
    top: (w, p) => `Top choice: <b>${w}</b> (${p})`,
    fixedFrom: (k) => `locked in from block ${k}`,
    notYet: 'changes later',
    beforeLookup: 'block 0 runs before the lookup',
    curveTitle: 'How sure is each model of the token that actually comes next, block by block?',
    step4: 'Step 4. The token it writes',
    step4Note: 'At the last block each model writes its top choice.',
    writes: (w, p, k) => `writes <b>${w}</b> (${p}), its top choice from block ${k} on`,
  },
  ja: {
    title: 'Engram の働きを見る',
    // the trace's variant labels are English; these replace them by what each variant does with Engram
    variants: { added: 'Engram あり', absent: 'Engram なしで学習したモデル', skipped: '表引きだけを止めた同じモデル' },
    compareWith: '比べる相手',
    tableView: '行番号の一覧',
    lang: 'English',
    headline: (n, sooner, later, b) => `Engram ありのモデルは、${n} か所のうち <b>${sooner}</b> か所で、${b}より早い段で 1 位の候補が固まりました。遅かったのは ${later} か所です。`,
    intro: '言語モデルは文章をトークンという単位で 1 つずつ書きます。トークンは語か、語のかけらです。たとえば「Sax」と「-on」で「Saxon」になります。モデルの中では、文章が 0 段目から 11 段目までの 12 段の計算を下から順に通ります。どの段のあとでも、次のトークンにどれを選びそうかを読み取れます。Engram は 1 段目の前に表引きを足す仕組みです。直前の 2〜3 トークンを合図にして、大きな表にその並びについて覚えておいた数字を取り出し、混ぜ込みます。次の語そのものを取り出すわけではありません。文章のトークンをクリックすると、そこで何が起きたかが右に出ます。',
    textLegend: '<span class="swatch" style="background:var(--engram)"></span>緑: Engram ありの方が早い段で 1 位が固まったトークン · <span class="swatch" style="background:var(--error)"></span>下線: 遅い段で固まったトークン · 灰色: 最初に与えた書き出し · 白: その続きを Engram ありのモデルが書いた部分 (毎回 1 位を選んで書いたもの)',
    chartTitle: '1 位の候補は何段目で固まった？',
    chartNote: '1 トークンにつき 1 列です。点は、そのモデルの 1 位の候補がそれ以上変わらなくなった段で、全部の段を見たあとで調べたものです。下ほど早く固まっています。点線は Engram の表引きを足す場所です。',
    engramLine: 'Engram を足す',
    blockAxis: '段',
    moreTitle: 'くわしく',
    more1: '各段の候補は、その段の途中の状態をモデルの最後の層に通して読み出しています。LogitLens という方法です。1 位が固まった段とは、1 位の候補が最後の段の 1 位と同じになり、そのあと変わらなくなった最初の段のことです。その段で計算が止まるわけではなく、上の段も動いていて、どれくらい自信があるかは変わり続けます。トークンの頭に「-」が付いているのは前のトークンの続きです。「@-@」はこのデータでのハイフンの書き方です。',
    more2: 'このページは文章 1 つ、学習 1 回分の結果です。白い部分は Engram ありのモデル自身が書いたので、そのモデルに有利です。小さなモデルなので同じ言い回しをくり返しています。検証用の文章全体を学習 3 回分で数えると、1 段目で答えが決まる位置は Engram ありで 14.5 %、なしで 4.8 % でした。くわしくは実験の README にあります。',
    countsAll: (c, b) => `全位置 (${c.n} か所) で${b}と比べると、早い ${c.sooner}、遅い ${c.later}、同じ ${c.same}。`,
    countsRight: (c) => `両方とも実際の次のトークンで終わった位置 (${c.n} か所) では、早い ${c.sooner}、遅い ${c.later}、同じ ${c.same}。`,
    where: (t, n, kind) => `${n} トークン中 ${t + 1} 番目 · ${kind}`,
    prompt: '書き出しの部分',
    generated: 'Engram ありのモデルが書いた部分',
    prevPos: '前のトークン', nextPos: '次のトークン', block: '段',
    step1: 'ステップ 1　ここまで読んだ文',
    question: 'この次に来るトークンは？',
    actualNext: '実際に次に来たトークン:',
    writtenNext: 'Engram ありのモデルが次に書いたトークン:',
    step2: 'ステップ 2　Engram が表を引く',
    lookupAdded: (rows, total) => ['Engram ありのモデルは ', ` を合図にして、${total} 行ある表から ${rows} 行を読み出し、1 段目の前に足します。`],
    gate: '混ぜる強さ (モデル自身が決めます。次の語の確率とは別です):',
    effect: (k) => `表引きで何が変わるか (同じモデルの ${k} 段目):`,
    effectOff: '表引きなし',
    effectOn: '表引きあり',
    effectRow: (w, p, next, q) => `1 位 <b>${w}</b> (${p}) · 実際の次 ${next}: ${q}`,
    lookupAbsent: () => 'このモデルは表なしで学習しました。同じことを段の計算だけで組み立てる必要があります。',
    lookupSkipped: () => '同じモデルのまま、表引きだけを止めたものです。',
    step3: 'ステップ 3　1 段ずつ: どのトークンを選びそう？',
    step3Note: 'スライダーを動かすか再生を押すと、段を上がっていきます。棒は、その段でそれぞれのモデルが次に来そうだと考えた上位 5 トークンと、それぞれへの自信 (確率) です。「その他」は残り全部です。',
    play: '再生', pause: '停止',
    blockLabel: (k) => `${k} 段目`,
    others: 'その他',
    nextLine: (w, p) => `実際の次 ${w}: ${p}`,
    top: (w, p) => `1 位: <b>${w}</b> (${p})`,
    fixedFrom: (k) => `${k} 段目から 1 位は固まったまま`,
    notYet: 'このあと 1 位が変わる',
    beforeLookup: '0 段目は表引きより前',
    curveTitle: '実際に次に来たトークンへの自信は、段ごとにどう変わったか',
    step4: 'ステップ 4　書くトークン',
    step4Note: '最後の段で 1 位のトークンを書きます。',
    writes: (w, p, k) => `<b>${w}</b> を書く (${p})。${k} 段目から 1 位は変わっていません`,
  },
};
const tr = (key, ...args) => {
  const value = TEXT[state.lang][key] ?? TEXT.en[key];
  return typeof value === 'function' ? value(...args) : value;
};

function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === 'class') node.className = v;
    else if (k === 'html') node.innerHTML = v;
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

// A token as a reader sees it: a piece that continues the previous one starts with "-".
const display = (text) => (text.startsWith(' ') ? text.slice(1) : `-${text}`).replace(/\n/g, '↵');
const word = (id) => display(state.trace.lens.vocab[id] ?? `#${id}`);
const raw = (token) => (token.display ?? token.text).replace(/\n/g, '↵');
const pct = (p) => `${(100 * p).toFixed(p >= 0.1 ? 0 : 1)} %`;
const mean = (xs) => xs.reduce((s, x) => s + x, 0) / Math.max(xs.length, 1);

// The block from which a variant's top-1 prediction equals its final one and stays equal.
export function settledBlock(blocks) {
  const final = blocks[blocks.length - 1].top[0][0];
  let k = blocks.length - 1;
  while (k > 0 && blocks[k - 1].top[0][0] === final) k -= 1;
  return k;
}

// Counts over a set of positions: where variant a settles sooner, later, or at the same block as b.
export function compare(sa, sb, positions) {
  const sooner = positions.filter((t) => sa[t] < sb[t]).length;
  const later = positions.filter((t) => sa[t] > sb[t]).length;
  return { n: positions.length, sooner, later, same: positions.length - sooner - later };
}

function checkLens(trace, name) {
  const lens = trace.lens;
  if (!lens) throw new TraceError(`${name}: this trace has no "lens" field; the ladder view needs one (the table view opens it)`);
  if (!Number.isInteger(lens.n_layer) || lens.n_layer < 1) throw new TraceError(`${name}: lens.n_layer must be a positive integer`);
  const names = (lens.variants ?? []).map((v) => v.name);
  if (!names.length || new Set(names).size !== names.length) throw new TraceError(`${name}: lens.variants must name each variant once`);
  if (!lens.variants.some((v) => v.engram === 'added') || names.length < 2) {
    throw new TraceError(`${name}: lens.variants needs one variant with "engram": "added" and one to compare it with`);
  }
  for (const v of lens.variants) {
    const values = lens.values?.[v.name];
    if (!Array.isArray(values) || values.length !== trace.tokens.length) {
      throw new TraceError(`${name}: lens.values.${v.name} has ${values?.length} positions, tokens has ${trace.tokens.length}`);
    }
    values.forEach((blocks, t) => {
      if (!Array.isArray(blocks) || blocks.length !== lens.n_layer) {
        throw new TraceError(`${name}: lens.values.${v.name}[${t}] has ${blocks?.length} blocks, expected ${lens.n_layer}`);
      }
      blocks.forEach((cell, b) => {
        const where = `${name}: lens.values.${v.name}[${t}][${b}]`;
        const top = Array.isArray(cell?.top) ? cell.top : [];
        if (!top.length || !top.every(([id]) => Number.isInteger(id) && String(id) in (lens.vocab ?? {}))) {
          throw new TraceError(`${where} has no top-1 token with text in lens.vocab`);
        }
        const probability = (p) => Number.isFinite(p) && p >= 0 && p <= 1;
        if (!top.every(([, p]) => probability(p)) || !probability(cell.p_final)) {
          throw new TraceError(`${where} needs a top-1 probability and p_final in [0, 1]`);
        }
      });
    });
  }
}

const variant = (name) => state.trace.lens.variants.find((v) => v.name === name) ?? { name };
const label = (name) => (state.lang === 'en' ? null : TEXT[state.lang].variants?.[variant(name).engram]) ?? variant(name).label ?? name;
const color = (name) => COLORS[variant(name).engram] ?? 'var(--ink)';
// positions that have a next token in the text; the last one has none
const scored = () => state.trace.tokens.length - 1;
const settled = (name) => state.trace.lens.values[name].map(settledBlock);
const prompt = () => state.trace.generation?.prompt_tokens ?? state.trace.tokens.length;
const lookupBlocks = () => state.trace.lens.engram_before_block ?? [];

function applyLanguage() {
  document.documentElement.lang = state.lang;
  for (const node of document.querySelectorAll('[data-i18n]')) node.textContent = tr(node.dataset.i18n);
  for (const node of document.querySelectorAll('[data-i18n-html]')) node.innerHTML = tr(node.dataset.i18nHtml);
  for (const node of document.querySelectorAll('[data-i18n-aria]')) node.setAttribute('aria-label', tr(node.dataset.i18nAria));
  $('lang').textContent = tr('lang');
}

function buildPicker() {
  const others = state.trace.lens.variants.filter((v) => v.name !== state.a);
  $('b').replaceChildren(...others.map((v) => new Option(label(v.name), v.name)));
  $('b').value = state.b;
}

function renderHeadline() {
  const { values, n_layer: blocks } = state.trace.lens;
  const sa = settled(state.a), sb = settled(state.b);
  const all = [...Array(scored()).keys()];
  const c = compare(sa, sb, all);
  $('headline').innerHTML = tr('headline', c.n, c.sooner, c.later, label(state.b));
  // both variants end on the token that actually comes next, so they settle on the same, right answer
  const finalTop = (name, t) => values[name][t][blocks - 1].top[0][0];
  const right = all.filter((t) => finalTop(state.a, t) === state.trace.tokens[t + 1].id && finalTop(state.b, t) === state.trace.tokens[t + 1].id);
  const r = compare(sa, sb, right);
  $('counts').replaceChildren(el('span', { class: 'stat' }, tr('countsAll', c, label(state.b))), el('br'),
    el('span', { class: 'stat' }, tr('countsRight', r)));
}

// The text, each token shaded by how many blocks sooner the Engram model settles than the other.
function renderText() {
  const { trace } = state;
  const sa = settled(state.a), sb = settled(state.b);
  const span = Math.max(1, trace.lens.n_layer - 1);
  const nodes = [];
  trace.tokens.forEach((token, t) => {
    if (t === prompt() && t > 0) nodes.push(el('span', { class: 'gen-mark' }, tr('generated')));
    const kind = t < prompt() ? 'prompt' : 'generated';
    if (t >= scored()) {
      nodes.push(el('span', { class: `tok last ${kind}` }, raw(token)));
      return;
    }
    const gain = sb[t] - sa[t];
    nodes.push(el('span', {
      class: `tok ${kind}`, 'data-t': t,
      style: { '--gain': String(Math.max(0, gain) / span), '--loss': String(Math.max(0, -gain) / span) },
    }, raw(token)));
  });
  $('text').replaceChildren(...nodes);
}

// Per position, the block each model settles at: block 0 at the bottom, the lookup as a dashed line.
function renderChart() {
  const { trace } = state;
  const n = scored();
  const blocks = trace.lens.n_layer;
  const width = LEFT + n * COL + 12;
  const y = (k) => TOP + HEIGHT - (k / Math.max(1, blocks - 1)) * HEIGHT;
  const chart = svg('svg', { width, height: TOP + HEIGHT + BOTTOM, class: 'chart' });
  const axis = svg('svg', { width: AXIS, height: TOP + HEIGHT + BOTTOM, class: 'chart' });
  for (let k = 0; k < blocks; k += 1) {
    chart.append(svg('line', { x1: LEFT, x2: width - 12, y1: y(k), y2: y(k), class: 'grid' }));
    axis.append(svg('text', { x: AXIS - 6, y: y(k) + 4, 'text-anchor': 'end' }, k));
  }
  axis.append(svg('text', { x: 10, y: TOP + HEIGHT / 2, transform: `rotate(-90 10 ${TOP + HEIGHT / 2})`, 'text-anchor': 'middle' }, tr('blockAxis')));
  $('chart-axis').replaceChildren(axis);
  for (const block of lookupBlocks()) {
    const ly = (y(block) + y(block - 1)) / 2;
    chart.append(svg('line', { x1: LEFT, x2: width - 12, y1: ly, y2: ly, class: 'lookup-line' }));
    chart.append(svg('text', { x: LEFT + 4, y: ly - 4, class: 'lookup-label' }, tr('engramLine')));
  }
  const sa = settled(state.a), sb = settled(state.b);
  for (let t = 0; t < n; t += 1) {
    const x = LEFT + t * COL + COL / 2;
    const group = svg('g', { class: 'col', 'data-t': t });
    group.append(svg('rect', { x: x - COL / 2, y: TOP - 6, width: COL, height: HEIGHT + BOTTOM, class: 'hit' }));
    group.append(svg('line', { x1: x, x2: x, y1: y(sa[t]), y2: y(sb[t]), stroke: 'var(--line)', 'stroke-width': 2 }));
    group.append(svg('circle', { cx: x, cy: y(sb[t]), r: 4.5, fill: color(state.b), class: 'dot-b' }));
    group.append(svg('circle', { cx: x, cy: y(sa[t]), r: 4.5, fill: color(state.a), class: 'dot-a' }));
    const text = display(trace.tokens[t].display ?? trace.tokens[t].text).slice(0, 9);
    group.append(svg('text', { x, y: TOP + HEIGHT + 12, class: 'tick', transform: `rotate(-55 ${x} ${TOP + HEIGHT + 12})`, 'text-anchor': 'end' }, text));
    chart.append(group);
  }
  chart.append(svg('rect', { id: 'cursor', x: LEFT, y: TOP - 8, width: COL, height: HEIGHT + 16, rx: 4, class: 'cursor' }));
  chart.onclick = (e) => {
    const t = e.target.closest('.col')?.dataset.t;
    if (t != null) go(Number(t), true);
  };
  $('chart').replaceChildren(chart);
  $('chart-legend').replaceChildren(...[state.a, state.b].map((name) =>
    el('span', {}, el('span', { class: 'swatch', style: { background: color(name) } }), label(name))));
}

function renderQuestion() {
  const { trace, t } = state;
  const from = Math.max(0, t - 8);
  $('question').replaceChildren(
    el('h3', {}, tr('step1')),
    el('div', { class: 'q' }, el('span', { class: 'ctx' }, `${from > 0 ? '…' : ''}${trace.tokens.slice(from, t).map(raw).join('')}`),
      el('span', { class: 'cur' }, raw(trace.tokens[t])), '  →  ?'),
    el('div', { class: 'small muted' }, tr('question')),
    el('div', {}, `${tr(t + 1 < prompt() ? 'actualNext' : 'writtenNext')} `, el('span', { class: 'chip' }, display(trace.tokens[t + 1].text))),
  );
}

function renderLookup() {
  const { trace, t } = state;
  const layerIndex = trace.layers.findIndex((l) => lookupBlocks().includes(l.layer_id));
  const index = Math.max(0, layerIndex);
  const layer = trace.layers[index];
  // the n-grams that end at this token, one key per n-gram size
  const keys = [...new Set(layer.columns.map((c) => c.n))].map((n) => el('span', { class: 'ngram' },
    trace.tokens.slice(Math.max(0, t - n + 1), t + 1).map(raw).join('').trim()));
  const [before, after] = tr('lookupAdded', layer.columns.length, layer.num_rows.toLocaleString());
  const added = el('div', { class: 'lookup-box' }, el('div', { class: 'who', style: { color: color(state.a) } }, label(state.a)),
    el('div', {}, before, ...keys, after));
  const copies = trace.gates?.[t]?.[index];
  if (copies?.length) {
    const gate = mean(copies);
    added.append(el('div', { class: 'gate' }, el('span', { class: 'small' }, tr('gate')),
      el('span', { class: 'gate-track' }, el('span', { class: 'gate-fill', style: { '--g': String(gate) } })), el('b', {}, pct(gate))));
  }
  // the same weights with and without the lookup, at the block the lookup is added before: what it changes
  const skipped = trace.lens.variants.find((v) => v.engram === 'skipped');
  if (skipped && layerIndex >= 0) {
    const k = layer.layer_id;
    const next = display(trace.tokens[t + 1].text);
    const row = (name, title) => {
      const cell = trace.lens.values[name][t][k];
      return el('div', { class: 'effect-row', 'data-variant': name }, el('span', { class: 'effect-title' }, title),
        el('span', { html: tr('effectRow', word(cell.top[0][0]), pct(cell.top[0][1]), next, pct(cell.p_next ?? 0)) }));
    };
    added.append(el('div', { class: 'effect' }, el('div', { class: 'small' }, tr('effect', k)),
      row(skipped.name, tr('effectOff')), row(state.a, tr('effectOn'))));
  }
  const other = el('div', { class: 'lookup-box off' }, el('div', { class: 'who', style: { color: color(state.b) } }, label(state.b)),
    el('div', {}, variant(state.b).engram === 'skipped' ? tr('lookupSkipped') : tr('lookupAbsent')));
  $('lookup').replaceChildren(el('h3', {}, tr('step2')), added, other);
}

function distribution(name) {
  const { trace, t, block } = state;
  const blocks = trace.lens.values[name][t];
  const cell = blocks[block];
  const next = trace.tokens[t + 1].id;
  const s = settledBlock(blocks);
  const bars = cell.top.map(([id, p]) => el('div', { class: `bar${id === next ? ' is-next' : ''}` },
    el('span', { class: 'w' }, word(id)),
    el('span', { class: 'track' }, el('span', { class: 'fill', style: { '--p': String(p), '--c': color(name) } })),
    el('span', { class: 'p' }, pct(p))));
  const rest = Math.max(0, 1 - cell.top.reduce((sum, [, p]) => sum + p, 0));
  bars.push(el('div', { class: 'bar other' }, el('span', { class: 'w' }, tr('others')),
    el('span', { class: 'track' }, el('span', { class: 'fill', style: { '--p': String(rest) } })), el('span', { class: 'p' }, pct(rest))));
  bars.push(el('div', { class: 'next-line' }, tr('nextLine', display(trace.tokens[t + 1].text), pct(cell.p_next ?? 0))));
  const notes = [tr('top', word(cell.top[0][0]), pct(cell.top[0][1])), block >= s ? tr('fixedFrom', s) : tr('notYet')];
  if (variant(name).engram === 'added' && lookupBlocks().some((b) => block < b)) notes.push(tr('beforeLookup'));
  return el('div', { class: 'dist', 'data-variant': name }, el('h5', { style: { color: color(name) } }, label(name)), ...bars,
    el('div', { class: 'dist-note', html: notes.join(' · ') }));
}

// The probability each model gives the actual next word after each block.
function renderCurve() {
  const { trace, t, block } = state;
  const blocks = trace.lens.n_layer;
  const W = 460, H = 120, L = 34, T = 8, B = 22;
  const x = (k) => L + (k / Math.max(1, blocks - 1)) * (W - L - 10);
  const y = (p) => T + (1 - p) * (H - T - B);
  const chart = svg('svg', { width: W, height: H, class: 'curve' });
  for (const p of [0, 0.5, 1]) {
    chart.append(svg('line', { x1: L, x2: W - 10, y1: y(p), y2: y(p), class: 'grid' }));
    chart.append(svg('text', { x: L - 6, y: y(p) + 4, 'text-anchor': 'end' }, `${100 * p}%`));
  }
  for (let k = 0; k < blocks; k += 1) chart.append(svg('text', { x: x(k), y: H - 6, 'text-anchor': 'middle' }, k));
  for (const b of lookupBlocks()) {
    const lx = (x(b) + x(b - 1)) / 2;
    chart.append(svg('line', { x1: lx, x2: lx, y1: T, y2: H - B, class: 'lookup-line' }));
  }
  chart.append(svg('line', { x1: x(block), x2: x(block), y1: T, y2: H - B, class: 'now' }));
  for (const name of [state.b, state.a]) {
    const ps = trace.lens.values[name][t].map((cell) => cell.p_next ?? 0);
    chart.append(svg('polyline', { points: ps.map((p, k) => `${x(k)},${y(p)}`).join(' '), fill: 'none', stroke: color(name), 'stroke-width': 2.5 }));
    chart.append(svg('circle', { cx: x(block), cy: y(ps[block]), r: 4.5, fill: color(name) }));
  }
  $('curve').replaceChildren(chart);
}

function renderAnswer() {
  const { trace, t } = state;
  const rows = [state.a, state.b].map((name) => {
    const blocks = trace.lens.values[name][t];
    const [id, p] = blocks[blocks.length - 1].top[0];
    return el('div', { class: 'answer-row', 'data-variant': name }, el('span', { class: 'who', style: { color: color(name) } }, label(name)),
      el('span', { html: tr('writes', word(id), pct(p), settledBlock(blocks)) }));
  });
  $('answer').replaceChildren(el('h3', {}, tr('step4')), el('p', { class: 'small' }, tr('step4Note')), ...rows);
}

function renderBlock() {
  $('block').value = state.block;
  $('block-label').textContent = tr('blockLabel', state.block);
  $('dists').replaceChildren(distribution(state.a), distribution(state.b));
  renderCurve();
  $('play').textContent = state.timer ? tr('pause') : tr('play');
  syncUrl();
}

function syncUrl() {
  const url = new URL(location.href);
  for (const [k, v] of Object.entries({ t: state.t, b: state.b, block: state.block, lang: state.lang })) url.searchParams.set(k, v);
  history.replaceState(null, '', url);
}

function setPlaying(playing) {
  clearInterval(state.timer);
  state.timer = null;
  if (playing && state.trace) {
    if (state.block >= state.trace.lens.n_layer - 1) state.block = 0;
    state.timer = setInterval(() => {
      if (state.block >= state.trace.lens.n_layer - 1) setPlaying(false);
      else { state.block += 1; renderBlock(); }
    }, STEP_MS);
  }
  if (state.trace) renderBlock();
}

// Select a position; from a click, climb the blocks from block 0.
function go(t, climb = false) {
  if (!state.trace) return;
  state.t = Math.max(0, Math.min(t, scored() - 1));
  $('where').textContent = tr('where', state.t, scored(), state.t < prompt() ? tr('prompt') : tr('generated'));
  document.querySelectorAll('.tok.current').forEach((n) => n.classList.remove('current'));
  document.querySelector(`.tok[data-t="${state.t}"]`)?.classList.add('current');
  $('cursor')?.setAttribute('x', LEFT + state.t * COL);
  const scroller = $('chart');
  scroller.scrollTo({ left: Math.max(0, LEFT + state.t * COL - scroller.clientWidth / 2), behavior: 'smooth' });
  renderQuestion();
  renderLookup();
  renderAnswer();
  if (climb) state.block = 0;
  setPlaying(climb);
}

function render() {
  applyLanguage();
  buildPicker();
  renderHeadline();
  renderText();
  renderChart();
  go(state.t);
}

async function open(source) {
  try {
    const trace = await loadTrace(source);
    checkLens(trace, String(source));
    if (trace.tokens.length < 2) throw new TraceError(`${source}: the ladder needs at least two tokens`);
    state.trace = trace;
    const variants = trace.lens.variants;
    state.a = variants.find((v) => v.engram === 'added').name;
    const others = variants.filter((v) => v.name !== state.a);
    const fallback = others.find((v) => v.engram === 'absent') ?? others[0];
    state.b = others.some((v) => v.name === params.get('b')) ? params.get('b') : fallback.name;
    const t = Number(params.get('t'));
    state.t = Number.isInteger(t) ? t : 0;
    const block = Number(params.get('block'));
    state.block = Number.isInteger(block) && block >= 0 && block < trace.lens.n_layer ? block : 0;
    $('block').max = trace.lens.n_layer - 1;
    $('title').textContent = trace.title || '';
    $('error').hidden = true;
    const tableView = new URL('index.html', location.href);
    tableView.searchParams.set('trace', source);
    $('table-link').href = tableView.href;
    render();
  } catch (error) {
    $('error').textContent = error instanceof TraceError ? error.message : `Could not open the trace: ${error.message}`;
    $('error').hidden = false;
  }
}

state.lang = params.get('lang') === 'ja' ? 'ja' : 'en';
applyLanguage();
$('lang').addEventListener('click', () => {
  state.lang = state.lang === 'en' ? 'ja' : 'en';
  if (state.trace) render(); else applyLanguage();
});
$('b').addEventListener('change', (e) => { state.b = e.target.value; render(); });
$('text').addEventListener('click', (e) => {
  const t = e.target.dataset?.t;
  if (t != null) go(Number(t), true);
});
$('prev').addEventListener('click', () => go(state.t - 1, true));
$('next').addEventListener('click', () => go(state.t + 1, true));
$('play').addEventListener('click', () => setPlaying(!state.timer));
$('block').addEventListener('input', (e) => { state.block = Number(e.target.value); setPlaying(false); });
document.addEventListener('keydown', (e) => {
  if (!state.trace || ['INPUT', 'SELECT', 'TEXTAREA'].includes(e.target.tagName)) return;
  if (e.key === 'ArrowRight') go(state.t + 1, true);
  if (e.key === 'ArrowLeft') go(state.t - 1, true);
  if (e.key === 'ArrowUp' && state.block < state.trace.lens.n_layer - 1) { state.block += 1; setPlaying(false); }
  if (e.key === 'ArrowDown' && state.block > 0) { state.block -= 1; setPlaying(false); }
  if (e.key === ' ' && !['BUTTON', 'SUMMARY', 'A'].includes(e.target.tagName)) { e.preventDefault(); setPlaying(!state.timer); }
});

open(params.get('trace') ?? 'traces/nanogpt-hastings.json');
