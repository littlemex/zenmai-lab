// Colours shared by the 3D scene and the panel, as CSS hex strings. No dependencies.

const NGRAM_COLORS = { 2: '#3bd16f', 3: '#2f9bff', 4: '#ff6fa0' };

function hslToHex(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const channel = (n) => {
    const k = (n + h * 12) % 12;
    const value = l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1));
    return Math.round(value * 255).toString(16).padStart(2, '0');
  };
  return `#${channel(0)}${channel(8)}${channel(4)}`;
}

export function ngramColor(n) {
  return NGRAM_COLORS[n] ?? hslToHex(((n * 67) % 360) / 360, 0.8, 0.62);
}

export function rankColor(rank, world) {
  return hslToHex((((rank * 360) / Math.max(world, 1) + 18) % 360) / 360, 0.7, 0.58);
}
