// Shared across the play + analysis screens: geometry, view navigation,
// the compact move-hash codec, and small config. Loaded FIRST.

const S = 28, SQ3 = Math.sqrt(3);
const URL_PREFIX = (window.__HEXO_CFG__ || {}).urlPrefix || "";
function axialToPixel(q, r) {
  return { x: S * (SQ3 * q + SQ3/2 * r), y: S * (3/2 * r) };
}
function hexCorners(cx, cy, k = 1) {
  let pts = [];
  for (let i = 0; i < 6; i++) {
    let a = Math.PI / 180 * (60 * i - 30);
    pts.push(`${cx + S * k * Math.cos(a)},${cy + S * k * Math.sin(a)}`);
  }
  return pts.join(" ");
}
function hexDist(q1, r1, q2 = 0, r2 = 0) {
  const dq = q1 - q2, dr = r1 - r2;
  return (Math.abs(dq) + Math.abs(dr) + Math.abs(dq + dr)) / 2;
}

// Winning-run highlight: a glowy LIGHT-TINT border around each hex of the winning
// run. The tint is deliberately lighter than the winner's own stone colour (a
// cyan ring on cyan stones would be invisible), so it reads clearly on top of the
// stones. Shared by the play + analysis boards so a win looks identical on both.
function winLineSvg(stones, winner, winLen) {
  const AXES = [[1, 0], [0, 1], [1, -1]];
  const won = new Set();
  for (const s of stones) if (s[1] === winner) won.add(`${s[0][0]},${s[0][1]}`);
  let line = null;
  for (const [dq, dr] of AXES) {
    if (line) break;
    for (const key of won) {
      const [sq, sr] = key.split(",").map(Number);
      if (won.has(`${sq - dq},${sr - dr}`)) continue;   // start at the run's head
      let q = sq, r = sr; const run = [];
      while (won.has(`${q},${r}`)) { run.push({ q, r }); q += dq; r += dr; }
      if (run.length >= winLen) { line = run.slice(0, winLen); break; }
    }
  }
  if (!line) return "";
  const ring = winner === "P1" ? "#ffe1b0" : "#c8f1fd";
  // Soft glow: a blurred copy of the border layered (twice, for intensity) under
  // the crisp border via feMerge. Filter defined once here, referenced by id.
  let out = `<defs><filter id="win-glow" x="-60%" y="-60%" width="220%" height="220%">`
          + `<feGaussianBlur stdDeviation="2.6" result="b"/>`
          + `<feMerge><feMergeNode in="b"/><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>`
          + `</filter></defs>`;
  for (const c of line) {
    const p = axialToPixel(c.q, c.r);
    out += `<polygon points="${hexCorners(p.x, p.y)}" fill="none" stroke="${ring}" stroke-width="2.6" opacity="0.98" filter="url(#win-glow)" pointer-events="none"/>`;
  }
  return out;
}


// Convert a side-to-move-perspective value to P1's perspective. Pure + tested
// (see test_serving_analysis.test_p1_perspective_sign).
function p1Perspective(value, currentPlayer) {
  if (currentPlayer === "P2") return -value;
  return value;  // P1 to move, or unknown/treat as P1
}

// Show/hide the two views under the shared topbar. body[data-view] drives the
// context-dependent topbar (CSS): the Analysis/Play toggle swaps, and play-only
// controls hide on the analysis screen. Pure UI — navigation goes via goToView().
function setView(mode) {   // "analysis" | "play"
  analysisMode = (mode === "analysis");
  document.body.dataset.view = mode;
  document.getElementById("analysis-panel").hidden = !analysisMode;
  document.getElementById("board-container").style.display = analysisMode ? "none" : "";
}

// --- View <-> URL path -------------------------------------------------------
// Play lives at "/", analysis at "/analysis" (both served the same shell). The
// line/game state stays in the hash (#c=/#a= for an analysis line, #g= for a
// game), so /analysis#c=... and /#g=... remain shareable deep links.
function viewPath(mode) { return URL_PREFIX + (mode === "analysis" ? "/analysis" : "/"); }
function currentPathView() {
  let p = location.pathname;
  if (URL_PREFIX && p.startsWith(URL_PREFIX)) p = p.slice(URL_PREFIX.length) || "/";
  return /^\/analysis\/?$/.test(p) ? "analysis" : "play";
}
// Carry only the hash that belongs to the target view across a navigation.
function relevantHash(mode) {
  const h = location.hash;
  if (mode === "analysis") return (h.startsWith("#c=") || h.startsWith("#a=")) ? h : "";
  return h.startsWith("#g=") ? h : "";
}
function goToView(mode) {
  try { history.pushState(null, "", viewPath(mode) + relevantHash(mode)); } catch (_e) { /* file:// */ }
  setView(mode);
}
function goToAnalysis() { goToView("analysis"); }
function goToPlay() { goToView("play"); }

// Back/forward between the two screens re-selects the view from the path.
window.addEventListener("popstate", () => setView(currentPathView()));

// --- Deep links: encode the analysis line into the URL hash so a position is
// shareable. Compact form `#c=`: each move's (q,r) is zigzag-varint-packed into
// bytes, then base64url'd — ~2 bytes/move for the usual small coords, roughly
// half the old decimal `#a=q.r_q.r` form on long games. The legacy `#a=`
// decimal decoder is kept so old shared links still open.
function _zig(n) { return n >= 0 ? n * 2 : -n * 2 - 1; }
function _unzig(z) { return (z % 2) ? -((z + 1) / 2) : z / 2; }
function _b64urlFromBytes(bytes) {
  let s = "";
  for (let i = 0; i < bytes.length; i++) s += String.fromCharCode(bytes[i]);
  return btoa(s).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}
function _bytesFromB64url(str) {
  let s = str.replace(/-/g, "+").replace(/_/g, "/");
  while (s.length % 4) s += "=";
  const bin = atob(s);
  const out = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) out[i] = bin.charCodeAt(i);
  return out;
}
function encodeMovesCompact(moves) {
  // moves includes the leading [0,0] seed; drop it. Math (not bitwise) keeps
  // the varint safe for any coordinate magnitude.
  const bytes = [];
  for (const m of moves.slice(1)) {
    for (const v of [_zig(m[0]), _zig(m[1])]) {
      let z = v;
      while (z > 0x7f) { bytes.push((z & 0x7f) | 0x80); z = Math.floor(z / 128); }
      bytes.push(z);
    }
  }
  return _b64urlFromBytes(bytes);
}
function decodeMovesCompact(s) {
  if (!s) return [];
  let bytes;
  try { bytes = _bytesFromB64url(s); } catch (_e) { return []; }
  const ints = [];
  let i = 0;
  while (i < bytes.length) {
    let shift = 0, val = 0, b;
    do {
      if (i >= bytes.length) return [];  // truncated varint
      b = bytes[i++];
      val += (b & 0x7f) * Math.pow(2, shift);
      shift += 7;
    } while (b & 0x80);
    ints.push(_unzig(val));
  }
  const moves = [];
  for (let j = 0; j + 1 < ints.length; j += 2) moves.push([ints[j], ints[j + 1]]);
  return moves.filter(m => Number.isFinite(m[0]) && Number.isFinite(m[1]));
}
// Legacy decoder for old `#a=q.r_q.r` links.
function decodeMovesHash(s) {
  if (!s) return [];
  return s.split("_").filter(Boolean).map(tok => {
    const [q, r] = tok.split(".").map(Number);
    return [q, r];
  }).filter(m => Number.isFinite(m[0]) && Number.isFinite(m[1]));
}
// Update the URL to the line ending at the current node, without spamming
// browser history (replaceState). Called on load and on every scrub/branch.
