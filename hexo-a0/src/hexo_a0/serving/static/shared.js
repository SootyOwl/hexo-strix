// Shared across the play + analysis screens: geometry, view navigation,
// the compact move-hash codec, and small config. Loaded FIRST.

// Hex size scales with viewport so the board feels right on phones: smaller
// hexes let more of the playfield fit on screen without panning. The constants
// mirror the CSS breakpoints in observatory.css — keep them in sync.
let S = 28;
const SQ3 = Math.sqrt(3);
const HEX_SIZES = [
  { maxWidth: 480, size: 20 },  // phones
  { maxWidth: 768, size: 24 },  // tablets / landscape phones
  { maxWidth: Infinity, size: 28 },  // desktop
];
function responsiveHexSize() {
  const w = window.innerWidth;
  return HEX_SIZES.find(t => w <= t.maxWidth).size;
}
function applyResponsiveHexSize() {
  const newSize = responsiveHexSize();
  if (newSize === S) return false;
  S = newSize;
  // Both boards (play + analysis) need to redraw with the new hex scale. The
  // draw functions are defined in play.js / analysis.js, so we dispatch a
  // custom event that those files listen for. They re-call updateTransform
  // and re-render their cells.
  window.dispatchEvent(new CustomEvent("hexo:hex-size-changed", { detail: { size: S } }));
  return true;
}
window.addEventListener("resize", applyResponsiveHexSize);
applyResponsiveHexSize();
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
  // Refresh the phone topbar overflow menu so play-only items appear/disappear
  // as the view changes. Safe on desktop — renderTopbarMenu is a no-op there
  // (the menu button is display:none and the menu is never opened).
  if (typeof renderTopbarMenu === "function") {
    const menu = document.getElementById("topbar-menu");
    if (menu && menu.classList.contains("open")) renderTopbarMenu();
  }
  // Notify view-specific code (analysis.js, play.js) so they can adapt
  // to the new screen — e.g. auto-open the analysis bottom sheet on
  // phones when entering the screen with no position loaded.
  window.dispatchEvent(new CustomEvent("hexo:view-changed", { detail: { view: mode } }));
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

// --- Topbar overflow menu (phone UI) ----------------------------------------
// On narrow screens the topbar hides everything except the title, status, the
// primary action, and a ⋮ button. The ⋮ opens a small panel that re-exposes
// the hidden buttons (Resign / Copy HTTTX / Analyze this game / GitHub) so
// nothing is actually removed — just one tap deeper. The menu rebuilds its
// contents every time it opens (cheap; the panel is small) so it always
// reflects the current view (e.g. Resign only shows when a game is active,
// Analyze this game only when the user is in a finished play game).
function topbarMenuItems() {
  const items = [];
  // Play-only items: only meaningful on the play screen and when a game
  // exists. We read the hidden topbar buttons directly — same source of
  // truth, no duplication. The "hidden" attribute is what the existing
  // setStatus/applyState code already toggles.
  if (currentPathView() === "play") {
    const resign = document.getElementById("resign-btn");
    if (resign && !resign.hidden) {
      items.push({ label: "Resign", onclick: "resign()" });
    }
    const copyBtn = document.getElementById("copy-htttx-btn");
    if (copyBtn && !copyBtn.hidden) {
      items.push({ label: "Copy HTTTX", onclick: "copyHtttx()" });
    }
    const analyzeBtn = document.getElementById("analyze-game-btn");
    if (analyzeBtn && !analyzeBtn.hidden) {
      items.push({ label: "Analyze this game", onclick: "analyzeThisGame()" });
    }
    // On phones the Analysis button is hidden from the topbar (it'd make the
    // row too wide) and re-exposed here so the user can still navigate to
    // the analysis screen. The button is always present in the DOM, just
    // display:none in the ≤480px media query.
    const analysisBtn = document.getElementById("analysis-btn");
    if (analysisBtn && window.innerWidth <= 480) {
      items.push({ label: "Analysis", onclick: "goToAnalysis()" });
    }
    // Difficulty badge: shown in the topbar on wider screens, re-exposed
    // here as a read-only label on phones so the user still sees what
    // strength they're playing against.
    const badge = document.getElementById("difficulty-badge");
    if (badge && !badge.hidden && window.innerWidth <= 480) {
      const diffLabel = (DIFFICULTY_LABELS && DIFFICULTY_LABELS[badge.textContent])
        || badge.textContent;
      items.push({ label: `Difficulty: ${diffLabel}`, disabled: true });
    }
  }
  // GitHub source link: present on both screens. Cloned as an <a> so the
  // external-link behavior (target=_blank, rel=noopener) is preserved.
  items.push({
    label: "Source on GitHub",
    href: "https://github.com/SootyOwl/hexo-strix",
    external: true,
  });
  return items;
}
function renderTopbarMenu() {
  const menu = document.getElementById("topbar-menu");
  if (!menu) return;
  menu.innerHTML = "";
  for (const it of topbarMenuItems()) {
    if (it.href) {
      const a = document.createElement("a");
      a.href = it.href;
      a.textContent = it.label;
      if (it.external) {
        a.target = "_blank";
        a.rel = "noopener noreferrer";
      }
      menu.appendChild(a);
    } else {
      const b = document.createElement("button");
      b.type = "button";
      b.textContent = it.label;
      b.setAttribute("role", "menuitem");
      if (it.disabled) {
        b.disabled = true;
        b.style.opacity = "0.6";
        b.style.cursor = "default";
      } else {
        // The original buttons use inline onclick="resign()". For the menu
        // we parse the "funcName()" string and call the global directly —
        // more reliable than eval (which depends on scope chain + strict mode)
        // and surfaces "function not found" as a real error instead of a
        // silent no-op.
        const callMatch = /^\s*([a-zA-Z_$][\w$]*)\s*\(\s*\)\s*$/.exec(it.onclick || "");
        if (callMatch) {
          const fnName = callMatch[1];
          b.onclick = (ev) => {
            // Stop the click from reaching the document outside-click handler,
            // which would otherwise close the menu on the same tap. We also
            // set a flag for one tick as belt-and-suspenders for browsers
            // where stopPropagation on a dynamically-created button doesn't
            // fully prevent the document handler from firing.
            if (ev) ev.stopPropagation();
            _topbarClosing = true;
            closeTopbarMenu();
            if (typeof window[fnName] === "function") {
              window[fnName]();
            } else {
              console.warn(`topbar menu: ${fnName}() not found on window`);
            }
            setTimeout(() => { _topbarClosing = false; }, 0);
          };
        } else {
          b.onclick = (ev) => { if (ev) ev.stopPropagation(); closeTopbarMenu(); };
        }
      }
      menu.appendChild(b);
    }
  }
}
function openTopbarMenu() {
  renderTopbarMenu();
  const menu = document.getElementById("topbar-menu");
  const btn = document.getElementById("topbar-menu-btn");
  if (!menu || !btn) return;
  menu.classList.add("open");
  btn.setAttribute("aria-expanded", "true");
}
function closeTopbarMenu() {
  const menu = document.getElementById("topbar-menu");
  const btn = document.getElementById("topbar-menu-btn");
  if (!menu || !btn) return;
  menu.classList.remove("open");
  btn.setAttribute("aria-expanded", "false");
}
function toggleTopbarMenu(event) {
  // Stop the click from bubbling to the document-level outside-click handler
  // below (which would immediately close the menu we just opened).
  if (event) event.stopPropagation();
  const menu = document.getElementById("topbar-menu");
  if (menu && menu.classList.contains("open")) closeTopbarMenu();
  else openTopbarMenu();
}
// Close the menu when clicking/tapping outside it, on Escape, or on resize.
// We use a short-delayed click handler (not pointerdown) because pointerdown
// fires before the button's onclick has a chance to run, which would close
// a freshly-opened menu on the same gesture. The delay also gives menu-item
// onclicks time to fire and call closeTopbarMenu() themselves, which sets
// _topbarClosing to suppress this handler for the same click.
let _topbarClosing = false;
document.addEventListener("click", (e) => {
  if (_topbarClosing) return;
  const menu = document.getElementById("topbar-menu");
  const btn = document.getElementById("topbar-menu-btn");
  if (!menu || !menu.classList.contains("open")) return;
  if (menu.contains(e.target) || (btn && btn.contains(e.target))) return;
  closeTopbarMenu();
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeTopbarMenu();
});
window.addEventListener("resize", () => {
  // If the topbar grows past 480px the menu button hides; the menu would
  // still be in the DOM. Close it to be safe, but keep the items for when
  // the user rotates back to portrait.
  if (window.innerWidth > 480) closeTopbarMenu();
});
// Re-render the menu when the view switches, so play-only items vanish on the
// analysis screen and vice versa. (Hooked into setView above; nothing more to
// do here.)
