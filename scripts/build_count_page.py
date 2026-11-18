"""Generate a self-contained click-to-count page for a blind hand-count set.

    python scripts/build_count_page.py labels/sacramento

Writes ``<dir>/count.html``. Open it directly in a browser (no server needed); it loads
``crops/<id>.jpg`` next to it. Only crop ids, sizes and lot outlines are embedded -- never the
year -- so counting stays blind. Click each vehicle inside the outlined lot; right-click (or
click an existing dot) removes one. Progress is kept in the browser; "Download counts" saves
``counts.json`` (clicks in native crop pixels) to put back in ``<dir>``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Hand Count</title>
<style>
:root{--bg:#f6f6f4;--fg:#1d1d1b;--mut:#6b6b66;--line:#d8d8d2;--acc:#0b6bcb;--dot:#ff2d55}
@media (prefers-color-scheme:dark){:root{--bg:#161615;--fg:#ececea;--mut:#9a9a94;--line:#33332f;--acc:#5aa9ff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.4 system-ui,sans-serif}
header{display:flex;flex-wrap:wrap;gap:12px;align-items:center;padding:10px 16px;border-bottom:1px solid var(--line)}
h1{font-size:16px;margin:0 12px 0 0}button{font:inherit;padding:6px 12px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);cursor:pointer}
button.primary{background:var(--acc);border-color:var(--acc);color:#fff}.mut{color:var(--mut)}
main{display:grid;grid-template-columns:200px 1fr;min-height:calc(100vh - 54px)}
nav{border-right:1px solid var(--line);overflow:auto;max-height:calc(100vh - 54px);padding:8px}
nav div{display:flex;justify-content:space-between;padding:4px 8px;border-radius:5px;cursor:pointer}
nav div.on{background:var(--acc);color:#fff}nav div.done span:first-child::after{content:" ✓"}
#stage{padding:16px;display:flex;flex-direction:column;align-items:center;gap:10px}
#wrap{position:relative;display:inline-block;cursor:crosshair;user-select:none}#wrap img{display:block;image-rendering:auto}
svg{position:absolute;inset:0;width:100%;height:100%}
.big{font-size:28px;font-weight:600}
label{display:flex;gap:6px;align-items:center}
@media (max-width:700px){main{grid-template-columns:1fr}nav{max-height:none;border-right:0;border-bottom:1px solid var(--line);display:flex;flex-wrap:wrap}}
</style></head><body>
<header><h1>Hand count</h1>
<span class="mut" id="prog"></span>
<label><input type="checkbox" id="outline" checked> outline</label>
<label>zoom <input type="range" id="zoom" min="1" max="3" step="0.25" value="1.5"></label>
<button id="prev">◀ Prev</button><button id="next">Next ▶</button>
<button id="unsure">Mark unsure</button>
<button class="primary" id="dl">Download counts</button></header>
<main><nav id="list"></nav>
<section id="stage">
 <div><span class="big" id="n">0</span> <span class="mut">vehicles in <b id="cid"></b></span> <span id="flag" class="mut"></span></div>
 <div id="wrap"><img id="img" alt=""><svg id="ov"></svg></div>
 <div class="mut">Click each car/truck inside the outline (count partially-inside ones if their centre is in). Click a dot or right-click near it to remove. Keys: ←/→ move, U unsure.</div>
</section></main>
<script>
const CROPS = __CROPS__;
const KEY = "handcount:" + __SET__;
let S = {}; try { S = JSON.parse(localStorage.getItem(KEY) || "{}"); } catch (e) {}
const ids = Object.keys(CROPS); let i = Math.max(0, ids.findIndex(id => !S[id]));
const $ = id => document.getElementById(id), img = $("img"), ov = $("ov");
function save(){ try { localStorage.setItem(KEY, JSON.stringify(S)); } catch (e) {} }
function rec(id){ return S[id] || (S[id] = {points: [], unsure: false, visited: true}); }
function zoom(){ return parseFloat($("zoom").value); }
function render(){
  const id = ids[i], c = CROPS[id], r = rec(id), z = zoom(), sc = c.upscale * z;
  save();
  img.src = "crops/" + id + ".jpg"; img.style.width = c.width * sc + "px"; img.style.height = c.height * sc + "px";
  ov.setAttribute("viewBox", `0 0 ${c.width} ${c.height}`);
  let h = "";
  if ($("outline").checked) for (const ring of c.outline_px)
    h += `<polygon points="${ring.map(p => p.join(",")).join(" ")}" fill="none" stroke="#ffd60a" stroke-width="${1.2/z}" stroke-dasharray="${3/z} ${2/z}"/>`;
  r.points.forEach((p, k) => h += `<circle data-k="${k}" cx="${p[0]}" cy="${p[1]}" r="${2.2/z+0.6}" fill="var(--dot)" stroke="#fff" stroke-width="${0.5/z}"/>`);
  ov.innerHTML = h;
  $("n").textContent = r.points.length; $("cid").textContent = id;
  $("flag").textContent = r.unsure ? "· marked unsure" : ""; $("unsure").textContent = r.unsure ? "Clear unsure" : "Mark unsure";
  const done = ids.filter(x => S[x] && S[x].visited).length;
  $("prog").textContent = `${i + 1} / ${ids.length} · ${done} visited`;
  $("list").innerHTML = ids.map((x, k) => `<div data-k="${k}" class="${k === i ? "on" : ""} ${S[x] && S[x].visited ? "done" : ""}"><span>${x}</span><span>${S[x] ? S[x].points.length : ""}</span></div>`).join("");
}
function pt(e){ const b = ov.getBoundingClientRect(), c = CROPS[ids[i]];
  return [+((e.clientX - b.left) / b.width * c.width).toFixed(2), +((e.clientY - b.top) / b.height * c.height).toFixed(2)]; }
function nearest(p){ const r = rec(ids[i]); let best = -1, d = 9 / zoom();
  r.points.forEach((q, k) => { const dd = Math.hypot(q[0]-p[0], q[1]-p[1]); if (dd < d) { d = dd; best = k; } }); return best; }
ov.addEventListener("click", e => { const r = rec(ids[i]);
  if (e.target.dataset.k !== undefined && e.target.tagName === "circle") r.points.splice(+e.target.dataset.k, 1);
  else r.points.push(pt(e)); render(); });
ov.addEventListener("contextmenu", e => { e.preventDefault(); const k = nearest(pt(e)); if (k >= 0) { rec(ids[i]).points.splice(k, 1); render(); } });
$("list").addEventListener("click", e => { const d = e.target.closest("[data-k]"); if (d) { i = +d.dataset.k; render(); } });
$("prev").onclick = () => { i = (i + ids.length - 1) % ids.length; render(); };
$("next").onclick = () => { i = (i + 1) % ids.length; render(); };
$("unsure").onclick = () => { const r = rec(ids[i]); r.unsure = !r.unsure; render(); };
$("outline").onchange = render; $("zoom").oninput = render;
document.addEventListener("keydown", e => { if (e.key === "ArrowRight") $("next").click(); if (e.key === "ArrowLeft") $("prev").click(); if (e.key.toLowerCase() === "u") $("unsure").click(); });
$("dl").onclick = () => {
  const out = {set: __SET__, saved: new Date().toISOString(), crops: {}};
  for (const id of ids) if (S[id] && S[id].visited) out.crops[id] = {count: S[id].points.length, unsure: !!S[id].unsure, points_px: S[id].points};
  const a = document.createElement("a"); a.href = URL.createObjectURL(new Blob([JSON.stringify(out, null, 1)], {type: "application/json"}));
  a.download = "counts.json"; a.click();
};
render();
</script></body></html>
"""


def main() -> None:
    d = Path(sys.argv[1])
    manifest = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    # Blind: no year, lot id, lot type or georeferencing in the page.
    crops = {cid: {k: m[k] for k in ("width", "height", "upscale", "outline_px")} for cid, m in sorted(manifest.items())}
    html = PAGE.replace("__CROPS__", json.dumps(crops, separators=(",", ":"))).replace("__SET__", json.dumps(d.name))
    (d / "count.html").write_text(html, encoding="utf-8")
    print(f"{len(crops)} crops -> {d / 'count.html'}")


if __name__ == "__main__":
    main()
