#!/usr/bin/env python
"""The whole surface, as slides: AutoGen core, AgentChat, OpenClaw, and Atlas.

    python docs/diagrams/make_slides.py

Writes `docs/pdf/slaytlar.html` — a 16:9 deck that reads on screen (arrow keys)
and prints to PDF (one slide per landscape page) without a second export path.

Figures come from `rough.py`, the same pen the two PDFs use, so a diagram here
and the same diagram in `agentchat-kilavuzu.pdf` are visibly the same drawing.

**Every claim on a slide is tagged.** `ölçüldü` means a number came out of a run
or out of source; `kaynak` means a document says it and the citation is on the
slide; `teyitsiz` means we believe it and have not checked. A deck is where
unmarked assertions do the most damage, because nobody stops to check a slide.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))


OUT = Path(__file__).resolve().parent.parent / "pdf" / "slaytlar.html"

from figures import (  # noqa: E402 — the shared drawing set
    f_actor, f_atlas, f_context, f_external_content, f_fanout, f_frozen_plan,
    f_gate, f_gateway, f_graphflow, f_identity, f_intervention,
    f_memory_tiers, f_message_types, f_send_vs_publish, f_skill_disclosure,
    f_task_lifecycle, f_task_stack, f_teams, f_thesis, f_three_axes,
    f_tool_loop, f_topic, f_two_ledgers,
)


# ─────────────────────────────────────────────────────────────── deck engine

SLIDES: list[str] = []
PARTS: list[tuple[str, str]] = []   # (numara, başlık) — içindekiler için


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def slide(part: str, title: str, body: str, *, kind: str = "", foot: str = "") -> None:
    cls = f"slide {kind}".strip()
    head = f'<div class="hd"><span>{part}</span><span class="n"></span></div>'
    ttl = f"<h2>{title}</h2>" if title else ""
    ft = f'<div class="ft">{foot}</div>' if foot else ""
    SLIDES.append(f'<section class="{cls}">{head}{ttl}<div class="bd">{body}</div>{ft}</section>')


def part(number: str, title: str, sub: str, contents: list[str]) -> None:
    PARTS.append((number, title))
    items = "".join(f"<li>{c}</li>" for c in contents)
    SLIDES.append(
        f'<section class="slide part"><div class="pnum">{number}</div>'
        f"<h1>{title}</h1><p class='psub'>{sub}</p>"
        f"<ol class='pcontents'>{items}</ol></section>"
    )


def fig(svg: str, caption: str = "", *, cap_mm: float = 86.0) -> str:
    """Size the figure to the slide, not to its own 600px drawing grid.

    A figure emitted at 600×180 renders at 600 CSS px — about a third of the
    slide's width, which is why the first render left every slide half empty.
    Width is therefore chosen so the drawing is as wide as it can be without
    its *height* pushing past `cap_mm`, which is the space a slide has under a
    title once the prose below it is accounted for.
    """
    import re

    box_mm = 250.0
    m = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg)
    if m:
        w, h = float(m.group(1)), float(m.group(2))
        box_mm = min(250.0, cap_mm * w / h)
    style = f' style="max-width:{box_mm:.0f}mm;margin:0 auto"'
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f"<figure{style}>{svg}{cap}</figure>"


def tag(kind: str) -> str:
    label = {"m": "ölçüldü", "k": "kaynak", "t": "teyitsiz"}[kind]
    return f'<span class="tag t-{kind}">{label}</span>'


def table(headers: list[str], rows: list[list[str]]) -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def cols(left: str, right: str) -> str:
    return f'<div class="cols"><div>{left}</div><div>{right}</div></div>'


def quote(s: str, cls: str = "") -> str:
    return f'<blockquote class="{cls}">{s}</blockquote>'


def big(s: str, sub: str = "") -> str:
    extra = f'<p class="bigsub">{sub}</p>' if sub else ""
    return f'<div class="bigstat"><p class="bignum">{s}</p>{extra}</div>'


def code(s: str) -> str:
    return f"<pre>{esc(s)}</pre>"


# ────────────────────────────────────────────────────────────── the content
#
# The slides live in `slides_a.py` / `slides_b.py` and run against this module's
# namespace. They were inline until the bodies grew long enough that the engine
# was hard to find among them; splitting keeps each file about one thing.

def load_content() -> None:
    """Run the slide files against this module's namespace.

    Called from `__main__` rather than at import time so the deck engine — CSS,
    the viewer script, `slide`/`part`/`fig` — can be imported by another
    generator without also building this deck.
    """
    for part_file in ("slides_a.py", "slides_b.py"):
        src = (Path(__file__).resolve().parent / part_file).read_text(encoding="utf-8")
        exec(compile(src, part_file, "exec"), globals())  # noqa: S102 — our own files


# ═══════════════════════════════════════════════════════════════ the shell
#
# One geometry for both media: the slide is a 297×167 mm box (16:9). Print
# gives that box its own page; screen scales the *same* box with a transform, so
# what a reviewer sees on a laptop and what comes out of the printer are the
# same layout rather than two layouts that drift apart.

CSS = """
:root{
  --ink:#1e1e1e; --ink2:#454c53; --ink3:#767d84; --rule:#dcdfd9; --rule2:#b6bab2;
  --panel:#f6f7f3; --panel2:#eceee8; --paper:#fbfbf8;
  --ochre:#8a5208; --ochre-bg:#f8f1e3; --green:#14594a; --green-bg:#e5f0ea;
  --blue:#2c4a6b; --blue-bg:#e4ebf3; --red:#8c2f1d; --red-bg:#f7e8e4;
  --mono:"DejaVu Sans Mono","Liberation Mono",monospace;
  --sans:"DejaVu Sans","Liberation Sans",Arial,sans-serif;
  --serif:"DejaVu Serif","Liberation Serif",Georgia,serif;
  --W:297mm; --H:167mm;
}
*{box-sizing:border-box}
html,body{margin:0;padding:0;background:#2a2c28;color:var(--ink)}
body{font-family:var(--serif);line-height:1.42}

.slide{
  width:var(--W); height:var(--H); background:#fff; overflow:hidden;
  padding:12mm 16mm 12mm; position:relative; font-size:13.4pt;
}
.hd{display:flex;justify-content:space-between;font-family:var(--mono);font-size:8.4pt;
    letter-spacing:.09em;text-transform:uppercase;color:var(--ink3);
    border-bottom:1.5px solid var(--rule);padding-bottom:1.6mm;margin-bottom:4mm}
.slide h2{font-family:var(--mono);font-size:23pt;font-weight:700;letter-spacing:-.02em;
          margin:0 0 5mm;line-height:1.08;text-wrap:balance}
.bd{font-size:13pt;line-height:1.45}
.bd p{margin:0 0 3.4mm}
.ft{position:absolute;left:16mm;right:16mm;bottom:7mm;font-family:var(--sans);
    font-size:9pt;color:var(--ink3);border-top:1px solid var(--rule);padding-top:1.4mm}
code{font-family:var(--mono);font-size:.88em;background:var(--panel2);padding:.05em .3em;
     border-radius:2px}
pre{font-family:var(--mono);font-size:10pt;line-height:1.45;background:var(--panel);
    border:1px solid var(--rule);border-left:2.5px solid var(--rule2);padding:2.6mm 3mm;
    margin:0 0 3mm;white-space:pre-wrap;border-radius:0 3px 3px 0}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:10.6pt;
      margin:1mm 0 3.4mm}
th{text-align:left;font-size:8pt;letter-spacing:.07em;text-transform:uppercase;
   color:var(--ink3);border-bottom:1px solid var(--rule2);padding:0 2mm .8mm 0;
   vertical-align:bottom}
td{border-bottom:1px solid var(--rule);padding:1.5mm 2.5mm 1.5mm 0;vertical-align:top;
   color:var(--ink2)}
td:first-child{color:var(--ink)}
blockquote{margin:0 0 3.4mm;padding:3mm 3.6mm;background:var(--ochre-bg);
           border-left:3.5px solid var(--ochre);font-family:var(--sans);font-size:11.6pt;
           line-height:1.42}
blockquote p:last-child{margin:0}
blockquote.g{background:var(--green-bg);border-left-color:var(--green)}
blockquote.r{background:var(--red-bg);border-left-color:var(--red)}
blockquote.b{background:var(--blue-bg);border-left-color:var(--blue)}
.cols{display:grid;grid-template-columns:1fr 1fr;gap:9mm}
.cols>div>*:last-child{margin-bottom:0}
figure{margin:0 0 2mm;text-align:center}
svg{display:block;margin:0 auto;width:100%;height:auto}
figcaption{font-family:var(--sans);font-size:9pt;color:var(--ink3);margin-top:1.6mm}
.lead{font-family:var(--sans);font-size:13pt;color:var(--ink2)}
.tag{font-family:var(--sans);font-size:7.6pt;letter-spacing:.05em;text-transform:uppercase;
     padding:.12em .42em;border-radius:2px;vertical-align:.18em;white-space:nowrap}
.t-m{background:#e5f0ea;color:#14594a}
.t-k{background:#e4ebf3;color:#2c4a6b}
.t-t{background:#f4f0e6;color:#6b5a2c}
.bigstat{text-align:center;margin:6mm 0 4mm}
.bignum{font-family:var(--mono);font-size:52pt;font-weight:700;letter-spacing:-.03em;
        margin:0;line-height:1;color:var(--ink)}
.bigsub{font-family:var(--sans);font-size:12.4pt;color:var(--ink2);max-width:150mm;
        margin:3mm auto 0}

/* cover + part dividers */
.cover{background:var(--paper);display:flex;flex-direction:column;justify-content:center;
       padding:20mm 22mm}
.cover h1{font-family:var(--mono);font-size:46pt;letter-spacing:-.035em;margin:0 0 5mm;
          line-height:1}
.ceyebrow{font-family:var(--mono);font-size:8.4pt;letter-spacing:.16em;
          text-transform:uppercase;color:var(--ink3);margin-bottom:7mm}
.csub{font-family:var(--sans);font-size:14pt;color:var(--ink2);max-width:190mm;
      line-height:1.4;margin:0 0 12mm}
.cmeta{font-family:var(--mono);font-size:8.4pt;color:var(--ink3);line-height:1.9;
       border-top:2.5px solid var(--ink);padding-top:3mm}
.part{background:var(--paper);display:flex;flex-direction:column;justify-content:center;
      padding:16mm 22mm}
.part .pnum{font-family:var(--mono);font-size:62pt;font-weight:700;color:var(--rule2);
            line-height:.9;margin-bottom:2mm}
.part h1{font-family:var(--mono);font-size:30pt;letter-spacing:-.03em;margin:0 0 3mm}
.psub{font-family:var(--sans);font-size:12pt;color:var(--ink2);margin:0 0 7mm}
.pcontents{font-family:var(--sans);font-size:11.4pt;color:var(--ink2);columns:2;
           column-gap:10mm;margin:0;padding-left:5mm}
.pcontents li{margin-bottom:1.6mm}

/* ── screen: one slide, scaled to fit ─────────────────────────────── */
@media screen{
  #deck{position:fixed;inset:0;display:grid;place-items:center}
  .slide{position:absolute;display:none;transform:scale(var(--s,1));
         box-shadow:0 8px 40px rgba(0,0,0,.45)}
  .slide.on{display:block}
  .slide.cover.on,.slide.part.on{display:flex}
  #bar{position:fixed;left:0;bottom:0;height:3px;background:#8a5208;z-index:9;
       transition:width .18s}
  #cnt{position:fixed;right:10px;bottom:10px;font-family:var(--mono);font-size:11px;
       color:#8b8f88;z-index:9}
  #help{position:fixed;left:12px;bottom:10px;font-family:var(--mono);font-size:11px;
        color:#8b8f88;z-index:9}
}
/* ── print: every slide, one per page ─────────────────────────────── */
@page{size:297mm 167mm;margin:0}
@media print{
  html,body{background:#fff}
  #bar,#cnt,#help{display:none}
  .slide{display:block!important;transform:none!important;box-shadow:none;
         break-after:page;page-break-after:always}
  .slide.cover,.slide.part{display:flex!important}
  .slide:last-child{break-after:auto;page-break-after:auto}
}
"""

JS = """
(function(){
  var s = Array.prototype.slice.call(document.querySelectorAll('.slide'));
  var i = 0, bar = document.getElementById('bar'), cnt = document.getElementById('cnt');
  s.forEach(function(el, n){
    var t = el.querySelector('.hd .n');
    if (t) t.textContent = (n + 1) + ' / ' + s.length;
  });
  function fit(){
    // The slide is authored at a fixed mm size; scale it rather than reflow it,
    // so screen and print never disagree about where a line breaks.
    var el = s[0], W = el.offsetWidth, H = el.offsetHeight;
    var k = Math.min(window.innerWidth / W, window.innerHeight / H) * 0.96;
    document.documentElement.style.setProperty('--s', k);
  }
  function show(n){
    i = Math.max(0, Math.min(s.length - 1, n));
    s.forEach(function(el, k){ el.classList.toggle('on', k === i); });
    if (bar) bar.style.width = ((i + 1) / s.length * 100) + '%';
    if (cnt) cnt.textContent = (i + 1) + ' / ' + s.length;
    if (location.hash !== '#' + (i + 1)) history.replaceState(null, '', '#' + (i + 1));
  }
  document.addEventListener('keydown', function(e){
    if (e.key === 'ArrowRight' || e.key === 'PageDown' || e.key === ' ') { show(i + 1); e.preventDefault(); }
    else if (e.key === 'ArrowLeft' || e.key === 'PageUp') { show(i - 1); e.preventDefault(); }
    else if (e.key === 'Home') show(0);
    else if (e.key === 'End') show(s.length - 1);
  });
  document.addEventListener('click', function(e){
    if (e.target.closest('a')) return;
    show(i + (e.clientX < window.innerWidth * 0.25 ? -1 : 1));
  });
  window.addEventListener('resize', fit);
  fit();
  var start = parseInt((location.hash || '').slice(1), 10);
  show(isNaN(start) ? 0 : start - 1);
})();
"""


def build() -> str:
    body = "".join(SLIDES)
    return (
        "<!doctype html>\n<html lang=\"tr\"><head><meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Bütün yüzey — AutoGen, OpenClaw, Atlas</title>\n"
        f"<style>{CSS}</style></head><body>\n"
        f'<div id="deck">{body}</div>\n'
        '<div id="bar"></div><div id="cnt"></div>'
        '<div id="help">← → gez · yazdır: Ctrl+P</div>\n'
        f"<script>{JS}</script>\n</body></html>\n"
    )


if __name__ == "__main__":
    load_content()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = build()
    OUT.write_text(html, encoding="utf-8")
    n_fig = html.count("<svg")
    print(f"{OUT}  ·  {len(SLIDES)} slayt  ·  {n_fig} şekil  ·  {len(html)/1024:.0f} KB")
