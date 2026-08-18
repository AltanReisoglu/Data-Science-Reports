#!/usr/bin/env python
"""A tutorial, not a reference: `docs/pdf/ogretici.pdf`.

    python docs/diagrams/make_ogretici.py

The other documents in `docs/pdf/` answer "what is this?" for someone who
already knows why they are asking. This one answers "how do I get from nothing
to a working agent system, and what will bite me on the way?" — so it is
ordered by *what you can do next*, not by what the API surface contains.

Three rules shape it, and they are the reason it is a separate document rather
than more chapters in the guide:

1. **Every chapter ends with something that runs.** A tutorial whose code
   fragments cannot be pasted into a file and executed teaches recognition, not
   ability.
2. **The gotcha comes right after the thing it bites.** `max_tool_iterations`
   is introduced two paragraphs after the tool loop, not in an appendix — you
   learn it while the mechanism is still in your head.
3. **Nothing is asserted that the repository cannot show.** Where a number
   appears, the run that produced it is named.

Figures come from `figures.py`, the same set the deck draws, so a diagram here
and the same diagram in the slides is the same drawing.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from figures import (  # noqa: E402
    f_actor, f_atlas, f_context, f_external_content, f_fanout, f_frozen_plan,
    f_gate, f_gateway, f_graphflow, f_identity, f_intervention, f_memory_tiers,
    f_message_types, f_send_vs_publish, f_skill_disclosure, f_teams, f_three_axes,
    f_tool_loop, f_topic, f_two_ledgers,
)

OUT = Path(__file__).resolve().parent.parent / "pdf" / "ogretici.html"

# ───────────────────────────────────────────────────────────── page furniture

BLOCKS: list[str] = []
TOC: list[tuple[str, str]] = []


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def part(number: str, title: str, blurb: str) -> None:
    """A part divider — a half page, not a full one; paper is not free."""
    TOC.append(("part", f"KISIM {number} · {title}"))
    BLOCKS.append(
        f'<section class="partdiv"><div class="pnum">KISIM {number}</div>'
        f"<h1>{esc(title)}</h1><p>{blurb}</p></section>"
    )


def chapter(number: str, title: str, learn: list[str], body: str) -> None:
    TOC.append(("ch", f"{number}. {title}"))
    items = "".join(f"<li>{x}</li>" for x in learn)
    BLOCKS.append(
        f'<section class="ch" id="b{number}">'
        f'<div class="chhold"><h2><span class="chn">{number}</span>{esc(title)}</h2>'
        f'<div class="learn"><b>Bu bölümde</b><ul>{items}</ul></div></div>'
        f"{body}</section>"
    )


def h3(s: str) -> str:
    return f"<h3>{esc(s)}</h3>"


def p(s: str) -> str:
    return f"<p>{s}</p>"


def code(s: str, caption: str = "") -> str:
    cap = f'<div class="codecap">{caption}</div>' if caption else ""
    return f'<div class="codeblk">{cap}<pre>{esc(s)}</pre></div>'


def shell(s: str) -> str:
    return f'<pre class="sh">{esc(s)}</pre>'


def out(s: str) -> str:
    """What the code above actually prints."""
    return f'<pre class="out">{esc(s)}</pre>'


def box(kind: str, title: str, body: str) -> str:
    """kind: dene | tuzak | neden | olcum"""
    return (f'<div class="box b-{kind}"><div class="boxt">{esc(title)}</div>'
            f"{body}</div>")


def dene(body: str, title: str = "Kendin dene") -> str:
    return box("dene", title, body)


def tuzak(body: str, title: str = "Tuzak") -> str:
    return box("tuzak", title, body)


def neden(body: str, title: str = "Neden böyle") -> str:
    return box("neden", title, body)


def olcum(body: str, title: str = "Ölçüm") -> str:
    return box("olcum", title, body)


def fig(svg: str, caption: str) -> str:
    return f'<figure>{svg}<figcaption>{caption}</figcaption></figure>'


def table(headers: list[str], rows: list[list[str]], caption: str = "") -> str:
    th = "".join(f"<th>{h}</th>" for h in headers)
    tr = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in r) + "</tr>" for r in rows)
    cap = f"<figcaption>{caption}</figcaption>" if caption else ""
    return f"<table><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>{cap}"


def two(left: str, right: str) -> str:
    return f'<div class="two"><div>{left}</div><div>{right}</div></div>'


# ──────────────────────────────────────────────────────────────────── styling

CSS = """
:root{
  --ink:#1e1e1e; --ink2:#454c53; --ink3:#767d84; --rule:#dcdfd9; --rule2:#b6bab2;
  --panel:#f6f7f3; --panel2:#eceee8;
  --ochre:#8a5208; --ochre-bg:#faf4e8; --green:#14594a; --green-bg:#e9f2ec;
  --blue:#2c4a6b; --blue-bg:#e8eef5; --red:#8c2f1d; --red-bg:#f9ece8;
  --mono:"DejaVu Sans Mono","Liberation Mono",monospace;
  --sans:"DejaVu Sans","Liberation Sans",Arial,sans-serif;
  --serif:"DejaVu Serif","Liberation Serif",Georgia,serif;
}
@page{size:A4;margin:19mm 17mm 20mm}
@page:first{margin:0}
*{box-sizing:border-box}
html{font-size:10.6pt}
body{margin:0;font-family:var(--serif);color:var(--ink);line-height:1.55;background:#fff;
     hyphens:auto}
h1,h2,h3,h4{font-family:var(--mono);font-weight:700;letter-spacing:-.02em;text-wrap:balance}
h2{font-size:16pt;margin:0 0 .5em;padding-top:.2em;border-top:2.5px solid var(--ink);
   break-after:avoid;line-height:1.15}
.chn{display:inline-block;min-width:1.9em;color:var(--ink3)}
h3{font-size:11.4pt;margin:1.7em 0 .45em;break-after:avoid;color:var(--ochre)}
h4{font-size:10.2pt;margin:1.2em 0 .3em;break-after:avoid;color:var(--ink2)}
p{margin:0 0 .62em;orphans:2;widows:2}
code{font-family:var(--mono);font-size:.87em;background:var(--panel2);padding:.05em .3em;
     border-radius:2px}
pre{font-family:var(--mono);font-size:8.1pt;line-height:1.5;background:var(--panel);
    border:1px solid var(--rule);border-left:2.5px solid var(--rule2);padding:.6em .75em;
    margin:0;white-space:pre-wrap;word-break:break-word;break-inside:avoid;
    border-radius:0 3px 3px 0}
.codeblk{margin:.6em 0 .9em;break-inside:avoid}
.codecap{font-family:var(--sans);font-size:7.2pt;letter-spacing:.06em;text-transform:uppercase;
         color:var(--ink3);margin-bottom:.25em}
pre.sh{background:#1e1e1e;color:#e6e6e6;border-color:#1e1e1e;border-left-color:#8a5208}
pre.out{background:#fff;border-style:dashed;border-left-style:solid;color:var(--ink2);
        margin:.35em 0 .9em}
table{border-collapse:collapse;width:100%;font-family:var(--sans);font-size:8.4pt;
      margin:.5em 0 .3em;break-inside:avoid}
th{text-align:left;font-size:7.2pt;letter-spacing:.07em;text-transform:uppercase;
   color:var(--ink3);border-bottom:1px solid var(--rule2);padding:0 .6em .25em 0;
   vertical-align:bottom}
td{border-bottom:1px solid var(--rule);padding:.3em .6em .3em 0;vertical-align:top;
   color:var(--ink2)}
td:first-child{color:var(--ink)}
ul,ol{margin:.2em 0 .7em;padding-left:1.2em}
li{margin-bottom:.24em}
figure{margin:.9em 0 .8em;break-inside:avoid;text-align:center}
svg{display:block;margin:0 auto;width:100%;max-width:132mm;height:auto}
figcaption{font-family:var(--sans);font-size:7.6pt;color:var(--ink3);margin-top:.4em;
           text-align:center}
.two{display:grid;grid-template-columns:1fr 1fr;gap:1.1em;margin-bottom:.4em}
.two>div>*:last-child{margin-bottom:0}

.learn{font-family:var(--sans);font-size:8.6pt;background:var(--panel);
       border:1px solid var(--rule);border-radius:3px;padding:.6em .8em;margin:0 0 1.1em;
       break-inside:avoid;color:var(--ink2)}
.learn b{font-size:7.2pt;letter-spacing:.08em;text-transform:uppercase;color:var(--ink3);
         display:block;margin-bottom:.3em}
.learn ul{margin:0;padding-left:1.1em}

.box{font-family:var(--sans);font-size:8.8pt;line-height:1.5;padding:.65em .85em;
     margin:.75em 0 1em;break-inside:avoid;border-left:3.5px solid;border-radius:0 3px 3px 0}
.box p{margin:0 0 .45em}.box p:last-child{margin:0}
.box pre{font-size:7.8pt;margin:.4em 0}
.boxt{font-size:7.2pt;letter-spacing:.09em;text-transform:uppercase;font-weight:700;
      margin-bottom:.35em}
.b-dene{background:var(--green-bg);border-color:var(--green)}
.b-dene .boxt{color:var(--green)}
.b-tuzak{background:var(--red-bg);border-color:var(--red)}
.b-tuzak .boxt{color:var(--red)}
.b-neden{background:var(--blue-bg);border-color:var(--blue)}
.b-neden .boxt{color:var(--blue)}
.b-olcum{background:var(--ochre-bg);border-color:var(--ochre)}
.b-olcum .boxt{color:var(--ochre)}

/* Chapters flow rather than each claiming a fresh page. Measured on the first
   build: forcing a break put 21 of 57 pages under eighteen lines, several of
   them holding a single box. A tutorial that wastes a third of its paper is
   harder to read, not more organised. The heading keeps its own block together
   so a chapter never opens with two lines at the foot of a page. */
.ch{margin-top:1.9em}
.ch>h2, .ch>h2+.learn{break-after:avoid}
.chhold{break-inside:avoid}
.partdiv{break-before:page;padding:22mm 0 0;border-top:none}
.partdiv .pnum{font-family:var(--mono);font-size:8.4pt;letter-spacing:.18em;
               color:var(--ink3);margin-bottom:.6em}
.partdiv h1{font-size:26pt;margin:0 0 .35em;line-height:1.05}
.partdiv p{font-family:var(--sans);font-size:10.4pt;color:var(--ink2);max-width:120mm}

.cover{height:297mm;padding:42mm 26mm 22mm;display:flex;flex-direction:column;
       justify-content:space-between;break-after:page;background:#fbfbf8}
.cover h1{font-size:34pt;line-height:1.05;margin:0 0 .3em}
.cover .sub{font-family:var(--sans);font-size:12.4pt;color:var(--ink2);max-width:120mm;
            line-height:1.45}
.cover .meta{font-family:var(--mono);font-size:8pt;color:var(--ink3);line-height:1.95;
             border-top:2.5px solid var(--ink);padding-top:.9em}
.toc{break-after:page}
.toc h2{border-top:none}
.tocl{font-family:var(--sans);font-size:9pt;column-count:2;column-gap:1.6em;
       list-style:none;padding:0;margin:0}
.tocl li{margin-bottom:.3em;color:var(--ink2);break-inside:avoid}
.tocl li.part{font-family:var(--mono);font-weight:700;color:var(--ink);
              margin:.9em 0 .35em;font-size:8.4pt;letter-spacing:.04em}
.tocl li.part:first-child{margin-top:0}
"""


def build() -> str:
    toc = "".join(
        f'<li class="{k}">{esc(t)}</li>' for k, t in TOC
    )
    cover = (
        '<section class="cover"><div>'
        '<p style="font-family:var(--mono);font-size:8.4pt;letter-spacing:.16em;'
        'text-transform:uppercase;color:var(--ink3);margin-bottom:9mm">öğretici · '
        "sıfırdan çalışır hâle</p>"
        "<h1>Ajan sistemleri:<br>motor, takım, kapı</h1>"
        '<p class="sub">AutoGen ile çalışan bir sistem kurmayı, OpenClaw\'ın onu '
        "kuşatan kontrol düzleminden ne öğrettiğini, ve kurumsal bir asistanın "
        "bunların hangisine ihtiyacı olduğunu adım adım anlatır.</p></div>"
        '<div class="meta">vc-agent · AutoGen v0.7.5 · OpenClaw @01cc7106<br>'
        "her bölüm çalışan kodla biter · her tuzak ısırdığı yerde durur</div></section>"
    )
    body = "".join(BLOCKS)
    return (
        '<!doctype html>\n<html lang="tr"><head><meta charset="utf-8">\n'
        "<title>Ajan sistemleri — öğretici</title>\n"
        f"<style>{CSS}</style></head><body>\n{cover}"
        f'<section class="toc"><h2>İçindekiler</h2><ul class="tocl">{toc}</ul></section>'
        f"{body}\n</body></html>\n"
    )


if __name__ == "__main__":
    for _f in ("ogretici_a.py", "ogretici_b.py"):
        _src = (Path(__file__).resolve().parent / _f).read_text(encoding="utf-8")
        exec(compile(_src, _f, "exec"), globals())  # noqa: S102 — our own files

    OUT.parent.mkdir(parents=True, exist_ok=True)
    html = build()
    OUT.write_text(html, encoding="utf-8")
    print(f"{OUT}  ·  {len(BLOCKS)} blok  ·  "
          f"{html.count('<svg')} şekil  ·  {len(html)/1024:.0f} KB")
