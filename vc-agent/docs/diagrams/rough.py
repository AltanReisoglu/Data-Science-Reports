"""Hand-drawn SVG primitives, in the Excalidraw manner.

### Why this exists rather than an SVG filter

The obvious way to get a sketchy look is `feTurbulence` + `feDisplacementMap`
over the shapes. It renders correctly in a browser window and is **silently
dropped when Chrome prints to PDF** — the boxes come out perfectly straight and
nothing warns you. Measured on Chrome 148 against this document: filtered and
unfiltered output were pixel-identical in the PDF.

So the wobble is baked into the path data instead. A rough rectangle here is a
real `<path>` with jittered corners and slightly bowed edges; nothing at render
time has to cooperate.

### What makes it read as hand-drawn

Three things, and the third is the one people miss:

1. **Corners miss.** Each vertex moves by a pixel or two.
2. **Edges bow.** A drawn line is never straight; each edge is a quadratic curve
   with a small perpendicular bulge.
3. **Every stroke is drawn twice**, with different jitter. This is Excalidraw's
   signature and the single biggest contributor — a pen goes back over a line.

### Determinism

The jitter comes from a seeded PRNG keyed per shape, so regenerating the document
produces byte-identical output. A diagram that reshuffles itself on every build
makes every rebuild look like a change in review.
"""

from __future__ import annotations

import math
import random

# Excalidraw's palette, near enough. Fills are the pale end of each hue.
INK = "#1e1e1e"
PALETTE = {
    "blue":   ("#1971c2", "#e7f5ff"),
    "green":  ("#2f9e44", "#ebfbee"),
    "red":    ("#c92a2a", "#fff5f5"),
    "orange": ("#e8590c", "#fff4e6"),
    "violet": ("#5f3dc4", "#f8f0fc"),
    "grey":   ("#868e96", "#f8f9fa"),
    "ink":    (INK, "#ffffff"),
}
HAND = "Comic Sans MS, Comic Neue, DejaVu Sans, sans-serif"


class Pen:
    """A seeded source of wobble. One Pen per figure keeps figures independent."""

    def __init__(self, seed: int, roughness: float = 1.0) -> None:
        self._rng = random.Random(seed)
        self.roughness = roughness

    def j(self, amount: float = 1.6) -> float:
        return self._rng.uniform(-amount, amount) * self.roughness

    # ---------------------------------------------------------------- edges

    def _edge(self, x1: float, y1: float, x2: float, y2: float, bow: float) -> str:
        """One bowed edge as a quadratic curve, endpoints jittered."""
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy) or 1.0
        # Perpendicular unit vector — the bulge goes across the line, not along it.
        nx, ny = -dy / length, dx / length
        swell = self.j(bow) * min(1.0, length / 90)
        mx = (x1 + x2) / 2 + nx * swell
        my = (y1 + y2) / 2 + ny * swell
        return (
            f"M{x1 + self.j():.1f},{y1 + self.j():.1f} "
            f"Q{mx:.1f},{my:.1f} {x2 + self.j():.1f},{y2 + self.j():.1f}"
        )

    # ---------------------------------------------------------------- shapes

    def rect(self, x, y, w, h, colour="ink", *, width=1.6, fill=True, dash="") -> str:
        stroke, wash = PALETTE[colour]
        out = []
        if fill:
            # The fill is a plain polygon set slightly inside the outline, so the
            # wobbly stroke reads as drawn *over* a wash rather than clipping it.
            out.append(
                f'<path d="M{x+1},{y+1} L{x+w-1},{y+1} L{x+w-1},{y+h-1} L{x+1},{y+h-1} Z" '
                f'fill="{wash}" stroke="none"/>'
            )
        corners = [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        for _pass in range(2):
            d = " ".join(
                self._edge(*corners[i], *corners[(i + 1) % 4], bow=1.8)
                for i in range(4)
            )
            out.append(
                f'<path d="{d}" fill="none" stroke="{stroke}" '
                f'stroke-width="{width}" stroke-linecap="round"{extra}/>'
            )
        return "".join(out)

    def circle(self, cx, cy, r, colour="ink", *, width=1.5, fill=True, dash="") -> str:
        stroke, wash = PALETTE[colour]
        out = []
        if fill:
            out.append(f'<circle cx="{cx}" cy="{cy}" r="{r - 0.6}" fill="{wash}" stroke="none"/>')
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        for _pass in range(2):
            pts = []
            steps = 11
            for i in range(steps + 1):
                a = 2 * math.pi * i / steps
                rr = r + self.j(1.1)
                pts.append((cx + rr * math.cos(a), cy + rr * math.sin(a)))
            d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f} " + " ".join(
                f"L{px:.1f},{py:.1f}" for px, py in pts[1:]
            ) + " Z"
            out.append(
                f'<path d="{d}" fill="none" stroke="{stroke}" '
                f'stroke-width="{width}" stroke-linejoin="round"{extra}/>'
            )
        return "".join(out)

    def line(self, x1, y1, x2, y2, colour="ink", *, width=1.4, arrow=True, dash="") -> str:
        stroke, _ = PALETTE[colour]
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        out = [
            f'<path d="{self._edge(x1, y1, x2, y2, bow=2.2)}" fill="none" '
            f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"{extra}/>'
            for _pass in range(2)
        ]
        if arrow:
            out.append(self._arrowhead(x1, y1, x2, y2, stroke, width))
        return "".join(out)

    def curve(self, points, colour="ink", *, width=1.4, arrow=True, dash="") -> str:
        """A drawn-through polyline: consecutive bowed edges."""
        stroke, _ = PALETTE[colour]
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        out = []
        for _pass in range(2):
            d = " ".join(
                self._edge(*points[i], *points[i + 1], bow=2.4)
                for i in range(len(points) - 1)
            )
            out.append(
                f'<path d="{d}" fill="none" stroke="{stroke}" '
                f'stroke-width="{width}" stroke-linecap="round"{extra}/>'
            )
        if arrow:
            out.append(self._arrowhead(*points[-2], *points[-1], stroke, width))
        return "".join(out)

    def _arrowhead(self, x1, y1, x2, y2, stroke, width) -> str:
        """Two strokes off the tip — drawn, not a filled triangle marker."""
        a = math.atan2(y2 - y1, x2 - x1)
        size = 7.5
        out = []
        for turn in (a + 2.55, a - 2.55):
            ex = x2 + size * math.cos(turn) + self.j(0.7)
            ey = y2 + size * math.sin(turn) + self.j(0.7)
            out.append(
                f'<path d="M{x2:.1f},{y2:.1f} L{ex:.1f},{ey:.1f}" fill="none" '
                f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>'
            )
        return "".join(out)

    def cross(self, cx, cy, r=11, colour="red", *, width=2.4) -> str:
        stroke, _ = PALETTE[colour]
        return "".join(
            f'<path d="{self._edge(cx - r, cy - r, cx + r, cy + r, 1.4)}" fill="none" '
            f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>'
            f'<path d="{self._edge(cx + r, cy - r, cx - r, cy + r, 1.4)}" fill="none" '
            f'stroke="{stroke}" stroke-width="{width}" stroke-linecap="round"/>'
            for _ in range(1)
        )


def text(x, y, s, *, size=8.4, colour="#454c53", weight="normal", anchor="start",
         family=HAND) -> str:
    esc = s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    w = f' font-weight="{weight}"' if weight != "normal" else ""
    a = f' text-anchor="{anchor}"' if anchor != "start" else ""
    return (
        f'<text x="{x}" y="{y}" font-size="{size}" fill="{colour}" '
        f'font-family="{family}"{w}{a}>{esc}</text>'
    )


def mono(x, y, s, **kw) -> str:
    kw.setdefault("family", "DejaVu Sans Mono, monospace")
    return text(x, y, s, **kw)


def figure(width, height, body) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}">'
        f"{body}</svg>"
    )
