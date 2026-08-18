/* Hand-drawn SVG primitives — the browser half of docs/diagrams/rough.py.
 *
 * The same three rules make a shape read as drawn rather than plotted, and the
 * third is the one people miss:
 *
 *   1. Corners miss      — every vertex moves a pixel or two
 *   2. Edges bow         — each edge is a quadratic curve with a small bulge
 *   3. Every stroke twice — with different jitter. A pen goes back over a line,
 *                           and this contributes more than the other two together
 *
 * Why a port rather than server-rendered SVG: this panel *highlights* — the
 * active node thickens as a turn moves through it — so the shapes have to be
 * redrawn in place. Shipping a static picture per state would mean shipping one
 * picture per state.
 *
 * Jitter is seeded per shape, so a redraw of the same diagram is identical. A
 * panel that reshuffles its own wobble on every repaint looks like it is doing
 * something, which is exactly the wrong signal from a monitor.
 */
(function (global) {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';

  /* Mulberry32 — small, fast, and deterministic from a single integer. */
  function seeded(seed) {
    var a = seed >>> 0;
    return function () {
      a = (a + 0x6D2B79F5) >>> 0;
      var t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  function Pen(seed, roughness) {
    this.rand = seeded(seed || 1);
    this.roughness = roughness == null ? 1 : roughness;
  }

  Pen.prototype.j = function (amount) {
    var a = amount == null ? 1.6 : amount;
    return (this.rand() * 2 - 1) * a * this.roughness;
  };

  /* One bowed edge. The bulge is perpendicular to the line, never along it —
     along it just makes the line look mis-measured rather than hand-drawn. */
  Pen.prototype.edge = function (x1, y1, x2, y2, bow) {
    var dx = x2 - x1, dy = y2 - y1;
    var len = Math.hypot(dx, dy) || 1;
    var nx = -dy / len, ny = dx / len;
    var swell = this.j(bow == null ? 1.8 : bow) * Math.min(1, len / 90);
    var mx = (x1 + x2) / 2 + nx * swell;
    var my = (y1 + y2) / 2 + ny * swell;
    return 'M' + (x1 + this.j()).toFixed(1) + ',' + (y1 + this.j()).toFixed(1) +
           ' Q' + mx.toFixed(1) + ',' + my.toFixed(1) +
           ' ' + (x2 + this.j()).toFixed(1) + ',' + (y2 + this.j()).toFixed(1);
  };

  function path(d, stroke, width, extra) {
    var node = document.createElementNS(NS, 'path');
    node.setAttribute('d', d);
    node.setAttribute('fill', 'none');
    node.setAttribute('stroke', stroke);
    node.setAttribute('stroke-width', String(width));
    node.setAttribute('stroke-linecap', 'round');
    if (extra) { node.setAttribute('stroke-dasharray', extra); }
    return node;
  }

  Pen.prototype.rect = function (x, y, w, h, opts) {
    var o = opts || {};
    var g = document.createElementNS(NS, 'g');
    if (o.fill) {
      var wash = document.createElementNS(NS, 'path');
      wash.setAttribute('d', 'M' + (x + 1) + ',' + (y + 1) + ' L' + (x + w - 1) + ',' + (y + 1) +
                             ' L' + (x + w - 1) + ',' + (y + h - 1) + ' L' + (x + 1) + ',' + (y + h - 1) + ' Z');
      wash.setAttribute('fill', o.fill);
      wash.setAttribute('stroke', 'none');
      g.appendChild(wash);
    }
    var corners = [[x, y], [x + w, y], [x + w, y + h], [x, y + h]];
    for (var pass = 0; pass < 2; pass++) {
      var d = '';
      for (var i = 0; i < 4; i++) {
        var a = corners[i], b = corners[(i + 1) % 4];
        d += this.edge(a[0], a[1], b[0], b[1], 1.8) + ' ';
      }
      g.appendChild(path(d, o.stroke || '#1e1e1e', o.width || 1.6, o.dash));
    }
    return g;
  };

  Pen.prototype.line = function (x1, y1, x2, y2, opts) {
    var o = opts || {};
    var g = document.createElementNS(NS, 'g');
    for (var pass = 0; pass < 2; pass++) {
      g.appendChild(path(this.edge(x1, y1, x2, y2, 2.2), o.stroke || '#1e1e1e', o.width || 1.4, o.dash));
    }
    if (o.arrow !== false) { g.appendChild(this.arrowhead(x1, y1, x2, y2, o)); }
    return g;
  };

  Pen.prototype.curve = function (points, opts) {
    var o = opts || {};
    var g = document.createElementNS(NS, 'g');
    for (var pass = 0; pass < 2; pass++) {
      var d = '';
      for (var i = 0; i < points.length - 1; i++) {
        d += this.edge(points[i][0], points[i][1], points[i + 1][0], points[i + 1][1], 2.4) + ' ';
      }
      g.appendChild(path(d, o.stroke || '#1e1e1e', o.width || 1.4, o.dash));
    }
    if (o.arrow !== false) {
      var n = points.length;
      g.appendChild(this.arrowhead(points[n - 2][0], points[n - 2][1], points[n - 1][0], points[n - 1][1], o));
    }
    return g;
  };

  /* Two strokes off the tip, not a filled marker — a drawn arrow has an open head. */
  Pen.prototype.arrowhead = function (x1, y1, x2, y2, opts) {
    var o = opts || {};
    var a = Math.atan2(y2 - y1, x2 - x1);
    var g = document.createElementNS(NS, 'g');
    var turns = [a + 2.55, a - 2.55];
    for (var i = 0; i < turns.length; i++) {
      var ex = x2 + 7.5 * Math.cos(turns[i]) + this.j(0.7);
      var ey = y2 + 7.5 * Math.sin(turns[i]) + this.j(0.7);
      g.appendChild(path('M' + x2.toFixed(1) + ',' + y2.toFixed(1) + ' L' + ex.toFixed(1) + ',' + ey.toFixed(1),
                         o.stroke || '#1e1e1e', o.width || 1.4));
    }
    return g;
  };

  function text(x, y, s, opts) {
    var o = opts || {};
    var node = document.createElementNS(NS, 'text');
    node.setAttribute('x', String(x));
    node.setAttribute('y', String(y));
    node.setAttribute('font-size', String(o.size || 8.4));
    node.setAttribute('fill', o.colour || '#454c53');
    node.setAttribute('font-family', o.mono
      ? 'ui-monospace, SFMono-Regular, Menlo, monospace'
      : '"Comic Sans MS", "Comic Neue", system-ui, sans-serif');
    if (o.weight) { node.setAttribute('font-weight', o.weight); }
    if (o.anchor) { node.setAttribute('text-anchor', o.anchor); }
    node.textContent = s;
    return node;
  }

  function svg(width, height) {
    var node = document.createElementNS(NS, 'svg');
    node.setAttribute('viewBox', '0 0 ' + width + ' ' + height);
    node.setAttribute('width', String(width));
    node.setAttribute('height', String(height));
    return node;
  }

  global.Rough = { Pen: Pen, text: text, svg: svg, NS: NS };
})(window);
