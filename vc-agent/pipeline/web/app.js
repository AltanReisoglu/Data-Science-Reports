/* Chat client for the local backend.

   The server owns routing and rendering: this file sends a question and injects
   the HTML that comes back. Answers arrive tagged with the path that produced
   them — `model` or `rules` — and the tag is shown, because a model answer and a
   deterministic one carry different warranties. */

(function () {
  var thread = document.getElementById('thread');
  var form = document.getElementById('ask');
  var input = document.getElementById('q');
  var chips = document.getElementById('chips');
  var disclaimer = document.getElementById('disclaimer');
  var picker = document.getElementById('scan-pick');
  var modeBadge = document.getElementById('mode-badge');
  var sheet = document.getElementById('scan-sheet');
  var tip = document.querySelector('.tooltip');
  var stopButton = document.getElementById('stop');
  var sendButton = document.getElementById('send');
  var resetButton = document.getElementById('reset-chat');

  var state = { scan: null, live: false, busy: false, pollTimer: null };

  var CHIPS = ['The funnel', 'What is missing', 'Cost', 'Candidates',
               'How it works', 'What is a workbench?'];

  // ---------------------------------------------------------------- helpers

  function el(tag, className, text) {
    var node = document.createElement(tag);
    if (className) { node.className = className; }
    if (text != null) { node.textContent = text; }
    return node;
  }

  function scrollToEnd(node) {
    node.scrollIntoView({ behavior: 'smooth', block: 'end' });
  }

  function addTurn(side, head, build) {
    var turn = el('div', 'turn turn--' + side);
    var bubble = el('div', 'bubble bubble--' + side);
    if (head) {
      var header = el('div', 'bubble__head');
      header.appendChild(el('span', null, head.title));
      if (head.path) {
        var badge = el('span', 'path path--' + head.path);
        badge.appendChild(el('span', 'path__dot'));
        var PATH_LABEL = {
          model: 'model', live: 'checked live',
          docs: 'from the docs', rules: 'from scan data',
          openclaw: 'straight to OpenClaw'
        };
        badge.appendChild(el('span', null, PATH_LABEL[head.path] || head.path));
        header.appendChild(badge);
      }
      bubble.appendChild(header);
    }
    build(bubble);
    turn.appendChild(bubble);
    thread.appendChild(turn);
    bind(bubble);
    scrollToEnd(turn);
    return bubble;
  }

  function bind(root) {
    // A "Check now" button is an action, not a question: it goes straight to the
    // live endpoint rather than asking the model to consider looking.
    root.querySelectorAll('[data-live]').forEach(function (node) {
      node.addEventListener('pointerdown', function (event) {
        event.preventDefault();
        liveCheck(node.getAttribute('data-live'));
      });
      node.addEventListener('click', function (event) { event.preventDefault(); });
    });
    root.querySelectorAll('[data-ask]').forEach(function (node) {
      // Commit on pointer-down: the response starts before the finger lifts.
      node.addEventListener('pointerdown', function (event) {
        event.preventDefault();
        ask(node.getAttribute('data-ask'));
      });
      node.addEventListener('click', function (event) { event.preventDefault(); });
      node.addEventListener('keydown', function (event) {
        if (event.key === 'Enter' || event.key === ' ') {
          event.preventDefault();
          ask(node.getAttribute('data-ask'));
        }
      });
    });
    root.querySelectorAll('[data-tip]').forEach(function (mark) {
      mark.addEventListener('pointermove', function (event) {
        tip.textContent = mark.getAttribute('data-tip');
        tip.setAttribute('data-show', 'true');
        var rect = tip.getBoundingClientRect();
        tip.style.left = Math.min(event.clientX + 14, window.innerWidth - rect.width - 8) + 'px';
        tip.style.top = Math.max(event.clientY - rect.height - 12, 8) + 'px';
      });
      mark.addEventListener('pointerleave', function () {
        tip.setAttribute('data-show', 'false');
      });
    });
  }

  // ---------------------------------------------------------------- asking

  /* A live turn. The agent streams tokens, and the tools it decides to call
     are shown as they happen — the same rule the audit log follows: the reader
     can see where an answer came from, not just what it said. */
  function streamAsk(question) {
    mech.begin('chat');
    addTurn('user', null, function (bubble) { bubble.appendChild(document.createTextNode(question)); });

    var tools = null;
    var text = null;
    var bubble = addTurn('bot', { title: 'Analyst', path: 'model' }, function (node) {
      tools = el('div', 'tools');
      text = el('div', 'streaming');
      node.appendChild(tools);
      node.appendChild(text);
    });

    setBusy(true);
    var buffer = '';

    fetch('/api/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question })
    }).then(function (response) {
      if (!response.ok || !response.body) { throw new Error('HTTP ' + response.status); }
      var reader = response.body.getReader();
      var decoder = new TextDecoder();

      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done) { finish(); return; }
          buffer += decoder.decode(chunk.value, { stream: true });
          var frames = buffer.split('\n\n');
          buffer = frames.pop();
          frames.forEach(function (frame) {
            var line = frame.replace(/^data: /, '').trim();
            if (!line) { return; }
            var event;
            try { event = JSON.parse(line); } catch (e) { return; }
            handle(event);
          });
          return pump();
        });
      }
      return pump();
    }).catch(function (error) {
      text.textContent += '\n[' + error.message + ']';
      finish();
    });

    function handle(event) {
      if (event.type === 'stage') {
        mech.stage(event);
      } else if (event.type === 'chunk') {
        text.textContent += event.text;
        scrollToEnd(bubble);
      } else if (event.type === 'tool') {
        var line = el('div', 'toolline');
        line.appendChild(el('span', 'toolline__name', event.name));
        line.appendChild(el('span', 'toolline__args', event.arguments || ''));
        tools.appendChild(line);
      } else if (event.type === 'tool_result') {
        var last = tools.lastElementChild;
        if (last) { last.setAttribute('data-tip', event.preview); bind(tools); }
        // A refusal is the one tool result that needs a decision, not a tooltip.
        // Leaving it as hover text meant the only way to act on it was curl.
        var refusal = parseRefusal(event.preview);
        if (refusal) {
          askApproval(refusal, function () { streamAsk(question); }, function (ran) {
            // Not a retry: the code already ran. This carries its output back so
            // the agent can answer with it rather than recomputing.
            streamAsk('Onayladığım kod konteynerde çalıştı. Çıktısı:\n\n' +
                      ran.output + '\n\nBuna göre cevabı yaz. Kodu tekrar ' +
                      'çalıştırma, sayıları da yeniden hesaplama.');
          });
        }
      } else if (event.type === 'done') {
        if (!text.textContent && event.text) { text.textContent = event.text; }
      } else if (event.type === 'cancelled') {
        text.textContent += '\n[stopped]';
        mech.ended('cancelled');
      } else if (event.type === 'error') {
        text.textContent += '\n[' + event.message + ']';
        mech.ended('error', event.message);
      }
    }

    function finish() {
      text.className = 'streaming streaming--done';
      setBusy(false);
      scrollToEnd(bubble);
    }
  }

  // --------------------------------------------------------------- approvals
  //
  // The gate refuses a call and records a request; until now the only way to
  // answer it was `curl`. A refusal is a question addressed to the operator, so
  // it belongs where the operator is looking.

  function parseRefusal(preview) {
    if (!preview || preview.indexOf('Refused:') !== 0) { return null; }
    var id = /Approve request ([0-9a-f]{6,})/.exec(preview);
    return { id: id ? id[1] : null, reason: preview.replace(/^Refused:\s*/, '') };
  }

  /* `retry` is a callback, not a question string. It used to be the question,
     which worked while the gate only ever stopped a chat turn. `/openclaw` lines
     are held by the same gate now and they do not go back through `streamAsk`,
     so what the approval has to remember is *how to try again* — not *what was
     asked*. */
  function askApproval(refusal, retry, onRan) {
    addTurn('bot', { title: 'Approval needed', path: 'gate' }, function (bubble) {
      bubble.appendChild(el('div', 'approval__why', refusal.reason));

      if (!refusal.id) {
        // A forbidden method has no request to approve — saying "approve this"
        // would offer a button that cannot exist.
        bubble.appendChild(el('div', 'approval__note',
          'This one has no approval path. It changes credentials, config or ' +
          'command approvals.'));
        return;
      }

      var row = el('div', 'approval__row');
      var yes = el('button', 'approval__btn approval__btn--yes', 'Approve and retry');
      var no = el('button', 'approval__btn', 'Deny');
      var status = el('span', 'approval__status', '');

      function decide(verb) {
        yes.disabled = no.disabled = true;
        status.textContent = verb === 'approve' ? 'approving…' : 'denying…';
        fetch('/api/approvals/' + refusal.id + '/' + verb, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ note: 'from the chat' })
        }).then(function (response) {
          if (!response.ok) { throw new Error('HTTP ' + response.status); }
          return response.json().catch(function () { return {}; });
        }).then(function (data) {
          status.textContent = verb === 'approve' ? 'approved' : 'denied';
          // Code is not retried, it is *replayed*: the server ran the exact text
          // that was on the card. Asking the model again would produce a
          // different program — measured — and then the thing approved would not
          // be the thing that ran.
          if (verb === 'approve' && data && data.ran) {
            status.textContent = 'onaylandı · konteynerde koştu';
            mech.term.code(data.ran.code);
            mech.term.result({ output: data.ran.output,
                               is_error: !data.ran.ok,
                               seconds: data.ran.seconds });
            // The container answered the person; it has not answered the agent.
            // The gate's refusal ended that turn, so the model never saw the
            // result and the thread was left holding a promise it could not
            // keep. Handing the output back as a new turn is what closes it.
            if (typeof onRan === 'function') { onRan(data.ran); }
            return;
          }
          // The grant covers exactly this call and is consumed by it, so the
          // work has to be attempted again for it to run.
          if (verb === 'approve' && typeof retry === 'function') {
            status.textContent = 'approved · asking again';
            setTimeout(retry, 250);
          }
        }).catch(function (error) {
          status.textContent = 'failed: ' + error.message;
          yes.disabled = no.disabled = false;
        });
      }

      yes.addEventListener('click', function () { decide('approve'); });
      no.addEventListener('click', function () { decide('deny'); });
      row.appendChild(yes);
      row.appendChild(no);
      row.appendChild(status);
      bubble.appendChild(row);
    });
  }

  function liveCheck(name) {
    if (!name || state.busy) { return; }
    addTurn('user', null, function (bubble) {
      bubble.appendChild(document.createTextNode('Check ' + name + ' now'));
    });
    var pending = addTurn('bot', { title: 'Checking live sources' }, function (bubble) {
      var dots = el('div', 'thinking');
      dots.appendChild(el('span')); dots.appendChild(el('span')); dots.appendChild(el('span'));
      bubble.appendChild(dots);
    });

    fetch('/api/live', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ company: name, scan: state.scan })
    }).then(function (response) {
      return response.json().then(function (body) {
        if (!response.ok) { throw new Error(body.detail || ('HTTP ' + response.status)); }
        return body;
      });
    }).then(function (result) {
      pending.parentNode.remove();
      addTurn('bot', { title: result.title, path: 'live' }, function (bubble) {
        var body = document.createElement('div');
        body.innerHTML = result.html;
        bubble.appendChild(body);
      });
    }).catch(function (error) {
      pending.parentNode.remove();
      addTurn('bot', { title: 'Live check failed' }, function (bubble) {
        bubble.appendChild(el('p', null, String(error.message || error)));
      });
    });
  }

  function setBusy(busy) {
    state.busy = busy;
    stopButton.hidden = !busy;
    sendButton.hidden = busy;
    input.disabled = busy;
  }

  stopButton.addEventListener('pointerdown', function () {
    fetch('/api/chat/stop', { method: 'POST' });
  });

  if (resetButton) {
    resetButton.addEventListener('pointerdown', function () {
      fetch('/api/chat/reset', { method: 'POST' }).then(function () { loadState(state.scan); });
    });
  }

  /* ------------------------------------------------------------- mechanism panel
   *
   * A turn is a sequence of named mechanisms, and until now every one of them was
   * invisible: the interface showed the answer and hid the machine. This draws the
   * machine while it runs, and labels each part with the class actually doing the
   * work so the drawing can be checked against the source.
   *
   * The labels come from `/api/mechanisms`, never from here — the catalogue lives
   * in `stages.py` next to the code it describes. Hardcoding the teaching text in
   * the browser is how a panel like this quietly starts lying.
   */
  /* ------------------------------------------------------------- mechanism panel
   *
   * Draws **only the mechanism that is running right now**, as its own figure —
   * the way the PDF draws one concept per diagram.
   *
   * It used to be a seven-box pipeline with nodes lighting up, which answered
   * "how far along is this" and nothing else. A progress bar is not a teaching
   * tool: the boxes never explained what a workbench is, or why a tool call and
   * a tool result are separate events. Each mechanism now gets a picture of
   * itself, redrawn when the active mechanism changes.
   *
   * The labels come from `/api/mechanisms`, never from here — the catalogue lives
   * in `stages.py` next to the code it describes.
   */
  var mech = (function () {
    var host = document.getElementById('mech');
    var canvas = document.getElementById('mech-canvas');
    var note = document.getElementById('mech-note');
    var counts = document.getElementById('mech-counts');
    var toggle = document.getElementById('mech-toggle');
    var body = document.getElementById('mech-body');
    // The static export has no panel. Every method the SSE loop calls must
    // exist here too, or a missing one throws mid-turn and takes the reply
    // with it — the panel is an explainer, it must never be load-bearing.
    if (!host) {
      return { stage: function () {}, begin: function () {},
               enable: function () {}, ended: function () {},
               cron: function () {} };
    }

    var catalogue = {}, runs = {}, runtime = null, coreNote = '';
    var mode = 'chat';
    var active = '';
    var last = null;                 // the last stage event, for redraws

    /* ── the running trace ───────────────────────────────────────────────
       The figure answers "what is happening right now"; a single figure can
       only ever answer that. The trace answers "what happened on the way
       here", which is the question you actually have once the answer has
       arrived and looks wrong.

       Deliberately terse: elapsed, lane, what, and whatever one number the
       stage carried. A log that needs to be read slowly does not get read. */
    var traceHost = document.getElementById('mech-trace');

    // ---------------------------------------------------------- terminal
    //
    // Read-only: it shows what the model ran and what came back. There is no
    // input, and that is the whole security story — the panel makes an existing
    // capability visible, it does not add one.
    var termHost = document.getElementById('term');
    var termBody = document.getElementById('term-body');
    var termMeta = document.getElementById('term-meta');

    function termLine(text, cls) {
      if (!termBody) { return; }
      // Measured before appending: adding the line changes scrollHeight, so a
      // check made afterwards would always say "not at the bottom". Same bug,
      // same fix as the trace strip — someone scrolled up is reading on purpose.
      var atBottom = termBody.scrollHeight - termBody.scrollTop
                     - termBody.clientHeight < 4;
      var row = el('div', cls || null, text);
      termBody.appendChild(row);
      while (termBody.children.length > 400) {
        termBody.removeChild(termBody.firstChild);
      }
      if (atBottom) { termBody.scrollTop = termBody.scrollHeight; }
    }

    function termOpen() {
      if (!termHost) { return; }
      termHost.hidden = false;
      document.body.classList.add('term-open');
    }

    function termClose() {
      if (!termHost) { return; }
      termHost.hidden = true;
      document.body.classList.remove('term-open');
    }

    function termCode(code) {
      termOpen();
      if (termMeta) { termMeta.textContent = 'çalışıyor…'; }
      termLine('$ python /workspace/tmp.py', 't-cmd');
      String(code || '').split('\n').forEach(function (line) {
        termLine('  ' + line, 't-dim');
      });
    }

    function termResult(meta) {
      termOpen();
      var ok = !meta.is_error;
      if (termMeta) {
        termMeta.textContent = (ok ? 'exit 0' : 'hata') +
          (meta.seconds != null ? ' · ' + meta.seconds + ' sn' : '');
      }
      String(meta.output == null ? '' : meta.output)
        .split('\n').forEach(function (line) {
          termLine(line, ok ? null : 't-err');
        });
      termLine('── ' + (ok ? 'bitti' : 'hata ile bitti') +
               (meta.seconds != null ? ' · ' + meta.seconds + ' sn' : '') + ' ──',
               't-dim');
    }

    if (document.getElementById('term-close')) {
      document.getElementById('term-close').addEventListener('click', termClose);
    }
    var traceT0 = 0;
    var TRACE_CAP = 60;

    /* `ours` is the label the catalogue uses; on screen it is the gateway —
       the gate, the context engine, our own counting. Saying "gateway" costs
       nothing and matches where the code lives. */
    var LANE_LABEL = { ours: 'gateway', core: 'core', agentchat: 'agentchat' };

    function traceReset() {
      if (!traceHost) { return; }
      traceHost.innerHTML = '';
      traceT0 = 0;
    }

    /* One field per stage, and every key below was read out of the emitter —
       `conversation.py`, `gateway/workbench.py`, `graph.py` — not guessed. A
       trace that prints an empty column because the key was renamed is worse
       than one that prints nothing, because it looks like the stage carried
       no information. `test_stages` pins the ids; these keys ride along. */
    function traceMeta(id, m) {
      switch (id) {
        case 'context':
          return (m.tools || 0) + ' tool · ' + (m.budget || 0) + ' tok bütçe';
        case 'compaction':
          return m.summarised + ' msj · ' + (m.before || '?') + '→' + (m.after || '?') +
                 ' tok · ' + (m.method || '');
        case 'model':
          return m.streaming ? 'stream' : '';
        case 'tool_request':
        case 'tool_result':
          return (m.tools || []).join(', ');
        case 'gate':
          return m.blocked ? ('RET · ' + (m.reason || '')) : ((m.tool || '') + ' · izin');
        case 'tool_exec':
          return (m.tool || '') + (m.kind ? ' · ' + m.kind : '');
        case 'code_request':
          // The code itself goes to the terminal; the strip gets its size, so a
          // twenty-line program does not push the trace off the panel.
          return String(m.code || '').split('\n').length + ' satır';
        case 'code_result':
          return (m.is_error ? 'hata' : 'exit 0') +
                 (m.seconds != null ? ' · ' + m.seconds + ' sn' : '');
        case 'loop':
          return m.limit ? ('tavan ' + m.limit) : '';
        case 'done':
          return (m.llm_calls || 0) + ' llm · ' + (m.tool_calls || 0) + ' tool · ' +
                 (m.tokens || 0) + ' tok';
        case 'graph_build':
          return (m.branches || []).length + ' dal · ' + (m.termination || '');
        case 'intervention':
          return m.handler || '';
        case 'analysts':
          return (m.branch || '') + ' · ' + (m.arrived || 0) + '/' + (m.expected || 0);
        case 'join':
          return (m.arrived || 0) + '/' + (m.expected || 0) + ' dal';
        case 'count':
          return (m.succeeded || 0) + '/' + (m.expected || 0) + ' başarılı';
        case 'graph_run':
        case 'runtime_stop':
          return m.company || '';
        default:
          return '';
      }
    }

    function traceAdd(event, meta) {
      if (!traceHost) { return; }
      var now = Date.now();
      if (!traceT0) { traceT0 = now; }
      var prev = traceHost.querySelector('li.is-live');
      if (prev) { prev.classList.remove('is-live'); }

      // Measured before appending: adding the row changes scrollHeight, so a
      // check made afterwards would always say "not at the bottom".
      var atBottom = traceHost.scrollHeight - traceHost.scrollTop
                     - traceHost.clientHeight < 4;

      var lane = LANE_LABEL[event.lane] || event.lane || '';
      var row = el('li', 'is-live' + (meta.blocked ? ' is-block' : ''));
      row.appendChild(el('span', 't-at', '+' + ((now - traceT0) / 1000).toFixed(2) + 's'));
      row.appendChild(el('span', 't-lane lane-' + lane, lane));
      var what = el('span', 't-what', event.title || event.id);
      what.title = (event.klass || '') + (event.module ? '  ·  ' + event.module : '');
      row.appendChild(what);
      row.appendChild(el('span', 't-meta', traceMeta(event.id, meta)));
      traceHost.appendChild(row);

      while (traceHost.children.length > TRACE_CAP) {
        traceHost.removeChild(traceHost.firstChild);
      }
      // Follow the newest row only when the reader is already at the bottom.
      // Someone scrolled up is reading an earlier stage on purpose — during a
      // demo, narrating the trace from the top — and yanking them back down
      // makes the strip unusable for exactly that.
      if (atBottom) { traceHost.scrollTop = traceHost.scrollHeight; }
    }
    var collapsed = localStorage.getItem('mech-collapsed') === '1';

    var INK = '#1e1e1e', GREY = '#868e96', MUTE = '#454c53', FAINT = '#adb5bd';
    var BLUE = '#1971c2', GREEN = '#2f9e44', RED = '#c92a2a';
    var ORANGE = '#e8590c', VIOLET = '#5f3dc4';
    var WASH = {
      blue: '#e7f5ff', green: '#ebfbee', red: '#fff5f5',
      orange: '#fff4e6', violet: '#f8f0fc', grey: '#f8f9fa'
    };

    function box(pen, svg, x, y, w, h, colour, label, sub, opts) {
      var o = opts || {};
      svg.appendChild(pen.rect(x, y, w, h, {
        stroke: colour, fill: o.fill || 'transparent',
        width: o.width || 1.6, dash: o.dash
      }));
      svg.appendChild(Rough.text(x + w / 2, y + (sub ? h / 2 - 2 : h / 2 + 4), label, {
        size: o.size || 9, anchor: 'middle', weight: '700', colour: o.ink || INK
      }));
      if (sub) {
        svg.appendChild(Rough.text(x + w / 2, y + h / 2 + 12, sub, {
          size: 7, anchor: 'middle', colour: GREY, mono: o.mono
        }));
      }
    }

    function line(pen, svg, x1, y1, x2, y2, colour, opts) {
      var o = opts || {};
      var n = pen.line(x1, y1, x2, y2, {
        stroke: colour || INK, width: o.width || 1.4, dash: o.dash, arrow: o.arrow
      });
      if (o.flow) { n.setAttribute('class', 'mech-flow'); }
      svg.appendChild(n);
    }

    function label(svg, x, y, s, colour, opts) {
      var o = opts || {};
      svg.appendChild(Rough.text(x, y, s, {
        size: o.size || 7.4, colour: colour || GREY, anchor: o.anchor,
        weight: o.weight, mono: o.mono
      }));
    }

    /* One drawing per mechanism. Each is a picture of that thing, not of where
       we are in a sequence. `m` is the meta the stage carried. */
    var FIGURES = {
      context: function (pen, svg, m) {
        var budget = m.budget || 10000;
        box(pen, svg, 16, 26, 150, 44, ORANGE, 'model_context', 'bütçeye göre seçer',
            { fill: WASH.orange });
        line(pen, svg, 166, 48, 214, 48, ORANGE, { flow: true });
        box(pen, svg, 218, 26, 130, 44, BLUE, 'AssistantAgent', null, { fill: WASH.blue });
        label(svg, 16, 92, 'bütçe: ' + budget + ' token · ' + (m.tools || 0) +
              ' tool şeması · ' + ((m.workbenches || []).length) + ' workbench', MUTE, { size: 8 });
        label(svg, 16, 108, 'Her tur bunların hepsi yeniden gönderiliyor — "merhaba" bile ' +
              'tam tarifeyi ödüyor.', GREY);
        box(pen, svg, 372, 26, 250, 44, GREY, 'BufferedChatCompletionContext',
            'AutoGen\'inki MESAJ sayar', { dash: '5 3', width: 1.2, ink: GREY, size: 7.6 });
      },
      compaction: function (pen, svg, m) {
        var blocks = [[16, 74, 'eski', GREY], [96, 52, 'tool call', ORANGE],
                      [152, 58, 'result', ORANGE], [216, 60, 'mesaj', GREEN]];
        blocks.forEach(function (b) {
          box(pen, svg, b[0], 30, b[1], 30, b[3], '', null, { width: 1.4 });
          label(svg, b[0] + b[1] / 2, 49, b[2], INK, { anchor: 'middle', size: 7 });
        });
        line(pen, svg, 124, 20, 124, 70, RED, { arrow: false, dash: '4 3', width: 1.8 });
        label(svg, 124, 15, 'yanlış', RED, { anchor: 'middle', weight: '700' });
        line(pen, svg, 212, 20, 212, 70, GREEN, { arrow: false, width: 2.2 });
        label(svg, 224, 15, 'kaydırılmış sınır', GREEN, { weight: '700' });
        label(svg, 16, 88, 'Bir tool çağrısı sonucundan asla ayrılmıyor.', MUTE, { size: 8 });
        label(svg, 16, 104, 'yöntem: ' + (m.method || '—') + ' · özetlenen: ' +
              (m.summarised || 0) + ' mesaj', GREY);
      },
      model: function (pen, svg, m) {
        box(pen, svg, 16, 26, 130, 44, BLUE, 'AssistantAgent', 'modeli görmez',
            { fill: WASH.blue });
        line(pen, svg, 146, 48, 200, 48, INK, { flow: true });
        box(pen, svg, 204, 26, 170, 44, VIOLET, 'ChatCompletionClient',
            'create_stream()', { fill: WASH.violet, mono: true });
        line(pen, svg, 374, 48, 428, 48, INK, { flow: true });
        box(pen, svg, 432, 26, 150, 44, GREEN, 'endpoint', 'OpenAI-uyumlu',
            { fill: WASH.green });
        label(svg, 16, 90, 'create() DEĞİL create_stream() — bu yüzden LLMCallEvent değil ' +
              'LLMStreamEndEvent yayılıyor.', MUTE, { size: 8 });
        label(svg, 16, 106, 'Maliyeti yalnız ilkini dinleyerek sayan, 0 görür.', RED,
              { size: 8, weight: '700' });
      },
      stream: function (pen, svg) {
        box(pen, svg, 16, 26, 150, 40, VIOLET, 'model', null, { fill: WASH.violet });
        for (var i = 0; i < 5; i++) {
          var x = 200 + i * 52;
          box(pen, svg, x, 32, 44, 28, GREEN, '', null, { width: 1.3, fill: WASH.green });
          label(svg, x + 22, 50, ['Mer', 'ha', 'ba', ' si', 'ze'][i], INK,
                { anchor: 'middle', size: 7, mono: true });
        }
        line(pen, svg, 166, 46, 196, 46, GREEN, { flow: true });
        label(svg, 16, 84, 'ModelClientStreamingChunkEvent — cevap token token geliyor.',
              MUTE, { size: 8 });
        label(svg, 16, 100, 'model_client_stream=True olmasaydı cevap tek parça hâlinde, ' +
              'model bitirdikten SONRA gelirdi.', GREY);
      },
      tool_request: function (pen, svg, m) {
        box(pen, svg, 16, 26, 140, 44, VIOLET, 'model', 'bir tool seçti',
            { fill: WASH.violet });
        line(pen, svg, 156, 48, 210, 48, ORANGE, { flow: true });
        box(pen, svg, 214, 22, 250, 52, ORANGE, (m.tools || ['?'])[0],
            'ToolCallRequestEvent', { fill: WASH.orange, mono: true, size: 8.6 });
        label(svg, 16, 92, 'Şema imzadan ve docstring\'den üretildi.', MUTE, { size: 8 });
        label(svg, 16, 108, 'Yani docstring dokümantasyon değil — ARAYÜZ. ' +
              'Yanlış yazılmış bir docstring, yanlış çağrılan bir tool demek.', GREY);
        box(pen, svg, 486, 22, 136, 52, GREY, 'henüz çalışmadı',
            'önce kapıdan geçecek', { dash: '5 3', width: 1.2, ink: GREY, size: 7.6 });
      },
      gate: function (pen, svg, m) {
        var blocked = !!m.blocked;
        box(pen, svg, 16, 28, 116, 42, ORANGE, 'tool çağrısı', m.tool || '',
            { fill: WASH.orange, mono: true });
        line(pen, svg, 132, 49, 180, 49, INK);
        box(pen, svg, 184, 20, 170, 58, RED, 'before_tool_call',
            'GatedWorkbench', { fill: WASH.red, mono: true, width: 2.2 });
        if (blocked) {
          line(pen, svg, 269, 78, 269, 104, RED, { dash: '4 3' });
          label(svg, 282, 100, 'REDDEDİLDİ', RED, { weight: '700', size: 8.4 });
          box(pen, svg, 400, 20, 222, 58, GREY, 'çalışmadı', 'iç workbench\'e hiç gitmedi',
              { dash: '5 3', width: 1.2, ink: GREY, size: 8 });
        } else {
          line(pen, svg, 354, 49, 400, 49, GREEN, { flow: true });
          box(pen, svg, 404, 28, 218, 42, GREEN, 'geçti',
              (m.hooks || 0) + ' hook koştu', { fill: WASH.green });
        }
        label(svg, 16, 100, 'AutoGen\'de böyle bir şey yok — cookbook 05:6638 nasıl ' +
              'yazılacağını gösteriyor.', MUTE, { size: 8 });
        label(svg, 16, 116, 'Kapı ajanın uyum göstermeyi SEÇMESİNE değil, hattın ' +
              'kendisine dayanıyor.', GREY);
      },
      tool_exec: function (pen, svg, m) {
        box(pen, svg, 16, 26, 120, 44, BLUE, 'workbench', 'call_tool()',
            { fill: WASH.blue, mono: true });
        var kinds = [['StaticWorkbench', 'yerel fonksiyonlar', GREEN],
                     ['McpWorkbench', 'OpenClaw · stdio', VIOLET],
                     ['McpWorkbench', 'DeepWiki · HTTP', VIOLET]];
        kinds.forEach(function (k, i) {
          var y = 14 + i * 34;
          var on = (m.kind || '') === k[0];
          box(pen, svg, 220, y, 200, 28, on ? k[2] : FAINT, k[0], null,
              { width: on ? 2 : 1.1, fill: on ? WASH.green : 'transparent',
                ink: on ? INK : GREY, size: 8 });
          label(svg, 430, y + 18, k[1], on ? MUTE : FAINT);
          line(pen, svg, 136, 48, 216, y + 14, on ? k[2] : '#dee2e6',
               { width: on ? 1.5 : 1 });
        });
        label(svg, 16, 128, 'Ajan için üçü de aynı arayüz: bir tool kaynağı. ' +
              'Şu an çalışan: ' + (m.tool || ''), MUTE, { size: 8 });
      },
      tool_result: function (pen, svg, m) {
        box(pen, svg, 16, 26, 150, 44, GREEN, (m.tools || ['sonuç'])[0], null,
            { fill: WASH.green, mono: true, size: 8 });
        line(pen, svg, 166, 48, 220, 48, GREEN, { flow: true });
        box(pen, svg, 224, 26, 150, 44, ORANGE, 'model_context', 'sonuç bağlama girdi',
            { fill: WASH.orange });
        line(pen, svg, 374, 48, 428, 48, ORANGE, { flow: true });
        box(pen, svg, 432, 26, 150, 44, VIOLET, 'model', 'tekrar çağrılacak',
            { fill: WASH.violet });
        label(svg, 16, 92, 'ToolCallExecutionEvent — istek ve sonuç AYRI olaylar.', MUTE,
              { size: 8 });
        label(svg, 16, 108, 'Ayrı olmalarının sebebi: aradaki kapı reddedebilir, ' +
              've o zaman sonuç hiç olmaz.', GREY);
      },
      loop: function (pen, svg, m) {
        var used = (m.used || 2), limit = m.limit || 6;
        for (var i = 0; i < limit; i++) {
          var on = i < used;
          box(pen, svg, 16 + i * 62, 30, 52, 32, on ? ORANGE : FAINT, String(i + 1), null,
              { width: on ? 1.8 : 1.1, fill: on ? WASH.orange : 'transparent',
                ink: on ? INK : FAINT });
        }
        line(pen, svg, 16, 76, 16 + limit * 62 - 10, 76, ORANGE,
             { arrow: false, dash: '5 3' });
        label(svg, 16, 100, 'max_tool_iterations = ' + limit + '   ·   varsayılan 1',
              ORANGE, { size: 8.4, weight: '700', mono: true });
        label(svg, 16, 116, 'Varsayılanla ajan bir tool çağırır, sonucu görür ve SUSAR — ' +
              'hata da vermez. Zincirleme davranış sessizce imkânsız.', GREY);
      },
      done: function (pen, svg, m) {
        box(pen, svg, 16, 24, 200, 48, GREEN, 'TaskResult', null,
            { fill: WASH.green, mono: true });
        box(pen, svg, 260, 14, 180, 30, BLUE, 'messages', 'bütün konuşma',
            { width: 1.3, size: 8 });
        box(pen, svg, 260, 52, 180, 30, BLUE, 'stop_reason', m.stop_reason || 'boş',
            { width: 1.3, size: 8 });
        line(pen, svg, 216, 40, 256, 29, BLUE);
        line(pen, svg, 216, 56, 256, 67, BLUE);
        label(svg, 466, 32, (m.llm_calls || 0) + ' LLM çağrısı', MUTE, { size: 8 });
        label(svg, 466, 48, (m.tool_calls || 0) + ' tool', MUTE, { size: 8 });
        label(svg, 466, 64, (m.tokens || 0) + ' token', INK, { size: 8.6, weight: '700' });
        label(svg, 16, 100, 'stop_reason boşsa takım değil tek ajan koştu — ' +
              'sonlandırma koşulu yalnız takımlarda var.', GREY);
      },

      /* ---- scan ---- */
      graph_build: function (pen, svg, m) {
        var b = (m.branches || ['A', 'B', 'C']);
        b.forEach(function (name, i) {
          box(pen, svg, 16, 14 + i * 34, 150, 28, BLUE, name, null,
              { width: 1.4, fill: WASH.blue, size: 7.6 });
          line(pen, svg, 166, 28 + i * 34, 236, 62, BLUE, { width: 1.2 });
        });
        box(pen, svg, 240, 46, 130, 32, ORANGE, 'RiskAuditor', null,
            { fill: WASH.orange, size: 8 });
        line(pen, svg, 370, 62, 414, 62, ORANGE);
        box(pen, svg, 418, 46, 130, 32, GREEN, 'Scorer', 'StructuredMessage',
            { fill: WASH.green, size: 8 });
        label(svg, 190, 40, 'join "all"', BLUE, { weight: '700', size: 7 });
        label(svg, 16, 126, 'DiGraphBuilder → GraphFlow · ' + (m.termination || ''), MUTE,
              { size: 8, mono: true });
        label(svg, 16, 142, 'custom_message_types beyanı olmadan takım StructuredMessage\'ı ' +
              'YÖNLENDİRMEZ.', RED, { size: 7.8, weight: '700' });
      },
      graph_run: function (pen, svg) {
        box(pen, svg, 16, 26, 180, 44, RED, 'run()', 'dal fırlatırsa HEPSİ gider',
            { dash: '5 3', width: 1.4, ink: RED });
        box(pen, svg, 240, 26, 200, 44, GREEN, 'run_stream()', 'ulaşanı elde tutar',
            { fill: WASH.green, width: 2.2 });
        label(svg, 470, 44, 'seçilen', GREEN, { weight: '700', size: 8 });
        label(svg, 470, 60, 'bu', GREEN, { weight: '700', size: 8 });
        label(svg, 16, 92, 'Akıştan okumak, gelmiş olan kısmi sonucu koruyor.', MUTE,
              { size: 8 });
        label(svg, 16, 108, 'Bu, ölçülmüş bir sessiz veri kaybının çaresi — tercih değil.',
              GREY);
      },
      analysts: function (pen, svg, m) {
        var expected = m.expected || 3, arrived = m.arrived || 1;
        for (var i = 0; i < expected; i++) {
          var on = i < arrived;
          box(pen, svg, 16 + i * 190, 26, 170, 44, on ? GREEN : FAINT,
              'AssistantAgent', on ? 'geldi' : 'bekleniyor',
              { width: on ? 2 : 1.1, fill: on ? WASH.green : 'transparent',
                ink: on ? INK : GREY });
        }
        label(svg, 16, 92, (m.branch || '') + ' geldi — ' + arrived + '/' + expected, INK,
              { size: 8.6, weight: '700' });
        label(svg, 16, 108, 'Çok ajanlı olmalarının sebebi zekâ değil AYRIŞTIRMA: ' +
              'üç ayrı kaynak, aynı anda.', GREY);
      },
      join: function (pen, svg, m) {
        var ok = (m.arrived || 0) >= (m.expected || 3);
        box(pen, svg, 16, 26, 200, 44, ok ? GREEN : RED,
            (m.arrived || 0) + ' / ' + (m.expected || 3) + ' dal',
            ok ? 'hepsi geldi' : 'eksik var',
            { fill: ok ? WASH.green : WASH.red, width: 2 });
        line(pen, svg, 216, 48, 264, 48, ok ? GREEN : RED);
        box(pen, svg, 268, 26, 354, 44, GREY, m.stop_reason ? 'stop_reason' : 'join',
            (m.stop_reason || '').slice(0, 46), { width: 1.3, size: 8 });
        label(svg, 16, 92, 'Bariyere SORULMUYOR — beklenen dal sayısı sayılıyor.', MUTE,
              { size: 8 });
        label(svg, 16, 108, 'stop_reason "abandoned" dese bile eldeki sonuç geçerli ' +
              'olabilir; karar sayıma ait.', GREY);
      },
      count: function (pen, svg, m) {
        var miss = (m.missing || []);
        box(pen, svg, 16, 26, 190, 44, GREEN, (m.succeeded || 0) + ' dal başarılı', null,
            { fill: WASH.green });
        box(pen, svg, 230, 26, 200, 44, miss.length ? RED : GREY,
            miss.length + ' eksik', 'missing_data',
            { fill: miss.length ? WASH.red : 'transparent',
              width: miss.length ? 1.8 : 1.1, ink: miss.length ? INK : GREY });
        label(svg, 16, 92, 'Sessiz eksik sonuç, BEYAN EDİLMİŞ bilgi yokluğuna çevriliyor.',
              MUTE, { size: 8 });
        label(svg, 16, 108, miss.length ? miss.join(' · ').slice(0, 96)
              : 'Bu koşuda eksik dal yok.', GREY);
      },
      intervention: function (pen, svg) {
        box(pen, svg, 16, 26, 130, 44, ORANGE, 'mesaj', null, { fill: WASH.orange });
        line(pen, svg, 146, 48, 194, 48, INK);
        box(pen, svg, 198, 20, 200, 58, RED, 'InterventionHandler',
            'on_send / on_publish', { fill: WASH.red, width: 2.2, mono: true, size: 8 });
        line(pen, svg, 398, 48, 446, 48, GREEN, { flow: true });
        box(pen, svg, 450, 26, 172, 44, GREEN, 'ajan', 'denetim kaydına yazıldı',
            { fill: WASH.green });
        label(svg, 16, 100, 'Runtime\'a takılan TEK kapı — ve takmak için runtime\'ı ' +
              'kendin kurmak zorundasın.', MUTE, { size: 8 });
        label(svg, 16, 116, 'Bedeli ölçüldü: kendi runtime\'ında çöken ajan fırlatmıyor, ' +
              'ASILIYOR.', RED, { size: 7.8, weight: '700' });
      },
      runtime_start: function (pen, svg) {
        box(pen, svg, 16, 26, 240, 44, VIOLET, 'SingleThreadedAgentRuntime',
            'start() — mesaj döngüsü', { fill: WASH.violet, mono: true, size: 8 });
        label(svg, 16, 92, 'Şirket başına kuruluyor ve kapatılıyor.', MUTE, { size: 8 });
        label(svg, 16, 108, 'Gateway\'in runtime\'ı ise sürekli koşuyor — ikisi ayrı ' +
              'ömür.', GREY);
      },
      runtime_stop: function (pen, svg) {
        box(pen, svg, 16, 26, 200, 44, ORANGE, 'stop_when_idle()', '5 sn kapaklı',
            { fill: WASH.orange, mono: true, size: 8 });
        line(pen, svg, 216, 48, 264, 48, ORANGE);
        box(pen, svg, 268, 26, 160, 44, RED, 'stop()', 'dolarsa sert kapat',
            { dash: '4 3', width: 1.4, ink: RED, size: 8 });
        line(pen, svg, 428, 48, 470, 48, INK);
        box(pen, svg, 474, 26, 148, 44, GREY, 'close()', null, { width: 1.3, size: 8 });
        label(svg, 16, 92, 'Bariyer burada KAPATMA aracı — sonuç toplamada kullanılmıyor.',
              MUTE, { size: 8 });
      }
    };

    function draw() {
      if (!window.Rough) { return; }
      canvas.textContent = '';
      var pen = new Rough.Pen(9137);
      var svg = Rough.svg(640, 150);
      var id = last ? last.id : '';
      var fn = FIGURES[id];
      if (fn) {
        fn(pen, svg, (last && last.meta) || {});
      } else {
        // Between turns there is no mechanism to draw, so draw the machine that
        // is standing by instead of an empty frame.
        var run = runs[mode] || {};
        box(pen, svg, 16, 24, 290, 46, GREY, run.team || '—', 'takım',
            { width: 1.3, ink: MUTE });
        box(pen, svg, 330, 24, 292, 46, GREY, run.pattern || '—', 'desen',
            { width: 1.3, ink: MUTE });
        label(svg, 16, 92, run.team_note || '', GREY);
        label(svg, 16, 108, run.pattern_note || '', GREY);
        label(svg, 16, 130, 'boşta — bir soru sor, o an ne koşuyorsa burada çizilir.',
              FAINT);
      }
      canvas.appendChild(svg);
    }

    /* The run header: which team, which pattern, which message types. Answers the
       questions people actually have about an AutoGen run, and the chat answer is
       the surprising one — there is no team. */
    function header() {
      var run = runs[mode];
      if (!run) { return; }
      var bar = document.getElementById('mech-run');
      if (!bar) { return; }
      bar.textContent = '';
      [['takım', run.team], ['desen', run.pattern],
       ['mesaj', (run.messages || []).join(' · ')]].forEach(function (pair) {
        var cell = el('span', 'mech__fact');
        cell.appendChild(el('span', 'mech__factkey', pair[0]));
        cell.appendChild(el('code', 'mech__factval', pair[1]));
        bar.appendChild(cell);
      });
    }

    var timer = null, since = 0;
    function tick() {
      if (!active) { return; }
      var slot = note.querySelector('.mech__clock');
      if (slot) { slot.textContent = ((Date.now() - since) / 1000).toFixed(1) + ' sn'; }
    }
    function startClock() {
      since = Date.now();
      if (timer) { clearInterval(timer); }
      timer = setInterval(tick, 100);
    }
    function stopClock() { if (timer) { clearInterval(timer); timer = null; } }

    function say(id, meta) {
      var m = catalogue[id];
      if (!m) { return; }
      note.textContent = '';
      note.appendChild(el('span', 'mech__lane mech__lane--' + m.lane, m.lane));
      note.appendChild(el('strong', null, m.title));
      note.appendChild(el('code', 'mech__class', m.klass));
      note.appendChild(el('span', 'mech__ref', m.ref));
      note.appendChild(el('span', 'mech__clock', '0.0 sn'));
      if (m.module) { note.appendChild(el('code', 'mech__mod', m.module)); }
      note.appendChild(el('span', 'mech__say', m.note));
      if (meta && meta.blocked) {
        note.appendChild(el('span', 'mech__meta', 'REDDEDİLDİ: ' + (meta.reason || '')));
      }
    }

    /* Between turns: rotate through what is actually attached, read from health. */
    var idleTimer = null, idleAt = 0;

    function idleFacts(h) {
      var oc = h.openclaw || {}, rt = h.runtime || {}, ctx = h.context || {};
      return [
        { lane: 'core', title: 'OpenClaw köprüsü', module: 'pipeline/openclaw.py',
          klass: oc.attached ? 'McpWorkbench · stdio' : 'openclaw mcp serve',
          note: oc.attached ? (oc.status || '') + ' · gönderme ' +
                (oc.outbound_gated ? 'KAPILI' : 'açık')
              : 'bağlı değil: ' + (oc.status || 'denenmedi') },
        { lane: 'core', title: 'DeepWiki', klass: 'McpWorkbench · streamable HTTP',
          module: 'pipeline/conversation.py', note: h.mcp || 'not attempted' },
        { lane: 'core', title: 'Gateway runtime', klass: 'SingleThreadedAgentRuntime',
          module: 'pipeline/gateway/runtime.py',
          note: (rt.running ? 'koşuyor' : 'kapalı') + ' · yönlendirilen: ' + (rt.routed || 0) },
        { lane: 'ours', title: 'Bağlam bütçesi', klass: 'CompactingChatCompletionContext',
          module: 'pipeline/context_engine.py',
          note: ctx.active ? (ctx.tokens || 0) + ' / ' + (ctx.budget || 0) + ' token'
              : 'bu oturumda henüz bağlam kurulmadı' },
        { lane: 'ours', title: 'Onay kapısı', klass: 'ApprovalGate',
          module: 'pipeline/gateway/approval.py',
          note: (h.approvals_pending || 0) + ' bekleyen · oturum: ' + (h.sessions || 0) }
      ];
    }

    function showIdle(f) {
      note.textContent = '';
      note.appendChild(el('span', 'mech__lane mech__lane--' + f.lane, f.lane));
      note.appendChild(el('strong', null, f.title));
      note.appendChild(el('code', 'mech__class', f.klass));
      note.appendChild(el('code', 'mech__mod', f.module));
      note.appendChild(el('span', 'mech__say', f.note));
    }

    function idleLoop() {
      if (active) { return; }
      fetch('/api/health').then(function (r) { return r.json(); }).then(function (h) {
        if (active) { return; }
        runtime = h.runtime || runtime;
        var facts = idleFacts(h);
        showIdle(facts[idleAt % facts.length]);
        idleAt += 1;
      }).catch(function () {});
    }

    function startIdle() {
      if (idleTimer) { clearInterval(idleTimer); }
      idleLoop();
      idleTimer = setInterval(idleLoop, 4000);
    }

    function apply() {
      body.hidden = collapsed;
      toggle.textContent = collapsed ? 'göster' : 'gizle';
      document.body.classList.toggle('mech-collapsed', collapsed);
    }

    toggle.addEventListener('click', function () {
      collapsed = !collapsed;
      localStorage.setItem('mech-collapsed', collapsed ? '1' : '0');
      apply();
    });
    apply();

    /* ── scheduled jobs ──────────────────────────────────────────────────
       Read from OpenClaw every time rather than cached here. A second copy
       would drift the moment somebody used `openclaw automations` directly,
       and a listing you cannot trust is worse than no listing.

       "Cannot reach the scheduler" is drawn as its own state, never as an
       empty list: the difference between "nothing is scheduled" and "I cannot
       tell you" is the whole reason to look. */
    var cronHost = document.getElementById('mech-cron');

    function drawCron(data) {
      if (!cronHost) { return; }
      cronHost.innerHTML = '';
      if (!data.reachable) {
        cronHost.hidden = false;
        cronHost.appendChild(el('div', 'cron-off',
          'Zamanlayıcıya ulaşılamıyor — ' + (data.note || 'OpenClaw Gateway kapalı')));
        return;
      }
      var jobs = data.jobs || [];
      if (!jobs.length) { cronHost.hidden = true; return; }

      cronHost.hidden = false;
      cronHost.appendChild(el('div', null, 'ZAMANLANMIŞ İŞ · ' + jobs.length));
      jobs.forEach(function (job) {
        var row = el('div', 'cron-row');
        row.appendChild(el('span', null, job.name));
        row.appendChild(el('span', 'cron-when', job.when));
        row.appendChild(el('span', null, job.enabled ? (job.last || '—') : 'kapalı'));
        cronHost.appendChild(row);
      });
      if (data.linger_warning) {
        cronHost.appendChild(el('div', 'cron-warn', data.linger_warning));
      }
    }

    function loadCron() {
      if (!cronHost) { return; }
      fetch('/api/schedule')
        .then(function (r) { return r.json(); })
        .then(drawCron)
        .catch(function () { /* the panel is an explainer; it never breaks a turn */ });
    }

    return {
      cron: loadCron,
      // The terminal lives inside this closure with the rest of the panel, but
      // the approval card is outside it and needs to replay a run into it.
      term: { code: termCode, result: termResult, close: termClose },
      enable: function () {
        loadCron();
        if (Object.keys(catalogue).length) {
          host.hidden = false;
          document.body.classList.add('has-mech');
          return;
        }
        fetch('/api/mechanisms').then(function (r) { return r.json(); }).then(function (d) {
          (d.mechanisms || []).forEach(function (m) { catalogue[m.id] = m; });
          runs = d.runs || {};
          coreNote = d.core_idle_note || '';
          runtime = d.runtime || null;
          host.hidden = false;
          document.body.classList.add('has-mech');
          apply();
          header();
          draw();
          startIdle();
        }).catch(function () {});
      },
      begin: function (which) {
        mode = which === 'scan' ? 'scan' : 'chat';
        active = ''; last = null; counts.textContent = '';
        traceReset();
        stopClock();
        header();
        draw();
      },
      /* A turn can end three ways and only one of them emits a stage. `done`
         arrives on success; a cancel or an error arrives as its own SSE event
         and the panel never hears about it — so the clock kept counting on a
         turn that had already failed. Twenty seconds of "still working" under
         a visible error message is the panel lying about the system it exists
         to explain. */
      ended: function (why, message) {
        stopClock();
        active = '';
        if (traceHost && traceT0) {
          var row = el('li', 'is-block');
          row.appendChild(el('span', 't-at',
            '+' + ((Date.now() - traceT0) / 1000).toFixed(2) + 's'));
          row.appendChild(el('span', 't-lane', why === 'cancelled' ? 'durdu' : 'hata'));
          row.appendChild(el('span', 't-what',
            why === 'cancelled' ? 'Kullanıcı durdurdu' : 'Tur düştü'));
          row.appendChild(el('span', 't-meta', (message || '').slice(0, 60)));
          traceHost.appendChild(row);
          traceHost.scrollTop = traceHost.scrollHeight;
        }
        draw();
        setTimeout(function () { if (!active) { idleLoop(); } }, 8000);
      },
      stage: function (event) {
        var meta = event.meta || {};
        var finished = event.id === 'done' || event.id === 'runtime_stop';
        last = event;
        active = finished ? '' : event.id;
        if (finished) {
          stopClock();
          if (event.id === 'done') {
            counts.textContent = (meta.llm_calls || 0) + ' LLM · ' +
              (meta.tool_calls || 0) + ' tool · ' + (meta.tokens || 0) + ' token';
          }
        }
        draw();
        say(event.id, meta);
        traceAdd(event, meta);
        // The terminal is fed by the two code stages and nothing else, so it
        // stays empty for every turn that did not run code.
        if (event.id === 'code_request') { termCode(meta.code); }
        else if (event.id === 'code_result') { termResult(meta); }
        if (finished) {
          setTimeout(function () { if (!active) { idleLoop(); } }, 8000);
        } else {
          startClock();
        }
      }
    };
  })();

  // The escape hatch. `/openclaw sessions.list` skips the model entirely and
  // prints what the Gateway actually returned — no paraphrase, because the
  // reason to type this instead of asking is that you want the bytes.
  var OPENCLAW = '/openclaw';

  function runOpenClaw(line) {
    addTurn('user', null, function (bubble) {
      bubble.appendChild(document.createTextNode(line));
    });
    var pending = addTurn('bot', { title: 'OpenClaw', path: 'openclaw' }, function (bubble) {
      bubble.appendChild(el('div', 'thinking', '…'));
    });

    state.busy = true;
    fetch('/api/openclaw', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ line: line })
    }).then(function (response) {
      return response.json();
    }).then(function (data) {
      // A held line never ran. Say so where the answer would have been, and hand
      // the decision to the same approval card the chat path uses — one queue,
      // one button, whichever way the call was made.
      if (data.held) {
        pending.textContent = '';
        pending.appendChild(el('div', 'thinking', 'onay bekliyor'));
        state.busy = false;
        askApproval(
          { id: data.approval_id, reason: data.reason },
          function () { runOpenClaw(line); }
        );
        return;
      }
      pending.textContent = '';
      var head = el('div', 'bubble__head');
      head.appendChild(el('span', null, data.method || OPENCLAW));
      if (data.tier) {
        var badge = el('span', 'path path--openclaw');
        badge.appendChild(el('span', 'path__dot'));
        badge.appendChild(el('span', null, data.tier));
        head.appendChild(badge);
      }
      pending.appendChild(head);

      var body = data.ok ? data.result : (data.error || 'no answer');
      // A frame is the one result worth showing rather than printing. The server
      // hands back an id and a URL; the id is what it is addressed by, and the
      // path never comes from anything typed here.
      if (data.ok && body && body.url && /^\/api\/shot\/[0-9a-f]{32}$/.test(body.url)) {
        var shot = el('img', 'shot');
        shot.src = body.url;
        shot.alt = 'kamera karesi';
        shot.loading = 'lazy';
        pending.appendChild(shot);
        var by = el('div', 'approval__note',
          body.by === 'local'
            ? 'kare yerel ffmpeg ile alındı — OpenClaw üretmedi'
            : 'kareyi OpenClaw çekti');
        pending.appendChild(by);
        if (data.note) { pending.appendChild(el('div', 'approval__note', data.note)); }
        return;
      }
      if (data.tier === 'chat' && typeof body === 'string') {
        // An answer from OpenClaw's agent is prose. Putting it in the raw box
        // would make a sentence scroll sideways for no reason — the raw box is a
        // promise about untouched bytes, and prose has no bytes worth guarding.
        pending.appendChild(el('div', 'prose', body));
      } else {
        var text = typeof body === 'string' ? body : JSON.stringify(body, null, 2);
        pending.appendChild(el('pre', 'raw', text));
      }
      if (data.usage) { pending.appendChild(el('div', 'approval__note', data.usage)); }
    }).catch(function (error) {
      pending.textContent = '';
      pending.appendChild(el('pre', 'raw', 'request failed: ' + error.message));
    }).then(function () {
      state.busy = false;
      scrollToEnd(pending);
    });
  }

  function ask(question) {
    if (!question || state.busy) { return; }
    // Checked before the scan guard on purpose: OpenClaw has nothing to do with
    // whether a scan has been loaded, and being unable to debug the bridge until
    // you run a scan would be a silly place to get stuck.
    if (question.indexOf(OPENCLAW) === 0) { runOpenClaw(question); return; }
    if (!state.scan) { return; }
    if (state.live) { streamAsk(question); return; }

    addTurn('user', null, function (bubble) { bubble.appendChild(document.createTextNode(question)); });

    var pending = addTurn('bot', { title: state.live ? 'Thinking' : 'Looking it up' }, function (bubble) {
      var dots = el('div', 'thinking');
      dots.appendChild(el('span')); dots.appendChild(el('span')); dots.appendChild(el('span'));
      bubble.appendChild(dots);
    });

    fetch('/api/ask', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question, scan: state.scan })
    })
      .then(function (response) { return response.json(); })
      .then(function (result) {
        pending.parentNode.remove();
        addTurn('bot', { title: result.title, path: result.path }, function (bubble) {
          if (result.text) { bubble.appendChild(el('p', 'prose', result.text)); }
          if (result.html) {
            var wrap = document.createElement('div');
            if (result.text && result.supporting_title) {
              wrap.className = 'evidence';
              wrap.appendChild(el('div', 'evidence__head', 'Evidence · ' + result.supporting_title));
            }
            var body = document.createElement('div');
            body.innerHTML = result.html;
            wrap.appendChild(body);
            bubble.appendChild(wrap);
          }
        });
      })
      .catch(function (error) {
        pending.parentNode.remove();
        addTurn('bot', { title: 'Could not reach the backend' }, function (bubble) {
          bubble.appendChild(el('p', null, String(error) + ' — is the server still running?'));
        });
      });
  }

  form.addEventListener('submit', function (event) {
    event.preventDefault();
    var value = input.value.trim();
    if (!value) { return; }
    input.value = '';
    ask(value);
  });

  // ---------------------------------------------------------------- state

  function renderChips() {
    chips.innerHTML = '';
    CHIPS.forEach(function (label) {
      var chip = el('button', 'chip', label);
      chip.type = 'button';
      chip.setAttribute('data-ask', label);
      chips.appendChild(chip);
    });
    bind(chips);
  }

  function loadState(scanName) {
    return fetch('/api/state' + (scanName ? '?scan=' + encodeURIComponent(scanName) : ''))
      .then(function (response) { return response.json(); })
      .then(function (data) {
        state.live = data.live_llm;
        modeBadge.hidden = false;
        modeBadge.textContent = data.live_llm ? 'live model' : 'no model';
        modeBadge.className = 'badge' + (data.live_llm ? ' badge--live' : '');

        if (resetButton) { resetButton.hidden = !data.live_llm; }
        // Only a live agent has mechanisms to report. Without a model the answers
        // come from scan data down a different path, and drawing an AutoGen turn
        // over it would describe machinery that did not run.
        if (data.live_llm) { mech.enable(); }
        var mcpNote = data.mcp && data.mcp.indexOf('connected') === 0
          ? ' DeepWiki attached.' : '';
        disclaimer.textContent = data.live_llm
          ? 'A live agent with memory, tools and a scan of its own.' + mcpNote
          : 'No model configured — answers are built from this scan’s data. '
            + 'Set ' + (data.missing_llm || []).join(', ') + ' to enable one.';

        var meta = document.getElementById('scan-meta');
        var stats = document.getElementById('rail-stats');
        if (data.has_scan && meta) {
          meta.textContent = 'last ' + data.days + ' days · ' + (data.mode || '—') + ' mode';
        } else if (meta) {
          meta.textContent = '';
        }
        if (stats) {
          var f = data.funnel || {};
          var rows = [
            ['Signals', f.signals], ['Companies', f.companies],
            ['Passed triage', f.triage_passed], ['Enriched', f.enriched],
            ['Memos', f.memos]
          ];
          stats.innerHTML = '';
          stats.hidden = !data.has_scan;
          rows.forEach(function (row) {
            var wrap = document.createElement('div');
            var dt = el('dt', null, row[0]);
            var dd = el('dd', row[1] ? null : 'is-zero', row[1] == null ? '—' : String(row[1]));
            wrap.appendChild(dt); wrap.appendChild(dd);
            stats.appendChild(wrap);
          });
        }

        picker.innerHTML = '';
        (data.scans || []).forEach(function (name) {
          var option = el('option', null, name.replace('scan-', '').replace('.json', ''));
          option.value = name;
          picker.appendChild(option);
        });
        picker.hidden = !(data.scans || []).length;

        thread.innerHTML = '';
        if (!data.has_scan) {
          document.getElementById('scan-query').textContent = 'no scan yet';
          addTurn('bot', { title: 'Nothing scanned yet' }, function (bubble) {
            var wrap = el('div', 'empty');
            wrap.appendChild(el('h2', null, 'Run the first scan'));
            wrap.appendChild(el('p', null,
              'Collectors are keyless: Hacker News, SEC Form D and GitHub. Choose a sector '
              + 'and a window, and the funnel runs top to bottom.'));
            var go = el('button', 'primary', 'New scan');
            go.type = 'button';
            go.addEventListener('pointerdown', openSheet);
            wrap.appendChild(go);
            bubble.appendChild(wrap);
          });
          state.scan = null;
          return;
        }

        state.scan = data.source;
        picker.value = data.source;
        document.getElementById('scan-query').textContent = data.query;

        addTurn('bot', { title: 'Scan report', path: 'rules' }, function (bubble) {
          if (data.banners) {
            var banners = document.createElement('div');
            banners.innerHTML = data.banners;
            bubble.appendChild(banners);
          }
          var body = document.createElement('div');
          body.innerHTML = data.opening.html;
          bubble.appendChild(body);
        });
      });
  }

  picker.addEventListener('change', function () { loadState(picker.value); });

  // ---------------------------------------------------------------- scanning

  function openSheet() { sheet.showModal(); }
  document.getElementById('new-scan').addEventListener('pointerdown', openSheet);
  document.getElementById('scan-cancel').addEventListener('pointerdown', function () { sheet.close(); });

  document.getElementById('scan-form').addEventListener('submit', function () {
    var body = {
      query: document.getElementById('f-query').value,
      days: parseInt(document.getElementById('f-days').value, 10),
      limit: parseInt(document.getElementById('f-limit').value, 10)
    };
    fetch('/api/scan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body)
    }).then(function (response) {
      if (!response.ok) { return response.json().then(function (e) { throw new Error(e.detail); }); }
      startLog(body);
    }).catch(function (error) {
      addTurn('bot', { title: 'Could not start the scan' }, function (bubble) {
        bubble.appendChild(el('p', null, String(error.message || error)));
      });
    });
  });

  function startLog(args) {
    var bubble = addTurn('bot', { title: 'Scanning' }, function (node) {
      node.appendChild(el('p', null,
        'Looking for ' + args.query + ' over the last ' + args.days + ' days.'));
      node.appendChild(el('pre', 'log', ''));
      node.appendChild(el('div', 'log__meta', 'running…'));
    });
    var log = bubble.querySelector('.log');
    var meta = bubble.querySelector('.log__meta');

    if (state.pollTimer) { clearInterval(state.pollTimer); }
    // The scan is the one place core actually runs, so the panel switches to the
    // pub/sub diagram for the duration and lights it from the subprocess's own
    // stage lines.
    mech.begin('scan');
    var seen = 0;
    state.pollTimer = setInterval(function () {
      fetch('/api/scan?since=' + seen).then(function (r) { return r.json(); }).then(function (run) {
        log.textContent = (run.lines || []).join('\n');
        log.scrollTop = log.scrollHeight;
        // Replay them in order rather than jumping to the last: the sequence is
        // the thing worth seeing, and a poll can carry several at once.
        (run.stages || []).forEach(function (s) { mech.stage(s); });
        if (typeof run.stage_count === 'number') { seen = run.stage_count; }
        if (!run.running) {
          clearInterval(state.pollTimer);
          state.pollTimer = null;
          var seconds = run.finished_at && run.started_at
            ? Math.round(run.finished_at - run.started_at) : 0;
          meta.textContent = run.exit_code === 0
            ? 'finished in ' + seconds + 's — loading the result'
            : 'exited with code ' + run.exit_code + ' after ' + seconds + 's';
          if (run.exit_code === 0) { loadState(); }
        }
      });
    }, 900);
  }

  // ---------------------------------------------------------------- boot

  var themeButton = document.getElementById('theme-toggle');
  if (themeButton) {
    // The product opens light. Dark is still designed and measured, so it stays
    // one click away rather than being deleted; the choice is remembered.
    var saved = null;
    try { saved = localStorage.getItem('vc-theme'); } catch (e) { saved = null; }
    var applyTheme = function (theme) {
      document.documentElement.setAttribute('data-theme', theme);
      themeButton.textContent = theme === 'dark' ? 'Light' : 'Dark';
      try { localStorage.setItem('vc-theme', theme); } catch (e) { /* private mode */ }
    };
    applyTheme(saved === 'dark' ? 'dark' : 'light');
    themeButton.addEventListener('pointerdown', function () {
      applyTheme(document.documentElement.getAttribute('data-theme') === 'dark' ? 'light' : 'dark');
    });
  }

  renderChips();
  loadState().catch(function (error) {
    disclaimer.textContent = 'Backend unreachable: ' + error;
  });
})();
