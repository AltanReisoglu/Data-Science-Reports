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
          docs: 'from the docs', rules: 'from scan data'
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
      if (event.type === 'chunk') {
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
        if (refusal) { askApproval(refusal, question); }
      } else if (event.type === 'done') {
        if (!text.textContent && event.text) { text.textContent = event.text; }
      } else if (event.type === 'cancelled') {
        text.textContent += '\n[stopped]';
      } else if (event.type === 'error') {
        text.textContent += '\n[' + event.message + ']';
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

  function askApproval(refusal, retryQuestion) {
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
          status.textContent = verb === 'approve' ? 'approved' : 'denied';
          // The grant covers exactly this call and is consumed by it, so the
          // question has to be asked again for the tool to run.
          if (verb === 'approve' && retryQuestion) {
            status.textContent = 'approved · asking again';
            setTimeout(function () { streamAsk(retryQuestion); }, 250);
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

  function ask(question) {
    if (!question || !state.scan || state.busy) { return; }
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
    state.pollTimer = setInterval(function () {
      fetch('/api/scan').then(function (r) { return r.json(); }).then(function (run) {
        log.textContent = (run.lines || []).join('\n');
        log.scrollTop = log.scrollHeight;
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
