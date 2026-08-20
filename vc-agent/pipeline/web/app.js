/* Chat client for the local backend.

   The server owns routing and rendering: this file sends a question and injects
   the HTML that comes back. Answers arrive tagged with the path that produced
   them — `model` or `rules` — and the tag is shown, because a model answer and a
   deterministic one carry different warranties. */

(function () {
  var thread = document.getElementById('thread');
  var deckHost = document.getElementById('deck');
  var deckBar = document.getElementById('deck-bar');
  var deckFrame = document.getElementById('deck-frame');
  var deckState = { id: null };
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
  var flowButton = document.getElementById('flow-open');

  var state = { scan: null, live: false, busy: false, pollTimer: null,
                run: null, framework: 'autogen', report: null };

  /* Rapor istendiğinde basılıyor. Açılışta değil: sunumda ekranın ortasını
     kaplıyordu ve orası artık slaytın. */
  function showReport() {
    var data = state.report;
    if (!data) { return; }
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
  }

  /* The flow screen draws one turn. It has to be *this* turn, so the id comes
     from the stream that produced it rather than from "whatever ran last" — two
     tabs asking at once would otherwise each open the other's question. */
  function armFlow(runId) {
    state.run = runId || null;
    if (flowButton) { flowButton.disabled = !state.run; }
  }

  /* Takım koşusu. Sohbetten ayrı bir yol, çünkü farklı bir şey: sohbet turu
     tek ajan ve öyle kalmalı — beş takım tipini görebilmek için ikinci bir
     düğme, birincinin dürüstlüğünü bozmaktan ucuz. */
  /* Çerçeve anahtarı. `state.framework` bütün soru yollarını etkiliyor:
     AutoGen'de `/api/chat`, MAF'ta `/api/maf`. Aynı kutuya aynı soruyu yazıp
     iki çerçeveyi yan yana görmek, bu ekranın en çok işe yarayan hâli. */
  var fwButton = document.getElementById('fw-toggle');

  function paintFramework() {
    if (!fwButton) { return; }
    var maf = state.framework === 'maf';
    fwButton.classList.toggle('is-maf', maf);
    fwButton.setAttribute('aria-pressed', maf ? 'true' : 'false');
    fwButton.querySelector('.fw__label').textContent = maf ? 'MAF' : 'AutoGen';
    if (teamPick) { teamPick.disabled = maf; }
    if (teamButton) { teamButton.disabled = maf; }
  }

  function loadFramework() {
    if (!fwButton) { return; }
    fetch('/api/maf').then(function (r) { return r.json(); }).then(function (d) {
      if (!d.available) { return; }         // kurulu değilse düğme hiç çıkmıyor
      fwButton.hidden = false;
      fwButton.title = 'AutoGen ↔ Microsoft Agent Framework';
      fwButton.addEventListener('click', function () {
        state.framework = state.framework === 'maf' ? 'autogen' : 'maf';
        paintFramework();
        addTurn('bot', { title: 'mod', path: 'docs' }, function (b) {
          b.appendChild(document.createTextNode(
            state.framework === 'maf'
              ? 'Microsoft Agent Framework moduna geçildi. Sorular ayrı bir '
                + 'sanal ortamdaki MAF ajanına gidiyor; takım tipleri AutoGen\'e '
                + 'ait olduğu için kapalı.'
              : 'AutoGen moduna dönüldü.'));
        });
      });
      paintFramework();
    }).catch(function () {});
  }

  /* MAF turu. Sohbetle aynı kutu, farklı uç — ve akış ekranı ikisini de aynı
     katalogdan çizdiği için karşılaştırma bedavaya geliyor. */
  function askMaf(question) {
    addTurn('user', null, function (b) {
      b.appendChild(document.createTextNode('[MAF] ' + question));
    });
    var bubble = addTurn('bot', { title: 'MAF', path: 'model' }, function (node) {
      node.appendChild(el('div', 'thinking', 'MAF koşuyor…'));
    });
    setBusy(true);
    fetch('/api/maf', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question: question })
    }).then(function (r) {
      var reader = r.body.getReader(), decoder = new TextDecoder(), buffer = '';
      var out = el('div', 'streaming');
      bubble.textContent = '';
      bubble.appendChild(out);
      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done) { setBusy(false); return; }
          buffer += decoder.decode(chunk.value, { stream: true });
          var parts = buffer.split('\n\n');
          buffer = parts.pop();
          parts.forEach(function (part) {
            if (part.indexOf('data: ') !== 0) { return; }
            var ev = JSON.parse(part.slice(6));
            if (ev.type === 'run') { armFlow(ev.id); }
            else if (ev.type === 'stage') { out.textContent += '· ' + ev.title + '\n'; }
            else if (ev.type === 'done') {
              out.textContent += '\n' + (ev.text
                || '(MAF bu turda metin döndürmedi — tool çağrısından sonra '
                 + 'response.text boş kalıyor; akış ekranında ölçümü var.)');
            } else if (ev.type === 'error') {
              out.textContent += '\n[' + ev.message + ']';
            }
            scrollToEnd(bubble);
          });
          return pump();
        });
      }
      return pump();
    }).catch(function (e) { bubble.textContent = String(e.message || e); setBusy(false); });
  }

  var teamPick = document.getElementById('team-kind');
  var teamButton = document.getElementById('team-run');

  function loadTeams() {
    if (!teamPick || !teamButton) { return; }
    fetch('/api/teams').then(function (r) { return r.json(); }).then(function (d) {
      if (!d.available) { return; }
      (d.kinds || []).forEach(function (k) {
        var o = el('option', null, k.id + ' · ' + k.picker);
        o.value = k.id;
        teamPick.appendChild(o);
      });
      teamPick.hidden = false;
      teamButton.hidden = false;
    }).catch(function () {});
  }

  function askTeam(question) {
    var kind = teamPick.value || 'roundrobin';
    addTurn('user', null, function (b) {
      b.appendChild(document.createTextNode('[' + kind + '] ' + question));
    });
    var bubble = addTurn('bot', { title: kind, path: 'model' }, function (node) {
      node.appendChild(el('div', 'thinking', 'takım koşuyor…'));
    });
    setBusy(true);
    fetch('/api/team', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ kind: kind, question: question })
    }).then(function (r) {
      var reader = r.body.getReader(), decoder = new TextDecoder(), buffer = '';
      var lines = el('div', 'streaming');
      bubble.textContent = '';
      bubble.appendChild(lines);
      function pump() {
        return reader.read().then(function (chunk) {
          if (chunk.done) { setBusy(false); return; }
          buffer += decoder.decode(chunk.value, { stream: true });
          var parts = buffer.split('\n\n');
          buffer = parts.pop();
          parts.forEach(function (part) {
            if (part.indexOf('data: ') !== 0) { return; }
            var ev = JSON.parse(part.slice(6));
            if (ev.type === 'run') { armFlow(ev.id); }
            else if (ev.type === 'message' && ev.source) {
              lines.textContent += ev.source + ': ' + (ev.text || '').slice(0, 400) + '\n\n';
            } else if (ev.type === 'done') {
              lines.textContent += '\n— ' + (ev.stop_reason || 'bitti') + ' —';
            } else if (ev.type === 'error') {
              lines.textContent += '\n[' + ev.message + ']';
            }
            scrollToEnd(bubble);
          });
          return pump();
        });
      }
      return pump();
    }).catch(function (e) {
      bubble.textContent = String(e.message || e);
      setBusy(false);
    });
  }

  if (teamButton) {
    teamButton.addEventListener('click', function () {
      var q = (input.value || '').trim();
      if (!q) { return; }
      input.value = '';
      askTeam(q);
    });
  }

  if (flowButton) {
    flowButton.addEventListener('click', function () {
      // Adlandırılmış hedef: ikinci basış yeni sekme açmıyor, açık olanı bu
      // turun akışına çeviriyor. Yoksa sunum ortasında altı akış sekmesi olur.
      window.open('/akis?run=' + encodeURIComponent(state.run || 'latest'), 'vc-akis');
    });
  }

  var CHIPS = ['Tarama raporu', 'The funnel', 'What is missing', 'Cost', 'Candidates',
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

  /* Deste görünümü. Varsayılan açılış burası — sunarken slayt ile PoC aynı
     pencerede duruyor. Soru sorulur sorulmaz sohbete geçiliyor: cevabı
     göstermeyen bir sohbet kutusu işe yaramaz. */
  function showDeck(on) {
    if (!deckHost) { return; }
    // Sohbet ARTIK gizlenmiyor: deste kendi sütununda duruyor ve soru sorulunca
    // kaybolmuyor. İlk hâlinde ikisi aynı yeri paylaşıyordu ve her mesajda
    // slayt yok oluyordu.
    deckHost.hidden = !on;
    document.body.classList.toggle('deck-open', !!on);
    if (on) {
      document.documentElement.style.setProperty('--deckw', deckWidth() + 'px');
    }
    if (!on && deckBar) {
      [].forEach.call(deckBar.querySelectorAll('.deck__tab'), function (b) {
        b.classList.remove('is-on');
      });
    }
  }

  function pickDeck(id) {
    if (!deckFrame) { return; }
    if (id === 'chat') { showDeck(false); return; }
    // `view=Fit` SAYFANIN TAMAMINI çerçeveye sığdırıyor. `FitH` genişliğe
    // sığdırıyordu ve geniş bir ekranda slayt taşıyordu — bir slaytta okunması
    // gereken şey sayfanın bütünü. `navpanes=0` küçük resim şeridini kapatıyor.
    deckState.id = id;
    // Gezinme Chrome'un kendi PDF görüntüleyicisinde: araç çubuğu, sayfa
    // numarası, yakınlaştırma, arama ve kaydırma onda zaten var. Kendi tıklama
    // katmanımız aynı işi daha kötü yapıyordu — her sayfa değişiminde PDF
    // yeniden yükleniyor ve göz kırpıyordu.
    // `view=Fit` sayfanın tamamını sığdırıyor; `navpanes=0` dar sütunda yer
    // kaplayan küçük resim şeridini kapatıyor (araç çubuğundan geri açılabilir).
    deckFrame.src = '/deck/' + encodeURIComponent(id) + '#view=Fit&navpanes=0';
    showDeck(true);
    [].forEach.call(deckBar.querySelectorAll('.deck__tab'), function (b) {
      b.classList.toggle('is-on', b.dataset.deck === id);
    });
  }

  /* Deste genişliği. Sabit 32rem iki işi de yarım yapıyordu — slayta bakarken
     dar, sohbeti okurken geniş. Sürükleyip bırakıyorsun ve tercih kalıyor.
     Sınırlar: sohbete en az 22rem, desteye en az 20rem. */
  var DECK_MIN = 20 * 16, CHAT_MIN = 22 * 16, DECK_DEFAULT = 32 * 16;
  var grip = document.getElementById('grip');

  function clampDeck(px) {
    // Sol ray 15.5rem; ondan ve sohbetin payından geriye kalanı deste alabilir.
    var room = window.innerWidth - 15.5 * 16 - CHAT_MIN;
    return Math.round(Math.min(Math.max(px, DECK_MIN), Math.max(DECK_MIN, room)));
  }

  function setDeckWidth(px, remember) {
    var w = clampDeck(px);
    document.documentElement.style.setProperty('--deckw', w + 'px');
    if (remember) {
      try { localStorage.setItem('deckw', String(w)); } catch (e) { /* yok say */ }
    }
    return w;
  }

  function deckWidth() {
    var saved = 0;
    try { saved = parseInt(localStorage.getItem('deckw') || '', 10) || 0; }
    catch (e) { saved = 0; }
    return clampDeck(saved || DECK_DEFAULT);
  }

  if (grip) {
    var dragFrom = 0, dragW = 0;
    var onMove = function (ev) {
      // Tutamak sağa gidince deste DARALIYOR: genişlik sağ kenardan ölçülüyor.
      setDeckWidth(dragW - (ev.clientX - dragFrom), false);
    };
    var onUp = function () {
      document.removeEventListener('pointermove', onMove);
      document.removeEventListener('pointerup', onUp);
      document.body.classList.remove('deck-drag');
      grip.classList.remove('is-drag');
      setDeckWidth(deckHost.getBoundingClientRect().width, true);
    };
    grip.addEventListener('pointerdown', function (ev) {
      dragFrom = ev.clientX;
      dragW = deckHost ? deckHost.getBoundingClientRect().width : DECK_DEFAULT;
      document.body.classList.add('deck-drag');
      grip.classList.add('is-drag');
      document.addEventListener('pointermove', onMove);
      document.addEventListener('pointerup', onUp);
      ev.preventDefault();
    });
    // Fareyle sürüklenebilen bir ayırıcı ok tuşlarıyla da ayarlanabilmeli,
    // yoksa yalnız fare kullananın özelliği olur.
    grip.addEventListener('keydown', function (ev) {
      var step = ev.shiftKey ? 96 : 24, w = deckHost.getBoundingClientRect().width;
      if (ev.key === 'ArrowLeft') { setDeckWidth(w + step, true); }
      else if (ev.key === 'ArrowRight') { setDeckWidth(w - step, true); }
      else if (ev.key === 'Home') { setDeckWidth(DECK_DEFAULT, true); }
      else { return; }
      ev.preventDefault();
    });
    grip.addEventListener('dblclick', function () {
      setDeckWidth(DECK_DEFAULT, true);
    });
  }
  // Pencere daralınca kayıtlı genişlik sınırı aşabilir; kırpılıyor ama
  // KAYDEDİLMİYOR — pencere büyüyünce eski tercih geri gelsin.
  window.addEventListener('resize', function () {
    if (deckHost && !deckHost.hidden) {
      setDeckWidth(deckHost.getBoundingClientRect().width, false);
    }
  });

  var deckMeta = {};

  function loadDecks() {
    if (!deckBar) { return; }
    fetch('/api/decks').then(function (r) { return r.json(); }).then(function (d) {
      deckBar.textContent = '';
      (d.decks || []).forEach(function (deck) {
        deckMeta[deck.id] = deck;
        var b = el('button', 'deck__tab', deck.label);
        b.type = 'button';
        b.dataset.deck = deck.id;
        b.addEventListener('click', function () { pickDeck(deck.id); });
        deckBar.appendChild(b);
      });
      var back = el('button', 'deck__tab deck__tab--chat', 'Kapat ×');
      back.type = 'button';
      back.dataset.deck = 'chat';
      back.addEventListener('click', function () { showDeck(false); });
      deckBar.appendChild(back);
      if ((d.decks || []).length) { pickDeck(d.default || d.decks[0].id); }
      else { showDeck(false); }
    }).catch(function (e) {
      // Sessiz yutmak, desteyi hiç açılmamış hâlde bırakıyordu ve konsolda da
      // iz kalmıyordu. Sebep artık görünüyor.
      console.warn('deste yüklenemedi:', e && e.message);
      showDeck(false);
    });
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
    // Çerçeve anahtarı açıksa soru MAF'a gidiyor. Tek yerde ayrılıyor ki
    // sohbetin geri kalanı hangi çerçevede olduğunu bilmek zorunda kalmasın.
    if (state.framework === 'maf') { askMaf(question); return; }
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
      if (event.type === 'run') {
        armFlow(event.id);
      } else if (event.type === 'stage') {
        term.feed(event);
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
            term.code(data.ran.code);
            term.result({ output: data.ran.output,
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

  /* ------------------------------------------------------------------ terminal
   *
   * Read-only: what the model ran inside the container, and what came back.
   * There is no input, and that is the whole security story — this makes an
   * existing capability visible, it does not add one.
   *
   * It used to be one corner of a much larger "what is happening right now"
   * panel, which drew the running mechanism, the stage trace and the scheduled
   * jobs above the composer. That panel is gone from the chat: the explaining
   * gets an interface of its own. The terminal stayed because it is not an
   * explainer — it is the only place the output of an approved program is
   * shown, and the approval card replays into it.
   *
   * To recover the panel it is whole in the commit before this one: the `mech`
   * closure in this file, figures and trace and cron together.
   */
  var term = (function () {
    var host = document.getElementById('term');
    var body = document.getElementById('term-body');
    var meta = document.getElementById('term-meta');

    function write(text, cls) {
      if (!body) { return; }
      // Measured before appending: adding the line changes scrollHeight, so a
      // check made afterwards would always say "not at the bottom". Someone who
      // scrolled up is reading on purpose and must not be yanked back down.
      var atBottom = body.scrollHeight - body.scrollTop - body.clientHeight < 4;
      var row = el('div', cls || null, text);
      body.appendChild(row);
      while (body.children.length > 400) { body.removeChild(body.firstChild); }
      if (atBottom) { body.scrollTop = body.scrollHeight; }
    }

    // The rail widens while the terminal has something to show and goes back
    // when it does not, so the panel never costs width it is not using.
    function show() {
      if (!host) { return; }
      host.hidden = false;
      document.body.classList.add('term-open');
    }

    function hide() {
      if (!host) { return; }
      host.hidden = true;
      document.body.classList.remove('term-open');
    }

    function code(src) {
      show();
      if (meta) { meta.textContent = 'çalışıyor…'; }
      write('$ python /workspace/tmp.py', 't-cmd');
      String(src || '').split('\n').forEach(function (line) {
        write('  ' + line, 't-dim');
      });
    }

    function result(m) {
      show();
      var ok = !m.is_error;
      if (meta) {
        meta.textContent = (ok ? 'exit 0' : 'hata') +
          (m.seconds != null ? ' · ' + m.seconds + ' sn' : '');
      }
      String(m.output == null ? '' : m.output).split('\n').forEach(function (line) {
        write(line, ok ? null : 't-err');
      });
      write('── ' + (ok ? 'bitti' : 'hata ile bitti') +
            (m.seconds != null ? ' · ' + m.seconds + ' sn' : '') + ' ──', 't-dim');
    }

    if (document.getElementById('term-close')) {
      document.getElementById('term-close').addEventListener('click', hide);
    }

    return {
      code: code, result: result, close: hide,
      /* Stages still stream for every turn and the server side is untouched —
         the terminal listens for exactly two of them and ignores the rest. */
      feed: function (event) {
        var m = event.meta || {};
        if (event.id === 'code_request') { code(m.code); }
        else if (event.id === 'code_result') { result(m); }
      }
    };
  })();

  // The escape hatch. `/openclaw sessions.list` skips the model entirely and
  // prints what the Gateway actually returned — no paraphrase, because the
  // reason to type this instead of asking is that you want the bytes.
  var OPENCLAW = '/openclaw';

  /* Bir Gateway cevabının tek satırlık hâli.
   *
   * Ham JSON'u okumak "iş oluştu mu, ne zaman koşacak" sorusunu cevaplamıyor;
   * cevabı üç alanda ama yirmi satırın içinde. Bilinmeyen bir metot için
   * genel kural yeterli: hangi dizi alanı varsa kaç eleman taşıdığını say.
   * Tanımadığı bir şekle `null` dönüyor ve o zaman yalnız ham kutu kalıyor —
   * uydurma bir özet, özetsizlikten kötüdür.
   */
  function openclawSummary(method, body) {
    if (!body || typeof body !== 'object') { return null; }

    // Zamanlanmış iş: demonun cevabını arayan tek yer burası.
    if (body.schedule && body.schedule.expr) {
      var bits = ['cron ' + body.schedule.expr];
      if (body.schedule.tz) { bits.push(body.schedule.tz); }
      if (body.nextRunAtMs) {
        bits.push('ilk koşu ' + new Date(body.nextRunAtMs)
          .toLocaleString('tr-TR', { dateStyle: 'short', timeStyle: 'short' }));
      }
      if (body.sessionTarget) { bits.push('oturum: ' + body.sessionTarget); }
      var what = body.name || (body.payload && body.payload.message) || '';
      return (what ? '“' + what + '” · ' : '') + bits.join(' · ');
    }

    // Genel kural: dizi alanlarını say. `commands.list` 89, `audit.list` 100.
    var counts = [];
    Object.keys(body).forEach(function (k) {
      if (Array.isArray(body[k])) { counts.push(body[k].length + ' ' + k); }
    });
    if (!counts.length) { return null; }
    if (body.nextCursor) { counts.push('devamı var'); }
    return counts.join(' · ');
  }

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
        // Özet satırı ham kutunun YERİNE değil, ÜSTÜNE geliyor. Ham kutu bu
        // arayüzdeki tek "baytlara dokunulmadı" sözü ve onu bir paragrafla
        // değiştirmek o sözü bozardı. Ama `cron.add` yirmi satır JSON basıyor
        // ve içinde okunmaya değer üç alan var — ölçüldü, ekran görüntüsünde
        // gövdenin tamamı kaydırma kutusuydu ve saat hiç görünmüyordu.
        // Adı `line` DEĞİL: bu geri çağırım `runOpenClaw(line)`'ın içinde ve
        // `var line` dış parametreyi gölgeliyordu. Aynı geri çağırımdaki
        // "tutuldu" dalı `runOpenClaw(line)` diye yeniden deniyor, ve
        // gölgelenen değişken `undefined` olduğu için istek gövdesi `{}`
        // gidiyordu: onaydan sonra "no answer". Ölçüldü — sunucu
        // `Field required: line` dedi, tarayıcı sessizce yuttu.
        var summary = openclawSummary(data.method, body);
        if (summary) { pending.appendChild(el('div', 'ocsum', summary)); }
        var pre = el('pre', 'raw', text);
        if (summary) { pre.classList.add('raw--short'); }
        pending.appendChild(pre);
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
    // Yerel kısayol: rapor zaten elimizde, modele sormaya gerek yok.
    if (question === 'Tarama raporu') { showReport(); return; }
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
        if (data.live_llm) { loadTeams(); loadFramework(); }
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

        // Tarama raporu artık AÇILIŞTA basılmıyor. Sunumda ekranın ortasını
        // kaplıyordu ve içeriği zaten sorulabilir: "the funnel", "what is
        // missing" çipleri aynı veriyi cevap olarak getiriyor. Silmedik,
        // istendiğinde geliyor — sunumun ortası slaytın.
        state.report = data;
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
    // A scan is the one path that really assembles a team, so its flow screen
    // has the most to show. `latest` rather than an id: the scan is a subprocess
    // and the server names the run, not this side.
    armFlow('latest');
    var seen = 0;
    state.pollTimer = setInterval(function () {
      fetch('/api/scan?since=' + seen).then(function (r) { return r.json(); }).then(function (run) {
        log.textContent = (run.lines || []).join('\n');
        log.scrollTop = log.scrollHeight;
        // Replay them in order rather than jumping to the last: a poll can
        // carry several at once and the code stages must arrive in sequence.
        (run.stages || []).forEach(function (s) { term.feed(s); });
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
  loadDecks();
  loadState().catch(function (error) {
    disclaimer.textContent = 'Backend unreachable: ' + error;
  });
})();
