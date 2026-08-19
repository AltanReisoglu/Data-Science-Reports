/* Akış ekranı — bir turun tamamı, çizilmiş hâliyle.
 *
 * Bu dosyada tek bir öğretici cümle yok, ve olmaması kural. Ekranın anlattığı
 * her şey `/api/run/<id>`'den geliyor; burada yalnız yerleşim ve çizim var.
 * Sebebi `stages.py`'nin docstring'inde yazılı: JavaScript'e gömülen açıklama
 * metni, anlattığı koddan sessizce uzaklaşıyor.
 *
 * Çizim `rough.js` ile, yani PDF'lerdeki elle çizilmiş şemalarla aynı elden.
 * Tertemiz kutular "bu bitmiş bir ürün" diye okunuyor; bu ekran bir düşünme
 * aracı ve öyle görünmeli.
 */
(function () {
  'use strict';

  var pickEl = document.getElementById('pick');
  var questionEl = document.getElementById('question');
  var totalsEl = document.getElementById('totals');
  var cardsEl = document.getElementById('cards');

  var INK = '#1e1e1e', GREY = '#6b7178', MUTE = '#4a4238';
  var LANE_INK = {
    agentchat: '#1971c2', core: '#5f3dc4', ext: '#2f9e44', ours: '#e8590c',
    // MAF kendi rengi: ayrı bir çerçeve, AutoGen'in katmanlarıyla aynı renge
    // boyamak ekranın anlattığı ayrımı silerdi.
    maf: '#9c36b5'
  };
  var KIND_INK = {
    user: '#495057', agent: '#1864ab', tool: '#5f3dc4', component: '#d9480f',
    exec: '#087f5b', gate: '#c92a2a'
  };
  /* Kutular DOLGULU. Yalnız kenarlıkla çizildiklerinde — ekran görüntüsünde
     ölçüldü — beyaz zemin üstünde soluk bir çerçeveden ibaret kalıyorlardı ve
     bir metre öteden hiçbiri diğerinden ayrılmıyordu. İç mimari şemaları zaten
     dolgu kullanıyordu; grafın onlardan daha az okunaklı olması için bir sebep
     yok. Dolgular çok açık: yazı hâlâ zeminle tam kontrastta duruyor. */
  var KIND_WASH = {
    user: '#f1f3f5', agent: '#e7f5ff', tool: '#f3f0ff', component: '#fff4e6',
    exec: '#e6fcf5', gate: '#fff5f5'
  };

  function el(tag, cls, text) {
    var n = document.createElement(tag);
    if (cls) { n.className = cls; }
    if (text != null) { n.textContent = text; }
    return n;
  }

  function clear(node) { while (node.firstChild) { node.removeChild(node.firstChild); } }

  /* Sunucudan gelen metinde `backtick` ile işaretlenmiş sınıf adları var — aynı
     metin PDF tarafında da öyle yazılıyor. innerHTML kullanmıyoruz: metin bizim
     olsa bile, bir gün olmayacağı gün için. */
  function rich(text) {
    var frag = document.createDocumentFragment();
    String(text || '').split('`').forEach(function (part, i) {
      if (!part) { return; }
      frag.appendChild(i % 2 ? el('code', null, part)
                             : document.createTextNode(part));
    });
    return frag;
  }

  /* Kelime kelime sarma. SVG metni kendi kendine sarmıyor, ve `foreignObject`
     yazdırma yolunda güvenilmez — şemaların PDF'te sessizce kaybolması bu
     projede bir kez ölçüldü. */
  function wrap(text, chars, maxLines) {
    var words = String(text || '').split(/\s+/), lines = [], line = '';
    words.forEach(function (w) {
      if (!w) { return; }
      if ((line + ' ' + w).trim().length > chars && line) { lines.push(line); line = w; }
      else { line = (line + ' ' + w).trim(); }
    });
    if (line) { lines.push(line); }
    if (lines.length > maxLines) {
      lines = lines.slice(0, maxLines);
      lines[maxLines - 1] = lines[maxLines - 1].replace(/.{2}$/, '…');
    }
    return lines;
  }

  /* Çizim tik'ler arasında YAŞIYOR. Kartların içeriği her yenilemede baştan
     kuruluyor — ucuz ve görünmez — ama SVG'yi yeniden yaratmak CSS animasyonunu
     her seferinde başa sardırırdı, yani "nerede koşuyor" ışığı hiç yanmazdı.
     Topoloji değişmediği sürece aynı düğüm taşınıyor. */
  /* `report` burada tutuluyor, fonksiyon argümanında değil. Sebebi ölçüldü:
     yürüyüş zamanlayıcısı kendisini kuran andaki raporu kapatıyordu. Tur bitip
     yoklama durunca zamanlayıcı o eski rapordaki satır sayısına yetişiyor ve
     susuyordu — zincir onuncu adımda değil ikincide kalıyordu. */
  var view = { topo: null, canvas: null, report: null };

  /* Işığın yürüyüşü.
   *
   * Ölçüldü: bir sohbet turunda sekiz aşama SSE'den tek seferde boşalıyor
   * (0 → 8 adım, 1.85s'de). Yalnız "en son aşama"yı yakan bir ekran, kapıyı ve
   * tool'u insan gözünün göremeyeceği kadar kısa gösteriyor — yani anlatmak
   * istediğimiz kısmı hiç göstermiyor.
   *
   * O yüzden ışık sırayı **yürüyor**: gelen her aşamayı sırayla, adım başına
   * sabit bir süreyle yakıyor. Sıra da zamanlar da kaydın kendisinden geliyor;
   * uydurulan tek şey izleme hızı. Kuyruk uzarsa adım kısalıyor, yoksa ekran
   * gerçeğin arkasında kalırdı.
   */
  /* `reached` yalnız büyüyor: şerit "şu ana kadar açılanlar", "geçtiklerim"
     değil. İkisini karıştırmak, ilk kutuya basınca şeridin tamamının silinmesi
     demekti — basılan kutudan öncesi boş küme olduğu için. */
  var play = { at: 0, reached: 0, timer: null, target: null, live: false,
               manual: false };

  /* Adım süresi. Geride iz kaldığı için hızlı olması bir şey kaybettirmiyor,
     ama gözün sahnedeki şemayı yakalamasına yetmeliydi — ilk hâli 90 ms'ye
     kadar iniyordu ve şemalar görülmeden geçiyordu. */
  function stepDelay(pending) {
    return pending > 6 ? 260 : pending > 3 ? 380 : 620;
  }

  function advance() {
    play.timer = null;
    var report = view.report;
    if (!report) { return; }
    var rows = report.timeline || [];
    play.live = report.status === 'running';
    play.reached = Math.max(play.reached, play.at);
    if (play.at >= rows.length) {
      play.reached = rows.length;
      // Yetiştik. Son şema sahnede kalıyor — bitmiş bir turda boş bir sahne,
      // az önce izlenen şeyin izini siliyordu. Graf ışığı yine de sönüyor:
      // sahnedeki resim "bu olmuştu" der, grafın ışığı "bu OLUYOR" der.
      play.target = rows.length ? rows[rows.length - 1] : null;
      // Sahnedeki şema kalıyor, grafın ışığı sönüyor — yukarıdaki cümlenin
      // gerçekten olması için ayrı bir bayrak gerekiyordu. `markGraph` tek
      // ölçüt olarak `play.target`e bakıyordu, ve o son satırda duruyor:
      // bitmiş bir turda `answer` kutusu nefes almaya devam ediyordu (ölçüldü,
      // DOM'da `is-live` kalıyordu), yani ekran zamanı yanlış söylüyordu.
      play.done = true;
      paint();
      return;
    }
    play.done = false;
    play.target = rows[play.at];
    play.at += 1;
    play.reached = Math.max(play.reached, play.at);
    paint();
    play.timer = setTimeout(advance, stepDelay(rows.length - play.at));
  }

  /* Bir aşamanın şeması. `null` dönebilir: her aşamanın ayrı resmi yok, ve
     olmayan bir resmin yerine boş çerçeve koymak hiçbir şey öğretmiyor. */
  function figureFor(row) {
    if (!row || !window.MechFigures) { return null; }
    return MechFigures.draw(row.id, row.meta);
  }

  /* Sahne + şerit.
   *
   * Şemalar sahnede büyük çıkıyor, sonra küçülüp alttaki şeride ok'la
   * ekleniyor. Sebebi ölçüldü: adımlar birbirini yüz milisaniyelerle kovalıyor
   * ve sahnede tek başına duran bir şema görülmeden geçiyordu. Şerit, geçenin
   * durduğu yer — üstüne basınca sahneye geri geliyor. */
  /* Işık bir kutuya gelince o kutunun içi kendiliğinden açılıyor — grafın
     altında, sabit bir yerde. Açılır pencere olarak yapmak, 260 ms'de bir
     ekranda zıplayan bir kutu demekti; sabit yer hem okunuyor hem sunumda
     gösterilebiliyor. Üstüne gelme yolu ayrıca duruyor. */
  function liveInner(report) {
    var host = document.querySelector('.liveinner');
    if (!host) { return; }
    clear(host);
    var cur = play.target;
    if (!cur) { host.hidden = true; return; }
    var ids = [].concat(cur.node == null ? [] : cur.node);
    var nodes = (report.graph && report.graph.nodes) || [];
    var node = null;
    ids.forEach(function (id) {
      if (node) { return; }
      nodes.forEach(function (n) { if (!node && n.id === id && n.inner) { node = n; } });
    });
    if (!node) { host.hidden = true; return; }
    host.hidden = false;
    host.appendChild(nodeBody(node));
  }

  /* Bir kutunun altına ne yazılacağı, tek yerde.
   *
   * Ajan kutularında önce **koştuğu desen** çiziliyor, sonra iç mimarisi.
   * Sırası bilinçli: "bu bir ajan" bilgi değil; sorulan soru, o ajanın hangi
   * desene göre çalıştığı. Sohbet turumuzda cevap sekiz desenden biri DEĞİL, ve
   * onu çizmek olmayan bir takımı iddia etmekten daha değerli. */
  function nodeBody(node) {
    var wrap = document.createDocumentFragment();
    var pat = node.pattern;
    var fig = pat && window.PatternFigures && PatternFigures.draw(pat.id);
    if (fig) {
      var head = el('div', 'liveinner__head');
      head.appendChild(el('span', 'liveinner__where', node.name + ' · deseni'));
      head.appendChild(el('span', 'liveinner__title', pat.name + ' · ' + pat.ref));
      wrap.appendChild(head);
      wrap.appendChild(fig);
      var role = el('p', 'liveinner__role');
      role.appendChild(rich(pat.role));
      wrap.appendChild(role);
    }
    if (node.inner) {
      var head2 = el('div', 'liveinner__head');
      head2.appendChild(el('span', 'liveinner__where',
                           fig ? node.name + ' · içi' : node.name));
      head2.appendChild(el('span', 'liveinner__title', node.inner.title));
      wrap.appendChild(head2);
      wrap.appendChild(drawInner(node.inner));
    }
    return wrap;
  }

  function paint() {
    var report = view.report;
    if (!report) { return; }
    markGraph(report);
    liveInner(report);
    var fig = document.querySelector('.fig');
    if (fig) {
      clear(fig);
      var stage = el('div', 'fig__stage');
      var svg = figureFor(play.target);
      if (svg) {
        stage.appendChild(svg);
      } else {
        stage.appendChild(el('p', 'fig__none', play.target
          ? play.target.name + ' — bu aşamanın ayrı bir şeması yok.'
          : 'Henüz bir aşama yok.'));
      }
      if (play.target) {
        stage.appendChild(el('p', 'fig__cap',
          play.target.name + (play.target.klass ? ' · ' + play.target.klass : '')));
      }
      fig.appendChild(stage);

      // Kutuya basınca o aşamanın tam anlatımı. Dört soru, hep aynı sırada:
      // ne · nasıl · neden · nerede ısırıyor. Dördüncüsü çoğu belgede yok ve
      // en çok işe yarayanı o.
      var info = (report.details || {})[play.target ? play.target.id : ''];
      if (info) {
        var det = el('div', 'detail');
        var head = el('div', 'detail__head');
        head.appendChild(el('span', 'detail__title', play.target.name));
        if (play.target.lane) { head.appendChild(lane(play.target.lane)); }
        if (play.target.ref) { head.appendChild(el('span', 'detail__ref', play.target.ref)); }
        if (play.target.module) {
          head.appendChild(el('code', 'detail__mod', play.target.module));
        }
        det.appendChild(head);
        [['NE', info.what], ['NASIL', info.how], ['NEDEN', info.why],
         ['NEREDE ISIRIYOR', info.trap]].forEach(function (row) {
          if (!row[1]) { return; }
          var block = el('div', 'detail__row' +
                              (row[0] === 'NEREDE ISIRIYOR' ? ' detail__row--trap' : ''));
          block.appendChild(el('span', 'detail__key', row[0]));
          var body = el('p', 'detail__text');
          body.appendChild(rich(row[1]));
          block.appendChild(body);
          det.appendChild(block);
        });
        fig.appendChild(det);
      }

      // Film şeridi: turun açılmış bütün adımları, sırayla. Sahnedeki adım
      // burada da duruyor ve işaretli — geri dönmek için basılacak yer o.
      var rows = (report.timeline || []).slice(0, play.reached);
      var trail = el('div', 'fig__trail');
      var current = null;
      rows.forEach(function (row, i) {
        if (i > 0) { trail.appendChild(el('span', 'fig__arrow', '→')); }
        var isNow = i === play.at - 1;
        var thumb = el('button', 'fig__thumb' + (isNow ? ' is-current' : ''));
        thumb.type = 'button';
        thumb.title = row.name + (row.note ? ' — ' + row.note : '');

        var head = el('span', 'fig__thumbhead');
        head.appendChild(el('span', 'fig__thumbno', String(i + 1)));
        head.appendChild(el('span', 'fig__thumblane lane lane--' +
                            (row.lane === 'ours' ? 'ours' : row.lane || 'agentchat'),
                            row.lane || ''));
        head.appendChild(el('span', 'fig__thumbat',
                            '+' + Number(row.at).toFixed(1) + 's'));
        thumb.appendChild(head);

        var small = figureFor(row);
        if (small) {
          thumb.appendChild(small);
        } else {
          // Şeması olmayan aşama da şeritte yerini alıyor: eksik bir kare,
          // sıranın kopuk görünmesine yol açardı.
          thumb.appendChild(el('span', 'fig__thumbnone', row.klass || '—'));
        }
        thumb.appendChild(el('span', 'fig__thumbcap', row.name));
        thumb.addEventListener('click', function () { park(row); });
        if (isNow) { current = thumb; }
        trail.appendChild(thumb);
      });
      if (rows.length) {
        fig.appendChild(trail);
        // Sahnedeki kare görünürde kalsın; sayfayı kaydırmadan, yalnız şeridi.
        if (current) {
          trail.scrollLeft = current.offsetLeft - trail.clientWidth / 2
                             + current.offsetWidth / 2;
        }
      }
    }
    var badge = document.querySelector('.card__live');
    if (badge) {
      badge.lastChild.textContent = play.target
        ? play.target.name + (play.target.klass ? ' · ' + play.target.klass : '')
        : '';
      badge.hidden = !play.target;
    }
    [].forEach.call(document.querySelectorAll('.tl li'), function (li, i) {
      li.classList.toggle('is-live', !!play.target && i === play.at - 1);
    });
  }

  function topoSignature(report) {
    return report.id + '|' +
      report.graph.nodes.map(function (n) { return n.id; }).join(',') + '|' +
      report.graph.edges.map(function (e) {
        return e.src + '>' + e.dst + ':' + e.message;
      }).join(',');
  }

  /* Grafın durumu: ne koştu, ne koşuyor, neye sıra gelmedi.
   *
   * Tek bir kutuyu yakmak "şu an neredeyiz"i söylüyordu ama "buraya nasıl
   * gelindi"i söylemiyordu. Zincir artık **doluyor**: geçilen her kutu ve kenar
   * yanık kalıyor, sırası gelmeyen sönük duruyor, o an koşan kalınlaşıp
   * nefes alıyor. Bir turun tamamı böylece tek bir resimde birikiyor.
   *
   * Hangi aşamanın grafta neresi olduğunu sunucu söylüyor; burada yalnız
   * biriktirme var.
   */
  function markGraph(report) {
    if (!view.canvas) { return; }
    var svg = view.canvas.querySelector('svg');
    if (!svg) { return; }

    var rows = (report.timeline || []).slice(0, play.at);
    var doneNodes = {}, doneEdges = {};
    rows.forEach(function (r) {
      [].concat(r.node == null ? [] : r.node).forEach(function (id) { doneNodes[id] = 1; });
      if (r.edge != null) { doneEdges[r.edge] = 1; }
    });
    // Kullanıcıdan ajana giden ok gibi, bir aşamaya bağlı OLMAYAN kenarlar da
    // var. Onlar iki ucu da geçilmişse geçilmiş sayılıyor — yoksa zincir hep
    // kopuk görünürdü.
    var edges = (report.graph && report.graph.edges) || [];
    edges.forEach(function (e, i) {
      if (doneNodes[e.src] && doneNodes[e.dst]) { doneEdges[i] = 1; }
    });
    // Ve simetriği: hiçbir aşamaya bağlı OLMAYAN kutular. `Soru` hiçbir
    // aşamanın hedefi değil — tur başlamadan önce oradaydı — ve yalnız
    // aşamalara bakan bir kural onu sonsuza kadar sönük bırakıyordu (ölçüldü:
    // bitmiş bir turda `is-pending` kalıyordu), yani zincirin başı hep eksik
    // görünüyordu.
    //
    // Yayılım GERİYE doğru: zincir senin ÜSTÜNDEN geçtiyse geçilmişsin.
    // İleriye doğru yaymak yanlış olurdu — ilk kutu yanar yanmaz bütün graf
    // yanardı ve zincirin adım adım dolması diye bir şey kalmazdı.
    // `İz` gibi uç kutular bilerek dışarıda: onlar gerçekten en sonda oluyor,
    // ve yürüyüş bittiğinde zaten hiçbir kutu sönük kalmıyor.
    for (var pass = 0; pass < 3; pass += 1) {
      edges.forEach(function (e) {
        if (doneNodes[e.dst]) { doneNodes[e.src] = 1; }
      });
    }

    // Yürüyüş bittiyse hiçbir yer yanmıyor ve hiçbir yer sönük değil: bütün
    // zincir eşit parlaklıkta duruyor, çünkü artık hepsi geçmiş.
    var cur = play.done ? null : play.target;
    var live = {};
    if (cur) {
      [].concat(cur.node == null ? [] : cur.node).forEach(function (id) { live[id] = 1; });
    }
    var liveEdge = cur && cur.edge != null ? String(cur.edge) : null;
    var walking = !!cur;

    [].forEach.call(svg.querySelectorAll('[data-node]'), function (g) {
      var id = g.getAttribute('data-node');
      g.classList.toggle('is-live', !!live[id]);
      g.classList.toggle('is-pending', walking && !doneNodes[id] && !live[id]);
    });
    [].forEach.call(svg.querySelectorAll('[data-edge]'), function (g) {
      var i = g.getAttribute('data-edge');
      g.classList.toggle('is-live', i === liveEdge);
      g.classList.toggle('is-pending', walking && !doneEdges[i] && i !== liveEdge);
    });
  }

  // ------------------------------------------------------------------ layout
  //
  // Katmanlar ileri kenarlardan hesaplanıyor. Dönüş kenarları (tool → ajan)
  // sunucuda `back` ile işaretli ve buraya karışmıyor — karışsaydı ajan kendi
  // tool'unun sağına düşer ve graf tersine dönerdi.
  // Sütun aralığı kutu genişliğinden çok daha geniş: aradaki boşluk kenar
  // etiketinin yeri. Dar bıraktığımızda `ToolCallRequestEvent` kutunun üstüne
  // biniyordu — etiket okunmuyorsa kenar da anlatmıyor.
  var W = 186, H = 64, COL = 330, GAP = 64, PAD = 24;
  // Kutunun ALTINDA, içinde değil: açıklama uzun ve kutuya sığdırmaya
  // çalışmak ya kutuyu şişiriyor ya da yazıyı okunmaz hâle getiriyor.
  var NOTE = 40, NOTE_LINES = 3, NOTE_CHARS = 44;
  // Dolanan okların en alta indiği nokta (aşağıdaki `dip`/`below` ile aynı
  // hesap). Bant yüksekliği buna yer ayırmazsa ok bir alttaki bandın kutusunun
  // içinden geçiyor.
  // Sol oluk bant adları için; DIP dönüş oklarının kutuların altında
  // kullandığı yer; BANDGAP iki bandı ayıran boşluk.
  var LABEL = 196, DIP = NOTE + 86, BANDGAP = 52;

  /* İki bant, iki ayrı yerleşim.
   *
   * Üst bant AutoGen'in yaptığı iş, alt bant bizim hattımız. Tek sıraya
   * dizildiklerinde kapı ile `AssistantAgent` aynı zincirin halkaları gibi
   * duruyordu, ve ekranın anlatması gereken en önemli ayrım tam olarak bu
   * değildi. Her bandın katmanları kendi içinde hesaplanıyor; bantlar arası
   * tek ok dikey ve kesikli, çünkü o bir sıra devri değil, bir iniş. */
  function layout(graph) {
    var bands = graph.bands || [{ id: 'agent' }];
    var layer = {}, bandOf = {};
    graph.nodes.forEach(function (n) {
      layer[n.id] = 0;
      bandOf[n.id] = n.band || 0;
    });

    // Uzun yol, bant içinde. Dönüş ve bantlar arası kenarlar sıraya karışmıyor.
    function intra(e) {
      return !e.back && !e.cross &&
             bandOf[e.src] === bandOf[e.dst] &&
             (e.src in layer) && (e.dst in layer);
    }
    // Emniyet: bir düğümün katmanı düğüm sayısını aşamaz. İşaretlenmemiş bir
    // döngü kaçarsa graf sonsuza kadar sağa uzuyor ve karta sığdırılınca
    // kılcal çizgilere dönüyor — bir kez ölçüldü, bir daha sessizce olmasın.
    var ceiling = graph.nodes.length;
    for (var pass = 0; pass < graph.edges.length + 1; pass++) {
      var moved = false;
      graph.edges.forEach(function (e) {
        if (!intra(e)) { return; }
        if (layer[e.dst] < layer[e.src] + 1 && layer[e.src] + 1 <= ceiling) {
          layer[e.dst] = layer[e.src] + 1;
          moved = true;
        }
      });
      if (!moved) { break; }
    }

    // Bitiş düğümü kendi bandının en sağında. Ajandan uzaklığı bir tool ile
    // aynı olabiliyor, ve aynı sütuna düşünce cevap tool'un kardeşi görünüyordu.
    // Bitiş düğümünün KENDİ katmanı bu hesaba girmemeli. Girdiğinde her
    // yeniden konumlandırma onu bir sütun daha ileri itiyordu: ölçüldü,
    // `Skorlayıcı → Skor` arasında boş bir sütun kalıyor ve ok, atlanacak bir
    // kutu varmış gibi alttan dolanıyordu.
    var deepest = {};
    graph.nodes.forEach(function (n) {
      if (n.terminal) { return; }
      var b = bandOf[n.id];
      deepest[b] = Math.max(deepest[b] || 0, layer[n.id]);
    });
    graph.nodes.forEach(function (n) {
      if (n.terminal) { layer[n.id] = deepest[bandOf[n.id]] + 1; }
    });

    var place = {}, rows = [], top = PAD, depth = 0;
    bands.forEach(function (band, bi) {
      var columns = {};
      graph.nodes.forEach(function (n) {
        if (bandOf[n.id] !== bi) { return; }
        (columns[layer[n.id]] = columns[layer[n.id]] || []).push(n);
      });
      var keys = Object.keys(columns);
      if (!keys.length) { return; }
      var tallest = Math.max.apply(null, keys.map(function (k) {
        return columns[k].length;
      }));
      depth = Math.max(depth, Math.max.apply(null, keys.map(Number)) + 1);
      // Dönüş okları kutuların altından geçiyor; bandın yüksekliği onlara da yer
      // ayırmazsa ok bir alttaki bandın kutusunun içinden geçiyor.
      var height = tallest * H + (tallest - 1) * GAP + DIP;
      keys.forEach(function (k) {
        var list = columns[k];
        var span = list.length * H + (list.length - 1) * GAP;
        var offset = (height - DIP - span) / 2;
        list.forEach(function (n, i) {
          place[n.id] = { x: LABEL + PAD + Number(k) * COL,
                          y: top + offset + i * (H + GAP), node: n };
        });
      });
      rows.push({ label: band.label || '', top: top, height: height });
      top += height + BANDGAP;
    });

    return { place: place, layer: layer, band: bandOf, rows: rows,
             width: LABEL + PAD * 2 + depth * COL - (COL - W),
             height: top - BANDGAP + PAD };
  }

  function drawGraph(graph) {
    var box = el('div', 'canvas');
    if (!graph.nodes.length) {
      box.appendChild(el('p', 'empty', 'Çizecek bir şey yok.'));
      return box;
    }

    var geo = layout(graph);
    var svg = Rough.svg(geo.width, geo.height);
    // Doğal genişliğin altına inme (bkz. `.canvas svg` yorumu). `width` niteliği
    // CSS'teki `width:100%` tarafından eziliyor, `min-width` ezilmiyor.
    svg.style.minWidth = geo.width + 'px';
    var pen = new Rough.Pen(7, 1);

    /* Her düğüm ve her kenar kendi `<g>`'sine giriyor. Sebep canlı vurgu:
       yanan yeri değiştirmek için grafı yeniden çizmek gerekmemeli — yeniden
       çizim, saniyede bir kıpırdayan bir resim demek olurdu. */
    function group(kind, key) {
      var g = document.createElementNS(Rough.NS, 'g');
      g.setAttribute('data-' + kind, String(key));
      svg.appendChild(g);
      return g;
    }

    // Bant adları ve ayırıcı çizgi, her şeyin altında.
    geo.rows.forEach(function (row, i) {
      if (i > 0) {
        svg.appendChild(pen.line(PAD, row.top - BANDGAP / 2,
                                 geo.width - PAD, row.top - BANDGAP / 2,
                                 { stroke: '#adb5bd', width: 1.2, dash: '4 6' }));
      }
      svg.appendChild(Rough.text(PAD, row.top + 18, row.label, {
        size: 13, colour: i === 0 ? LANE_INK.agentchat : LANE_INK.ours,
        weight: '700', mono: true
      }));
    });

    graph.edges.forEach(function (e, i) {
      var a = geo.place[e.src], b = geo.place[e.dst];
      if (!a || !b) { return; }
      var svg = group('edge', i);
      var ink = LANE_INK.agentchat;
      if (e.message && e.message.indexOf('Structured') === 0) { ink = LANE_INK.ext; }
      if (geo.band[e.src] === 1 || geo.band[e.dst] === 1) { ink = LANE_INK.ours; }

      if (e.cross) {
        // Bantlar arası iniş. Kesikli ve dikey: bu bir sıra devri değil, aynı
        // çağrının bir alt kata inmesi.
        var cx = a.x + W / 2, cy = a.y + H, tx = b.x + W / 2, ty = b.y;
        svg.appendChild(pen.curve([[cx, cy], [cx, (cy + ty) / 2],
                                   [tx, (cy + ty) / 2], [tx, ty]],
                                  { stroke: LANE_INK.ours, width: 1.7, dash: '6 4',
                                    arrow: true }));
        // Not satırları kutunun altında NOTE kadar yer kaplıyor; etiket onun
        // ALTINA iniyor, yoksa üstüne biniyor (ölçüldü: `takım koşusu`,
        // `Researcher`ın "10 kez konuştu…" notunun üstünde duruyordu).
        svg.appendChild(Rough.text((cx + tx) / 2, cy + NOTE + 26, e.message,
          { size: 9.4, anchor: 'middle', colour: LANE_INK.ours, mono: true, weight: '600' }));
        return;
      }

      if (e.back) {
        var dip = Math.max(a.y, b.y) + H + NOTE + 16 + (i % 2) * 16;
        svg.appendChild(pen.curve([
          [a.x + W / 2, a.y + H], [a.x, dip], [b.x + W, dip], [b.x + W, b.y + H - 6]
        ], { stroke: ink, width: 1.6, dash: '5 4' }));
        svg.appendChild(Rough.text((a.x + b.x + W) / 2, dip + 11, e.message,
          { size: 9.4, anchor: 'middle', colour: ink, mono: true, weight: '600' }));
        return;
      }

      var x1 = a.x + W, y1 = a.y + H / 2, x2 = b.x, y2 = b.y + H / 2;
      var mx, my;

      if (geo.layer[e.dst] - geo.layer[e.src] > 1) {
        // Bir sütun atlayan kenar. Düz çizilince aradaki kutunun tam içinden
        // geçiyordu — ölçüldü: `TaskResult` oku `scan_facts`'in başlığının
        // üstüne biniyordu. Alttan dolanmak aynı zamanda doğru şeyi söylüyor:
        // bu mesaj o kutuya uğramadı.
        var below = Math.max(a.y, b.y) + H + NOTE + 54 + (i % 2) * 16;
        svg.appendChild(pen.curve([
          [x1, y1], [x1 + 30, below], [x2 - 30, below], [x2, y2]
        ], { stroke: ink, width: 1.8, arrow: true }));
        svg.appendChild(Rough.text((x1 + x2) / 2, below + 12, e.message,
          { size: 9.4, anchor: 'middle', colour: ink, mono: true, weight: '600' }));
        return;
      }

      svg.appendChild(pen.line(x1, y1, x2, y2,
        { stroke: ink, width: e.join ? 2.3 : 1.8, arrow: true,
          dash: e.join ? '6 3' : null }));

      // Etiket okun üstünde, ve komşu oklarla çakışmasın diye sıra sıra kaydırılıyor.
      mx = x1 + (x2 - x1) * 0.5; my = y1 + (y2 - y1) * 0.5;
      svg.appendChild(Rough.text(mx, my - 8 - (i % 2) * 13, e.message,
        { size: 9.4, anchor: 'middle', colour: ink, mono: true, weight: '600' }));
      if (e.gate) {
        svg.appendChild(Rough.text(mx, my + 11, 'kapı · ' + e.gate, {
          size: 6.8, anchor: 'middle',
          colour: e.gate === 'red' ? '#c92a2a' : LANE_INK.ours, weight: '700'
        }));
      }
    });

    graph.nodes.forEach(function (n) {
      var p = geo.place[n.id];
      if (!p) { return; }
      var svg = group('node', n.id);
      var ink = KIND_INK[n.kind] || INK;
      if (n.inner || n.pattern) {
        // İçi olan kutu, olduğunu belli etsin: imleç değişiyor ve alt kenarına
        // ince bir çizgi düşüyor. Keşfedilmeyen bir etkileşim, olmayan bir
        // etkileşimle aynı şey.
        svg.setAttribute('class', 'has-inner');
        svg.addEventListener('mouseenter', function () { showInner(n, svg); });
        svg.addEventListener('mouseleave', hideInner);
      }
      svg.appendChild(pen.rect(p.x, p.y, W, H,
        { stroke: ink, width: 2, fill: KIND_WASH[n.kind] || '#f8f9fa' }));
      svg.appendChild(Rough.text(p.x + W / 2, p.y + 24, n.name,
        { size: 13, anchor: 'middle', weight: '700', colour: ink }));
      if (n.sub) {
        svg.appendChild(Rough.text(p.x + W / 2, p.y + 40, n.sub,
          { size: 8.6, anchor: 'middle', colour: GREY, mono: true }));
      }
      // Kutunun altında ne olduğu. Kutu adı NE olduğunu söylüyor, bu satırlar
      // orada NE YAPILDIĞINI — ikisi ayrı soru.
      wrap(n.note, NOTE_CHARS, NOTE_LINES).forEach(function (row, li) {
        svg.appendChild(Rough.text(p.x + W / 2, p.y + H + 14 + li * 11.5, row,
          { size: 8, anchor: 'middle', colour: MUTE }));
      });
    });

    box.appendChild(svg);
    return box;
  }

  /* ------------------------------------------------------------- iç mimari
   *
   * Kutunun üstüne gelince içi açılıyor. Graf, kutuların birbirine nasıl
   * bağlandığını anlatıyor; bu, kutunun kendi içinde ne olduğunu — ajanlarda
   * özellikle işe yarıyor, çünkü dışarıdan tek kutu görünen şeyin içinde dört
   * ayrı karar noktası var.
   *
   * Tek çizici, veriyi sunucudan alıyor. On ayrı elle çizilmiş şema yazmak,
   * on tanesini ayrı ayrı bakımda tutmak demekti.
   */
  // Sütun aralığı (COL - W) etiketin yeri. 48 px'te `seçilmiş mesajlar` hiçbir
  // yazı boyunda sığmıyordu ve kutuların üstüne biniyordu; 94 px'te sığıyor.
  var IN = { W: 124, H: 40, COL: 218, GAP: 20, PAD: 14 };
  // Destedeki palet: çizgi rengi + dolgu. Dolgusuz kutular, destedeki dolgulu
  // kutularla yan yana durunca iki ayrı belge gibi okunuyordu.
  var IN_INK = {
    in: ['#868e96', '#f8f9fa'], ours: ['#e8590c', '#fff4e6'],
    core: ['#5f3dc4', '#f8f0fc'], ext: ['#2f9e44', '#ebfbee'],
    agent: ['#1971c2', '#e7f5ff'], block: ['#c92a2a', '#fff5f5']
  };
  function inInk(k) { return (IN_INK[k] || IN_INK.in)[0]; }
  function inWash(k) { return (IN_INK[k] || IN_INK.in)[1]; }

  function drawInner(inner) {
    var layer = {}, byId = {};
    inner.nodes.forEach(function (n) { layer[n.id] = 0; byId[n.id] = n; });
    for (var pass = 0; pass < inner.edges.length + 1; pass++) {
      var moved = false;
      inner.edges.forEach(function (e) {
        if (e.back || !(e.src in layer) || !(e.dst in layer)) { return; }
        if (layer[e.dst] < layer[e.src] + 1) { layer[e.dst] = layer[e.src] + 1; moved = true; }
      });
      if (!moved) { break; }
    }
    var cols = {};
    inner.nodes.forEach(function (n) {
      (cols[layer[n.id]] = cols[layer[n.id]] || []).push(n);
    });
    var keys = Object.keys(cols);
    var depth = Math.max.apply(null, keys.map(Number)) + 1;
    var tall = Math.max.apply(null, keys.map(function (k) { return cols[k].length; }));
    var height = IN.PAD * 2 + tall * IN.H + (tall - 1) * IN.GAP + 20;
    var width = IN.PAD * 2 + depth * IN.COL - (IN.COL - IN.W);
    var svg = Rough.svg(width, height);
    var pen = new Rough.Pen(33, 0.9);
    var place = {};
    keys.forEach(function (k) {
      var list = cols[k];
      var span = list.length * IN.H + (list.length - 1) * IN.GAP;
      var top = (height - 20 - span) / 2;
      list.forEach(function (n, i) {
        place[n.id] = { x: IN.PAD + Number(k) * IN.COL, y: top + i * (IN.H + IN.GAP) };
      });
    });

    inner.edges.forEach(function (e, ei) {
      var a = place[e.src], b = place[e.dst];
      if (!a || !b) { return; }
      var colour = inInk((byId[e.dst] || {}).kind);
      if (e.back) {
        // Geri besleme: sonucun bağlama dönmesi bir sonraki adım değil, aynı
        // döngünün kapanması. Düz okla çizmek onu ileri bir adım gibi gösterirdi.
        var dip = Math.max(a.y, b.y) + IN.H + 10;
        svg.appendChild(pen.curve([[a.x + IN.W / 2, a.y + IN.H], [a.x, dip],
                                   [b.x + IN.W, dip], [b.x + IN.W, b.y + IN.H - 4]],
                                  { stroke: colour, width: 1, dash: '3 3' }));
        svg.appendChild(Rough.text((a.x + b.x + IN.W) / 2, dip + 9, e.label,
          { size: 5.8, anchor: 'middle', colour: colour, mono: true }));
        return;
      }
      var x1 = a.x + IN.W, y1 = a.y + IN.H / 2, x2 = b.x, y2 = b.y + IN.H / 2;
      svg.appendChild(pen.line(x1, y1, x2 - 4, y2,
        { stroke: colour, width: 1.1, arrow: true }));
      // Etiket iki kutunun arasındaki boşlukta. Birden çok ok aynı kutuya
      // giriyorsa etiketler sıra sıra kaydırılıyor, yoksa üst üste binerler.
      svg.appendChild(Rough.text((x1 + x2) / 2, (y1 + y2) / 2 - 5 - (ei % 2) * 8,
        e.label, { size: 5.6, anchor: 'middle', colour: colour, mono: true }));
    });

    inner.nodes.forEach(function (n) {
      var p = place[n.id], colour = inInk(n.kind);
      svg.appendChild(pen.rect(p.x, p.y, IN.W, IN.H,
        { stroke: colour, fill: inWash(n.kind),
          width: n.kind === 'block' ? 1.8 : 1.3,
          dash: n.kind === 'block' ? '4 3' : null }));
      svg.appendChild(Rough.text(p.x + IN.W / 2, p.y + 16, n.name,
        { size: 7.4, anchor: 'middle', weight: '700', colour: colour }));
      if (n.sub) {
        svg.appendChild(Rough.text(p.x + IN.W / 2, p.y + 28, n.sub,
          { size: 5.8, anchor: 'middle', colour: '#868e96', mono: true }));
      }
    });
    return svg;
  }

  /* Tek bir açılır kutu, tekrar tekrar kullanılıyor. Her kutuya bir tane
     yaratmak, grafı yeniden çizen her tik'te onları da yeniden yaratmak
     demekti. */
  var pop = null;

  function popover() {
    if (!pop) {
      pop = el('div', 'pop');
      pop.hidden = true;
      document.body.appendChild(pop);
    }
    return pop;
  }

  /* Açılır kutuya ne konursa konsun konumlandırma aynı. `build`, kutunun
     içini dolduran fonksiyon. */
  function showPop(build, anchor) {
    if (!window.Rough) { return; }
    var box = popover();
    clear(box);
    if (build(box) === false) { return; }
    box.hidden = false;

    // Kutunun altına, ekrandan taşmayacak şekilde.
    var r = anchor.getBoundingClientRect();
    var w = box.offsetWidth, h = box.offsetHeight;
    var left = Math.min(Math.max(8, r.left + r.width / 2 - w / 2),
                        window.innerWidth - w - 8);
    var top = r.bottom + 10;
    if (top + h > window.innerHeight - 8) { top = Math.max(8, r.top - h - 10); }
    box.style.left = (left + window.scrollX) + 'px';
    box.style.top = (top + window.scrollY) + 'px';
  }

  function showInner(node, anchor) {
    if (!node.inner && !node.pattern) { return; }
    showPop(function (box) {
      box.appendChild(nodeBody(node));
      if (node.inner) {
        var note = el('p', 'pop__note');
        note.appendChild(rich(node.inner.note));
        box.appendChild(note);
      }
    }, anchor);
  }

  /* Desenin kendi şeması — destedeki slaytın aynısı. Açıklama satırları
     çizimin içinde, çünkü destede de orada: şemayı ondan ayırmak, iki ayrı
     yerde bakım demek. */
  function showPattern(pattern, anchor) {
    showPop(function (box) {
      var fig = window.PatternFigures && PatternFigures.draw(pattern.id);
      if (!fig) { return false; }
      box.appendChild(el('div', 'pop__title',
        pattern.name + ' · ' + pattern.ref));
      box.appendChild(fig);
      var note = el('p', 'pop__note');
      note.appendChild(rich(pattern.used ? 'Bu turda KULLANILDI. ' + pattern.why
                                         : pattern.why));
      box.appendChild(note);
    }, anchor);
  }

  function hideInner() { if (pop) { pop.hidden = true; } }

  /* ------------------------------------------------------------ sıra diyagramı
   *
   * Graf **yapıyı** gösteriyor: hangi kutu neye bağlı. Bu **zamanı** gösteriyor:
   * aynı ajan üç kez konuşuyorsa grafta tek kutu var, burada üç ok. Bir sohbet
   * turunda asıl merak edilen ikincisi, ve grafın üstünde onu göstermenin yolu
   * yok — okları üst üste bindirmeden sıra anlatılamıyor.
   */
  // GUTTER zaman damgalarının kendi sütunu. Onları şeritlerin payına
  // sıkıştırmak, `+38.2s`'i sol kenardan taşırıp ilk hanesini kesiyordu —
  // ekranda `8.2s` yazıyordu, yani diyagram yanlış bir sayı gösteriyordu.
  var SEQ = { LANE: 152, ROW: 34, HEAD: 46, TOP: 26, PADX: 18, GUTTER: 52 };

  function drawSequence(seq) {
    var box = el('div', 'canvas');
    if (!seq || !seq.steps.length) {
      box.appendChild(el('p', 'empty', 'Henüz mesaj geçmedi.'));
      return box;
    }
    var lanes = seq.lanes, steps = seq.steps;
    var width = SEQ.GUTTER + SEQ.PADX * 2 + lanes.length * SEQ.LANE;
    var first = SEQ.TOP + SEQ.HEAD + 34;
    var height = first + steps.length * SEQ.ROW + SEQ.HEAD + 24;
    var svg = Rough.svg(width, height);
    var pen = new Rough.Pen(21, 0.9);

    function x(id) {
      var i = 0;
      lanes.forEach(function (l, k) { if (l.id === id) { i = k; } });
      return SEQ.GUTTER + SEQ.PADX + i * SEQ.LANE + SEQ.LANE / 2;
    }
    function ink(id) {
      var lane = '';
      lanes.forEach(function (l) { if (l.id === id) { lane = l.lane; } });
      return LANE_INK[lane] || GREY;
    }

    // Kutular önce, en altta: okların üstüne binmemeliler.
    (seq.groups || []).forEach(function (g, gi) {
      var top = first + g.from * SEQ.ROW - 20;
      var bottom = first + g.to * SEQ.ROW + 12;
      var pad = 8 + gi * 7;              // iç içe kutular birbirini yemesin
      var left = SEQ.GUTTER + SEQ.PADX + pad;
      var node = pen.rect(left, top, width - left - SEQ.PADX - pad,
                          bottom - top,
                          { stroke: g.kind === 'alt' ? LANE_INK.ours : LANE_INK.agentchat,
                            width: 1, dash: '5 4' });
      node.setAttribute('opacity', '0.65');
      svg.appendChild(node);
      svg.appendChild(Rough.text(left + 8, top + 11, g.label, {
        size: 7, colour: g.kind === 'alt' ? LANE_INK.ours : LANE_INK.agentchat,
        weight: '700', mono: true
      }));
    });

    // Şeritler: başlık üstte ve altta, arada kesikli hayat çizgisi.
    lanes.forEach(function (l) {
      var cx = x(l.id), colour = LANE_INK[l.lane] || GREY;
      [SEQ.TOP, height - SEQ.HEAD - 12].forEach(function (top) {
        svg.appendChild(pen.rect(cx - SEQ.LANE / 2 + 10, top, SEQ.LANE - 20, 30,
                                 { stroke: colour, width: 1.4 }));
        svg.appendChild(Rough.text(cx, top + 13, l.name,
          { size: 8.4, anchor: 'middle', weight: '700', colour: colour }));
        if (l.sub) {
          svg.appendChild(Rough.text(cx, top + 24, l.sub,
            { size: 6.2, anchor: 'middle', colour: GREY, mono: true }));
        }
      });
      svg.appendChild(pen.line(cx, SEQ.TOP + 30, cx, height - SEQ.HEAD - 12,
                               { stroke: '#c9c2b4', width: 0.8, dash: '3 5' }));
    });

    steps.forEach(function (s, i) {
      var y = first + i * SEQ.ROW;
      var colour = s.blocked ? '#c92a2a' : ink(s.src);
      var g = document.createElementNS(Rough.NS, 'g');
      g.setAttribute('data-seq', String(i));
      if (s.stage) { g.setAttribute('data-stage', s.stage); }
      svg.appendChild(g);

      if (s.kind === 'self') {
        // Kendine mesaj: küçük bir kanca. Ajanın kendi içinde yaptığı iş de bir
        // adım, ve okun olmaması onu görünmez yapardı.
        var cx = x(s.src);
        g.appendChild(pen.curve([[cx, y - 8], [cx + 34, y - 10], [cx + 34, y + 6],
                                 [cx + 4, y + 6]],
                                { stroke: colour, width: 1.2, arrow: true }));
        g.appendChild(Rough.text(cx + 42, y + 2, s.label,
          { size: 7, colour: colour, mono: true }));
      } else {
        var x1 = x(s.src), x2 = x(s.dst);
        var dir = x2 > x1 ? -6 : 6;
        g.appendChild(pen.line(x1, y, x2 + dir, y,
          { stroke: colour, width: s.blocked ? 1.8 : 1.3, arrow: true,
            dash: s.kind === 'return' ? '5 4' : null }));
        g.appendChild(Rough.text((x1 + x2) / 2, y - 5, s.label,
          { size: 7, anchor: 'middle', colour: colour, mono: true }));
      }
      g.appendChild(Rough.text(SEQ.GUTTER - 6, y + 2,
                               '+' + Number(s.at).toFixed(1) + 's',
                               { size: 6.4, anchor: 'end', colour: GREY, mono: true }));
    });

    box.appendChild(svg);
    return box;
  }

  /* Şelale. Kasıtlı olarak SVG değil, düz kutular: çubuklar dikdörtgen, ve
     onları elle çizmek okunurluk kazandırmıyor — yalnız bakım maliyeti ekliyor.
     Girinti, span'in ağaçtaki derinliği; genişlik, turun içindeki payı. */
  function waterfall(spans) {
    var host = el('div', 'wf');
    var depth = {}, byId = {};
    spans.forEach(function (s) { byId[s.id] = s; });
    spans.forEach(function (s) {
      var d = 0, cur = s;
      while (cur && cur.parent && byId[cur.parent] && d < 8) { d++; cur = byId[cur.parent]; }
      depth[s.id] = d;
    });
    var slowest = spans.reduce(function (a, b) { return b.ms > a ? b.ms : a; }, 0);
    spans.forEach(function (s) {
      var row = el('div', 'wf__row');
      var name = el('span', 'wf__name', s.name);
      name.style.paddingLeft = (depth[s.id] * 0.7) + 'rem';
      name.title = Object.keys(s.attrs || {}).map(function (k) {
        return k + '=' + s.attrs[k];
      }).join('\n') || s.name;
      row.appendChild(name);
      var track = el('span', 'wf__track');
      var bar = el('span', 'wf__bar' + (s.ms >= slowest * 0.5 ? ' is-slow' : ''));
      bar.style.left = (s.offset * 100) + '%';
      // Görünürlük tabanı: 0.2 %'lik bir çubuk ekranda hiç yok demek, ve o
      // span'in olmadığı anlamına gelmiyor.
      bar.style.width = Math.max(s.width * 100, 0.6) + '%';
      track.appendChild(bar);
      row.appendChild(track);
      row.appendChild(el('span', 'wf__ms', s.ms + ' ms'));
      host.appendChild(row);
    });
    return host;
  }

  /* Zamanlayıcı. Ayrı bir istek, çünkü tura ait değil — ve başarısız olursa
     kart bunu söylüyor: erişilemeyen bir zamanlayıcı, boş bir zamanlayıcıyla
     aynı şey değil. */
  function loadSchedule(host) {
    fetch('/api/schedule').then(function (r) { return r.json(); }).then(function (d) {
      clear(host);
      if (!d.reachable) {
        host.appendChild(el('p', 'empty', 'Zamanlayıcıya ulaşılamadı — '
          + 'OpenClaw Gateway ayakta değil. Bu, "kayıtlı iş yok" ile aynı şey değil.'));
        return;
      }
      var jobs = d.jobs || [];
      if (!jobs.length) {
        host.appendChild(el('p', 'empty', 'Kayıtlı iş yok.'));
      }
      jobs.forEach(function (j) {
        var row = el('div', 'sched__row' + (j.enabled ? '' : ' is-off'));
        row.appendChild(el('span', 'sched__mark', j.enabled ? '●' : '○'));
        var body = el('div');
        body.appendChild(el('span', 'sched__name', j.name || j.id));
        body.appendChild(el('span', 'sched__when', j.when || ''));
        if (j.last) { body.appendChild(el('span', 'sched__last', 'son: ' + j.last)); }
        row.appendChild(body);
        host.appendChild(row);
      });
      if (d.linger_warning) {
        var warn = el('p', 'sched__warn');
        warn.appendChild(rich(d.linger_warning));
        host.appendChild(warn);
      }
    }).catch(function () {
      clear(host);
      host.appendChild(el('p', 'empty', 'Zamanlayıcı okunamadı.'));
    });
  }

  // ------------------------------------------------------------------- cards
  function card(title, hint, wide) {
    var c = el('div', 'card' + (wide ? ' card--wide' : ''));
    var head = el('div', 'card__head');
    head.appendChild(el('span', 'card__title', title));
    if (hint) { head.appendChild(el('span', 'card__hint', hint)); }
    c.appendChild(head);
    return c;
  }

  function lane(name) {
    return el('span', 'lane lane--' + (name || 'agentchat'), name || '');
  }

  /* Kullanılan ile kullanılmayan aynı listede duruyor. Bu bilinçli: "hangisi
     koştu" sorusunun cevabı ancak koşmayanların yanında bir anlam taşıyor. */
  function markedRow(item, opts) {
    var row = el('div', 'row' + (item.used ? ' is-used' : ''));
    row.appendChild(el('span', 'row__mark', item.used ? '●' : '○'));
    var body = el('div');
    body.appendChild(el('span', 'row__name', item.name));
    if (opts && opts.badge && item[opts.badge]) {
      body.appendChild(el('span', 'row__meta', item[opts.badge]));
    }
    if (item.lane) { body.appendChild(document.createTextNode(' ')); body.appendChild(lane(item.lane)); }
    if (item.ref && item.ref !== '—') { body.appendChild(el('span', 'row__meta', item.ref)); }
    if (item.cost) { body.appendChild(el('span', 'row__meta', item.cost + ' token')); }
    if (item.what) { body.appendChild(el('span', 'row__what', item.what)); }
    var why = item.why || item.did;
    if (why) { body.appendChild(el('span', 'row__why', why)); }
    row.appendChild(body);
    return row;
  }

  function totals(report) {
    clear(totalsEl);
    var t = report.totals;
    var cells = [
      ['durum', report.status, report.status === 'error'],
      ['süre', t.seconds + ' sn', false],
      ['llm çağrısı', t.llm_calls, false],
      ['token', t.tokens, false],
      ['tool · istendi', t.tools_requested, false],
      ['tool · koştu', t.tools_ran, false],
      ['tool · kapıda tutuldu', t.tools_blocked, t.tools_blocked > 0],
      ['adım', report.steps, false]
    ];
    cells.forEach(function (c) {
      var cell = el('div', 'tot__cell');
      cell.appendChild(el('span', 'tot__key', c[0]));
      cell.appendChild(el('span', 'tot__val' + (c[2] ? ' is-warn' : ''), String(c[1])));
      totalsEl.appendChild(cell);
    });
  }

  /* Tepegöz: runtime'ın kendi dört işi, canlı.
     Graf mesajın nereye gittiğini çiziyor. Bu şerit, o mesajı taşırken
     runtime'ın ne yaptığını — ve dördü de kılavuzun kendi cümlesinden geliyor,
     bizim uydurduğumuz bir sınıflandırma değil. Sayaçlar önemli: "güvenlik
     sınırı 0" diyen bir tur, kapının o turda hiç devreye girmediğini söylüyor
     ve bunu grafta aramak kutu kutu dolaşmak demek. */
  function overhead(report) {
    var host = document.getElementById('overhead');
    if (!host) { return; }
    var data = report.overhead;
    if (!data || !data.cells) { host.hidden = true; return; }
    host.hidden = false;
    clear(host);
    data.cells.forEach(function (c) {
      var cell = el('div', 'oh__cell');
      cell.setAttribute('data-lane', c.lane || '');
      if (c.live) { cell.classList.add('is-live'); }
      if (!c.hits) { cell.classList.add('is-idle'); }
      var head = el('div', 'oh__head');
      head.appendChild(el('span', 'oh__name', c.name));
      head.appendChild(el('span', 'oh__hits', c.hits ? c.hits + '×' : '—'));
      cell.appendChild(head);
      cell.appendChild(el('code', 'oh__sub', c.sub));
      cell.appendChild(el('p', 'oh__note', c.note));
      host.appendChild(cell);
    });
    // Alıntı şeridin altında ve kısaltılmamış: dört hücrenin nereden geldiğini
    // sormak sunumda kesin gelen bir soru, ve cevabı ekranda durmalı.
    host.appendChild(el('div', 'oh__src',
      'runtime · ' + data.ref + '  ·  ' + data.spans + ' span  ·  “' +
      data.quote + '”'));
  }

  /* O an hangi ajan tasarımı koşuyor. Ekranın en üstünde ve her zaman görünür:
     "hangi desen" sorusu sunumda ilk sorulan soru, ve cevabı bir kartın içinde
     aranacak bir şey olmamalı. */
  function design(report) {
    var host = document.getElementById('design');
    if (!host) { return; }
    clear(host);
    [['TAKIM', report.design.team, report.design.team_note],
     ['DESEN', report.design.pattern, report.design.pattern_note],
     ['BEYAN', report.design.declared, '']].forEach(function (row) {
      if (!row[1]) { return; }
      var cell = el('div', 'design__cell');
      cell.appendChild(el('span', 'design__key', row[0]));
      cell.appendChild(el('span', 'design__val', row[1]));
      if (row[2]) { cell.appendChild(el('span', 'design__note', row[2])); }
      host.appendChild(cell);
    });
  }

  function render(report) {
    view.report = report;
    questionEl.textContent = report.question || '—';
    questionEl.title = report.question || '';
    totals(report);
    overhead(report);
    design(report);
    clear(cardsEl);

    // ---- the drawing ------------------------------------------------------
    var g = card('Bu soruda ne koştu', report.graph.shape, true);
    var sig = topoSignature(report);
    if (sig !== view.topo || !view.canvas) {
      view.canvas = drawGraph(report.graph);
      view.topo = sig;
    }
    // Yanan kutunun adı yazıyla da duruyor: animasyon nereye bakılacağını
    // söylüyor, bu satır orada ne olduğunu.
    var live = el('span', 'card__live');
    live.hidden = true;
    live.appendChild(el('span', 'card__dot'));
    live.appendChild(document.createTextNode(''));
    g.querySelector('.card__head').appendChild(live);
    g.appendChild(view.canvas);
    // Işığın bulunduğu kutunun içi buraya düşüyor.
    var slot = el('div', 'liveinner');
    slot.hidden = true;
    g.appendChild(slot);
    cardsEl.appendChild(g);

    // ---- the sequence -----------------------------------------------------
    // Grafın altına: aynı tur, bu kez zaman ekseninde.
    var seqCard = card('Sıra diyagramı', 'kim kime, hangi sırayla', true);
    seqCard.appendChild(drawSequence(report.sequence));
    cardsEl.appendChild(seqCard);

    // ---- the mechanism drawing --------------------------------------------
    // Hap destesindeki şemanın ta kendisi, o an koşan mekanizma için.
    var figCard = card('Mekanizma şeması', 'hap destesiyle aynı el', true);
    figCard.appendChild(el('div', 'fig'));
    cardsEl.appendChild(figCard);

    // ---- teams ------------------------------------------------------------
    var teams = card('Takım', 'beş tipten hangisi kuruldu');
    report.teams.forEach(function (t) {
      teams.appendChild(markedRow(t, { badge: 'picker' }));
    });
    cardsEl.appendChild(teams);

    // ---- patterns ---------------------------------------------------------
    var pats = card('Desen', 'üstüne gel: destedeki şeması');
    report.patterns.forEach(function (p) {
      var row = markedRow(p);
      if (window.PatternFigures && PatternFigures.has(p.id)) {
        row.classList.add('has-fig');
        row.addEventListener('mouseenter', function () { showPattern(p, row); });
        row.addEventListener('mouseleave', hideInner);
      }
      pats.appendChild(row);
    });
    cardsEl.appendChild(pats);

    // ---- components -------------------------------------------------------
    var comps = card('Bileşenler', 'kurulu olan ile bu turda iş yapan ayrı');
    report.components.forEach(function (c) { comps.appendChild(markedRow(c)); });
    cardsEl.appendChild(comps);

    // ---- message types ----------------------------------------------------
    var msgs = card('Mesaj tipleri', 'kenarların üstünde yazan şey');
    if (!report.messages.length) {
      msgs.appendChild(el('p', 'empty', 'Henüz mesaj uçmadı.'));
    }
    report.messages.forEach(function (m) {
      msgs.appendChild(markedRow({
        used: true, name: m.name, lane: m.lane, ref: m.ref, what: m.what,
        why: m.count + ' kez'
      }));
    });
    cardsEl.appendChild(msgs);

    // ---- topics -----------------------------------------------------------
    var top = card('Topic iletişimi', report.topics.active ? 'var' : 'bu turda yok');
    var t = el('div', 'row is-used');
    t.appendChild(el('span', 'row__mark', report.topics.active ? '●' : '○'));
    var tb = el('div');
    tb.appendChild(el('span', 'row__name', report.topics.topic));
    if (report.topics.ref) { tb.appendChild(el('span', 'row__meta', report.topics.ref)); }
    tb.appendChild(el('span', 'row__what', report.topics.note));
    t.appendChild(tb);
    top.appendChild(t);
    cardsEl.appendChild(top);

    // ---- code -------------------------------------------------------------
    if (report.code && report.code.length) {
      var codeCard = card('Konteynerde koşan kod', 'terminalin kalıcı hâli', true);
      report.code.forEach(function (run) {
        codeCard.appendChild(el('pre', 'code', run.code));
        if (run.running) {
          codeCard.appendChild(el('p', 'note', 'çalışıyor…'));
        } else {
          codeCard.appendChild(el('pre', 'code code--out' + (run.is_error ? ' code--err' : ''),
            run.output || '(çıktı yok)'));
          codeCard.appendChild(el('p', 'note',
            (run.is_error ? 'hata ile bitti' : 'exit 0') +
            (run.seconds != null ? ' · ' + run.seconds + ' sn' : '')));
        }
      });
      cardsEl.appendChild(codeCard);
    }

    // ---- scheduled work ---------------------------------------------------
    // Bu turun parçası DEĞİL, ve tam da bu yüzden burada: zamanlanmış iş,
    // kimse sormadan koşacak olan şey. Turun izini okuyup "sistem yalnız
    // sorulunca çalışıyor" diye düşünmek, en kolay yanlış çıkarım.
    var sched = el('div', 'sched');
    var schedCard = card('Zamanlanmış iş', 'bu turun dışında, kimse sormadan');
    schedCard.appendChild(sched);
    cardsEl.appendChild(schedCard);
    loadSchedule(sched);

    // ---- telemetry --------------------------------------------------------
    // Tepegöz: turun içindeki her işin başlangıç–bitiş aralığı. Zaman çizgisi
    // "ne oldu"yu sıralıyor, bu "ne kadar sürdü ve neyin içinde oldu"yu.
    var spans = report.spans || [];
    if (spans.length) {
      var tel = card('Telemetri · şelale',
                     spans.length + ' span · OpenTelemetry gen_ai', true);
      tel.appendChild(waterfall(spans));
      cardsEl.appendChild(tel);
    }

    // ---- timeline ---------------------------------------------------------
    var tl = card('Zaman çizgisi', report.timeline.length + ' adım', true);
    var list = el('ol', 'tl');
    report.timeline.forEach(function (s, index) {
      var li = el('li');
      // Bitmiş bir turu adım adım gezmek: sunumda asıl kullanılacak şey bu.
      li.addEventListener('click', function () { park(s); });
      li.appendChild(el('span', 't-at', '+' + Number(s.at).toFixed(2) + 's'));
      li.appendChild(el('span', 't-lane lane-' + (s.lane || ''), s.lane || ''));
      li.appendChild(el('span', 't-what', s.name + (s.note ? ' — ' + s.note : '')));
      li.appendChild(el('span', 't-ref', s.klass || ''));
      list.appendChild(li);
    });
    if (!report.timeline.length) { list.appendChild(el('li', 'empty', 'Aşama yok.')); }
    tl.appendChild(list);
    cardsEl.appendChild(tl);

    // Yürüyüş sürüyorsa devam, durduysa yeniden başlat. Yeni aşamalar geldiyse
    // kuyruk kendiliğinden uzuyor — `play.at` nerede kaldığını hatırlıyor.
    if (play.timer) { paint(); } else { advance(); }
  }

  /* Yürüyüşü durdurup belli bir aşamada beklemek. Hem şerit hem zaman çizgisi
     buraya geliyor: iki ayrı yerden aynı şeyin iki kopyası yazılmasın. */
  function park(row) {
    if (play.timer) { clearTimeout(play.timer); play.timer = null; }
    var rows = (view.report && view.report.timeline) || [];
    var index = rows.indexOf(row);
    if (index < 0) { index = rows.length - 1; }
    play.at = index + 1;
    play.target = row;
    play.reached = Math.max(play.reached, play.at);   // şerit hiç kısalmıyor
    play.manual = true;
    paint();
  }

  // Yeni bir tura geçildiğinde ışık baştan yürümeli.
  function resetPlay() {
    if (play.timer) { clearTimeout(play.timer); }
    play = { at: 0, reached: 0, timer: null, target: null, live: false,
             manual: false };
  }

  // -------------------------------------------------------------------- load
  function param(name) {
    var m = new RegExp('[?&]' + name + '=([^&]*)').exec(window.location.search);
    return m ? decodeURIComponent(m[1]) : null;
  }

  /* Hangi tura bakıldığı hatırlanıyor. Sekme kapanıp yeniden açıldığında
     `latest`'e düşmek, üstünde konuşulan turu elden kaçırmak demekti. */
  var wanted = param('run') || (function () {
    try { return localStorage.getItem('akis-run') || 'latest'; }
    catch (e) { return 'latest'; }
  })();

  var backButton = document.getElementById('back');
  if (backButton) {
    backButton.addEventListener('click', function () {
      // Sohbet bu sekmeyi açtıysa kapatmak oraya döndürüyor. Açmadıysa (yer
      // imi, ikinci ekran) gidecek bir sohbet yok, o yüzden gidiyoruz.
      if (window.opener && !window.opener.closed) { window.close(); }
      else { window.location.href = '/'; }
    });
  }
  var timer = null;

  function loadList(current) {
    fetch('/api/runs').then(function (r) { return r.json(); }).then(function (d) {
      clear(pickEl);
      (d.runs || []).forEach(function (r) {
        var o = el('option', null,
          r.id + ' · ' + (r.question || '').slice(0, 42) + ' · ' + r.seconds + 's');
        o.value = r.id;
        if (r.id === current) { o.selected = true; }
        pickEl.appendChild(o);
      });
      if (!pickEl.children.length) {
        pickEl.appendChild(el('option', null, 'henüz tur yok'));
      }
    }).catch(function () {});
  }

  function load(id) {
    fetch('/api/run/' + encodeURIComponent(id))
      .then(function (r) {
        if (!r.ok) { throw new Error('HTTP ' + r.status); }
        return r.json();
      })
      .then(function (report) {
        render(report);
        loadList(report.id);
        try { localStorage.setItem('akis-run', report.id); } catch (e) { /* özel mod */ }
        // Tur sürüyorsa ekran onunla birlikte doluyor. SSE değil çünkü bu sayfa
        // turu başlatan taraf değil — okuyan taraf, ve saniyede bir yeterli.
        if (timer) { clearTimeout(timer); }
        // Koşarken hızlı: bir aşama çoğu zaman bir saniyeden kısa sürüyor ve
        // saniyede bir bakan bir ışık, aşamaların yarısını hiç göstermez.
        if (report.status === 'running') {
          timer = setTimeout(function () { load(report.id); }, 500);
        }
      })
      .catch(function (e) {
        clear(cardsEl);
        var c = card('Tur bulunamadı', String(e.message || e), true);
        c.appendChild(el('p', 'note',
          'Sohbette bir soru sor, sonra "Akış" düğmesine bas.'));
        cardsEl.appendChild(c);
        loadList(null);
      });
  }

  pickEl.addEventListener('change', function () {
    if (pickEl.value) { resetPlay(); view.topo = null; load(pickEl.value); }
  });

  window.addEventListener('scroll', hideInner, { passive: true });

  load(wanted);
})();
