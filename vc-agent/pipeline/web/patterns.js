/* Sekiz resmî desenin şemaları — destedekilerin ta kendisi.
 *
 * `docs/diagrams/figures.py`'deki `f_fanout` … `f_codeexec_pattern`
 * fonksiyonlarının birebir karşılığı: aynı koordinatlar, aynı palet, aynı
 * açıklama satırları. Bilerek kopya değil **taşıma**: iki ayrı çizim, aynı
 * iddiayı bir süre sonra iki farklı şekilde anlatmaya başlıyor, ve destede
 * gördüğün şema ile ekranda gördüğün şema aynı olmak zorunda.
 *
 * Koordinatlar Python tarafındaki gibi elle yazılmış. Bir yerleşim algoritması
 * bunları daha "düzgün" dizerdi ama destedekinden farklı dizerdi.
 */
(function (global) {
  'use strict';

  var PALETTE = {
    blue: ['#1971c2', '#e7f5ff'], green: ['#2f9e44', '#ebfbee'],
    red: ['#c92a2a', '#fff5f5'], orange: ['#e8590c', '#fff4e6'],
    violet: ['#5f3dc4', '#f8f0fc'], grey: ['#868e96', '#f8f9fa'],
    ink: ['#1e1e1e', '#ffffff']
  };
  var MUTE = '#767d84';

  function box(pen, svg, x, y, w, h, title, sub, colour, opts) {
    var o = opts || {}, pair = PALETTE[colour || 'ink'];
    svg.appendChild(pen.rect(x, y, w, h,
      { stroke: pair[0], fill: pair[1], width: 1.6, dash: o.dash }));
    var cy = y + h / 2 + (sub ? -4 : 0);
    svg.appendChild(Rough.text(x + w / 2, cy + 3.2, title,
      { size: o.size || 9.4, colour: '#1e1e1e', weight: '700', anchor: 'middle' }));
    if (sub) {
      svg.appendChild(Rough.text(x + w / 2, cy + 14, sub,
        { size: 6.8, colour: MUTE, anchor: 'middle', mono: true }));
    }
  }

  function arrow(pen, svg, x1, y1, x2, y2, label, colour, opts) {
    var o = opts || {}, pair = PALETTE[colour || 'ink'];
    svg.appendChild(pen.line(x1, y1, x2, y2,
      { stroke: pair[0], width: 1.4, dash: o.dash, arrow: true }));
    if (label) {
      svg.appendChild(Rough.text((x1 + x2) / 2, (y1 + y2) / 2 - 5, label,
        { size: 7, colour: pair[0], anchor: 'middle', mono: true }));
    }
  }

  function note(svg, x, y, s, colour, size, anchor) {
    svg.appendChild(Rough.text(x, y, s,
      { size: size || 7.6, colour: colour || MUTE, anchor: anchor }));
  }

  var FIGURES = {
    /* Sekizden biri DEĞİL — ve tam da bu yüzden çizilmesi gerekiyor. Bizim
       sohbet turumuzun koştuğu şey bu: takım yok, `agent.run_stream()` doğrudan
       çağrılıyor ve ajan kendi döngüsünde modeli ve tool'ları sırayla işletiyor.
       Sekiz desenden birini buraya yazmak, olmayan bir takımı iddia etmek olurdu. */
    toolloop: function (pen, svg) {
      box(pen, svg, 18, 66, 136, 50, 'AssistantAgent', 'tek ajan', 'blue', { size: 8.6 });
      box(pen, svg, 226, 18, 146, 46, 'model_client', 'create_stream()', 'green',
          { size: 8.2 });
      box(pen, svg, 226, 122, 146, 46, 'workbench', 'call_tool()', 'violet',
          { size: 8.2 });
      box(pen, svg, 438, 66, 142, 50, 'TaskResult', 'tur biter', 'grey', { size: 8.2 });

      arrow(pen, svg, 156, 82, 224, 48, 'bağlam');
      arrow(pen, svg, 374, 44, 436, 80, 'cevap yazarsa');
      // Dikey: istek aşağı iner, sonuç sola döner. Yatay çizilince ikisi de
      // aynı bantta kalıyordu ve etiketleri birbirinin üstüne biniyordu.
      arrow(pen, svg, 299, 66, 299, 120, '', 'orange', { dash: '3 3' });
      note(svg, 308, 96, 'tool isterse', '#e8590c', 7.2);
      arrow(pen, svg, 224, 152, 158, 114, '', 'violet', { dash: '3 3' });
      note(svg, 150, 160, 'sonuç bağlama girer', '#5f3dc4', 7.2, 'end');

      note(svg, 18, 190, 'Takım YOK: beş takım tipinden hiçbiri kurulmuyor — ' +
                         '`agent.run_stream()` doğrudan çağrılıyor.');
      note(svg, 18, 206, 'Döngü en çok 6 tur: max_tool_iterations. Varsayılan 1\'dir, ' +
                         'yani zincir ilk tool\'dan sonra durur.', '#e8590c');
      note(svg, 18, 222, 'Sekiz resmî desenin hiçbiri değil — ve bunu yazmak, ' +
                         'olmayan bir takımı iddia etmekten daha değerli.', MUTE);
      return [600, 234];
    },
    concurrent: function (pen, svg) {
      box(pen, svg, 20, 56, 104, 44, 'koordinatör', '', 'ink', { size: 8.4 });
      box(pen, svg, 196, 8, 116, 36, 'arXiv', 'RoutedAgent', 'blue', { size: 8 });
      box(pen, svg, 196, 58, 116, 36, 'HN', 'RoutedAgent', 'blue', { size: 8 });
      box(pen, svg, 196, 108, 116, 36, 'GitHub', 'RoutedAgent', 'blue', { size: 8 });
      [26, 76, 126].forEach(function (ty) { arrow(pen, svg, 126, 78, 194, ty); });
      box(pen, svg, 392, 56, 128, 44, 'ClosureAgent', 'toplayıcı', 'green', { size: 8.4 });
      [26, 76, 126].forEach(function (fy) { arrow(pen, svg, 314, fy, 390, 78); });
      note(svg, 196, 162, 'hepsi AYNI topic\'e abone — tek publish üçünü birden uyandırır');
      note(svg, 392, 116, 'sayaç 3\'e ulaşınca');
      note(svg, 392, 130, 'kuyruğa yazar');
      note(svg, 20, 116, '1 publish', '#2f9e44');
      note(svg, 20, 130, '0 dönüş değeri', '#c92a2a');
      return [600, 172];
    },
    sequential: function (pen, svg) {
      var stages = [['Concept Extractor', 'özellik · hedef kitle'],
                    ['Writer', 'pazarlama metni'],
                    ['Format & Proof', 'dilbilgisi · ton'],
                    ['User', 'sunum']];
      stages.forEach(function (s, i) {
        var x = 14 + i * 148;
        box(pen, svg, x, 26, 124, 44, s[0], '', 'blue', { size: 8 });
        note(svg, x, 84, s[1], MUTE, 7);
        if (i < 3) { arrow(pen, svg, x + 126, 48, x + 144, 48); }
      });
      note(svg, 14, 116, 'Sıra DETERMİNİSTİK: her ajan bir alt görevi yapıp bir ' +
                         'sonrakine devrediyor. Kim konuşacak diye kimse karar vermiyor.');
      note(svg, 14, 132, 'core\'da bu, her ajanın bir sonrakinin topic\'ine yayın ' +
                         'yapmasıyla kuruluyor — zincir aboneliklerde yazılı.', MUTE);
      return [600, 144];
    },
    groupchat: function (pen, svg) {
      box(pen, svg, 232, 12, 136, 40, 'ORTAK TOPIC', 'hepsi abone + yayıncı',
          'orange', { size: 8.4 });
      [['yazar', 26, 96], ['çizer', 180, 110], ['editör', 334, 110],
       ['insan', 470, 96]].forEach(function (r) {
        box(pen, svg, r[1], r[2], 104, 34, r[0], '', 'blue', { size: 8 });
        arrow(pen, svg, r[1] + 52, r[2], 300, 56);
      });
      box(pen, svg, 232, 156, 136, 30, 'yönetici', 'sırayı dağıtır', 'violet',
          { size: 7.8 });
      note(svg, 14, 176, 'Tek bir mesaj dizisi: herkes aynı konuşmayı görüyor.', '#454c53');
      note(svg, 14, 192, 'Bedeli: bağlam herkes için aynı ve büyük. Ayırmaya değer ' +
                         'bir bağlam varsa yanlış desen.', '#c92a2a');
      return [600, 204];
    },
    handoffs: function (pen, svg) {
      box(pen, svg, 16, 40, 128, 46, 'triyaj ajanı', '', 'blue', { size: 8.4 });
      arrow(pen, svg, 146, 52, 196, 32, 'transfer_to_x');
      arrow(pen, svg, 146, 74, 196, 96);
      box(pen, svg, 198, 14, 128, 38, 'iade ajanı', '', 'green', { size: 8 });
      box(pen, svg, 198, 78, 128, 38, 'satış ajanı', '', 'green', { size: 8 });
      note(svg, 350, 30, 'devretme ÖZEL BİR TOOL ÇAĞRISI', '#8a5208', 7.6);
      note(svg, 350, 46, 'modelin kendi kararı — dışarıdan', MUTE, 7.2);
      note(svg, 350, 58, 'bir yönlendirici yok', MUTE, 7.2);
      note(svg, 350, 82, 'ölçülen: en pahalı desen, 334 token', '#c92a2a', 7.6);
      note(svg, 350, 96, 'her devirde bağlam yeniden kuruluyor', MUTE, 7.2);
      note(svg, 16, 140, 'OpenAI\'ın Swarm projesinden geliyor. AutoGen\'in eklediği: ' +
                         'dağıtık runtime\'a ölçeklenebilmesi.');
      return [600, 152];
    },
    mixture: function (pen, svg) {
      box(pen, svg, 14, 62, 92, 40, 'görev', '', 'grey', { size: 8 });
      [0, 1].forEach(function (layer) {
        var x = 138 + layer * 150;
        for (var j = 0; j < 3; j++) {
          var y = 14 + j * 52;
          box(pen, svg, x, y, 116, 38, 'işçi ' + (layer + 1) + '.' + (j + 1), '',
              'blue', { size: 7.6 });
          if (layer === 0) { arrow(pen, svg, 108, 82, 136, y + 19); }
          else { arrow(pen, svg, 256, 33 + j * 52, 286, y + 19, '', 'ink', { dash: '2 3' }); }
        }
      });
      box(pen, svg, 444, 62, 142, 40, 'orkestratör', 'birleştirir', 'green', { size: 8.4 });
      for (var j = 0; j < 3; j++) { arrow(pen, svg, 404, 33 + j * 52, 442, 82); }
      note(svg, 14, 128, 'İleri-beslemeli sinir ağı mimarisinden modellenmiş: katman ' +
                         'katman işçiler, bir önceki katmanın çıktıları BİRLEŞTİRİLİP ' +
                         'sonrakine gidiyor.');
      note(svg, 14, 144, 'arXiv:2406.04692 · aynı soru, farklı uzmanlıklar, tek ' +
                         'birleştirici.', MUTE);
      return [600, 156];
    },
    debate: function (pen, svg) {
      for (var t = 0; t < 3; t++) {
        var x = 20 + t * 152;
        note(svg, x + 50, 18, 'tur ' + (t + 1), MUTE, 7.4, 'middle');
        for (var j = 0; j < 2; j++) {
          box(pen, svg, x, 26 + j * 48, 104, 36, 'çözücü ' + (j + 1), '', 'blue',
              { size: 7.4 });
        }
        if (t < 2) {
          arrow(pen, svg, x + 106, 44, x + 150, 44, '', 'ink', { dash: '3 3' });
          arrow(pen, svg, x + 106, 92, x + 150, 92, '', 'ink', { dash: '3 3' });
          arrow(pen, svg, x + 106, 52, x + 150, 84, 'çapraz', 'ink', { dash: '2 3' });
        }
      }
      box(pen, svg, 480, 50, 106, 40, 'toplayıcı', '', 'green', { size: 8 });
      arrow(pen, svg, 428, 44, 478, 62);
      arrow(pen, svg, 428, 92, 478, 78);
      note(svg, 20, 122, 'Her turda ajanlar cevaplarını DEĞİŞ TOKUŞ edip birbirinin ' +
                         'cevabına göre kendilerininkini düzeltiyor.');
      note(svg, 20, 138, 'Çözücüler seyrek bağlı — herkes herkesle değil. GSM8K ' +
                         'matematik problemleri üstünde gösteriliyor.', MUTE);
      return [600, 150];
    },
    reflection: function (pen, svg) {
      box(pen, svg, 60, 40, 160, 54, 'ÜRETİCİ', 'kod yazar', 'blue', { size: 9 });
      box(pen, svg, 372, 40, 160, 54, 'ELEŞTİRMEN', 'kritik üretir', 'orange', { size: 9 });
      arrow(pen, svg, 222, 56, 370, 56, 'taslak');
      svg.appendChild(pen.curve([[370, 82], [300, 108], [224, 82]],
        { stroke: PALETTE.orange[0], width: 1.4, arrow: true }));
      note(svg, 296, 124, 'düzeltme isteği', '#8a5208', 7.4, 'middle');
      note(svg, 60, 150, 'İkinci LLM üretimi, birincinin ÇIKTISINA koşullanmış. Döngü ' +
                         'eleştirmen tatmin olana kadar sürüyor.');
      note(svg, 60, 166, 'Bizde karşılığı: RiskAuditor — üç analizi çelişki ve ' +
                         'kaynaksız iddia için çapraz kontrol ediyor.', '#2f9e44');
      return [600, 178];
    },
    codeexec: function (pen, svg) {
      box(pen, svg, 30, 46, 150, 52, 'Assistant', 'kodu YAZAR', 'blue', { size: 9 });
      box(pen, svg, 390, 46, 150, 52, 'Executor', 'kodu KOŞTURUR', 'green', { size: 9 });
      arrow(pen, svg, 182, 60, 388, 60, 'kod bloğu');
      svg.appendChild(pen.curve([[388, 88], [285, 116], [182, 88]],
        { stroke: PALETTE.green[0], width: 1.4, arrow: true }));
      note(svg, 285, 132, 'çıktı ya da hata', '#14594a', 7.4, 'middle');
      note(svg, 30, 158, 'İki ayrı ajan, tek bir Message veri sınıfı. AgentChat\'te ' +
                         'hazır karşılığı var (CodeExecutorAgent) ama kılavuz burada ' +
                         'elle yazmayı gösteriyor.');
      note(svg, 30, 174, 'Bizde: kod bir kaçış kapağı — tool\'u olmayan iş için, ve ' +
                         'aynı kapıdan geçerek.', MUTE);
      return [600, 186];
    }
  };

  // Deste bunları 1..8 diye numaralandırıyor; ekranda da aynı sırayla duruyorlar.
  var SEED = { toolloop: 77, concurrent: 15, sequential: 125, groupchat: 126, handoffs: 127,
               mixture: 128, debate: 129, reflection: 130, codeexec: 131 };

  function draw(id) {
    if (!global.Rough || !FIGURES[id]) { return null; }
    var svg = Rough.svg(600, 200);
    var pen = new Rough.Pen(SEED[id] || 7, 1);
    var size = FIGURES[id](pen, svg);
    svg.setAttribute('viewBox', '0 0 ' + size[0] + ' ' + size[1]);
    svg.setAttribute('width', String(size[0]));
    svg.setAttribute('height', String(size[1]));
    return svg;
  }

  global.PatternFigures = { draw: draw, has: function (id) { return !!FIGURES[id]; } };
})(window);
