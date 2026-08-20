/* Mekanizma şemaları — her biri tek bir kavramın resmi.
 *
 * Bunlar hap destesindeki (`docs/pdf/hap-autogen.pdf`) şemaların tarayıcıdaki
 * karşılığı ve aynı elden çıkıyor: `rough.js`, yani `docs/diagrams/rough.py`'nin
 * JS ikizi. Ekranda gördüğün şema ile destede gördüğün şema aynı olmalı — iki
 * ayrı çizim, aynı iddiayı iki farklı şekilde anlatmaya başlıyor.
 *
 * Sohbetin içindeki eski panelden buraya taşındılar. Panel kalktı, şemalar
 * kalmalıydı: bir turda ne olduğunu anlatan asıl şey bunlar, ve `3af7313`
 * commit'inden bire bir alındılar.
 *
 * `m`, aşamanın taşıdığı meta — şema kendi sayılarını oradan okuyor, buraya
 * gömülü hiçbir sayı yok.
 */
(function (global) {
  'use strict';

  var Rough = global.Rough;
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

    /* Her mekanizmaya bir çizim. Her biri o şeyin resmi, bir sıradaki
       yerinin değil. `m`, aşamanın taşıdığı meta. */
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
      code_request: function (pen, svg, m) {
        box(pen, svg, 16, 24, 150, 44, BLUE, 'model', 'kodu YAZDI',
            { fill: WASH.blue, size: 8.4 });
        line(pen, svg, 166, 46, 210, 46, ORANGE, { arrow: true });
        box(pen, svg, 214, 20, 176, 52, ORANGE, 'onay kartı', 'kodun KENDİSİ',
            { fill: WASH.orange, width: 2.2, size: 8.4 });
        line(pen, svg, 390, 46, 434, 46, GREY, { arrow: true, dash: '4 3' });
        box(pen, svg, 438, 24, 150, 44, GREY, 'konteyner', 'henüz çalışmadı',
            { dash: '4 3', width: 1.3, size: 8 });
        label(svg, 16, 96, 'Onay ÇALIŞACAK METNE bağlanıyor, modele değil.', INK,
              { size: 8 });
        label(svg, 16, 112, 'Ölçüldü: model aynı soruya her seferinde farklı bir ' +
              'program yazıyor — onayı yeniden üretilene bağlarsan hiç tüketilemiyor.',
              RED, { size: 7.6 });
        label(svg, 16, 130, (m.code ? String(m.code).split('\n')[0].slice(0, 74)
                                    : ''), MUTE, { mono: true, size: 7 });
      },
      code_result: function (pen, svg, m) {
        var ok = !m.is_error;
        box(pen, svg, 16, 24, 168, 46, VIOLET, 'python:3-slim', 'izole konteyner',
            { fill: WASH.violet, size: 8.4 });
        line(pen, svg, 184, 47, 228, 47, ok ? GREEN : RED, { arrow: true });
        box(pen, svg, 232, 22, 176, 50, ok ? GREEN : RED,
            ok ? 'exit 0' : 'hata',
            (m.seconds != null ? m.seconds + ' sn' : ''),
            { fill: ok ? WASH.green : WASH.red, size: 8.4 });
        box(pen, svg, 432, 24, 156, 46, RED, 'ağ', 'AÇIK',
            { dash: '4 3', width: 1.8, ink: RED, size: 8 });
        line(pen, svg, 184, 60, 430, 56, RED, { dash: '3 3' });
        label(svg, 16, 98, 'Konteyner izole ama ağ erişimi VAR.', RED,
              { size: 8, weight: '700' });
        label(svg, 16, 114, 'DockerCommandLineCodeExecutor\'da network_mode diye ' +
              'bir parametre yok — ölçüldü, kaynakta ağ ile ilgili tek kelime geçmiyor.',
              MUTE, { size: 7.6 });
        label(svg, 16, 132, String(m.output || '').split('\n')[0].slice(0, 76),
              GREY, { mono: true, size: 7 });
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

  /* Bir aşamanın şemasını çizip döndürür; şeması olmayan aşama için null.
     Çağıran taraf null'ı boş kutu çizmeden geçiyor: her aşamanın resmi yok, ve
     olmayan bir resmin yerine boş çerçeve koymak hiçbir şey öğretmiyor. */
  function draw(id, meta) {
    if (!Rough || !FIGURES[id]) { return null; }
    var pen = new Rough.Pen(9137);
    var svg = Rough.svg(640, 150);
    FIGURES[id](pen, svg, meta || {});
    return svg;
  }

  global.MechFigures = {
    draw: draw,
    has: function (id) { return !!FIGURES[id]; },
    ids: function () { return Object.keys(FIGURES); }
  };
})(window);
