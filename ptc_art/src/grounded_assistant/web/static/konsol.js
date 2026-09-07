/* Konsol — sekmeler, hat çalıştırma, depo, soy. Hepsi CANLI.
 *
 * `app.js` bu sayfada da yüklü: sohbet sekmesi ve sol-alt PTC paneli oradan
 * geliyor, tek satırı bile kopyalanmadı. Buradaki tek ortak nokta, hat
 * çalıştırmasının aynı PTC paneline yazması — iki farklı yerde iki farklı
 * log penceresi olması kafa karıştırırdı.
 *
 * Sahte veri yok: depo `/api/depo`, soy `/api/depo/<id>/soy`, çalıştırma
 * `/ws/pipeline` üzerinden gerçek Kubernetes Job'larından geliyor.
 */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]));
const kb = b => !b ? "0 B" : b >= 1e6 ? (b / 1048576).toFixed(2) + " MB" : b >= 1e3 ? (b / 1024).toFixed(0) + " KB" : b + " B";
const kisa = s => String(s || "").slice(0, 8);

const S = {
  hatlar: [], hat: "a", adim: null, durum: {}, calisiyor: false,
  wfSon: {}, kayitlar: [], filtre: "hepsi", acik: {}, art: null, soyId: null,
};

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

const uyari = m => `<div class="uyari">${esc(m)}</div>`;

/* PTC paneline yaz — `app.js`in kullandığı DOM'un aynısı. */
function ptcYaz(metin, sinif = "info") {
  const log = $("#ptc-panel-log");
  if (!log) return;
  const d = document.createElement("div");
  d.className = "line " + sinif;
  d.textContent = metin;
  log.appendChild(d);
  log.scrollTop = log.scrollHeight;
}
function ptcTemizle() { const l = $("#ptc-panel-log"); if (l) l.innerHTML = ""; }

/* ══ üst çubuk ═══════════════════════════════════════════════════ */

async function yenileSayac() {
  try {
    const d = await getJSON("/api/depo?limit=1000");
    if (d.error) throw new Error(d.error);
    S.kayitlar = d.kayitlar;
    $("#k-art").textContent = d.kayitlar.length;
    const wf = new Set(d.kayitlar.map(a => a.workflow_id)).size;
    $("#k-servis").textContent = `${d.kayitlar.length} artifact · ${wf} çalıştırma`;
  } catch {
    $("#k-art").textContent = "—";
    $("#k-servis").textContent = "servis kapalı";
  }
}

/* ══ 1 · HATLAR ══════════════════════════════════════════════════ */

function ciz_hatlar() {
  const box = $("#hatGrid");
  if (!S.hatlar.length) { box.innerHTML = uyari("Hat tanımları alınamadı."); return; }

  box.innerHTML = S.hatlar.map(h => {
    const wf = S.wfSon[h.key];
    const uretilen = S.kayitlar.filter(a => a.workflow_id === wf).length;
    const podlar = h.nodes.filter(n => n.tur === "sandbox").length;
    return `<button class="hat" data-hat="${h.key}">
      <h3>${esc(h.ad)} <span class="badge ${wf ? "ok" : ""}">${wf ? "çalıştı" : "hiç çalışmadı"}</span></h3>
      <p>${esc(h.aciklama)}</p>
      <div class="hat-meta">
        <div><b>${h.nodes.length}</b><span>adım</span></div>
        <div><b>${podlar}</b><span>sandbox pod'u</span></div>
        <div><b>${uretilen}</b><span>üretilen artifact</span></div>
        <div><b>${wf ? kisa(wf) : "—"}</b><span>son çalıştırma</span></div>
      </div>
    </button>`;
  }).join("");

  $$("#hatGrid .hat").forEach(b => b.onclick = () => {
    S.hat = b.dataset.hat; S.adim = null; S.durum = {}; git("calistir");
  });
}

/* ══ 2 · ÇALIŞTIRMA ══════════════════════════════════════════════ */

const hatOf = k => S.hatlar.find(h => h.key === k);

function ciz_calistir() {
  const h = hatOf(S.hat);
  if (!h) { $("#adimlar").innerHTML = uyari("Hat tanımı yüklenemedi."); return; }

  $("#runTitle").textContent = h.ad;
  $("#runDesc").textContent = h.aciklama;

  const bitti = h.nodes.every(n => S.durum[n.n]?.status === "success");
  const p = $("#runPill");
  p.className = "badge " + (S.calisiyor ? "live" : bitti ? "ok" : "");
  p.textContent = S.calisiyor ? "çalışıyor" : bitti ? "tamamlandı" : "hazır";
  $("#btnRun").disabled = S.calisiyor;

  $("#adimlar").innerHTML = h.nodes.map((n, i) => {
    const d = S.durum[n.n] || {};
    const cls = d.status === "success" ? "bitti" : d.status === "running" ? "calisiyor"
              : d.status === "error" ? "hata" : "";
    const bag = i ? `<div class="baglanti ${S.durum[h.nodes[i - 1].n]?.status === "success" ? "bitti" : ""}"></div>` : "";
    const sag = d.dur ? `<span class="badge">${esc(d.dur)}</span>`
              : n.tur === "sandbox" ? `<span class="badge">pod</span>`
              : `<span class="badge">sorgu</span>`;
    const cikti = (d.artifacts || []).map(a => `<span class="badge art">${esc(a.name)}</span>`).join(" ");
    return `${bag}<button class="adim ${S.adim === n.n ? "on" : ""}" data-tur="${n.tur}" data-n="${n.n}">
      <span class="baloncuk ${cls}">${n.n}</span>
      <span>
        <span class="adim-ad">${esc(n.ad)}</span>
        <span class="adim-alt">${cikti || esc(n.aciklama)}</span>
      </span>
      ${sag}
    </button>`;
  }).join("");

  $$("#adimlar .adim").forEach(b => b.onclick = () => {
    const n = +b.dataset.n;
    S.adim = S.adim === n ? null : n; ciz_calistir();
  });
  ciz_adimDetay();
}

function ciz_adimDetay() {
  const kutu = $("#adimDetay"), h = hatOf(S.hat);
  if (!S.adim || !h) {
    kutu.innerHTML = `<p class="bos">Bir adım seçin: kodunu, beyan ettiği girdileri ve ürettiği artifact'i gösterir.</p>`;
    return;
  }
  const n = h.nodes.find(x => x.n === S.adim), d = S.durum[n.n] || {};

  kutu.innerHTML = `
    <div class="card-h">
      <h2>${esc(n.ad)}</h2>
      <span class="badge ${d.status === "success" ? "ok" : d.status === "error" ? "err" : ""}">${
        n.tur === "sandbox" ? "sandbox pod'u" : "kayıt defteri sorgusu"}</span>
    </div>

    <div class="not" style="margin-bottom:.9rem">${n.tur === "sandbox"
      ? `<b>Gerçek pod.</b> Ayrı bir Job açılıyor; kapsam jetonu sidecar'da, sandbox'ta S3 anahtarı yok, ağı kapalı.`
      : `<b>Pod açılmıyor.</b> Kayıt defterine HTTP sorgusu — keşif sandbox'ta değil, host tarafında olur.`}</div>

    <p class="muted" style="margin:0 0 .9rem">${esc(n.aciklama)}</p>

    <dl class="kv" style="margin-bottom:.9rem">
      <dt>Beyan edilen girdi</dt><dd>${n.inputs.length ? esc(JSON.stringify(n.inputs)) : "[] — yok"}</dd>
      ${n.sorgu ? `<dt>Sorgu</dt><dd>?${esc(new URLSearchParams(n.sorgu).toString())}</dd>` : ""}
      <dt>Durum</dt><dd class="duz">${d.status === "success" ? "tamamlandı " + (d.dur || "")
        : d.status === "running" ? "çalışıyor…" : d.status === "error" ? "hata" : "beklemede"}</dd>
      ${d.run_id ? `<dt>Pod</dt><dd>ptc-sandbox-${esc(d.run_id)}</dd>` : ""}
    </dl>

    ${(d.artifacts || []).length ? `<div style="margin-bottom:.9rem">
      <div class="muted" style="margin-bottom:.35rem">Üretilen artifact</div>
      ${d.artifacts.map(a => `<button class="plaka" data-art="${esc(a.artifact_id)}" style="width:100%;margin-bottom:.35rem">
        <span class="ad">${esc(a.name)}</span>
        <span class="alt">${esc(a.artifact_id)} · ${kb(a.size_bytes)} · depoda aç →</span></button>`).join("")}
    </div>` : ""}

    ${d.sonuc ? `<div style="margin-bottom:.9rem"><div class="muted" style="margin-bottom:.35rem">Sonuç</div>
      <pre class="kod">${esc(typeof d.sonuc === "string" ? d.sonuc : JSON.stringify(d.sonuc, null, 2))}</pre></div>` : ""}

    ${(d.kod || n.kod) ? `<details><summary class="muted" style="cursor:pointer;font-size:.83rem">Sandbox'ta çalışan kod</summary>
      <pre class="kod" style="margin-top:.5rem">${esc((d.kod || n.kod).trim())}</pre></details>` : ""}`;

  $$("[data-art]", kutu).forEach(b => b.onclick = () => acArtifact(b.dataset.art));
}

let WS = null;

function calistir() {
  if (S.calisiyor) return;
  S.durum = {}; S.calisiyor = true; S.adim = null;
  ptcTemizle();
  ptcYaz(`▶ ${hatOf(S.hat).ad} başlatılıyor…`, "info");
  ciz_calistir();

  const proto = location.protocol === "https:" ? "wss" : "ws";
  WS = new WebSocket(`${proto}://${location.host}/ws/pipeline`);
  WS.onopen = () => WS.send(JSON.stringify({ type: "run", key: S.hat }));
  WS.onerror = () => { S.calisiyor = false; ptcYaz("Bağlantı hatası", "denied"); ciz_calistir(); };
  WS.onclose = () => { if (S.calisiyor) { S.calisiyor = false; ciz_calistir(); } };

  WS.onmessage = ev => {
    const m = JSON.parse(ev.data);
    const d = n => (S.durum[n] = S.durum[n] || { loglar: [], artifacts: [] });

    if (m.type === "pipeline_start") {
      S.wfSon[m.key] = m.workflow_id;
      ptcYaz(`workflow ${m.workflow_id}`, "info");
    } else if (m.type === "node_start") {
      d(m.n).status = "running"; S.adim = m.n;
      ptcYaz(`\n[${m.n}] ${m.ad}  ·  ${m.tur === "sandbox" ? "sandbox pod'u" : "kayıt defteri sorgusu"}`, "info");
      ciz_calistir();
    } else if (m.type === "log") {
      d(m.n).loglar.push(m);
      ptcYaz(`    ${m.ts}  ${m.msg}`, m.cls === "fail" ? "denied" : "info");
    } else if (m.type === "code") {
      d(m.n).kod = m.kod;
    } else if (m.type === "artifact") {
      if (m.op === "produced") {
        d(m.n).artifacts.push({ artifact_id: m.artifact_id, name: m.name, size_bytes: m.size_bytes });
        yenileSayac();
      }
      ciz_calistir();
    } else if (m.type === "node_done") {
      Object.assign(d(m.n), {
        status: m.status === "success" ? "success" : "error",
        dur: m.dur, sonuc: m.sonuc, run_id: m.run_id,
      });
      ciz_calistir();
    } else if (m.type === "pipeline_done") {
      S.calisiyor = false;
      ptcYaz(`\n■ ${m.status === "success" ? "tamamlandı" : "hata"}`, m.status === "success" ? "info" : "denied");
      ciz_calistir(); ciz_hatlar(); yenileSayac();
      WS && WS.close();
    }
  };
}

/* ══ 3 · DEPO ════════════════════════════════════════════════════ */

async function ciz_depo() {
  const cek = $("#depoCekmece"), suz = $("#depoSuzgec");
  let d;
  try { d = await getJSON("/api/depo?limit=1000"); }
  catch (e) { cek.innerHTML = uyari(e.message); suz.innerHTML = ""; return; }
  if (d.error) { cek.innerHTML = uyari(d.error); suz.innerHTML = ""; return; }

  S.kayitlar = d.kayitlar;
  const arts = d.kayitlar;
  const tipler = [...new Set(arts.map(a => a.type))];

  suz.innerHTML =
    `<button class="${S.filtre === "hepsi" ? "on" : ""}" data-f="hepsi">Hepsi · ${arts.length}</button>` +
    tipler.map(t => `<button class="${S.filtre === "t:" + t ? "on" : ""}" data-f="t:${esc(t)}">${esc(t.replace("system.", ""))} · ${arts.filter(a => a.type === t).length}</button>`).join("") +
    (Object.values(S.wfSon).filter(Boolean).length
      ? `<button class="${S.filtre === "benim" ? "on" : ""}" data-f="benim">Bu oturum</button>` : "");

  let gosterilen = arts;
  if (S.filtre.startsWith("t:")) gosterilen = arts.filter(a => a.type === S.filtre.slice(2));
  if (S.filtre === "benim") gosterilen = arts.filter(a => Object.values(S.wfSon).includes(a.workflow_id));

  const grup = {};
  gosterilen.forEach(a => (grup[a.workflow_id] = grup[a.workflow_id] || []).push(a));
  const sirali = Object.entries(grup).sort((x, y) => y[1].length - x[1].length);

  cek.innerHTML = sirali.length ? sirali.map(([wf, list]) => {
    const acik = S.acik[wf] !== false;
    const benim = Object.values(S.wfSon).includes(wf);
    return `<div class="cekmece ${acik ? "acik" : ""}" data-wf="${esc(wf)}">
      <button class="cekmece-yuz">
        <span class="cekmece-ok">▸</span>
        <span style="flex:1;min-width:0">
          <span class="mono" style="font-size:.82rem">${esc(wf)}</span>
          <span class="muted" style="display:block;font-size:.74rem">${list.length} artifact · ${kb(list.reduce((s, a) => s + (a.size_bytes || 0), 0))}</span>
        </span>
        ${benim ? `<span class="badge live">bu oturum</span>` : ""}
      </button>
      <div class="raf">${list.map(plaka).join("")}</div>
    </div>`;
  }).join("") : `<p class="bos">Bu süzgeçle eşleşen artifact yok.</p>`;

  $$("#depoSuzgec button").forEach(b => b.onclick = () => { S.filtre = b.dataset.f; ciz_depo(); });
  $$("#depoCekmece .cekmece-yuz").forEach(b => b.onclick = () => {
    const wf = b.closest(".cekmece").dataset.wf;
    S.acik[wf] = S.acik[wf] === false; ciz_depo();
  });
  $$("#depoCekmece .plaka").forEach(b => b.onclick = () => { S.art = b.dataset.art; ciz_depo(); ciz_artDetay(); });
  ciz_artDetay();
}

function plaka(a) {
  return `<button class="plaka ${S.art === a.artifact_id ? "on" : ""}" data-art="${esc(a.artifact_id)}">
    <span class="ad">${esc(a.name)}</span>
    <span class="alt">
      <span class="badge">${esc((a.type || "").replace("system.", ""))}</span>
      ${kb(a.size_bytes)}
      ${a.parents?.length ? `· ${a.parents.length} ebeveyn` : ""}
      ${a.alias ? `<span class="badge live">@${esc(a.alias)}</span>` : ""}
    </span></button>`;
}

async function ciz_artDetay() {
  const kutu = $("#artDetay");
  const a = S.kayitlar.find(x => x.artifact_id === S.art);
  if (!a) { kutu.innerHTML = `<p class="bos">Bir çekmece açıp artifact seçin.</p>`; return; }

  kutu.innerHTML = `
    <div class="card-h"><h2 class="mono" style="font-size:.92rem;word-break:break-all">${esc(a.name)}</h2></div>
    <dl class="kv" style="margin-bottom:1rem">
      <dt>ID</dt><dd>${esc(a.artifact_id)}</dd>
      <dt>Tip</dt><dd class="duz">${esc(a.type)}</dd>
      <dt>Boyut</dt><dd>${kb(a.size_bytes)}</dd>
      <dt>Çalıştırma</dt><dd>${esc(a.workflow_id)}</dd>
      <dt>Node</dt><dd>${esc(a.node_id || "—")}</dd>
      <dt>Zaman</dt><dd>${esc((a.created_at || "").replace("T", " ").slice(0, 19))}</dd>
      <dt>Alias</dt><dd>${a.alias ? "@" + esc(a.alias) : "—"}</dd>
    </dl>

    <div style="margin-bottom:1rem">
      <div class="muted" style="margin-bottom:.35rem">Beyan edilen ebeveynler</div>
      ${a.parents?.length ? a.parents.map(p => {
        const pa = S.kayitlar.find(x => x.artifact_id === p);
        const capraz = pa && pa.workflow_id !== a.workflow_id;
        return `<button class="plaka" data-art="${esc(p)}" style="width:100%;margin-bottom:.35rem">
          <span class="ad">${esc(pa ? pa.name : p)}</span>
          <span class="alt">${capraz ? `<span class="badge live">çapraz workflow</span>` : "aynı çalıştırma"}</span></button>`;
      }).join("") : `<p class="bos">Kök artifact — girdisi yok.</p>`}
    </div>

    <div style="margin-bottom:1rem">
      <div class="muted" style="margin-bottom:.35rem">Sürümü sabitle</div>
      <div style="display:flex;gap:.5rem">
        <input id="aliasIn" class="k-input" value="${esc(a.alias || "")}" placeholder="ör. onaylanmis" />
        <button class="k-btn ikincil" id="aliasBtn">Ata</button>
      </div>
      <div id="aliasSonuc" style="margin-top:.5rem"></div>
    </div>

    <div style="display:flex;gap:.5rem;margin-bottom:1rem">
      <button class="k-btn ikincil" id="soyBtn">Soy ağacını aç</button>
    </div>

    <div class="muted" style="margin-bottom:.35rem">Önizleme</div>
    <div id="onizleme"><p class="bos">Yükleniyor…</p></div>`;

  $$("[data-art]", kutu).forEach(b => b.onclick = () => { S.art = b.dataset.art; ciz_depo(); });
  $("#soyBtn", kutu).onclick = () => { S.soyId = a.artifact_id; git("soy"); };
  $("#aliasBtn", kutu).onclick = async () => {
    const v = $("#aliasIn").value.trim(), out = $("#aliasSonuc");
    out.innerHTML = `<p class="bos">Gönderiliyor…</p>`;
    try {
      const r = await getJSON(`/api/depo/${a.artifact_id}/alias${v ? "?alias=" + encodeURIComponent(v) : ""}`, { method: "PUT" });
      out.innerHTML = r.error ? uyari(r.error)
        : `<div class="not"><b>Atandı.</b> Artık <span class="mono">${esc(a.name)}@${esc(r.alias || v)}</span> ile çözülüyor.</div>`;
      if (!r.error) ciz_depo();
    } catch (e) { out.innerHTML = uyari(e.message); }
  };

  try {
    const o = await getJSON(`/api/artifact/${a.artifact_id}?session=${encodeURIComponent(a.workflow_id)}`);
    const el = $("#onizleme"); if (!el) return;
    const cerceve = "width:100%;border-radius:10px;border:1px solid var(--border)";
    if (o.hata) el.innerHTML = `<p class="bos">${esc(o.hata)}</p>`;
    else if (o.bilgi) el.innerHTML = `<p class="bos">${esc(o.bilgi)}</p>`;
    else if (o.gorsel) el.innerHTML = `<img src="${o.gorsel}" style="${cerceve}" alt="${esc(a.name)}" />`;
    else if (o.pdf) el.innerHTML = `<embed src="${o.pdf}" type="application/pdf" style="${cerceve};height:22rem" />`;
    else el.innerHTML = `<pre class="kod">${esc(JSON.stringify(o.tablo ?? o.metin ?? o, null, 2)).slice(0, 1200)}</pre>`;
  } catch { const el = $("#onizleme"); if (el) el.innerHTML = `<p class="bos">Önizleme alınamadı.</p>`; }
}

function acArtifact(id) { S.art = id; S.filtre = "hepsi"; git("depo"); }

/* ══ 4 · SOY ═════════════════════════════════════════════════════ */

async function ciz_soy() {
  const alan = $("#soyAlan");
  if (!S.soyId) {
    const aday = S.kayitlar.filter(a => a.parents?.length);
    if (!aday.length) { alan.innerHTML = `<div class="card"><p class="bos">Henüz soyu olan bir artifact yok. Bir hat çalıştırın.</p></div>`; return; }
    S.soyId = aday[0].artifact_id;
  }
  alan.innerHTML = `<div class="card"><p class="bos">Yükleniyor…</p></div>`;
  let g;
  try { g = await getJSON(`/api/depo/${S.soyId}/soy`); }
  catch (e) { alan.innerHTML = `<div class="card">${uyari(e.message)}</div>`; return; }
  if (g.error || g.hata) { alan.innerHTML = `<div class="card">${uyari(g.error || g.hata)}</div>`; return; }

  const dugum = g.nodes || [], kenar = g.edges || [];
  const merkez = dugum.find(n => n.artifact_id === S.soyId) || dugum[0];
  const ata = new Set(), urun = new Set();
  kenar.forEach(e => { if (e.to === S.soyId) ata.add(e.from); if (e.from === S.soyId) urun.add(e.to); });

  const kat = [
    dugum.filter(n => ata.has(n.artifact_id)),
    dugum.filter(n => n.artifact_id === S.soyId),
    dugum.filter(n => urun.has(n.artifact_id)),
  ];
  const W = 900, Y = [80, 210, 340], yer = {};
  kat.forEach((satir, si) => satir.forEach((n, i) =>
    (yer[n.artifact_id] = { x: W / (satir.length + 1) * (i + 1), y: Y[si] })));

  const kutular = kat.flatMap((satir, si) => satir.map(n => {
    const { x, y } = yer[n.artifact_id];
    const capraz = merkez && n.workflow_id && n.workflow_id !== merkez.workflow_id;
    return `<g data-art="${esc(n.artifact_id)}" style="cursor:pointer">
      <rect x="${x - 95}" y="${y - 20}" width="190" height="40" rx="9" class="kutu ${si === 1 ? "merkez" : ""}"/>
      <text x="${x}" y="${y - 2}" text-anchor="middle" class="ad">${esc(n.name)}</text>
      <text x="${x}" y="${y + 12}" text-anchor="middle" class="alt">${kb(n.size_bytes)}${capraz ? " · başka çalıştırma" : ""}</text>
    </g>`;
  })).join("");

  const cizgi = kenar.filter(e => yer[e.from] && yer[e.to]).map(e => {
    const a = yer[e.from], b = yer[e.to];
    const na = dugum.find(n => n.artifact_id === e.from), nb = dugum.find(n => n.artifact_id === e.to);
    const capraz = na && nb && na.workflow_id !== nb.workflow_id;
    return `<path class="kenar ${capraz ? "capraz" : ""}" d="M${a.x} ${a.y + 20} C${a.x} ${a.y + 60} ${b.x} ${b.y - 60} ${b.x} ${b.y - 20}"/>`;
  }).join("");

  alan.innerHTML = `
    <div class="card">
      <div class="card-h">
        <h2 class="mono" style="font-size:.92rem">${esc(merkez?.name || S.soyId)}</h2>
        <span class="badge">${dugum.length} düğüm · ${kenar.length} kenar</span>
      </div>
      <svg class="soy-svg" viewBox="0 0 ${W} 400" role="img" aria-label="Soy ağacı">
        <text x="8" y="24" class="bant">ATALAR — beyan edilen girdiler</text>
        <text x="8" y="160" class="bant">BU ARTIFACT</text>
        <text x="8" y="292" class="bant">ÜRÜNLER — bunu girdi alanlar</text>
        ${cizgi}${kutular}
        ${!kat[0].length ? `<text x="${W / 2}" y="80" text-anchor="middle" class="alt">kök — girdisi yok</text>` : ""}
        ${!kat[2].length ? `<text x="${W / 2}" y="340" text-anchor="middle" class="alt">henüz kimse tüketmedi</text>` : ""}
      </svg>
      <p class="muted" style="margin:.6rem 0 0">Kesikli mavi kenar, iki farklı çalıştırma arasında geçen bağı gösterir.</p>
    </div>
    <div class="card">
      <div class="card-h"><h2>Soyu olan diğer artifact'ler</h2></div>
      <div class="suzgec">${S.kayitlar.filter(a => a.parents?.length).slice(0, 14)
        .map(a => `<button data-soy="${esc(a.artifact_id)}" class="${a.artifact_id === S.soyId ? "on" : ""}">${esc(a.name)}</button>`).join("")}</div>
    </div>`;

  $$("#soyAlan [data-art]").forEach(g2 => g2.onclick = () => { S.soyId = g2.dataset.art; ciz_soy(); });
  $$("#soyAlan [data-soy]").forEach(b => b.onclick = () => { S.soyId = b.dataset.soy; ciz_soy(); });
}

/* ══ sekmeler ════════════════════════════════════════════════════ */

function git(t) {
  $$(".k-view").forEach(v => v.classList.toggle("on", v.id === "k-" + t));
  $$(".k-tab").forEach(b => b.classList.toggle("on", b.dataset.t === t));
  if (t === "hatlar") ciz_hatlar();
  if (t === "calistir") ciz_calistir();
  if (t === "depo") ciz_depo();
  if (t === "soy") ciz_soy();
  window.scrollTo({ top: 0 });
}

$$(".k-tab").forEach(b => b.onclick = () => git(b.dataset.t));
$("#btnRun").onclick = calistir;
$("#btnTemizle").onclick = () => { if (!S.calisiyor) { S.durum = {}; S.adim = null; ciz_calistir(); } };

(async () => {
  await yenileSayac();
  try { S.hatlar = (await getJSON("/api/pipelines")).pipelines; } catch { /* kart alanı gösterir */ }
  setInterval(yenileSayac, 20000);
})();
