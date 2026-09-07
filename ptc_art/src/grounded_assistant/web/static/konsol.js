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
  hatlar: [], hat: "a", adim: null, icTab: "genel", durum: {}, calisiyor: false,
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
    const acik = S.adim === n.n;
    const cls = d.status === "success" ? "bitti" : d.status === "running" ? "calisiyor"
              : d.status === "error" ? "hata" : "";
    const bag = i ? `<div class="baglanti ${S.durum[h.nodes[i - 1].n]?.status === "success" ? "bitti" : ""}"></div>` : "";
    const sag = d.dur ? `<span class="badge">${esc(d.dur)}</span>`
              : n.tur === "sandbox" ? `<span class="badge">pod</span>`
              : `<span class="badge">sorgu</span>`;
    const cikti = (d.artifacts || []).map(a => `<span class="badge art">${esc(a.name)}</span>`).join(" ");

    return `${bag}<div class="adim-blok ${acik ? "acik" : ""}">
      <button class="adim ${acik ? "on" : ""}" data-tur="${n.tur}" data-n="${n.n}">
        <span class="baloncuk ${cls}">${n.n}</span>
        <span>
          <span class="adim-ad">${esc(n.ad)}</span>
          <span class="adim-alt">${cikti || esc(n.aciklama)}</span>
        </span>
        ${sag}
      </button>
      <div class="adim-ic">${acik ? adimIci(n, d) : ""}</div>
    </div>`;
  }).join("");

  $$("#adimlar .adim").forEach(b => b.onclick = () => {
    const n = +b.dataset.n;
    S.adim = S.adim === n ? null : n; S.icTab = "genel"; ciz_calistir();
  });
  $$("#adimlar .ic-tab").forEach(b => b.onclick = e => {
    e.stopPropagation(); S.icTab = b.dataset.ic; ciz_calistir();
  });
  $$("#adimlar .adim-ic [data-art]").forEach(b => b.onclick = e => {
    e.stopPropagation(); acArtifact(b.dataset.art);
  });

  // Çalışırken log'un dibinde kal.
  const lg = $("#adimlar .ic-log");
  if (lg && S.calisiyor) lg.scrollTop = lg.scrollHeight;
}

/* Bir adımın İÇİ — kod, kendi log'u, girdi/çıktı, pod bilgisi.
 *
 * Log'lar zaten adım başına toplanıyordu ama hiçbir yerde gösterilmiyordu:
 * yalnızca alttaki ORTAK panele akıyordu ve orada hangi satırın hangi adıma
 * ait olduğu kayboluyordu. Artık her adım kendi satırının altında açılıyor.
 */
function adimIci(n, d) {
  const tab = S.icTab || "genel";
  const sekmeler = [
    ["genel", "Genel"],
    n.tur === "sandbox" ? ["kod", "Kod"] : ["sorgu", "Sorgu"],
    ["log", `Log${(d.loglar || []).length ? " · " + d.loglar.length : ""}`],
    ["cikti", `Çıktı${(d.artifacts || []).length ? " · " + d.artifacts.length : ""}`],
  ];

  let govde = "";

  if (tab === "genel") {
    govde = `
      <p class="muted" style="margin:0 0 .8rem">${esc(n.aciklama)}</p>
      <div class="not" style="margin-bottom:.8rem">${n.tur === "sandbox"
        ? `<b>Gerçek pod.</b> Bu adım için ayrı bir Kubernetes Job açılıyor. Kapsam jetonu
           sidecar'da; sandbox container'ında S3 anahtarı yok ve ağı kapalı — baytları
           içeri/dışarı sidecar taşıyor.`
        : `<b>Pod açılmıyor.</b> Bu adım kayıt defterine bir HTTP sorgusu. Keşif sandbox'ta
           değil host tarafında olur; sandbox'ın listeleme yolu hiç yok.`}</div>
      <dl class="kv">
        <dt>Beyan edilen girdi</dt><dd>${n.inputs.length ? esc(JSON.stringify(n.inputs)) : "[] — yok"}</dd>
        <dt>Beklenen çıktı</dt><dd>${n.bekleniyor.length ? esc(n.bekleniyor.join(", ")) : "— üretmiyor"}</dd>
        <dt>Durum</dt><dd class="duz">${
          d.status === "success" ? "tamamlandı · " + (d.dur || "")
          : d.status === "running" ? "çalışıyor…"
          : d.status === "error" ? "hata" : "beklemede"}</dd>
        ${d.run_id ? `<dt>Pod</dt><dd>ptc-sandbox-${esc(d.run_id)}</dd>` : ""}
      </dl>
      ${d.sonuc ? `<div style="margin-top:.8rem">
        <div class="muted" style="margin-bottom:.3rem">Adımın döndürdüğü</div>
        <pre class="kod">${esc(typeof d.sonuc === "string" ? d.sonuc : JSON.stringify(d.sonuc, null, 2))}</pre>
      </div>` : ""}`;
  }

  else if (tab === "kod") {
    const kod = d.kod || n.kod || "";
    govde = kod
      ? `<pre class="kod">${esc(kod.trim())}</pre>
         <p class="muted" style="margin:.6rem 0 0">Düz Python. Artifact API'si yok —
         <span class="mono">/output</span>'a dosya yazmak yeterli.
         ${d.kod && d.kod !== n.kod
            ? " Bu, <b>gerçekten çalıştırılan</b> hâli: çapraz workflow kimliği yerine konmuş."
            : ""}</p>`
      : `<p class="bos">Bu adım kod çalıştırmıyor.</p>`;
  }

  else if (tab === "sorgu") {
    govde = `<pre class="kod">GET /artifacts?${esc(new URLSearchParams(n.sorgu || {}).toString())}</pre>
      <p class="muted" style="margin:.6rem 0 0">Kayıt defterine ada göre sorgu. Cevaptaki
      <span class="mono">workflow_id</span> bir sonraki adıma veriliyor — kimlik hiçbir yere gömülü değil.</p>
      ${d.sonuc && typeof d.sonuc === "object" ? `<div style="margin-top:.7rem">
        <div class="muted" style="margin-bottom:.3rem">Cevap</div>
        <pre class="kod">${esc(JSON.stringify(d.sonuc, null, 2))}</pre></div>` : ""}`;
  }

  else if (tab === "log") {
    const L = d.loglar || [];
    govde = L.length
      ? `<div class="ic-log">${L.map(l =>
          `<div><span class="t">${esc(l.ts)}</span><span class="m ${esc(l.cls || "")}">${esc(l.msg)}</span></div>`).join("")}
         ${d.status === "running" ? `<div><span class="t">—</span><span class="imlec">▌</span></div>` : ""}</div>
         <p class="muted" style="margin:.6rem 0 0">Bu satırlar bu adımın pod'undan geliyor.
         Alttaki panelde bütün adımlar iç içe akıyor; burada yalnızca bu adım var.</p>`
      : `<p class="bos">Bu adım henüz çalışmadı.</p>`;
  }

  else {
    const A = d.artifacts || [];
    govde = A.length
      ? A.map(a => `<button class="plaka" data-art="${esc(a.artifact_id)}" style="width:100%;margin-bottom:.4rem">
          <span class="ad">${esc(a.name)}</span>
          <span class="alt">${esc(a.artifact_id)} · ${kb(a.size_bytes)} · depoda aç →</span></button>`).join("")
        + `<p class="muted" style="margin:.4rem 0 0">Bunlar pod ölmeden önce sidecar tarafından
           süpürüldü; kod hiçbir yükleme çağrısı yapmadı.</p>`
      : `<p class="bos">${n.bekleniyor.length
          ? "Henüz üretilmedi. Beklenen: " + esc(n.bekleniyor.join(", "))
          : "Bu adım artifact üretmiyor — sonucu bellekte kalıyor ve depoya girmiyor."}</p>`;
  }

  return `<div class="ic-tabs">${sekmeler.map(([k, l]) =>
      `<button class="ic-tab ${tab === k ? "on" : ""}" data-ic="${k}">${l}</button>`).join("")}
    </div>${govde}`;
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
      d(m.n).status = "running"; S.adim = m.n; S.icTab = "log";
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
  const merkez = dugum.find(n => n.artifact_id === g.root) || dugum[0];

  // ZİNCİRİN TAMAMI çiziliyor. Servis her düğümün `depth`ini veriyor
  // (negatif = ata, 0 = bu artifact, pozitif = ürün); önceden yalnızca ±1
  // komşular çizildiği için 5 düğümlük bir zincirin 3'ü kayboluyordu — ve
  // kaybolanlar tam da workflow sınırını geçenlerdi.
  const seviyeler = [...new Set(dugum.map(n => n.depth))].sort((a, b) => a - b);
  const W = 900, satirH = 108, ustBosluk = 54;
  const yuk = ustBosluk + seviyeler.length * satirH + 26;
  const yer = {};
  seviyeler.forEach((d, si) => {
    const satir = dugum.filter(n => n.depth === d);
    satir.forEach((n, i) => (yer[n.artifact_id] = {
      x: W / (satir.length + 1) * (i + 1) + 60,
      y: ustBosluk + si * satirH,
    }));
  });

  const wfRenk = {};
  [...new Set(dugum.map(n => n.workflow_id))].forEach((w, i) => (wfRenk[w] = i));

  const satirEtiket = seviyeler.map((d, si) => {
    const y = ustBosluk + si * satirH;
    const ad = d === 0 ? "BU ARTIFACT" : d < 0 ? `GİRDİ · ${-d} adım geride` : `TÜREV · ${d} adım ileride`;
    return `<text x="8" y="${y - 26}" class="bant">${ad}</text>`;
  }).join("");

  const kutular = dugum.map(n => {
    const { x, y } = yer[n.artifact_id];
    const capraz = merkez && n.workflow_id !== merkez.workflow_id;
    return `<g data-art="${esc(n.artifact_id)}" style="cursor:pointer">
      <rect x="${x - 105}" y="${y - 21}" width="210" height="42" rx="9"
            class="kutu ${n.depth === 0 ? "merkez" : ""}"/>
      <text x="${x}" y="${y - 3}" text-anchor="middle" class="ad">${esc(n.name)}</text>
      <text x="${x}" y="${y + 12}" text-anchor="middle" class="alt">${kb(n.size_bytes)} · ${esc(kisa(n.workflow_id))}${capraz ? " ⟂" : ""}</text>
    </g>`;
  }).join("");

  const cizgi = kenar.filter(e => yer[e.from] && yer[e.to]).map(e => {
    const a = yer[e.from], b = yer[e.to];
    const na = dugum.find(n => n.artifact_id === e.from), nb = dugum.find(n => n.artifact_id === e.to);
    const capraz = na && nb && na.workflow_id !== nb.workflow_id;
    return `<path class="kenar ${capraz ? "capraz" : ""}"
      d="M${a.x} ${a.y + 21} C${a.x} ${a.y + 60} ${b.x} ${b.y - 60} ${b.x} ${b.y - 21}"/>`;
  }).join("");

  const wfSayi = new Set(dugum.map(n => n.workflow_id)).size;

  alan.innerHTML = `
    <div class="card">
      <div class="card-h">
        <h2 class="mono" style="font-size:.92rem">${esc(merkez?.name || S.soyId)}</h2>
        <span class="badge">${dugum.length} düğüm · ${kenar.length} kenar</span>
        ${wfSayi > 1 ? `<span class="badge live">${wfSayi} farklı çalıştırma</span>` : ""}
      </div>

      <div class="not" style="margin-bottom:.9rem">
        <b>Nasıl okunur.</b> Yukarıdan aşağı veri akıyor: en üstteki kutu en eski girdi,
        ortadaki (mavi çerçeveli) seçtiğiniz artifact, altındakiler ondan türeyenler.
        Her kutunun altında boyutu ve <b>hangi çalıştırmadan geldiği</b> yazıyor.
        ${wfSayi > 1 ? `<b>Kesikli mavi çizgi</b>, bağın iki farklı çalıştırma arasında
        kurulduğunu gösterir — <span class="mono">⟂</span> işaretli kutular seçtiğinizden
        başka bir hatta üretilmiş.` : `Bu zincirin tamamı tek bir çalıştırmada üretilmiş.`}
      </div>

      <div style="overflow-x:auto">
        <svg class="soy-svg" viewBox="0 0 ${W + 120} ${yuk}" style="min-width:36rem"
             role="img" aria-label="Soy ağacı">
          ${satirEtiket}${cizgi}${kutular}
        </svg>
      </div>

      <p class="muted" style="margin:.7rem 0 0">
        Kenarlar kayıt defterindeki <span class="mono">parents</span> alanından geliyor;
        o alanı da her adımın <span class="mono">inputs=[…]</span> beyanı dolduruyor.
        Bir kutuya basınca grafiğin merkezi oraya kayar.
      </p>
    </div>

    <div class="card">
      <div class="card-h"><h2>Başka bir artifact'in soyuna bak</h2></div>
      <div class="suzgec">${S.kayitlar.filter(a => a.parents?.length).slice(0, 16)
        .map(a => `<button data-soy="${esc(a.artifact_id)}" class="${a.artifact_id === S.soyId ? "on" : ""}">
          ${esc(a.name)} <span class="mono" style="opacity:.55">${esc(kisa(a.workflow_id))}</span></button>`).join("")}</div>
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
