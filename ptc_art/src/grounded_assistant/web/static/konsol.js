/* Konsol — dört sekme, hepsi canlı (2026-09-07).
 *
 * Burada sahte veri YOK. Depo `/api/depo`'dan, soy `/api/depo/<id>/soy`'dan,
 * çalıştırma `/ws/pipeline` üzerinden gerçek Kubernetes Job'larından geliyor.
 * Bir şey görünmüyorsa gerçekten yok demektir; hata metinleri de gizlenmiyor.
 */

const $  = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const esc = s => String(s ?? "").replace(/[&<>"]/g, c => ({ "&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;" }[c]));
const ico = id => `<svg><use href="#${id}"/></svg>`;
const kb = b => !b ? "0 B" : b >= 1e6 ? (b/1048576).toFixed(2) + " MB" : b >= 1e3 ? (b/1024).toFixed(0) + " KB" : b + " B";
const kisa = s => String(s || "").slice(0, 8);

const S = {
  view: "hatlar",
  hatlar: [],
  hat: null,               // seçili hat (a/b)
  nodeSec: null,           // seçili node numarası
  tab: "genel",
  durum: {},               // node no -> {status, dur, sonuc, run_id, loglar[], kod, artifacts[]}
  calisiyor: false,
  wfSon: {},               // hat key -> son workflow_id
  kaynakWf: null,          // B'nin çözdüğü A çalıştırması
  kayitlar: [],
  filtre: "hepsi",
  acik: {},
  art: null,
  soyId: null,
  soyVeri: null,
};

/* ══ ortak ═══════════════════════════════════════════════════════ */

async function getJSON(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

function hata(mesaj) {
  return `<div class="callout" style="border-color:rgba(224,100,95,.4);background:rgba(224,100,95,.07)">
    <b style="color:var(--fail)">Ulaşılamadı.</b> ${esc(mesaj)}</div>`;
}

/* ══ dock ════════════════════════════════════════════════════════ */

async function yenileDock() {
  try {
    const d = await getJSON("/api/depo?limit=1000");
    if (d.error) throw new Error(d.error);
    S.kayitlar = d.kayitlar;
    $("#dockCount").textContent = d.kayitlar.length;
    $("#navArt").textContent = d.kayitlar.length;
    $("#statBytes").textContent = kb(d.kayitlar.reduce((s, a) => s + (a.size_bytes || 0), 0));
    $("#statWf").textContent = new Set(d.kayitlar.map(a => a.workflow_id)).size;
    $("#statLive").textContent = "•"; $("#statLive").style.color = "var(--ok)";
    $("#statLiveL").textContent = "servis açık";
  } catch (e) {
    $("#dockCount").textContent = "—";
    $("#statLive").textContent = "•"; $("#statLive").style.color = "var(--fail)";
    $("#statLiveL").textContent = "servis kapalı";
  }
}

function pulseDock() {
  const d = $("#dockRepo");
  d.classList.add("pulse");
  setTimeout(() => d.classList.remove("pulse"), 900);
}

function ucur(nodeNo) {
  const src = $(`#flowline .step[data-n="${nodeNo}"] .orb`), dst = $("#dockRepo");
  if (!src || !dst || matchMedia("(prefers-reduced-motion: reduce)").matches) return pulseDock();
  const a = src.getBoundingClientRect(), b = dst.getBoundingClientRect();
  const f = document.createElement("div");
  f.className = "flyer"; f.innerHTML = ico("i-pkg");
  f.style.left = (a.left + a.width/2 - 15) + "px";
  f.style.top = (a.top + a.height/2 - 15) + "px";
  document.body.appendChild(f);
  requestAnimationFrame(() => {
    f.style.transform = `translate(${b.left + 24 - (a.left + a.width/2)}px, ${b.top + b.height/2 - (a.top + a.height/2)}px) scale(.55)`;
    f.style.opacity = "0.15";
  });
  setTimeout(() => { f.remove(); pulseDock(); }, 950);
}

/* ══ 1 · HATLAR ══════════════════════════════════════════════════ */

async function ciz_hatlar() {
  const box = $("#hatGrid");
  if (!S.hatlar.length) {
    try { S.hatlar = (await getJSON("/api/pipelines")).pipelines; }
    catch (e) { box.innerHTML = hata(e.message); return; }
  }
  box.innerHTML = S.hatlar.map(h => {
    const wf = S.wfSon[h.key];
    const uretilen = S.kayitlar.filter(a => a.workflow_id === wf);
    const strip = h.nodes.map((n, i) =>
      (i ? `<span class="strip-l"></span>` : "") +
      `<span class="strip-n ${n.tur === "sandbox" ? "ptc" : ""}">${n.n}</span>`).join("");
    const cikti = h.nodes.flatMap(n => n.bekleniyor).map(a =>
      `<span class="chip">${ico("i-pkg")}${esc(a)}</span>`).join("");
    const podlar = h.nodes.filter(n => n.tur === "sandbox").length;

    return `<button class="wf-card" data-hat="${h.key}">
      <div class="wf-top">
        <span class="wf-code">${esc(h.kod)}</span>
        <div style="flex:1;min-width:0"><h2>${esc(h.ad)}</h2><p>${esc(h.aciklama)}</p></div>
        <span class="pill ${wf ? "ok" : "idle"}"><i></i><span>${wf ? "çalıştı" : "hiç çalışmadı"}</span></span>
      </div>
      <div class="strip">${strip}</div>
      <div class="wf-meta">
        <div><b>${h.nodes.length}</b><span>adım</span></div>
        <div><b>${podlar}</b><span>sandbox pod'u</span></div>
        <div><b>${uretilen.length}</b><span>üretilen artifact</span></div>
        <div><b class="mono">${wf ? kisa(wf) : "—"}</b><span>son çalıştırma</span></div>
      </div>
      ${cikti ? `<div><div class="eyebrow" style="margin-bottom:6px">Beklenen çıktılar</div><div class="wf-arts">${cikti}</div></div>` : ""}
    </button>`;
  }).join("");

  $$("#hatGrid .wf-card").forEach(c => c.onclick = () => {
    S.hat = c.dataset.hat; S.nodeSec = null; S.durum = {}; git("calistir");
  });
}

/* ══ 2 · ÇALIŞTIRMA ══════════════════════════════════════════════ */

function hatOf(key) { return S.hatlar.find(h => h.key === key); }

function ciz_calistir() {
  if (!S.hat) S.hat = "a";
  const h = hatOf(S.hat);
  if (!h) { $("#flowline").innerHTML = hata("Hat tanımı yüklenemedi"); return; }

  $("#runCrumb").textContent = h.kod;
  $("#runTitle").textContent = h.ad;
  $("#runDesc").textContent = h.aciklama;

  const bitti = h.nodes.every(n => S.durum[n.n]?.status === "success");
  const p = $("#runPill");
  p.className = "pill " + (S.calisiyor ? "run" : bitti ? "ok" : "idle");
  p.querySelector("span").textContent = S.calisiyor ? "çalışıyor" : bitti ? "tamamlandı" : "hazır";
  $("#btnRun").disabled = S.calisiyor;

  $("#flowline").innerHTML = h.nodes.map((n, i) => {
    const d = S.durum[n.n] || {};
    const st = d.status === "success" ? "done" : d.status === "running" ? "running"
             : d.status === "error" ? "fail" : "pending";
    const sel = S.nodeSec === n.n ? " sel" : "";
    const dim = (S.nodeSec && S.nodeSec !== n.n && !S.calisiyor) ? " dim" : "";
    const kon = i ? `<div class="conduit ${S.durum[h.nodes[i-1].n]?.status === "success" ? "done" : ""} ${d.status === "running" ? "live" : ""}"></div><div></div>` : "";

    const etiket = [
      n.inputs.length ? `<span class="tag in">girdi · ${n.inputs.map(esc).join(", ")}</span>`
                      : `<span class="tag">girdi yok</span>`,
      n.tur === "sandbox" ? `<span class="tag ptc">sandbox pod'u</span>`
                          : `<span class="tag">kayıt defteri sorgusu · pod yok</span>`,
      ...(d.artifacts || []).map(a => `<span class="tag out" data-art="${a.artifact_id}">çıktı · ${esc(a.name)}</span>`),
      (!d.artifacts?.length && n.bekleniyor.length)
        ? `<span class="tag">bekleniyor · ${n.bekleniyor.map(esc).join(", ")}</span>` : "",
    ].join("");

    return `${kon}
      <div class="step${dim}" data-kind="${n.tur === "sandbox" ? "ptc" : "io"}" data-n="${n.n}">
        <div class="orb-wrap">
          <div class="orb ${st}${sel}" role="button" tabindex="0" aria-label="${esc(n.ad)}">
            ${ico(n.ikon)}<span class="orb-n">${n.n}</span>
          </div>
        </div>
        <button class="step-body${sel}">
          <div class="step-name">${esc(n.ad)}${d.dur ? ` <span class="tag">${esc(d.dur)}</span>` : ""}</div>
          <div class="step-desc">${esc(n.aciklama)}</div>
          <div class="step-io">${etiket}</div>
        </button>
      </div>`;
  }).join("");

  $$("#flowline .step").forEach(el => {
    const sec = e => {
      const c = e.target.closest("[data-art]");
      if (c) { e.stopPropagation(); acArtifact(c.dataset.art); return; }
      const n = +el.dataset.n;
      S.nodeSec = S.nodeSec === n ? null : n; S.tab = "genel"; ciz_calistir();
    };
    el.querySelector(".orb").onclick = sec;
    el.querySelector(".orb").onkeydown = e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); sec(e); } };
    el.querySelector(".step-body").onclick = sec;
  });
  ciz_nodeInsp();
}

function ciz_nodeInsp() {
  const box = $("#nodeInsp");
  const h = hatOf(S.hat);
  if (!S.nodeSec || !h) {
    box.innerHTML = `<div class="insp-empty">${ico("i-cpu")}
      <div style="font-family:var(--f-display);font-weight:700;color:var(--ink-2);margin-bottom:4px">Adım seçilmedi</div>
      <div style="font-size:12.5px">Grafikte bir daireye bas: o adımın kodunu, gerçek pod log'unu ve ürettiği artifact'i gösterir.</div></div>`;
    return;
  }
  const n = h.nodes.find(x => x.n === S.nodeSec), d = S.durum[n.n] || {};
  const tabs = [["genel","genel"],["girdi","girdi"],["kod","kod"],["log","log"],["cikti","çıktı"]];

  box.innerHTML = `<div class="insp-h">
      <div class="insp-title"><h3>${esc(n.ad)}</h3>
        ${n.tur === "sandbox" ? `<span class="tag ptc">sandbox</span>` : `<span class="tag">sorgu</span>`}</div>
      <div class="insp-sub mono">${esc(h.kod)} · adım ${n.n}/${h.nodes.length} · ${
        d.status === "success" ? "tamamlandı " + (d.dur || "")
        : d.status === "running" ? "çalışıyor…"
        : d.status === "error" ? "hata" : "beklemede"}${d.run_id ? " · pod " + esc(d.run_id) : ""}</div>
      <div class="tabs">${tabs.map(([k, l]) => `<button class="tab ${S.tab === k ? "on" : ""}" data-tab="${k}">${l}</button>`).join("")}</div>
    </div>
    <div class="insp-body">${nodeGovde(n, d)}</div>`;

  $$(".tab", box).forEach(b => b.onclick = () => { S.tab = b.dataset.tab; ciz_nodeInsp(); });
  $$("[data-art]", box).forEach(b => b.onclick = () => acArtifact(b.dataset.art));
  const lg = $(".logs", box); if (lg && S.calisiyor) lg.scrollTop = lg.scrollHeight;
}

function nodeGovde(n, d) {
  if (S.tab === "genel") {
    return `<div class="sec"><p class="sec-l">Bu adım ne yapıyor</p>
        <p class="prose">${esc(n.aciklama)}</p></div>
      <div class="sec"><p class="sec-l">Nerede çalışıyor</p>
        <div class="callout">${n.tur === "sandbox"
          ? `<b>Gerçek sandbox pod'u.</b> Ayrı bir Kubernetes Job açılıyor: kapsam jetonu sidecar'da, sandbox'ta S3 anahtarı yok, ağı kapalı. Çıktılar pod ölmeden önce sidecar tarafından süpürülüyor.`
          : `<b>Pod açılmıyor.</b> Bu adım kayıt defterine bir HTTP sorgusu — keşif sandbox'ta değil, host tarafında olur. Sandbox'ın listeleme yolu hiç yok.`}</div></div>
      ${d.sonuc ? `<div class="sec"><p class="sec-l">Sonuç</p><pre class="code">${esc(typeof d.sonuc === "string" ? d.sonuc : JSON.stringify(d.sonuc, null, 2))}</pre></div>` : ""}`;
  }
  if (S.tab === "girdi") {
    return `<div class="sec"><p class="sec-l">Beyan edilen girdiler</p>
        ${n.inputs.length
          ? `<pre class="code">inputs = ${esc(JSON.stringify(n.inputs))}</pre>
             <p class="prose" style="margin-top:9px">Bu liste iki işi birden yapıyor: sidecar yalnızca bunları <code class="mono">/output</code>'a yerleştiriyor, ve soy ağacında bu adımın ebeveynleri bunlar oluyor (MLMD'nin <code class="mono">DECLARED_INPUT</code>'u).</p>`
          : `<p class="none">Bu adım girdi beyan etmiyor — <code class="mono">inputs=[]</code>, yani hiçbir eski çıktı yerleştirilmiyor.</p>`}
      </div>
      ${n.sorgu ? `<div class="sec"><p class="sec-l">Kayıt defteri sorgusu</p>
        <pre class="code">GET /artifacts?${esc(new URLSearchParams(n.sorgu).toString())}</pre></div>` : ""}`;
  }
  if (S.tab === "kod") {
    const kod = d.kod || n.kod;
    return `<div class="sec"><p class="sec-l">${n.tur === "sandbox" ? "Sandbox'ta çalışan kod" : "Bu adım kod çalıştırmıyor"}</p>
      ${kod ? `<pre class="code">${esc(kod.trim())}</pre>
        <p class="prose" style="margin-top:9px">Düz Python. Artifact API'si yok; <code class="mono">/output</code>'a dosya yazmak yeterli.</p>`
      : `<p class="none">Bu adım host tarafında bir HTTP sorgusu yapıyor, kod çalıştırmıyor.</p>`}</div>`;
  }
  if (S.tab === "log") {
    const L = d.loglar || [];
    return `<div class="sec"><p class="sec-l">Canlı log</p>
      ${L.length ? `<div class="logs">${L.map(l =>
        `<div><span class="t">${esc(l.ts)}</span><span class="m ${esc(l.cls || "")}">${esc(l.msg)}</span></div>`).join("")}
        ${d.status === "running" ? `<div><span class="t">—</span><span class="cursor">▌</span></div>` : ""}</div>`
      : `<p class="none">Henüz çalışmadı. <b>Hattı çalıştır</b>'a basınca gerçek pod log'u buraya akar.</p>`}</div>`;
  }
  const arts = d.artifacts || [];
  return `<div class="sec"><p class="sec-l">Üretilen artifact</p>
    ${arts.length ? arts.map(a => `<button class="art-card" data-art="${a.artifact_id}" style="margin-bottom:8px">
        <span class="art-ico">${ico("i-pkg")}</span>
        <span style="flex:1;min-width:0">
          <span class="art-name" style="display:block">${esc(a.name)}</span>
          <span class="art-sub">${esc(a.artifact_id)} · ${kb(a.size_bytes)} · depoda aç</span></span>
        <span class="art-go">${ico("i-arrow")}</span></button>`).join("")
      : `<p class="none">${n.bekleniyor.length ? "Henüz üretilmedi. Beklenen: " + n.bekleniyor.map(esc).join(", ") : "Bu adım artifact üretmiyor — sonucu bellekte kalıyor."}</p>`}
  </div>`;
}

/* ── gerçek çalıştırma ── */
let WS = null;

function calistir() {
  if (S.calisiyor) return;
  const h = hatOf(S.hat);
  S.durum = {}; S.calisiyor = true; S.nodeSec = null; ciz_calistir();

  const proto = location.protocol === "https:" ? "wss" : "ws";
  WS = new WebSocket(`${proto}://${location.host}/ws/pipeline`);
  WS.onopen = () => WS.send(JSON.stringify({ type: "run", key: S.hat, kaynak_wf: null }));
  WS.onerror = () => { S.calisiyor = false; ciz_calistir(); };
  WS.onclose = () => { if (S.calisiyor) { S.calisiyor = false; ciz_calistir(); } };
  WS.onmessage = ev => {
    const m = JSON.parse(ev.data);
    const d = n => (S.durum[n] = S.durum[n] || { loglar: [], artifacts: [] });

    if (m.type === "pipeline_start") { S.wfSon[m.key] = m.workflow_id; }
    else if (m.type === "node_start") { d(m.n).status = "running"; S.nodeSec = m.n; S.tab = "log"; ciz_calistir(); }
    else if (m.type === "log") { d(m.n).loglar.push(m); ciz_nodeInsp(); }
    else if (m.type === "code") { d(m.n).kod = m.kod; }
    else if (m.type === "artifact") {
      if (m.op === "produced") {
        d(m.n).artifacts.push({ artifact_id: m.artifact_id, name: m.name, size_bytes: m.size_bytes });
        ucur(m.n); yenileDock();
      }
      ciz_calistir();
    }
    else if (m.type === "node_done") {
      Object.assign(d(m.n), { status: m.status === "success" ? "success" : "error",
                              dur: m.dur, sonuc: m.sonuc, run_id: m.run_id });
      ciz_calistir();
    }
    else if (m.type === "pipeline_done") {
      S.calisiyor = false;
      if (m.kaynak_wf) S.kaynakWf = m.kaynak_wf;
      ciz_calistir(); ciz_hatlar(); yenileDock();
      WS && WS.close();
    }
  };
}

/* ══ 3 · DEPO ════════════════════════════════════════════════════ */

async function ciz_depo() {
  const kutu = $("#depoCekmece"), yan = $("#depoFiltre");
  let d;
  try { d = await getJSON("/api/depo?limit=1000"); }
  catch (e) { kutu.innerHTML = hata(e.message); yan.innerHTML = ""; return; }
  if (d.error) { kutu.innerHTML = hata(d.error); yan.innerHTML = ""; return; }

  S.kayitlar = d.kayitlar;
  const arts = d.kayitlar;
  const wfler = [...new Set(arts.map(a => a.workflow_id))];
  const tipler = [...new Set(arts.map(a => a.type))];

  yan.innerHTML = `
    <div class="eyebrow" style="padding:8px 11px 6px">Süzgeç</div>
    <button class="filter ${S.filtre === "hepsi" ? "on" : ""}" data-f="hepsi"><i></i>Hepsi<span class="nav-n">${arts.length}</span></button>
    <div class="eyebrow" style="padding:14px 11px 6px">Tipe göre</div>
    ${tipler.map(t => `<button class="filter ${S.filtre === "t:" + t ? "on" : ""}" data-f="t:${esc(t)}"><i></i>${esc(t.replace("system.", ""))}<span class="nav-n">${arts.filter(a => a.type === t).length}</span></button>`).join("")}
    <div class="eyebrow" style="padding:14px 11px 6px">Çalıştırmaya göre</div>
    ${wfler.slice(0, 12).map(w => `<button class="filter ${S.filtre === "w:" + w ? "on" : ""}" data-f="w:${esc(w)}"><i></i><span class="mono" style="font-size:11px">${kisa(w)}</span><span class="nav-n">${arts.filter(a => a.workflow_id === w).length}</span></button>`).join("")}`;

  let gosterilen = arts;
  if (S.filtre.startsWith("t:")) gosterilen = arts.filter(a => a.type === S.filtre.slice(2));
  if (S.filtre.startsWith("w:")) gosterilen = arts.filter(a => a.workflow_id === S.filtre.slice(2));

  const grup = {};
  gosterilen.forEach(a => (grup[a.workflow_id] = grup[a.workflow_id] || []).push(a));
  const sirali = Object.entries(grup).sort((x, y) => y[1].length - x[1].length);

  kutu.innerHTML = sirali.length ? sirali.map(([wf, list]) => {
    const acik = S.acik[wf] !== false;
    const bayt = list.reduce((s, a) => s + (a.size_bytes || 0), 0);
    const benim = Object.values(S.wfSon).includes(wf);
    return `<div class="drawer ${acik ? "open" : ""}" data-wf="${esc(wf)}">
      <button class="drawer-face">
        <span class="drawer-pull"></span>
        <span class="drawer-t">
          <b>${benim ? "Bu oturumun çalıştırması" : "Çalıştırma"} <span class="mono" style="font-weight:400;font-size:12px;color:var(--ink-3)">${esc(wf)}</span></b>
          <span class="mono">${list.length} artifact · ${kb(bayt)}</span>
        </span>
        ${benim ? `<span class="badge shared">yeni</span>` : ""}
        <span class="drawer-caret">${ico("i-caret")}</span>
      </button>
      <div class="shelf">${list.map(plaka).join("")}</div>
    </div>`;
  }).join("") : `<p class="none">Bu süzgeçle eşleşen artifact yok.</p>`;

  $$("#depoFiltre .filter").forEach(b => b.onclick = () => { S.filtre = b.dataset.f; ciz_depo(); });
  $$("#depoCekmece .drawer-face").forEach(b => b.onclick = () => {
    const wf = b.closest(".drawer").dataset.wf;
    S.acik[wf] = S.acik[wf] === false; ciz_depo();
  });
  $$("#depoCekmece .slab").forEach(b => b.onclick = () => { S.art = b.dataset.art; ciz_depo(); ciz_artInsp(); });
  ciz_artInsp();
}

function plaka(a) {
  return `<button class="slab ${S.art === a.artifact_id ? "on" : ""}" data-art="${esc(a.artifact_id)}">
    <span class="slab-top"><span class="art-ico">${ico("i-pkg")}</span>
      <span style="min-width:0"><span class="slab-name" style="display:block">${esc(a.name)}</span>
      <span class="slab-id">${esc(a.artifact_id)}</span></span></span>
    <span class="slab-foot">
      <span class="badge">${esc((a.type || "").replace("system.", ""))}</span>
      <span>${kb(a.size_bytes)}</span>
      ${a.parents?.length ? `<span class="dot"></span><span>${a.parents.length} ebeveyn</span>` : ""}
      ${a.alias ? `<span class="badge shared" style="margin-left:auto">@${esc(a.alias)}</span>` : ""}
    </span></button>`;
}

async function ciz_artInsp() {
  const box = $("#artInsp");
  const a = S.kayitlar.find(x => x.artifact_id === S.art);
  if (!a) {
    box.innerHTML = `<div class="insp-empty">${ico("i-cabinet")}
      <div style="font-family:var(--f-display);font-weight:700;color:var(--ink-2);margin-bottom:4px">Hiçbiri açık değil</div>
      <div style="font-size:12.5px">Bir çekmece açıp artifact seç: künyesi, soyu ve önizlemesi gelsin.</div></div>`;
    return;
  }
  box.innerHTML = `<div class="insp-h">
      <div class="insp-title"><h3 style="font-family:var(--f-mono);font-size:14px">${esc(a.name)}</h3></div>
      <div class="insp-sub mono">${esc(a.artifact_id)} · ${kb(a.size_bytes)}</div>
      <div class="tabs"><span class="tab on">künye</span></div>
    </div>
    <div class="insp-body">
      <div class="sec"><p class="sec-l">Kayıt defteri</p>
        <dl class="kv">
          <dt>Artifact ID</dt><dd>${esc(a.artifact_id)}</dd>
          <dt>Tip</dt><dd class="plain">${esc(a.type)}</dd>
          <dt>MIME</dt><dd>${esc(a.content_type)}</dd>
          <dt>Boyut</dt><dd>${kb(a.size_bytes)}</dd>
          <dt>Üreten çalıştırma</dt><dd>${esc(a.workflow_id)}</dd>
          <dt>Node</dt><dd>${esc(a.node_id || "—")}</dd>
          <dt>Oluşturma</dt><dd>${esc((a.created_at || "").replace("T", " ").slice(0, 19))}</dd>
          <dt>İçerik özeti</dt><dd>${esc((a.content_hash || "").slice(0, 26))}…</dd>
          <dt>Alias</dt><dd>${a.alias ? "@" + esc(a.alias) : "—"}</dd>
        </dl></div>

      <div class="sec"><p class="sec-l">Beyan edilen ebeveynler</p>
        ${a.parents?.length ? `<div class="rel-list">${a.parents.map(p => {
          const pa = S.kayitlar.find(x => x.artifact_id === p);
          return `<button class="rel" data-art="${esc(p)}">${ico("i-link")}
            <span class="n">${esc(pa ? pa.name : p)}</span>
            <span class="w">${pa && pa.workflow_id !== a.workflow_id ? "çapraz workflow" : "aynı çalıştırma"}</span></button>`;
        }).join("")}</div>` : `<p class="none">Kök artifact — girdisi yok.</p>`}
      </div>

      <div class="sec"><p class="sec-l">Sürümü sabitle</p>
        <div style="display:flex;gap:8px;align-items:center">
          <input id="aliasIn" class="mono" value="${esc(a.alias || "")}" placeholder="ör. onaylanmis"
            style="flex:1;background:#0A0E13;border:1px solid var(--line);border-radius:var(--r-sm);color:var(--ink);padding:7px 10px;font-size:12px" />
          <button class="btn ghost" id="aliasBtn">${ico("i-tag")}Ata</button>
        </div>
        <p class="prose" style="margin-top:8px;font-size:12px">MLflow'un <code class="mono">models:/&lt;ad&gt;@&lt;alias&gt;</code>'ı. Aynı ad birçok çalıştırmada varken "en yeni kazanır" kuralından kaçırır. Sandbox atayamaz — bu yol insanın.</p>
        <div id="aliasSonuc" style="margin-top:8px"></div>
      </div>

      <div class="sec"><p class="sec-l">Soy ağacı</p>
        <button class="rel" id="soyBtn">${ico("i-share")}<span class="n">Grafiği aç</span><span class="w">canlı</span></button>
      </div>

      <div class="sec"><p class="sec-l">Önizleme</p>
        <div id="onizleme"><p class="none">Yükleniyor…</p></div>
      </div>
    </div>`;

  $$("[data-art]", box).forEach(b => b.onclick = () => { S.art = b.dataset.art; ciz_depo(); });
  $("#soyBtn", box).onclick = () => { S.soyId = a.artifact_id; git("soy"); };
  $("#aliasBtn", box).onclick = async () => {
    const v = $("#aliasIn").value.trim();
    const out = $("#aliasSonuc");
    out.innerHTML = `<p class="none">Gönderiliyor…</p>`;
    try {
      const r = await getJSON(`/api/depo/${a.artifact_id}/alias${v ? "?alias=" + encodeURIComponent(v) : ""}`, { method: "PUT" });
      out.innerHTML = r.error
        ? hata(r.error)
        : `<div class="callout" style="border-color:var(--flow-dim);background:rgba(52,211,198,.07)"><b style="color:var(--flow)">Atandı.</b> Artık <code class="mono">${esc(a.name)}@${esc(r.alias || v)}</code> ile çözülüyor.</div>`;
      if (!r.error) { await ciz_depo(); }
    } catch (e) { out.innerHTML = hata(e.message); }
  };

  // içerik önizlemesi — mevcut panelin ucu, aynen kullanılıyor
  try {
    const o = await getJSON(`/api/artifact/${a.artifact_id}?session=${encodeURIComponent(a.workflow_id)}`);
    const el = $("#onizleme");
    if (!el) return;
    // `durum.py` iki ikili biçim döndürüyor: `gorsel` (data: URI) ve `pdf`
    // (ilk sayfa data: URI + sayfa sayısı). Metin/tablo yolu ayrı.
    const cerceve = "width:100%;border-radius:var(--r);border:1px solid var(--line);background:#fff";
    if (o.hata) el.innerHTML = `<p class="none">${esc(o.hata)}</p>`;
    else if (o.bilgi) el.innerHTML = `<p class="none">${esc(o.bilgi)}</p>`;
    else if (o.gorsel) el.innerHTML = `<img src="${o.gorsel}" style="${cerceve}" alt="${esc(a.name)}" />`;
    else if (o.pdf) el.innerHTML = `<embed src="${o.pdf}" type="application/pdf" style="${cerceve};height:420px" />
        <p class="none" style="margin-top:6px">PDF${o.sayfa ? " · " + o.sayfa + " sayfa" : ""} · ${kb(o.bayt)}</p>`;
    else el.innerHTML = `<pre class="code">${esc(JSON.stringify(o.tablo ?? o.metin ?? o, null, 2)).slice(0, 1400)}</pre>`;
  } catch (e) { const el = $("#onizleme"); if (el) el.innerHTML = `<p class="none">Önizleme alınamadı.</p>`; }
}

async function acArtifact(id) { S.art = id; S.filtre = "hepsi"; git("depo"); }

/* ══ 4 · SOY ═════════════════════════════════════════════════════ */

async function ciz_soy() {
  const alan = $("#soyAlan");
  if (!S.soyId) {
    const aday = S.kayitlar.filter(a => a.parents?.length);
    if (!aday.length) {
      alan.innerHTML = `<div class="canvas"><p class="none">Henüz soyu olan bir artifact yok. Bir hat çalıştırın; beyan edilen girdiler soyu oluşturur.</p></div>`;
      return;
    }
    S.soyId = aday[0].artifact_id;
  }
  alan.innerHTML = `<div class="canvas"><p class="none">Yükleniyor…</p></div>`;
  let g;
  try { g = await getJSON(`/api/depo/${S.soyId}/soy`); }
  catch (e) { alan.innerHTML = `<div class="canvas">${hata(e.message)}</div>`; return; }
  if (g.error || g.hata) { alan.innerHTML = `<div class="canvas">${hata(g.error || g.hata)}</div>`; return; }

  const dugumler = g.nodes || [], kenarlar = g.edges || [];
  const merkez = dugumler.find(n => n.artifact_id === S.soyId) || dugumler[0];

  // katmanlara ayır: ata / merkez / ürün
  const atalar = new Set(), urunler = new Set();
  kenarlar.forEach(e => { if (e.to === S.soyId) atalar.add(e.from); if (e.from === S.soyId) urunler.add(e.to); });
  const kat = [
    dugumler.filter(n => atalar.has(n.artifact_id)),
    dugumler.filter(n => n.artifact_id === S.soyId),
    dugumler.filter(n => urunler.has(n.artifact_id)),
  ];
  const genis = 1000, satirY = [90, 240, 390];
  const kutu = (n, x, y, vurgu) => `<g data-art="${esc(n.artifact_id)}" style="cursor:pointer">
    <rect x="${x-100}" y="${y-22}" width="200" height="44" rx="8"
      fill="${vurgu ? "rgba(224,162,75,.2)" : "rgba(224,162,75,.08)"}"
      stroke="${vurgu ? "#E0A24B" : "#7A5824"}" stroke-width="${vurgu ? 1.8 : 1.2}"/>
    <text x="${x}" y="${y-3}" text-anchor="middle" class="lbl" style="fill:var(--brass)">${esc(n.name)}</text>
    <text x="${x}" y="${y+13}" text-anchor="middle" class="lbl-s">${esc(n.artifact_id)} · ${kb(n.size_bytes)}</text>
    ${n.workflow_id && merkez && n.workflow_id !== merkez.workflow_id
      ? `<text x="${x}" y="${y-30}" text-anchor="middle" class="lbl-s" style="fill:var(--flow)">başka çalıştırma · ${esc(kisa(n.workflow_id))}</text>` : ""}
    </g>`;

  const yerler = {};
  kat.forEach((satir, si) => satir.forEach((n, i) => {
    yerler[n.artifact_id] = { x: genis / (satir.length + 1) * (i + 1), y: satirY[si] };
  }));

  const cizgiler = kenarlar.filter(e => yerler[e.from] && yerler[e.to]).map(e => {
    const a = yerler[e.from], b = yerler[e.to];
    return `<path class="edge art" d="M${a.x} ${a.y+22} C${a.x} ${a.y+70} ${b.x} ${b.y-70} ${b.x} ${b.y-22}" marker-end="url(#arb)"/>`;
  }).join("");

  alan.innerHTML = `
    <div class="canvas" style="margin-bottom:16px">
      <div class="canvas-h"><h2>${esc(merkez?.name || S.soyId)}</h2>
        <span class="pill ok"><i></i><span>${dugumler.length} düğüm · ${kenarlar.length} kenar</span></span></div>
      <div class="map-wrap" style="border:0;background:none;padding:0">
        <svg class="map" viewBox="0 0 ${genis} 470" role="img" aria-label="Soy ağacı">
          <defs><marker id="arb" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
            <path d="M0 0 10 5 0 10z" fill="#7A5824"/></marker></defs>
          <text x="24" y="34" class="lbl-h" style="fill:var(--ink-3)">ATALAR — beyan edilen girdiler</text>
          <text x="24" y="188" class="lbl-h" style="fill:var(--brass)">BU ARTIFACT</text>
          <text x="24" y="338" class="lbl-h" style="fill:var(--ink-3)">ÜRÜNLER — bunu girdi olarak alanlar</text>
          ${cizgiler}
          ${kat.flatMap((satir, si) => satir.map(n => kutu(n, yerler[n.artifact_id].x, yerler[n.artifact_id].y, si === 1))).join("")}
          ${!kat[0].length ? `<text x="${genis/2}" y="90" text-anchor="middle" class="lbl-s">kök — girdisi yok</text>` : ""}
          ${!kat[2].length ? `<text x="${genis/2}" y="390" text-anchor="middle" class="lbl-s">henüz kimse tüketmedi</text>` : ""}
        </svg>
      </div>
    </div>
    <div class="wf-grid">
      ${S.kayitlar.filter(a => a.parents?.length).slice(0, 8).map(a =>
        `<button class="wf-card" data-soy="${esc(a.artifact_id)}" style="padding:14px 16px">
          <div class="wf-top"><span class="wf-code">${esc((a.type||"").replace("system.",""))}</span>
            <div style="flex:1;min-width:0"><h2 style="font-family:var(--f-mono);font-size:13px">${esc(a.name)}</h2>
            <p style="font-size:12px">${a.parents.length} beyan edilen girdi · ${kb(a.size_bytes)}</p></div></div>
        </button>`).join("")}
    </div>`;

  $$("#soyAlan [data-art]").forEach(g2 => g2.onclick = () => { S.soyId = g2.dataset.art; ciz_soy(); });
  $$("#soyAlan [data-soy]").forEach(b => b.onclick = () => { S.soyId = b.dataset.soy; ciz_soy(); });
}

/* ══ yönlendirme ═════════════════════════════════════════════════ */

function git(v) {
  S.view = v;
  $$(".view").forEach(x => x.classList.toggle("on", x.id === "view-" + v));
  $$(".nav").forEach(b => b.classList.toggle("on", b.dataset.view === v));
  if (v === "hatlar") ciz_hatlar();
  if (v === "calistir") ciz_calistir();
  if (v === "depo") ciz_depo();
  if (v === "soy") ciz_soy();
  const av = $("#view-" + v); if (av) av.scrollTop = 0;
}

$$(".nav").forEach(b => b.onclick = () => git(b.dataset.view));
$$("[data-view]").forEach(b => { if (!b.classList.contains("nav")) b.onclick = () => git(b.dataset.view); });
$("#dockRepo").onclick = () => git("depo");
$("#btnRun").onclick = calistir;
$("#btnTemizle").onclick = () => { if (!S.calisiyor) { S.durum = {}; S.nodeSec = null; ciz_calistir(); } };

(async () => {
  await yenileDock();
  try { S.hatlar = (await getJSON("/api/pipelines")).pipelines; } catch (e) { /* kart alanı hatayı gösterir */ }
  ciz_hatlar();
  setInterval(yenileDock, 15000);
})();
