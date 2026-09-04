// Durum paneli — /api/durum'u periyodik çeker ve çizer.
//
// Framework yok (Principle V): sohbet ekranıyla aynı ilke. Tek bir uç noktadan
// gelen JSON'u üç bölüme dağıtıyoruz; her bölüm KENDİ hatasını gösteriyor, biri
// başarısız olunca sayfa kararmıyor.

const YENILEME_MS = 5000;

// Oturum kimliği sohbet ekranıyla AYNI yerden (localStorage) okunuyor — böylece
// panel, o oturumun artifact'lerini gösteriyor.
//
// URL'deki ?session= önceliklidir: böylece belirli bir workflow'un durumu
// LİNK OLARAK paylaşılabiliyor ("şu çalıştırmaya bak"). Tarayıcı kimliği
// olmayan biri de o linki açıp görebilir.
function oturum() {
  const url = new URLSearchParams(location.search).get("session");
  if (url) return url;
  try { return localStorage.getItem("ptc_session_id"); } catch (e) { return null; }
}

const $ = (id) => document.getElementById(id);
const esc = (t) => String(t ?? "—").replace(/[<>&"]/g, c =>
  ({ "<": "&lt;", ">": "&gt;", "&": "&amp;", '"': "&quot;" }[c]));

function hata(el, mesaj) { el.innerHTML = `<p class="hata">${esc(mesaj)}</p>`; }
function bos(el, mesaj) { el.innerHTML = `<p class="bos">${esc(mesaj)}</p>`; }

// ===========================================================================
// SERVİS HARİTASI — Hubble'ın hizmet haritasının bizim yapımıza uyarlanmışı
//
// Kenarlar POLİTİKADAN çiziliyor ("neye izin var"), üzerlerinde akan
// parçacıklar HUBBLE'DAN geliyor ("ne oldu"). İkisi aynı resimde: bir paket
// izinli bir kenardan akıyorsa yeşil kayar; izinsiz bir hedefe gidiyorsa
// kırmızı ve kenar yok — kesildiği yer düğümün kenarında kalır.
// ===========================================================================

const NS = "http://www.w3.org/2000/svg";

// Sabit yerleşim: düğüm sayısı az ve değişmiyor, otomatik yerleşim
// algoritmasına gerek yok — elle konumlandırmak daha okunaklı bir resim veriyor.
const DUGUMLER = {
  "sandbox":          { x:  95, y: 215, e: "Sandbox",        alt: "LLM'in kodu",       renk: "#0071e3" },
  "tool-gateway":     { x: 370, y: 100, e: "Tool Gateway",   alt: "10 tool",           renk: "#1d1d1f" },
  "artifact-service": { x: 370, y: 320, e: "Artifact Service", alt: "artifact + kayıt", renk: "#1d1d1f" },
  "minio":            { x: 650, y: 320, e: "MinIO",          alt: "nesne deposu",      renk: "#1d1d1f" },
  "coredns":          { x: 650, y: 215, e: "coredns",        alt: "DNS",               renk: "#8e8e93" },
  "world":            { x: 660, y: 100, e: "internet",       alt: "onaylı hedefler",   renk: "#8e8e93" },
};
const KUTU_G = 150, KUTU_Y = 54;

// Aynı bileşen üç yerde üç farklı adla geçiyor:
//   Cilium politikası → `ptc-sandbox`, `kube-dns`
//   Hubble akışı      → `sandbox`, `coredns`
//   Harita düğümü     → `sandbox`, `coredns`
// Bu tablo üçünü tek isme indiriyor; olmadığında politika kenarları haritada
// hiç çizilmiyordu (kaynak adı hiçbir düğümle eşleşmiyordu).
const ESANLAM = {
  "ptc-sandbox": "sandbox",
  "kube-dns": "coredns",
  "kube-system/coredns": "coredns",
};
const norm = (ad) => ESANLAM[ad] || ad;

// Politikadan gelen kenarlar burada saklanıyor: "kaynak>hedef" -> path elemanı
const kenarYolu = {};

function merkez(ad) {
  const d = DUGUMLER[ad];
  return d ? { x: d.x + KUTU_G / 2, y: d.y + KUTU_Y / 2 } : null;
}

// İki kutu arasındaki çizgiyi kutuların KENARINDA başlatıp bitirir — yoksa ok
// kutunun içinden çıkmış gibi görünüyor.
function kenarNoktalari(a, b) {
  const A = merkez(a), B = merkez(b);
  if (!A || !B) return null;
  const dx = B.x - A.x, dy = B.y - A.y;
  const kes = (m, sx, sy) => {
    const yG = KUTU_G / 2 + 6, yY = KUTU_Y / 2 + 6;
    const t = Math.min(Math.abs(dx) > 1 ? yG / Math.abs(dx) : 9, Math.abs(dy) > 1 ? yY / Math.abs(dy) : 9);
    return { x: m.x + sx * dx * t, y: m.y + sy * dy * t };
  };
  return { a: kes(A, 1, 1), b: kes(B, -1, -1) };
}

function haritayiKur(kenarlar) {
  const gK = $("kenarlar"), gD = $("dugumler");
  if (!gK || gD.childElementCount) return;      // bir kez kur

  // --- kenarlar (politikadan) ---
  const gorulen = new Set();
  kenarlar.forEach(k => {
    const kaynak = norm(k.kaynak);
    const h = norm(k.hedef);
    const hedef = DUGUMLER[h] ? h : (k.tur === "dis" ? "world" : null);
    if (!hedef || !DUGUMLER[kaynak] || gorulen.has(`${kaynak}>${hedef}`)) return;
    gorulen.add(`${kaynak}>${hedef}`);
    const n = kenarNoktalari(kaynak, hedef);
    if (!n) return;
    const p = document.createElementNS(NS, "path");
    p.setAttribute("d", `M${n.a.x},${n.a.y} L${n.b.x},${n.b.y}`);
    p.setAttribute("class", "kenar");
    p.setAttribute("marker-end", "url(#ucu)");
    gK.appendChild(p);
    kenarYolu[`${kaynak}>${hedef}`] = p;
    kenarYolu[`${k.kaynak}>${k.hedef}`] = p;    // ham adıyla da bulunabilsin
  });

  // --- düğümler ---
  Object.entries(DUGUMLER).forEach(([ad, d]) => {
    const g = document.createElementNS(NS, "g");
    g.setAttribute("class", "dugum-g");
    g.setAttribute("data-ad", ad);
    g.innerHTML =
      `<rect x="${d.x}" y="${d.y}" width="${KUTU_G}" height="${KUTU_Y}" rx="10"
             class="dugum-kutu"/>
       <circle cx="${d.x + 14}" cy="${d.y + 18}" r="4" class="dugum-nokta" fill="#c7c7cc"/>
       <text x="${d.x + 26}" y="${d.y + 22}" class="dugum-baslik">${esc(d.e)}</text>
       <text x="${d.x + 14}" y="${d.y + 40}" class="dugum-alt2">${esc(d.alt)}</text>`;
    gD.appendChild(g);
  });
}

// Pod sağlığını haritadaki noktalara işle
function haritaSagligi(p) {
  if (p.error || !p.kalici) return;
  const durum = {};
  p.kalici.forEach(k => { durum[k.ad] = k.hazir; });
  const sandboxAktif = (p.sandboxlar || []).length > 0;
  document.querySelectorAll(".dugum-g").forEach(g => {
    const ad = g.dataset.ad, n = g.querySelector(".dugum-nokta");
    // Sandbox efemer: boşta olması NORMAL, hata değil. Kırmızı yerine gri.
    if (ad === "sandbox") {
      n.setAttribute("fill", sandboxAktif ? "#34c759" : "#c7c7cc");
      return;
    }
    if (!(ad in durum)) { n.setAttribute("fill", "#c7c7cc"); return; }
    n.setAttribute("fill", durum[ad] ? "#34c759" : "#ff3b30");
  });
}

// --- akış animasyonu -------------------------------------------------------
//
// Her akış için kenar boyunca kayan bir daire. SMIL `animateMotion`
// kullanılıyor: tarayıcı animasyonu kendisi yürütüyor, JS her karede
// çalışmıyor — panel saatlerce açık kalabildiği için bu önemli.

function akisiCiz(f) {
  const gP = $("parcaciklar");
  if (!gP) return;
  const dusuk = f.verdict === "DROPPED";
  const kaynak = norm(f.kaynak), hedef = norm(f.hedef);
  const yol = kenarYolu[`${kaynak}>${hedef}`]
           || kenarYolu[`${f.kaynak}>${f.hedef}`]
           || (!DUGUMLER[hedef] ? kenarYolu[`${kaynak}>world`] : null);

  if (!yol) {
    // İzinli kenarı olmayan bir hedef: paket kaynağın kenarında kesiliyor.
    // Görsel olarak da öyle gösteriliyor — kısa kırmızı bir çıkıntı.
    if (dusuk) kesikGoster(kaynak);
    return;
  }

  yol.classList.add(dusuk ? "kenar-red" : "kenar-aktif");
  setTimeout(() => yol.classList.remove("kenar-red", "kenar-aktif"), 900);

  const c = document.createElementNS(NS, "circle");
  c.setAttribute("r", dusuk ? 6 : 5);
  c.setAttribute("class", dusuk ? "parcacik red" : "parcacik");
  const m = document.createElementNS(NS, "animateMotion");
  m.setAttribute("dur", "1.1s");
  m.setAttribute("fill", "freeze");
  m.setAttribute("path", yol.getAttribute("d"));
  c.appendChild(m);
  gP.appendChild(c);
  setTimeout(() => c.remove(), 1200);
}

function kesikGoster(kaynak) {
  const gP = $("parcaciklar"), m = merkez(kaynak);
  if (!m) return;
  const c = document.createElementNS(NS, "circle");
  c.setAttribute("cx", m.x + KUTU_G / 2 + 16);
  c.setAttribute("cy", m.y);
  c.setAttribute("r", 7);
  c.setAttribute("class", "parcacik red kesik");
  gP.appendChild(c);
  setTimeout(() => c.remove(), 1100);
}

// --- sandbox listesi -------------------------------------------------------
function ciz_sandboxlar(p) {
  const el = $("sandboxlar");
  if (p.error) return hata(el, p.error);
  if (!p.sandboxlar.length)
    return bos(el, "Şu an çalışan sandbox yok — pod'lar saniyeler içinde doğup ölüyor.");
  el.innerHTML = `<table class="veri"><thead><tr>
      <th>Pod</th><th>run_id</th><th>Durum</th><th>Yaş</th></tr></thead><tbody>` +
    p.sandboxlar.map(s => `<tr>
      <td class="mono">${esc(s.pod)}</td>
      <td class="mono">${esc(s.run_id)}</td>
      <td>${esc(s.durum)}</td>
      <td class="sayi">${esc(s.yas)}</td></tr>`).join("") +
    `</tbody></table>`;
}

// --- izinli akışlar --------------------------------------------------------
function ciz_akislar(a) {
  const el = $("akislar");
  if (a.error) return hata(el, a.error);
  $("politika-sayisi").textContent = `${a.politika_sayisi} politika`;
  const alt = $("harita-alt");
  if (alt) alt.innerHTML = "Gri oklar <strong>izin verilmiş</strong> kenarlar (politikadan). "
    + "Üzerlerinde kayan noktalar <strong>gerçek paketler</strong> (Hubble'dan) — "
    + "<span style=\"color:var(--izin)\">yeşil geçti</span>, "
    + "<span style=\"color:var(--yasak)\">kırmızı engellendi</span>.";
  if (!a.kenarlar.length) return bos(el, "Politika bulunamadı.");
  el.innerHTML = a.kenarlar.map(k => `<div class="akis-satir">
      <span class="mono">${esc(k.kaynak)}</span>
      <span class="akis-ok">→</span>
      <span class="${k.tur === "dis" ? "dis" : "mono"}">${esc(k.hedef)}</span>
    </div>`).join("");
}

// --- artifact deposu -------------------------------------------------------
function boyut(b) {
  if (!b) return "0 B";
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

function ciz_artifactler(a) {
  const el = $("artifactler");
  const kapsam = $("artifact-kapsam");
  if (a.error) { kapsam.textContent = ""; return hata(el, a.error); }
  if (a.not) {
    kapsam.textContent = "";
    return bos(el, "Bu tarayıcıda oturum yok. Önce sohbet ekranını açın, ya da "
      + "adres satırına ?session=<uuid> ekleyin.");
  }
  kapsam.textContent = `workflow ${String(a.workflow_id).slice(0, 8)}… · ${a.kayitlar.length} artifact`;
  if (!a.kayitlar.length)
    return bos(el, "Bu oturumda henüz artifact üretilmedi.");
  el.innerHTML = `<table class="veri"><thead><tr>
      <th>Ad</th><th>Tip</th><th>Boyut</th><th>Node</th><th>Üretilme</th><th>ID</th>
    </tr></thead><tbody>` +
    a.kayitlar.map(k => `<tr class="artifact-satir" data-id="${esc(k.artifact_id)}"
        title="İçeriği görmek için tıkla">
      <td><strong>${esc(k.name)}</strong></td>
      <td><span class="tip">${esc((k.type || "").replace("system.", ""))}</span></td>
      <td class="sayi">${boyut(k.size_bytes)}</td>
      <td class="mono">${esc(k.node_id || "—")}</td>
      <td class="sayi">${esc((k.created_at || "").slice(11, 19))}</td>
      <td class="mono">${esc(k.artifact_id)}</td>
    </tr>`).join("") + `</tbody></table>`;

  // Satıra tıklayınca içeriği aç. Tablo her yenilemede yeniden çiziliyor,
  // o yüzden dinleyici tabloya (kapsayıcıya) bağlanıyor — tek tek satırlara
  // değil; yoksa 5 saniyede bir yeniden bağlamak gerekirdi.
  el.querySelectorAll(".artifact-satir").forEach(tr => {
    tr.addEventListener("click", () => onizlemeAc(tr.dataset.id, tr));
  });

  // `?onizle=<artifact_id>` — belirli bir artifact'in içeriğine DERİN BAĞLANTI.
  // "Şu çıktıya bak" demenin en kısa yolu; ayrıca ilk çizimde bir kez açılıyor,
  // sonraki 5 saniyelik yenilemelerde tekrar açılıp kullanıcıyı rahatsız etmiyor.
  const istenen = new URLSearchParams(location.search).get("onizle");
  if (istenen && !derinBaglantiAcildi) {
    const satir = el.querySelector(`.artifact-satir[data-id="${CSS.escape(istenen)}"]`);
    if (satir) { derinBaglantiAcildi = true; onizlemeAc(istenen, satir); }
  }
}

let derinBaglantiAcildi = false;

// --- artifact içeriği önizlemesi ------------------------------------------

let acikArtifact = null;   // aynı satıra ikinci tıklama kapatsın

async function onizlemeAc(id, satir) {
  const kutu = $("onizleme");
  if (!kutu) return;
  if (acikArtifact === id) { kutu.innerHTML = ""; acikArtifact = null; return; }
  acikArtifact = id;

  const ad = satir.querySelector("strong")?.textContent || id;
  kutu.innerHTML = `<div class="onizleme-kutu"><div class="onizleme-baslik">
      <strong>${esc(ad)}</strong><span class="onizleme-kapat">kapat ×</span></div>
      <p class="bos">Yükleniyor…</p></div>`;
  kutu.querySelector(".onizleme-kapat").onclick = () => {
    kutu.innerHTML = ""; acikArtifact = null;
  };

  const s = oturum();
  let d;
  try {
    const r = await fetch(`/api/artifact/${encodeURIComponent(id)}?session=${encodeURIComponent(s || "")}`);
    d = await r.json();
  } catch (e) { d = { hata: "İstek başarısız" }; }
  if (acikArtifact !== id) return;          // arada başkasına tıklanmış

  const govde = kutu.querySelector(".onizleme-kutu");
  const bas = govde.querySelector(".onizleme-baslik").outerHTML;
  govde.innerHTML = bas + onizlemeGovdesi(d);
  govde.querySelector(".onizleme-kapat").onclick = () => {
    kutu.innerHTML = ""; acikArtifact = null;
  };
}

function onizlemeGovdesi(d) {
  if (d.hata) return `<p class="hata">${esc(d.hata)}</p>`;
  if (d.bilgi) return `<p class="bos">${esc(d.bilgi)}</p>`;
  if (d.metin) return `<pre class="onizleme-metin">${esc(d.metin)}</pre>`;
  if (d.tablo) {
    const t = d.tablo;
    const not = t.toplam > t.gosterilen
      ? `<p class="kart-not" style="text-align:left">${t.toplam} satırın ilk ${t.gosterilen}'i</p>` : "";
    return `<div class="onizleme-kaydir"><table class="veri onizleme-tablo"><thead><tr>` +
      t.sutunlar.map(c => `<th>${esc(c)}</th>`).join("") + `</tr></thead><tbody>` +
      t.satirlar.map(r => `<tr>` + r.map(v => `<td>${esc(v)}</td>`).join("") + `</tr>`).join("") +
      `</tbody></table></div>` + not;
  }
  return `<p class="bos">Önizlenecek içerik yok.</p>`;
}

// --- döngü -----------------------------------------------------------------
async function yenile() {
  const s = oturum();
  try {
    const r = await fetch("/api/durum" + (s ? `?session=${encodeURIComponent(s)}` : ""));
    const d = await r.json();
    haritaSagligi(d.podlar);
    ciz_sandboxlar(d.podlar);
    ciz_akislar(d.akislar);
    if (!d.akislar.error) haritayiKur(d.akislar.kenarlar);
    ciz_artifactler(d.artifactler);
    $("yenileme").textContent = "güncellendi " + new Date().toLocaleTimeString("tr-TR");
  } catch (e) {
    $("yenileme").textContent = "sunucuya ulaşılamıyor";
  }
}

yenile();
setInterval(yenile, YENILEME_MS);

// --- canlı akış (Hubble → SSE) --------------------------------------------
//
// `EventSource` kullanılıyor çünkü akış TEK YÖNLÜ ve tarayıcı yeniden
// bağlanmayı kendisi hallediyor — WebSocket'te bunu elle yazmak gerekirdi.

const AZAMI_SATIR = 60;   // DOM'u sınırla: sayfa saatlerce açık kalabilir

function akisSatiri(f) {
  const dusuk = f.verdict === "DROPPED";
  const el = document.createElement("div");
  el.className = "akis-satiri" + (dusuk ? " dropped" : "");
  const port = f.port ? `:${f.port}` : "";
  el.innerHTML =
    `<span class="akis-zaman">${esc(f.zaman)}</span>` +
    `<span class="akis-yol">${esc(f.kaynak)} <span class="oku">${dusuk ? "⊘" : "→"}</span> ` +
      `${esc(f.hedef)}${esc(port)}</span>` +
    `<span class="akis-etiketler">` +
      (dusuk ? `<span class="akis-rozet red">ENGELLENDİ</span>` : "") +
      `<span class="akis-rozet">${esc(f.tur)}</span>` +
    `</span>`;
  return el;
}

function akisiBaslat() {
  const liste = $("akis-listesi");
  const durumEl = $("akis-durum");
  const kaynak = new EventSource("/api/akis");

  kaynak.onopen = () => {
    durumEl.textContent = "canlı";
    durumEl.className = "akis-durum canli";
  };

  kaynak.onmessage = (olay) => {
    let f;
    try { f = JSON.parse(olay.data); } catch (e) { return; }
    if (f.hata) {
      durumEl.textContent = "kapalı";
      durumEl.className = "akis-durum kopuk";
      hata(liste, f.hata + " — `kubectl port-forward -n kube-system svc/hubble-relay 4245:80` gerekiyor.");
      kaynak.close();
      return;
    }
    const bosMesaj = liste.querySelector(".bos");
    if (bosMesaj) bosMesaj.remove();
    liste.prepend(akisSatiri(f));               // en yeni en üstte
    akisiCiz(f);                                // haritada da göster
    while (liste.children.length > AZAMI_SATIR) liste.lastChild.remove();
  };

  kaynak.onerror = () => {
    durumEl.textContent = "yeniden bağlanıyor…";
    durumEl.className = "akis-durum kopuk";
  };
}

// Akış paneli `?akis=kapali` ile kapatılabilir.
//
// İki gerçek ihtiyaç: (a) Hubble kurulu olmayan bir kümede panel boş bir kutu
// olarak durmasın, (b) uzun süre açık bırakılan bir ekranda sürekli bir SSE
// bağlantısı tutmak istenmeyebilir.
if (new URLSearchParams(location.search).get("akis") === "kapali") {
  const kart = document.querySelector(".durum-sag");
  if (kart) kart.remove();
  document.querySelector(".durum-govde").style.gridTemplateColumns = "1fr";
} else {
  akisiBaslat();
}
