// Veri akışı — artifact soy ağacının canlı mermaid diyagramı.
//
// ## Ne gösteriyor
//
// Topoloji haritası "hangi SERVİS hangi servisle konuşabilir" diyor. Burası
// bir kat aşağısı: "hangi VERİ hangi veriden türedi". İkisi farklı sorular —
// biri altyapı, diğeri iş akışı.
//
// Kenarlar kayıt defterindeki `parents` alanından geliyor. O alan uzun süre
// boştu (dolması için LLM'in elle `parents=[...]` yazması gerekiyordu);
// 2026-09-05'te sandbox soyu otomatik doldurmaya başladı — bir çalıştırmada
// OKUNAN artifact'ler, o çalıştırmada ÜRETİLENLERİN ebeveyni sayılıyor.
// Bu diyagram o veriyi görünür kılan taraf.
//
// ## İki mod
//
//   TÜMÜ    — workflow'un bütün grafiği, `node_id`'ye göre kutulanmış.
//             Veri hangi düğümlere UĞRAMIŞ, tek bakışta.
//   ODAK    — tek bir artifact'in atalarıyla ürünleri. Tablodaki "⑂" düğmesi
//             buraya geçirir; sunucudaki `/lineage` uç noktasından gelir
//             (döngü koruması ve kapsam denetimi orada).

const SOY_YENI_MS = 30000;   // bu kadar yeni bir artifact "az önce" sayılır

let soyKip = { tur: "tumu", id: null };
let soySonMetin = null;      // aynı grafiği yeniden çizme (mermaid pahalı)
let soySayac = 0;
let mermaidHazir = false;

function soyKur() {
  if (typeof mermaid === "undefined") return false;
  if (!mermaidHazir) {
    // `securityLevel: strict` — etiketlerdeki HTML çalıştırılmaz. Etiketler
    // artifact ADLARINDAN geliyor; adlar serviste doğrulanıyor ama diyagram
    // katmanında da güvenmemek doğru olan.
    mermaid.initialize({
      startOnLoad: false,
      securityLevel: "strict",
      theme: "base",
      themeVariables: {
        fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
        fontSize: "13px",
        primaryColor: "#f5f5f7",
        primaryTextColor: "#1d1d1f",
        primaryBorderColor: "#d2d2d7",
        lineColor: "#86868b",
      },
      flowchart: { curve: "basis", nodeSpacing: 28, rankSpacing: 46, padding: 8 },
    });
    mermaidHazir = true;
  }
  return true;
}

// Mermaid düğüm kimliği: artifact_id doğrudan kullanılamaz (nokta, tire
// mermaid'de anlamlı). Sıralı bir takma ad üretip eşlemede tutuyoruz.
function soyKimlikleri(kayitlar) {
  const harita = new Map();
  kayitlar.forEach((k, i) => harita.set(k.artifact_id, "n" + i));
  return harita;
}

// Etiket metni: mermaid `["..."]` içinde tırnak ve köşeli parantez kırıcı.
function soyEtiket(t) {
  return String(t ?? "")
    .replace(/&/g, "&amp;").replace(/"/g, "&quot;")
    .replace(/[[\]{}()<>|]/g, "·")
    .slice(0, 48);
}

function soyBoyut(b) {
  if (!b) return "0 B";
  if (b < 1024) return b + " B";
  if (b < 1048576) return (b / 1024).toFixed(1) + " KB";
  return (b / 1048576).toFixed(1) + " MB";
}

function soyYeniMi(k) {
  const t = Date.parse(k.created_at);
  return Number.isFinite(t) && (Date.now() - t) < SOY_YENI_MS;
}

function soyDugumMetni(k) {
  const tip = String(k.type || k.artifact_type || "").replace("system.", "") || "Artifact";
  return `["${soyEtiket(k.name)}<br/><small>${soyEtiket(tip)} · ${soyBoyut(k.size_bytes)}</small>"]`;
}

// --- grafik metni: TÜMÜ modu ----------------------------------------------

function soyMetniTumu(kayitlar, calisanPod) {
  const kimlik = soyKimlikleri(kayitlar);
  const satir = ["graph LR"];

  // `node_id`'ye göre kutulama: "veri hangi adımlara uğradı" sorusunun
  // görsel cevabı. node_id'si olmayanlar (süpürmeden gelenler) kutusuz.
  const gruplar = new Map();
  for (const k of kayitlar) {
    const g = k.node_id || "";
    if (!gruplar.has(g)) gruplar.set(g, []);
    gruplar.get(g).push(k);
  }

  for (const [ad, uyeler] of gruplar) {
    const icerik = uyeler.map(k => `    ${kimlik.get(k.artifact_id)}${soyDugumMetni(k)}`);
    if (ad) {
      satir.push(`  subgraph sg_${ad.replace(/[^A-Za-z0-9_]/g, "_")}["${soyEtiket(ad)}"]`);
      satir.push(...icerik);
      satir.push("  end");
    } else {
      satir.push(...icerik);
    }
  }

  // Kenarlar: yalnızca iki ucu da grafikte olanlar. TTL reaper ebeveyni
  // silmişse çocuk öksüz görünür — kopuk ok çizmiyoruz.
  let kenarSayisi = 0;
  for (const k of kayitlar) {
    for (const e of (k.parents || [])) {
      if (kimlik.has(e)) {
        satir.push(`  ${kimlik.get(e)} --> ${kimlik.get(k.artifact_id)}`);
        kenarSayisi++;
      }
    }
  }

  // "ŞU AN" katmanı: çalışan bir sandbox varsa diyagrama giriyor. Kesikli
  // ok, henüz TAMAMLANMAMIŞ bir üretimi anlatıyor — hangi artifact'i
  // üreteceğini bilmediğimiz için hedefi yok, düğüm tek başına duruyor.
  if (calisanPod) {
    satir.push(`  sbx(["⚙ sandbox çalışıyor<br/><small>${soyEtiket(calisanPod)}</small>"])`);
    satir.push("  class sbx calisan");
  }

  const yeniler = kayitlar.filter(soyYeniMi).map(k => kimlik.get(k.artifact_id));
  if (yeniler.length) satir.push(`  class ${yeniler.join(",")} yeni`);

  satir.push("  classDef yeni fill:#e8f5e9,stroke:#34c759,stroke-width:2px");
  satir.push("  classDef calisan fill:#e3f2fd,stroke:#0071e3,stroke-width:2px,stroke-dasharray:4 3");
  return { metin: satir.join("\n"), kenarSayisi };
}

// --- grafik metni: ODAK modu ----------------------------------------------

function soyMetniOdak(soy) {
  const kimlik = soyKimlikleri(soy.nodes);
  const satir = ["graph LR"];
  for (const n of soy.nodes) satir.push(`  ${kimlik.get(n.artifact_id)}${soyDugumMetni(n)}`);
  for (const e of soy.edges) {
    if (kimlik.has(e.from) && kimlik.has(e.to))
      satir.push(`  ${kimlik.get(e.from)} --> ${kimlik.get(e.to)}`);
  }
  const grup = (yon) => soy.nodes.filter(n => n.yon === yon).map(n => kimlik.get(n.artifact_id));
  const ata = grup("ata"), urun = grup("urun");
  if (ata.length) satir.push(`  class ${ata.join(",")} ata`);
  if (urun.length) satir.push(`  class ${urun.join(",")} urun`);
  satir.push(`  class ${kimlik.get(soy.root)} kok`);
  satir.push("  classDef kok fill:#0071e3,stroke:#0071e3,color:#fff,stroke-width:2px");
  satir.push("  classDef ata fill:#f5f5f7,stroke:#c7c7cc,color:#6e6e73");
  satir.push("  classDef urun fill:#e8f5e9,stroke:#34c759");
  return { metin: satir.join("\n"), ata: ata.length, urun: urun.length };
}

// --- çizim ----------------------------------------------------------------

async function soyBoya(metin, altYazi) {
  const kutu = document.getElementById("soy-diyagram");
  const alt = document.getElementById("soy-alt");
  if (!kutu) return;
  if (alt) alt.innerHTML = altYazi;

  // Aynı grafiği yeniden çizmiyoruz: `mermaid.render` 5 saniyede bir
  // çağrılırsa diyagram gözle görülür şekilde titriyor ve seçim kayboluyor.
  if (metin === soySonMetin) return;
  soySonMetin = metin;

  try {
    const { svg } = await mermaid.render("soy_svg_" + (++soySayac), metin);
    kutu.innerHTML = svg;
  } catch (e) {
    kutu.innerHTML = `<p class="hata">Diyagram çizilemedi: ${String(e).slice(0, 200)}</p>`;
  }
}

/** Grafiği BU OTURUMLA ve onun ATALARIYLA sınırlar.
 *
 * Keşif tenant genelinde olduğu için liste 80+ artifact'e çıkabiliyor; hepsini
 * çizmek okunmaz bir duvar üretiyordu. Kullanıcının sorusu "benim işim neyden
 * türedi" — o yüzden bu oturumun çıktılarından başlayıp ebeveyn zincirini
 * yukarı yürüyoruz. Ata BAŞKA bir çalıştırmadan geliyorsa grafiğe giriyor
 * (asıl anlatılmak istenen şey zaten o bağ).
 */
function soyKapsami(kayitlar, workflowId) {
  const hepsi = new Map(kayitlar.map(k => [k.artifact_id, k]));
  const secili = new Map();
  const sira = kayitlar.filter(k => k.workflow_id === workflowId);

  // Bu oturumda hiç çıktı yoksa en yeni 25'i göster — boş ekran vermektense.
  if (!sira.length) return kayitlar.slice(0, 25);

  const yigin = [...sira];
  while (yigin.length) {
    const k = yigin.pop();
    if (secili.has(k.artifact_id)) continue;
    secili.set(k.artifact_id, k);
    for (const e of (k.parents || [])) {
      const ata = hepsi.get(e);
      if (ata && !secili.has(e)) yigin.push(ata);
    }
  }
  return [...secili.values()];
}

/** Panelin her yenilemesinde çağrılır. `a` = /api/durum'un artifactler bölümü. */
async function soyCiz(a, podlar) {
  const kutu = document.getElementById("soy-diyagram");
  if (!kutu || !soyKur()) return;

  if (soyKip.tur === "odak") return;   // odak modu kendi verisini çeker

  if (a.error || a.not || !a.kayitlar || !a.kayitlar.length) {
    soySonMetin = null;
    kutu.innerHTML = `<p class="bos">${a.error || a.not
      || "Henüz artifact yok — bir soru sorunca akış burada belirir."}</p>`;
    return;
  }

  const calisan = (podlar && !podlar.error && podlar.sandboxlar && podlar.sandboxlar[0])
    ? podlar.sandboxlar[0].pod : null;
  const kapsam = soyKapsami(a.kayitlar, a.workflow_id);
  const { metin, kenarSayisi } = soyMetniTumu(kapsam, calisan);
  const yeni = kapsam.filter(soyYeniMi).length;
  const disaridan = kapsam.filter(k => k.workflow_id !== a.workflow_id).length;
  await soyBoya(metin,
    `Bu oturumun ${kapsam.length - disaridan} çıktısı`
    + (disaridan ? ` + türedikleri <strong>${disaridan} dış artifact</strong>` : "")
    + ` · <strong>${kenarSayisi} soy bağı</strong>`
    + (yeni ? ` · <span style="color:var(--izin)">${yeni} tanesi az önce üretildi</span>` : "")
    + (calisan ? ` · <span style="color:#0071e3">sandbox çalışıyor</span>` : "")
    + `. Kutular <code>node_id</code>, oklar "şundan türedi". `
    + `Tenant'ın tamamı için tablodaki <strong>⑂</strong> düğmelerini kullan.`);
}

/** Tablodaki "⑂" düğmesi — tek artifact'in atalarına + ÜRÜNLERİNE odaklan. */
async function soyOdakla(artifactId, ad) {
  if (!soyKur()) return;
  soyKip = { tur: "odak", id: artifactId };
  soySonMetin = null;
  const kutu = document.getElementById("soy-diyagram");
  const baslik = document.getElementById("soy-mod");
  if (baslik) {
    baslik.innerHTML = `<strong>${ad}</strong> odakta`
      + ` <button class="soy-geri" id="soy-geri">← tüm akışa dön</button>`;
    document.getElementById("soy-geri").onclick = soyTumuneDon;
  }
  kutu.innerHTML = `<p class="bos">Soy ağacı yükleniyor…</p>`;

  const s = (typeof oturum === "function") ? oturum() : null;
  let d;
  try {
    const r = await fetch(`/api/artifact/${encodeURIComponent(artifactId)}/soy`
      + `?session=${encodeURIComponent(s || "")}`);
    d = await r.json();
  } catch (e) { d = { hata: "İstek başarısız" }; }

  if (soyKip.id !== artifactId) return;      // arada başkasına tıklanmış
  if (d.hata) { kutu.innerHTML = `<p class="hata">${d.hata}</p>`; return; }

  const { metin, ata, urun } = soyMetniOdak(d);
  await soyBoya(metin,
    `<strong>${urun} ürün</strong> (bu veriden türeyenler) · ${ata} ata (bunun türediği veriler).`
    + (urun === 0 ? " Bu artifact'ten henüz bir şey türetilmemiş." : ""));
}

function soyTumuneDon() {
  soyKip = { tur: "tumu", id: null };
  soySonMetin = null;
  const baslik = document.getElementById("soy-mod");
  if (baslik) baslik.textContent = "";
  if (typeof yenile === "function") yenile();
}
