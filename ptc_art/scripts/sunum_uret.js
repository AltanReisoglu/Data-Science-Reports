/* PTC_Sunum_Karsilastirma.md  →  PTC_Sunum_Karsilastirma.pptx
 *
 * Kaynak TEK: markdown. Slaytlar elle tutulmuyor, md'den üretiliyor — iki
 * yerde iki farklı içerik olmasın diye. `node scripts/sunum_uret.js`
 *
 * Deck'in yükünü TABLOLAR taşıyor (her sayfa bir karşılaştırma), o yüzden
 * markdown tabloları gerçek pptx tablosuna çevriliyor; metne dökülmüyor.
 *
 * Tema, ürünün kendi arayüzünden: açık zemin, #0071e3 vurgu, mono kod.
 */

const fs = require("fs");
const path = require("path");
const pptxgen = require("pptxgenjs");

const KOK = path.resolve(__dirname, "..");
const MD = path.join(KOK, "PTC_Sunum_Karsilastirma.md");
const CIKTI = path.join(KOK, "PTC_Sunum_Karsilastirma.pptx");

/* ── tema — style.css'teki değerlerle aynı ─────────────────────── */
const T = {
  zemin: "FFFFFF",
  panel: "F5F5F7",
  metin: "1D1D1F",
  soluk: "6E6E73",
  vurgu: "0071E3",
  cizgi: "D8D8DC",
  basliksatiri: "EEF4FD",
  vurguYumusak: "E8F1FD",
  iyi: "1E7E42",
  kotu: "C0392B",
  sans: "Inter",
  mono: "JetBrains Mono",
};

const W = 13.333, H = 7.5;              // LAYOUT_WIDE
const KENAR = 0.62;
const ICERIK_G = W - KENAR * 2;

/* ── markdown'ı sayfalara böl ──────────────────────────────────── */

function sayfalariAyikla(md) {
  const satirlar = md.split("\n");
  const sayfalar = [];
  let simdiki = null;

  for (const s of satirlar) {
    const m = s.match(/^##\s+Sayfa\s+(\d+)\s+[—-]\s+(.+?)\s*$/);
    if (m) {
      if (simdiki) sayfalar.push(simdiki);
      simdiki = { no: +m[1], baslik: m[2], satirlar: [] };
      continue;
    }
    if (simdiki) simdiki.satirlar.push(s);
  }
  if (simdiki) sayfalar.push(simdiki);
  return sayfalar;
}

/* Bir sayfanın gövdesini bloklara ayır: tablo / kod / alıntı / paragraf */
function bloklaraAyir(satirlar) {
  const bloklar = [];
  let i = 0;

  while (i < satirlar.length) {
    const s = satirlar[i];

    if (!s.trim() || s.trim() === "---") { i++; continue; }

    // <!-- diyagram: ad --> → docs/diyagram/ad.png
    const dg = s.match(/^\s*<!--\s*diyagram:\s*([\w-]+)\s*-->\s*$/);
    if (dg) { bloklar.push({ tur: "diyagram", ad: dg[1] }); i++; continue; }

    // kod bloğu
    if (s.trim().startsWith("```")) {
      const govde = [];
      i++;
      while (i < satirlar.length && !satirlar[i].trim().startsWith("```")) govde.push(satirlar[i++]);
      i++;
      bloklar.push({ tur: "kod", satirlar: govde });
      continue;
    }

    // tablo
    if (s.trim().startsWith("|") && satirlar[i + 1] && /^\s*\|[\s:|-]+\|\s*$/.test(satirlar[i + 1])) {
      const ham = [];
      while (i < satirlar.length && satirlar[i].trim().startsWith("|")) ham.push(satirlar[i++]);
      const hucreler = ham
        .filter((r, idx) => idx !== 1)
        .map(r => r.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map(c => c.trim()));
      bloklar.push({ tur: "tablo", satirlar: hucreler });
      continue;
    }

    // alıntı
    if (s.trim().startsWith(">")) {
      const govde = [];
      while (i < satirlar.length && satirlar[i].trim().startsWith(">")) {
        govde.push(satirlar[i++].trim().replace(/^>\s?/, ""));
      }
      bloklar.push({ tur: "alinti", metin: govde.join(" ").trim() });
      continue;
    }

    // paragraf (boş satıra kadar)
    const govde = [];
    while (i < satirlar.length && satirlar[i].trim() && !satirlar[i].trim().startsWith("|")
           && !satirlar[i].trim().startsWith(">") && !satirlar[i].trim().startsWith("```")
           && satirlar[i].trim() !== "---") {
      govde.push(satirlar[i++].trim());
    }
    bloklar.push({ tur: "metin", metin: govde.join(" ") });
  }
  return bloklar;
}

/* ── satır içi markdown → pptx zengin metin parçaları ──────────── */

function parcala(metin, taban = {}) {
  const parcalar = [];
  // **kalın**, `kod`, ~~üstü çizili~~
  const re = /(\*\*[^*]+\*\*|`[^`]+`|~~[^~]+~~)/g;
  let son = 0, m;
  while ((m = re.exec(metin)) !== null) {
    if (m.index > son) parcalar.push({ text: metin.slice(son, m.index), options: { ...taban } });
    const t = m[0];
    if (t.startsWith("**")) {
      parcalar.push({ text: t.slice(2, -2), options: { ...taban, bold: true, color: taban.color || T.metin } });
    } else if (t.startsWith("~~")) {
      parcalar.push({ text: t.slice(2, -2), options: { ...taban, strike: true, color: T.soluk } });
    } else {
      parcalar.push({ text: t.slice(1, -1), options: { ...taban, fontFace: T.mono, color: T.vurgu } });
    }
    son = m.index + t.length;
  }
  if (son < metin.length) parcalar.push({ text: metin.slice(son), options: { ...taban } });
  return parcalar.length ? parcalar : [{ text: metin, options: { ...taban } }];
}

/* Hücre içeriğinden rengi çıkar — "BİZ", "YOK", "✅" gibi işaretler
   tabloyu okunur yapan asıl şey; düz siyah bırakmak bilgiyi saklardı. */
function hucreRengi(ham) {
  const d = ham.replace(/[*`~]/g, "");
  if (/\bBİZ\b|BİZİM/.test(d)) return T.vurgu;
  if (/^✅|KAPANDI|\bvar\b.*✓|✓$/.test(d)) return T.iyi;
  if (/^⚠️|^❌|^🔴|YOK$|Yok$|yok$|Kimse/.test(d)) return T.kotu;
  return T.metin;
}

/* Diyagramın slayttaki yeri. PNG'nin gerçek oranı korunuyor — esnetilmiş
   bir çizim, elle çizilmiş görünümü ilk bozan şey. */
function diyagramOlcu(ad) {
  const yol = path.join(KOK, "docs", "diyagram", `${ad}.png`);
  if (!fs.existsSync(yol)) return null;
  const boyut = require("image-size").imageSize
    ? require("image-size").imageSize(fs.readFileSync(yol))
    : require("image-size")(yol);
  const enBoy = boyut.width / boyut.height;
  // Azami yükseklik, ALTINDAKİ CÜMLEYE yer bırakacak kadar. 4.6 iken
  // diyagram slaydı doldurup açıklamayı devam slaydına atıyordu — diyagram
  // ile onu okutan cümlenin ayrı sayfalara düşmesi en kötü sonuçtu.
  const azamiG = ICERIK_G, azamiY = 3.95;
  let w = azamiG, h = w / enBoy;
  if (h > azamiY) { h = azamiY; w = h * enBoy; }
  return { yol, w, h, x: KENAR + (ICERIK_G - w) / 2 };
}

/* ── slayt çizimi ──────────────────────────────────────────────── */

function kunye(slayt, no, toplam) {
  slayt.addShape("rect", { x: 0, y: 0, w: W, h: 0.055, fill: { color: T.vurgu } });
  slayt.addText("PTC · Artifact Persistence", {
    x: KENAR, y: H - 0.44, w: 6, h: 0.28, fontSize: 9, color: T.soluk, fontFace: T.sans,
  });
  slayt.addText(`${no} / ${toplam}`, {
    x: W - KENAR - 1.2, y: H - 0.44, w: 1.2, h: 0.28, fontSize: 9, color: T.soluk,
    align: "right", fontFace: T.mono,
  });
}

function kapak(pres, toplam) {
  const s = pres.addSlide();
  s.background = { color: T.zemin };
  s.addShape("rect", { x: 0, y: 0, w: 0.16, h: H, fill: { color: T.vurgu } });
  s.addText("PTC Artifact Persistence", {
    x: 1.1, y: 2.15, w: W - 2.2, h: 0.9, fontSize: 40, bold: true, color: T.metin,
    fontFace: T.sans, charSpacing: -0.6,
  });
  s.addText("Karşılaştırmalı sunum — her sayfa bir özellik, her sayfada bir tablo", {
    x: 1.1, y: 3.05, w: W - 2.2, h: 0.5, fontSize: 16, color: T.soluk, fontFace: T.sans,
  });
  s.addShape("rect", { x: 1.12, y: 3.75, w: 2.2, h: 0.03, fill: { color: T.cizgi } });
  s.addText([
    { text: "Sandbox'ın ölmesi güvenlik için gerekli.", options: { breakLine: true } },
    { text: "Ürettiğinin kalması iş için gerekli.", options: {} },
  ], {
    x: 1.1, y: 4.0, w: 8, h: 0.9, fontSize: 17, color: T.metin, fontFace: T.sans,
    lineSpacingMultiple: 1.25,
  });
  s.addText(`2026-09-07  ·  ${toplam} sayfa  ·  bütün ölçümler canlı cluster'dan`, {
    x: 1.1, y: H - 1.0, w: 9, h: 0.3, fontSize: 11, color: T.soluk, fontFace: T.mono,
  });
  return s;
}

function tabloCiz(slayt, hucreler, y, genislik) {
  const [basliklar, ...govde] = hucreler;
  const sut = basliklar.length;
  const satirlar = [
    basliklar.map(h => ({
      text: h.replace(/[*`]/g, ""),
      options: { bold: true, color: T.metin, fill: { color: T.basliksatiri }, fontSize: 11.5 },
    })),
    ...govde.map((r, ri) => r.map(c => ({
      text: parcala(c, { fontSize: 11.5, color: hucreRengi(c) }),
      options: { fill: { color: ri % 2 ? T.panel : T.zemin } },
    }))),
  ];
  // İlk sütun genelde etiket; biraz daha geniş olsun.
  const kalan = genislik - genislik * 0.055 * (sut - 1);
  const ilk = sut > 2 ? kalan * 0.28 : kalan * 0.4;
  const digerG = (genislik - ilk) / (sut - 1);

  slayt.addTable(satirlar, {
    x: KENAR, y, w: genislik,
    colW: [ilk, ...Array(sut - 1).fill(digerG)],
    border: { type: "solid", pt: 0.5, color: T.cizgi },
    fontFace: T.sans, fontSize: 11.5, color: T.metin,
    valign: "middle", margin: [6, 9, 6, 9], autoPage: false,
  });
  // yaklaşık yükseklik: başlık + satırlar
  return 0.36 + govde.length * 0.34;
}

function slaytBasligi(s, sayfa, devam) {
  s.background = { color: T.zemin };
  s.addText(`SAYFA ${sayfa.no}${devam ? " · devam" : ""}`, {
    x: KENAR, y: 0.28, w: 3.4, h: 0.24, fontSize: 9.5, bold: true, color: T.vurgu,
    fontFace: T.sans, charSpacing: 1.6,
  });
  s.addText(sayfa.baslik, {
    x: KENAR, y: 0.52, w: ICERIK_G, h: 0.56, fontSize: 27, bold: true, color: T.metin,
    fontFace: T.sans, charSpacing: -0.4,
  });
  s.addShape("rect", { x: KENAR, y: 1.10, w: 1.5, h: 0.028, fill: { color: T.vurgu } });
}

/* Bloğun kaplayacağı yükseklik + altındaki boşluk. Çizim matematiğiyle
   AYNI formüller; ikisi ayrışırsa ya taşma ya boş slayt olur. */
function blokYuksekligi(b) {
  if (b.tur === "diyagram") return diyagramOlcu(b.ad).h + 0.3;
  if (b.tur === "tablo") return 0.36 + (b.satirlar.length - 1) * 0.34 + 0.28;
  if (b.tur === "kod") return b.satirlar.length * 0.215 + 0.26 + 0.22;
  if (b.tur === "alinti") {
    return Math.max(0.5, Math.ceil(b.metin.length / 118) * 0.26 + 0.24) + 0.22;
  }
  if (!b.metin || !b.metin.trim()) return 0;
  const h = Math.max(0.3, Math.ceil(b.metin.length / 132) * 0.25 + 0.1);
  return /^\*\*/.test(b.metin.trim()) ? h + 0.32 : h + 0.16;
}

/* Bir md sayfası bir slayda sığmazsa DÜŞÜRÜLMÜYOR, devam slaydına taşınıyor.
   Sessizce kaybolması, md'ye eklenen bir tablonun kimse fark etmeden yok
   olması demekti — bu projede en çok kovaladığımız hata sınıfı. */
function icerikSlaydi(pres, sayfa, toplam, uretilen) {
  let s = pres.addSlide();
  slaytBasligi(s, sayfa, false);
  uretilen.push(s);

  let y = 1.34;
  const TABAN = H - 0.72;

  const bloklar = bloklaraAyir(sayfa.satirlar);

  for (const b of bloklar) {
    const gerekli = blokYuksekligi(b);
    // Slaydın başındaysak kırmanın anlamı yok — blok tek başına taşıyorsa
    // taşsın, ikinci boş slayt üretmekten iyi.
    if (gerekli && y > 1.34 && y + gerekli > TABAN) {
      kunye(s, sayfa.no, toplam);
      s = pres.addSlide();
      slaytBasligi(s, sayfa, true);
      uretilen.push(s);
      y = 1.34;
    }

    if (b.tur === "diyagram") {
      const o = diyagramOlcu(b.ad);
      if (!o) { console.warn(`  ! diyagram yok: ${b.ad}`); continue; }
      s.addImage({ path: o.yol, x: o.x, y, w: o.w, h: o.h });
      y += o.h + 0.3;
    }

    else if (b.tur === "tablo") {
      const h = tabloCiz(s, b.satirlar, y, ICERIK_G);
      y += h + 0.26;
    }

    else if (b.tur === "alinti") {
      const satirSayisi = Math.ceil(b.metin.length / 118);
      const h = Math.max(0.5, satirSayisi * 0.26 + 0.24);
      s.addShape("rect", { x: KENAR, y, w: 0.045, h, fill: { color: T.vurgu } });
      s.addText(parcala(b.metin, { fontSize: 13, color: T.metin, italic: true }), {
        x: KENAR + 0.22, y, w: ICERIK_G - 0.3, h, fontFace: T.sans, valign: "middle",
      });
      y += h + 0.22;
    }

    else if (b.tur === "kod") {
      const h = b.satirlar.length * 0.215 + 0.26;
      s.addShape("rect", { x: KENAR, y, w: ICERIK_G, h, fill: { color: T.panel },
                           line: { color: T.cizgi, width: 0.5 } });
      s.addText(b.satirlar.join("\n"), {
        x: KENAR + 0.14, y: y + 0.1, w: ICERIK_G - 0.28, h: h - 0.2,
        fontSize: 10, fontFace: T.mono, color: T.metin, lineSpacingMultiple: 1.1,
      });
      y += h + 0.22;
    }

    else {
      if (!b.metin.trim()) continue;
      const vurgulu = /^\*\*/.test(b.metin.trim());
      const satirSayisi = Math.ceil(b.metin.length / 132);
      const h = Math.max(0.3, satirSayisi * 0.25 + 0.1);
      if (vurgulu) {
        s.addShape("rect", { x: KENAR, y, w: ICERIK_G, h: h + 0.14,
                             fill: { color: T.vurguYumusak } });
        s.addText(parcala(b.metin, { fontSize: 13, color: T.metin }), {
          x: KENAR + 0.16, y: y + 0.06, w: ICERIK_G - 0.32, h, fontFace: T.sans, valign: "middle",
        });
        y += h + 0.32;
      } else {
        s.addText(parcala(b.metin, { fontSize: 13, color: T.soluk }), {
          x: KENAR, y, w: ICERIK_G, h, fontFace: T.sans, valign: "top",
        });
        y += h + 0.16;
      }
    }
  }

  kunye(s, sayfa.no, toplam);
  return s;
}

/* ── çalıştır ──────────────────────────────────────────────────── */

const md = fs.readFileSync(MD, "utf8");
const sayfalar = sayfalariAyikla(md);
if (!sayfalar.length) { console.error("md'de 'Sayfa N —' başlığı bulunamadı"); process.exit(1); }

const pres = new pptxgen();
pres.layout = "LAYOUT_WIDE";
pres.author = "PTC";
pres.title = "PTC Artifact Persistence — Karşılaştırmalı Sunum";

kapak(pres, sayfalar.length);
const uretilen = [];
for (const s of sayfalar) icerikSlaydi(pres, s, sayfalar.length, uretilen);
const devam = uretilen.length - sayfalar.length;

pres.writeFile({ fileName: CIKTI })
  .then(() => console.log(
    `yazıldı: ${path.basename(CIKTI)}  ·  ${uretilen.length + 1} slayt` +
    (devam ? `  (${devam} devam slaydı)` : "")))
  .catch(e => { console.error(e); process.exit(1); });
