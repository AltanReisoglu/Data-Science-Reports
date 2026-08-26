"""
Tarayıcı aracı — DOM + ekran görüntüsü + etkileşim.

SOTA computer-use ajanlarının tarayıcı tarafı üç şeye dayanır ve üçü de burada:
  goto/screenshot   VLM'in GÖRDÜĞÜ  — piksel
  dom               VLM'in OKUDUĞU  — yapı, tıklanabilir öğeler numaralı
  click/type/scroll VLM'in YAPTIĞI  — eylem

`dom()` ham HTML dökmüyor. Ham HTML bir sayfada 200 KB olabiliyor ve bağlamı
şişirip döngü üretiyor. Onun yerine **etkileşilebilir öğeleri numaralandırıyor**
(set-of-marks deseni): model `click(3)` diyor, koordinat hesaplamıyor. Bu hem
ucuz hem koordinat kayması hatasını tamamen ortadan kaldırıyor.
"""

from __future__ import annotations

import base64
import json

from .cdp import Chrome

# Etkilesilebilir ogeleri toplayan ve NUMARALAYAN betik.
# Gorunur olmayanlar (display:none, sifir boyut, ekran disi) atiliyor.
TOPLA = r"""
(() => {
  const secici = 'a,button,input,textarea,select,summary,[role=button],[role=link],[onclick],[contenteditable=true]';
  const cikti = [];
  document.querySelectorAll(secici).forEach(el => {
    const r = el.getBoundingClientRect();
    const s = getComputedStyle(el);
    if (r.width < 2 || r.height < 2) return;
    if (s.display === 'none' || s.visibility === 'hidden' || +s.opacity === 0) return;
    if (r.bottom < 0 || r.top > innerHeight) return;
    const etiket = (el.innerText || el.value || el.placeholder ||
                    el.getAttribute('aria-label') || el.name || '').trim().slice(0, 70);
    cikti.push({
      i: cikti.length,
      tag: el.tagName.toLowerCase(),
      tur: el.type || '',
      etiket: etiket.replace(/\s+/g, ' '),
      x: Math.round(r.x + r.width / 2),
      y: Math.round(r.y + r.height / 2)
    });
  });
  return JSON.stringify({
    url: location.href,
    baslik: document.title,
    metin: document.body ? document.body.innerText.replace(/\n{3,}/g, '\n\n').slice(0, 1800) : '',
    ogeler: cikti.slice(0, 60)
  });
})()
"""


class Browser:
    """Ayrı bir headless Chrome. Kullanıcının tarayıcısına dokunmaz."""

    def __init__(self, port: int = 9222, width: int = 1280, height: int = 800,
                 headless: bool = True, konum: str | None = "sag"):
        self.chrome = Chrome(port=port, width=width, height=height,
                             headless=headless, konum=konum)
        self.width, self.height = width, height
        self._ogeler: list[dict] = []
        self._acik = False

    def start(self) -> None:
        self.chrome.start()
        # Gercek pencere boyutu Chrome tarafinda degismis olabilir
        # (gorunur modda ekranin yarisina yerlesiyor) — geri al.
        self.width, self.height = self.chrome.width, self.chrome.height
        self._acik = True

    def stop(self) -> None:
        if self._acik:
            self.chrome.stop()
            self._acik = False

    # -- eylemler ----------------------------------------------------------

    def goto(self, url: str) -> str:
        if not url.startswith(("http://", "https://", "data:", "file://", "about:")):
            url = "https://" + url
        self.chrome.cmd("Page.navigate", url=url)
        self._bekle()
        return f"acildi: {url}"

    def _bekle(self, saniye: float = 6.0) -> None:
        """readyState complete olana kadar bekle — sabit sleep yerine."""
        import time
        son = time.monotonic() + saniye
        while time.monotonic() < son:
            try:
                r = self.chrome.cmd("Runtime.evaluate",
                                    expression="document.readyState",
                                    returnByValue=True)
                if r["result"].get("value") == "complete":
                    time.sleep(0.25)      # son boyama
                    return
            except Exception:
                pass
            time.sleep(0.2)

    def dom(self) -> dict:
        """Sayfanın MODELE GİDECEK özeti — ham HTML değil, numaralı öğeler."""
        r = self.chrome.cmd("Runtime.evaluate", expression=TOPLA, returnByValue=True)
        d = json.loads(r["result"]["value"])
        self._ogeler = d["ogeler"]
        return d

    def screenshot(self) -> bytes:
        r = self.chrome.cmd("Page.captureScreenshot", format="png")
        return base64.b64decode(r["data"])

    def click(self, i: int) -> str:
        """Numaraya tıkla. Koordinat MODELDEN gelmiyor — DOM'dan geliyor,
        yani ölçek/kayma hatası hiç doğmuyor."""
        oge = self._oge(i)
        for tur in ("mousePressed", "mouseReleased"):
            self.chrome.cmd("Input.dispatchMouseEvent", type=tur,
                            x=oge["x"], y=oge["y"], button="left", clickCount=1)
        self._bekle(3.0)
        return f"tiklandi: [{i}] {oge['tag']} \"{oge['etiket'][:40]}\""

    def type(self, i: int, metin: str) -> str:
        oge = self._oge(i)
        self.click(i)
        self.chrome.cmd("Runtime.evaluate", expression=(
            "(()=>{const e=document.activeElement;"
            "if(e){e.value='';e.dispatchEvent(new Event('input',{bubbles:true}));}})()"))
        for ch in metin:
            self.chrome.cmd("Input.dispatchKeyEvent", type="char", text=ch)
        return f"yazildi: [{i}] \"{metin[:40]}\""

    def key(self, tus: str) -> str:
        kod = {"Enter": 13, "Tab": 9, "Escape": 27, "Backspace": 8}.get(tus, 0)
        for tur in ("rawKeyDown", "keyUp"):
            self.chrome.cmd("Input.dispatchKeyEvent", type=tur, key=tus,
                            windowsVirtualKeyCode=kod, nativeVirtualKeyCode=kod)
        self._bekle(3.0)
        return f"tus: {tus}"

    def scroll(self, dy: int = 400) -> str:
        self.chrome.cmd("Input.dispatchMouseEvent", type="mouseWheel",
                        x=self.width // 2, y=self.height // 2, deltaX=0, deltaY=dy)
        return f"kaydirildi: {dy}px"

    # -- ARAMA ve OKUMA ----------------------------------------------------

    def find(self, metin: str) -> str:
        """Metne göre öğe BUL — numarasını döndür.

        Bunsuz ajan DOM listesini gözle tarayıp numara tahmin ediyordu ve
        yanlış öğeye tıklayıp duruyordu: ölçtük, alphaXiv koşumunda 12 adım
        bu yüzden harcandı. Artık `find("Search")` → `[7]`.
        """
        d = self.dom()
        arama = metin.lower().strip()
        tam = [o for o in d["ogeler"] if o["etiket"].lower() == arama]
        kismi = [o for o in d["ogeler"] if arama in o["etiket"].lower()]
        bulunan = tam or kismi
        if not bulunan:
            yakin = ", ".join(f"[{o['i']}] {o['etiket'][:28]}"
                              for o in d["ogeler"][:8])
            return f"'{metin}' bulunamadi. Gorunen ogeler: {yakin or '(yok)'}"
        return " · ".join(f"[{o['i']}] {o['tag']} \"{o['etiket'][:40]}\""
                          for o in bulunan[:5])

    def read(self, sayfa: int = 1, boyut: int = 2500) -> str:
        """Sayfa metnini SAYFALI oku. `dom()` 1800 karakterde kesiyor;
        uzun bir makalede asıl içerik oradan sonra başlıyor."""
        r = self.chrome.cmd("Runtime.evaluate", returnByValue=True, expression=(
            "document.body ? document.body.innerText"
            ".replace(/\\n{3,}/g,'\\n\\n') : ''"))
        tam = r["result"].get("value") or ""
        n = max(1, -(-len(tam) // boyut))
        s = max(1, min(int(sayfa), n))
        parca = tam[(s - 1) * boyut: s * boyut]
        return f"[sayfa {s}/{n}, {len(tam)} karakter]\n{parca}"

    def links(self, filtre: str = "") -> str:
        """Sayfadaki bağlantılar — metin + hedef. Arama sonucu sayfalarında
        tıklamadan önce nereye gideceğini görmek için."""
        r = self.chrome.cmd("Runtime.evaluate", returnByValue=True, expression=(
            "JSON.stringify([...document.querySelectorAll('a[href]')]"
            ".filter(a=>a.innerText.trim()).slice(0,120)"
            ".map(a=>({t:a.innerText.trim().replace(/\\s+/g,' ').slice(0,80),"
            "h:a.href})))"))
        try:
            bag = json.loads(r["result"]["value"])
        except Exception:
            return "baglanti okunamadi"
        f = filtre.lower().strip()
        if f:
            bag = [b for b in bag if f in b["t"].lower() or f in b["h"].lower()]
        if not bag:
            return f"'{filtre}' iceren baglanti yok"
        return "\n".join(f"  {b['t'][:64]}  →  {b['h'][:70]}" for b in bag[:25])

    def wait_for(self, metin: str, saniye: float = 10.0) -> str:
        """Metin belirene kadar bekle. Model bunu ZATEN uydurmuştu
        (`browser.wait`) — demek ki ihtiyaç gerçek."""
        import time
        son = time.monotonic() + saniye
        while time.monotonic() < son:
            r = self.chrome.cmd("Runtime.evaluate", returnByValue=True,
                                expression="document.body?document.body.innerText:''")
            if metin.lower() in (r["result"].get("value") or "").lower():
                return f"gorundu: '{metin}'"
            time.sleep(0.5)
        return f"ZAMAN ASIMI: '{metin}' {saniye:g}sn icinde gorunmedi"

    def back(self) -> str:
        self.chrome.cmd("Runtime.evaluate", expression="history.back()")
        self._bekle(4.0)
        return f"geri: {self.dom()['url'][:70]}"

    def scroll_to(self, i: int) -> str:
        """Bir öğeyi görünür alana getir — ekran dışındaki öğe DOM'a girmiyor."""
        oge = self._oge(i)
        self.chrome.cmd("Runtime.evaluate", expression=(
            f"window.scrollTo(0, {oge['y']} + scrollY - innerHeight/2)"))
        return f"kaydirildi: [{i}] gorunur alanda"

    # -- yardımcı ----------------------------------------------------------

    def _oge(self, i: int) -> dict:
        if not self._ogeler:
            self.dom()
        for o in self._ogeler:
            if o["i"] == int(i):
                return o
        raise ValueError(f"[{i}] numarali oge yok — once dom() cagir")

    def durum_hash(self) -> str:
        """İlerleme sinyali: URL + başlık + SAYFA METNİ + öğe imzaları.

        Sayfa metni şart. Yalnız etkileşilebilir öğeleri hash'lersek sayfaya
        eklenen bir sonuç satırı ("GIRIS YAPILDI") hash'i değiştirmez ve
        ilerleme dedektörü ilerlemeyi göremez — ölçüldü, tam olarak bu oldu.

        Metin `\\n{3,}` sadeleştirilmiş ve 1800 karaktere kırpılmış hâliyle
        geliyor; saat/sayaç gibi sürekli değişen öğeler yine hash'i oynatabilir,
        o yüzden eşik yüksek tutulmalı (bkz. loopguard `no_progress=8`).
        """
        import hashlib
        try:
            d = self.dom()
        except Exception:
            return "?"
        blob = (d["url"] + "|" + d["baslik"] + "|" + d["metin"] + "|"
                + "|".join(f"{o['tag']}:{o['etiket']}" for o in d["ogeler"]))
        return hashlib.sha256(blob.encode()).hexdigest()[:12]
