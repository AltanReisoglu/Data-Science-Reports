"""
VLM istemcisi — bu araç kümesi için.

`cua_lab/hf_model.py` ile aynı taşıyıcı (HF router, OpenAI uyumlu, stdlib
urllib). Fark ARAÇ ŞEMASI: orada 17 eylemli computer-use uzayı vardı, burada
üç araç var — terminal · browser · finish.

KOORDİNAT YOK: model `click(3)` diyor, `click(x=840, y=412)` demiyor.
Koordinatı DOM veriyor. Ölçek kayması hatası bu tasarımda hiç doğmuyor.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
VLM = "Qwen/Qwen2.5-VL-72B-Instruct"

SISTEM = """Bir bilgisayari kullanan ajansin. Ucu arac var. Her turda TEK arac cagir.

Yalnizca tek satirlik JSON yaz, baska hicbir sey yazma:

{"dusunce":"kisa gerekce","arac":"browser.goto","url":"example.com"}
{"dusunce":"...","arac":"browser.dom"}
{"dusunce":"...","arac":"browser.click","i":3}
{"dusunce":"...","arac":"browser.type","i":0,"metin":"altan"}
{"dusunce":"...","arac":"browser.key","tus":"Enter"}
{"dusunce":"...","arac":"browser.scroll","dy":400}
{"dusunce":"...","arac":"browser.find","metin":"Search"}
{"dusunce":"...","arac":"browser.read","sayfa":1}
{"dusunce":"...","arac":"browser.links","filtre":"decoding"}
{"dusunce":"...","arac":"browser.wait_for","metin":"Sonuclar"}
{"dusunce":"...","arac":"browser.back"}
{"dusunce":"...","arac":"terminal","komut":"ls -la"}
{"dusunce":"...","arac":"terminal.yaz","ad":"notlar.md","icerik":"cok\nsatirli"}
{"dusunce":"...","arac":"terminal.oku","ad":"notlar.md","bas":1}
{"dusunce":"...","arac":"terminal.listele"}
{"dusunce":"...","arac":"terminal.ara","desen":"TODO"}
{"dusunce":"...","arac":"desktop.screenshot"}
{"dusunce":"...","arac":"desktop.click","x":840,"y":412}
{"dusunce":"...","arac":"desktop.type","metin":"merhaba"}
{"dusunce":"...","arac":"desktop.key","tus":"Return"}
{"dusunce":"...","arac":"desktop.pencereler"}
{"dusunce":"...","arac":"desktop.pencere","ad":"Firefox"}
{"dusunce":"...","arac":"desktop.odakla","ad":"Firefox"}
{"dusunce":"...","arac":"finish","cevap":"kisa ozet"}

KURALLAR
- ONCELIK: gorevde bir adres/site geciyorsa ILK ISIN browser.goto olmali.
  Acik olan sayfa onceki bir gorevden kalmis olabilir; ONA GUVENME.
  Sayfadaki metni ASLA kendi cevabin gibi tekrarlama — o senin bulgun degil.
- ARAMA: bir seyi bulmak icin DOM listesini gozle tarama — browser.find kullan.
  Sayfa metnini okumak icin browser.read, baglantilar icin browser.links.
- Dosya yazacaksan `terminal` ile echo yerine terminal.yaz kullan (tirnak derdi yok).
- desktop.pencere ile TEK BIR pencerenin goruntusunu al — tum ekran yerine.
- Tiklamak/yazmak icin ONCE browser.dom cagir; ogeler NUMARALI gelir.
  Koordinat kullanma, numara kullan: {"arac":"browser.click","i":3}
- terminal KISITLI: tek dizine kilitli, rm/sudo/dd yok. Reddedilirsen baska yol dene.
- desktop.* GERCEK masaustunu kullanir. Koordinat SANA VERILEN GORUNTUNUN
  piksel uzayinda olmali. Silme tuslari ve terminal pencereleri ENGELLIDIR.
  Tarayici isi yapiyorsan desktop degil browser.* kullan — orada numara var,
  koordinat hatasi olmaz.
- Ayni eylemi tekrar tekrar deneme. Ekran/sayfa degismiyorsa BASKA bir yol dene.
- Is gercekten bittiyse finish ver; bitmediyse verme.
"""


def token_oku() -> str | None:
    for k in ("HF_TOKEN", "HF_Token", "HUGGINGFACE_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    p = Path(__file__).resolve().parents[2] / ".env"
    if p.is_file():
        for satir in p.read_text().splitlines():
            if "=" in satir and not satir.lstrip().startswith("#"):
                k, v = satir.split("=", 1)
                if k.strip().lower() in ("hf_token", "huggingface_token"):
                    return v.strip().strip('"').strip("'")
    return None


class VLMModel:
    def __init__(self, model: str = VLM, token: str | None = None,
                 max_tokens: int = 220, timeout: float = 120.0,
                 gorsel: bool = True):
        self.model, self.ad = model, f"hf:{model.split('/')[-1]}"
        self._token = token or token_oku()
        if not self._token:
            raise RuntimeError("HF token yok — .env icine HF_Token=... koy")
        self.max_tokens, self.timeout, self.gorsel = max_tokens, timeout, gorsel
        self.cagri = self.ayristirma_hatasi = 0
        self._uyardi = False

    def dusun(self, istem: str, png: bytes | None = None) -> tuple[dict, int, float]:
        icerik: list[dict] = [{"type": "text", "text": istem}]
        if png and self.gorsel:
            if not self._uyardi:
                self._uyardi = True
                print(f"     ! ekran görüntüsü {self.model} servisine gönderiliyor "
                      f"(~{len(png)//1024} KB/adım)")
            icerik.append({"type": "image_url", "image_url": {
                "url": "data:image/png;base64," + base64.b64encode(png).decode()}})

        govde = json.dumps({"model": self.model, "max_tokens": self.max_tokens,
                            "temperature": 0,
                            "messages": [{"role": "system", "content": SISTEM},
                                         {"role": "user", "content": icerik}]}).encode()
        req = urllib.request.Request(ENDPOINT, data=govde, headers={
            "Authorization": f"Bearer {self._token}", "Content-Type": "application/json"})
        self.cagri += 1
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                d = json.load(r)
        except urllib.error.HTTPError as e:
            return ({"arac": "hata", "dusunce": f"HTTP{e.code}: "
                     f"{e.read()[:150].decode(errors='replace')}"}, 0, 0.0)
        except Exception as e:
            return ({"arac": "hata", "dusunce": f"{type(e).__name__}: {e}"}, 0, 0.0)

        u = d.get("usage") or {}
        tok, cost = int(u.get("total_tokens") or 0), float(u.get("estimated_cost") or 0.0)
        metin = ((d.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        karar = _json_cikar(metin)
        if karar is None:
            self.ayristirma_hatasi += 1
            # Gizleme: metni olay akisina birak, monolog dedektoru gorsun.
            karar = {"arac": "soyle", "dusunce": metin.strip()[:300] or "(bos yanit)"}
        return karar, tok, cost


def _json_cikar(metin: str) -> dict | None:
    for m in re.finditer(r"\{.*?\}", metin, re.S):
        try:
            d = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if isinstance(d, dict) and "arac" in d:
            return d
    return None
