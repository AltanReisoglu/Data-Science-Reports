"""
VLM üzerinden gerçek computer use — HF Inference API, stdlib `urllib`.

KLASİK COMPUTER-USE DÖNGÜSÜ:

    ekran goruntusu  ->  VLM  ->  {"act":"left_click","x":..,"y":..}  ->  xdotool
           ^                                                                 |
           +-----------------------------------------------------------------+

Model ekranı GERÇEKTEN görüyor: PNG, `image_url` içerik bloğu olarak gidiyor.
Metin bağlamı (imleç yeri, aktif pencere, ekran boyutu) yanında taşınıyor —
görüntüden okunamayacak şeyler.

NEDEN Qwen2.5-VL: router'da erişimi açık olan ve GUI grounding'de çalışan model
bu. Denendi ve düğme üzerindeki yazıyı doğru okudu. `Llama-3.2-Vision` ve
`Qwen2.5-VL-7B` bu token'la "not supported by any provider" dönüyor.

KOORDİNAT ÖLÇEĞİ — sessiz hata kaynağı:
Görüntüyü küçültüp gönderiyoruz (token maliyeti). Model koordinatı KÜÇÜLTÜLMÜŞ
görüntü uzayında veriyor. Gerçek ekrana çevirmezseniz her tıklama sol üste
kayar ve neden olduğunu anlamazsınız. `image_scale` ile çarpılıyor.

ÖLÇÜM DÜRÜSTLÜĞÜ: `tokens` ve `cost_usd` uydurulmuyor — API'nin `usage`
bloğundan okunuyor. Görsel token'lar da oraya dahil, yani bütçe ekseni ekran
görüntüsünün gerçek maliyetini sayıyor.

GİZLİLİK: gerçek masaüstünde bu ekranını dış bir servise yüklüyor. Koşum
başlarken bir kez açıkça uyarılıyor.
"""

from __future__ import annotations

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

from .events import Act, ComputerCall, Finish, Say

ENDPOINT = "https://router.huggingface.co/v1/chat/completions"
DEFAULT_MODEL = "Qwen/Qwen2.5-VL-72B-Instruct"      # vision
TEXT_MODEL = "meta-llama/Llama-3.1-8B-Instruct"     # görüntüsüz ortamlar için

SYSTEM = """Bir bilgisayari GRAFIK ARAYUZUNDEN kullanan ajansin.
Sana her turda ekranin goruntusu veriliyor. Goruntuye bak, TEK bir eylem sec.

Yalnizca tek satirlik JSON yaz, baska hicbir sey yazma:

{"act":"left_click","x":420,"y":310}
{"act":"double_click","x":420,"y":310}
{"act":"type","text":"merhaba"}
{"act":"key","text":"Return"}
{"act":"scroll","x":600,"y":400,"scroll_direction":"down","scroll_amount":3}
{"act":"mouse_move","x":420,"y":310}
{"act":"screenshot"}
{"act":"wait","duration":1}
{"act":"finish","answer":"kisa ozet"}

KURALLAR
- Koordinatlar SANA VERILEN GORUNTUNUN piksel uzayinda olmali.
- Bir alana yazmadan once ona tiklayarak odaklan.
- Ayni eylemi tekrar tekrar deneme; ekran degismiyorsa BASKA bir yol dene.
- Is gercekten bittiyse finish ver; bitmediyse verme.
- Silme, kapatma ve sistem komutlari ENGELLIDIR; denersen reddedilirsin.
"""

_ACTS = {a.value: a for a in Act}


def read_token(env_path: str | os.PathLike | None = None) -> str | None:
    """Token: önce ortam değişkeni, sonra `.env`. Değer asla loglanmıyor."""
    for k in ("HF_TOKEN", "HF_Token", "HUGGINGFACE_TOKEN", "HF_API_TOKEN"):
        if os.environ.get(k):
            return os.environ[k].strip()
    p = Path(env_path) if env_path else Path(__file__).resolve().parents[2] / ".env"
    if p.is_file():
        for line in p.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                if k.strip().lower() in ("hf_token", "huggingface_token", "hf_api_token"):
                    return v.strip().strip('"').strip("'")
    return None


class HFInferenceModel:
    """VLM istemcisi. `act()` sözleşmesi `ScriptedModel` ile aynı."""

    def __init__(self, model: str = DEFAULT_MODEL, token: str | None = None,
                 max_tokens: int = 160, temperature: float = 0.0,
                 timeout: float = 120.0):
        self.model = model
        self.name = f"hf:{model.split('/')[-1]}"
        self._token = token or read_token()
        if not self._token:
            raise RuntimeError(
                "HF token bulunamadi. `.env` icine HF_Token=... koy ya da "
                "HF_TOKEN ortam degiskenini ayarla.")
        self.max_tokens, self.temperature, self.timeout = max_tokens, temperature, timeout
        self.calls = 0
        self.parse_failures = 0
        self.pixels_sent = 0
        self._uyardi = False

    # -- ağ ---------------------------------------------------------------

    def _post(self, messages: list[dict]) -> dict:
        body = json.dumps({
            "model": self.model, "messages": messages,
            "max_tokens": self.max_tokens, "temperature": self.temperature,
        }).encode()
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Authorization": f"Bearer {self._token}",
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.load(r)

    def _icerik(self, req) -> list[dict]:
        """Kullanıcı mesajı: metin bağlamı + (varsa) ekran görüntüsü."""
        parcalar: list[dict] = [{"type": "text", "text": req.prompt()}]
        if req.image:
            if not self._uyardi:
                self._uyardi = True
                print("  ! EKRAN GORUNTUSU DIS SERVISE GONDERILIYOR "
                      f"({len(req.image)//1024} KB/adim, {self.model})")
            b64 = base64.b64encode(req.image).decode()
            self.pixels_sent += len(req.image)
            parcalar.append({"type": "image_url",
                             "image_url": {"url": f"data:image/png;base64,{b64}"}})
        return parcalar

    # -- ayrıştırma --------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> dict | None:
        """Modelin etrafına yazdığı laf kalabalığını tolere et."""
        for m in re.finditer(r"\{[^{}]*\}", text, re.S):
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                continue
            if isinstance(d, dict) and "act" in d:
                return d
        return None

    # -- sözleşme ----------------------------------------------------------

    def act(self, req):
        self.calls += 1
        try:
            data = self._post([{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": self._icerik(req)}])
        except urllib.error.HTTPError as e:
            detay = e.read()[:200].decode(errors="replace")
            return Say(f"MODEL HATASI HTTP{e.code}: {detay}", tokens=0, cost_usd=0.0)
        except Exception as e:                       # ağ kesildi, zaman aşımı...
            return Say(f"MODEL HATASI {type(e).__name__}: {e}", tokens=0, cost_usd=0.0)

        usage = data.get("usage") or {}
        tok = int(usage.get("total_tokens") or 0)
        cost = float(usage.get("estimated_cost") or 0.0)
        text = ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""

        d = self._extract_json(text)
        if d is None:
            # Ayrıştırılamadı. GİZLEME — metin olayı olarak akışa gir ki
            # monolog dedektörü görebilsin.
            self.parse_failures += 1
            return Say(text.strip()[:300] or "(bos yanit)", tokens=tok, cost_usd=cost)

        act_adi = str(d.get("act", "")).lower().strip()
        if act_adi == "finish":
            return Finish(str(d.get("answer", ""))[:300], tokens=tok, cost_usd=cost)
        if act_adi not in _ACTS:
            self.parse_failures += 1
            return Say(f"BILINMEYEN EYLEM: {act_adi}", tokens=tok, cost_usd=cost)

        args = {k: v for k, v in d.items() if k != "act"}
        # KOORDINAT OLCEGI: model kucultulmus goruntuye gore konusuyor.
        # Cevirmezsek her tiklama sol uste kayar ve sebebi gorunmez.
        for eksen in ("x", "y"):
            if eksen in args:
                try:
                    args[eksen] = int(round(float(args[eksen]) * req.image_scale))
                except (TypeError, ValueError):
                    args.pop(eksen)
        return ComputerCall(_ACTS[act_adi], args, tokens=tok, cost_usd=cost)
