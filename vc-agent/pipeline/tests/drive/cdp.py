"""PoC'yi gerçek bir insan gibi sürmek: fare, klavye, bekleme.

`--dump-dom` ve `--screenshot` sayfanın *sonucunu* gösteriyor ama ona nasıl
gelindiğini göstermiyor. Bir düğmenin gerçekten tıklanabilir olduğunu, girdinin
odağı aldığını ve cevabın ekrana düştüğünü ancak olayları göndererek öğrenirsin.

Sürücü CDP üstünde: `Input.dispatchMouseEvent` gerçek koordinata gerçek tık
gönderiyor, `Input.insertText` metni kutuya yazıyor. `Runtime.evaluate` yalnız
**okumak** için — bir düğmeye `el.click()` demek, o düğmenin görünür ve
tıklanabilir olduğunu doğrulamıyor, ki asıl soru o.
"""

from __future__ import annotations

import asyncio
import base64
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import websockets

HERE = Path(__file__).resolve().parent
PORT = 9333
CHROME = "/usr/bin/google-chrome"


class Page:
    def __init__(self, ws):
        self.ws = ws
        self.n = 0
        self.log: list[str] = []

    async def call(self, method: str, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method,
                                       "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    async def js(self, expr: str):
        """JS çalıştır ve DEĞERİ döndür — istisnayı yutmadan.

        İlk hâli yalnız `result.value` okuyordu, ve JS bir hata fırlattığında
        sessizce `None` dönüyordu. Ölçüldü: takım beklemesi 200 saniye boyunca
        `None` alıp zaman aşımı raporladı, oysa koşu 110 saniyede bitmişti —
        yani sürücü uygulamayı değil kendini ölçüyordu. Yutulan bir istisna,
        olmayan bir hata gibi görünür ve yanlış yerde arattırır.
        """
        r = await self.call("Runtime.evaluate", expression=expr,
                            returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            desc = (r["exceptionDetails"].get("exception") or {}).get(
                "description", r["exceptionDetails"].get("text", "?"))
            self.log.append(f"  ✘ JS HATASI {str(desc)[:120]}")
            return None
        return r.get("result", {}).get("value")

    async def goto(self, url: str, wait: float = 2.5):
        await self.call("Page.navigate", url=url)
        await asyncio.sleep(wait)

    async def box(self, selector: str):
        """Öğenin tıklanacak noktası. `null` dönerse öğe yok ya da görünmez —
        ikisi de tıklamadan önce bilinmesi gereken şey.

        Nokta, öğenin merkezi DEĞİL: öğenin **görünür kısmının** merkezi.
        Ölçüldü — 4638 px boyundaki bir ayırıcının merkezi y=2319'daydı, yani
        ekranın çok altında; tık oraya gidiyor, hiçbir olay düşmüyor ve sürücü
        "tıkladım" diyordu. Uzun bir öğeye merkezinden tıklamak, sessizce
        ıskalamanın en kolay yolu.
        """
        return await self.js(f"""(() => {{
          const el = document.querySelector({selector!r});
          if (!el) return null;
          const r = el.getBoundingClientRect();
          if (r.width === 0 || r.height === 0) return null;
          const cs = getComputedStyle(el);
          if (cs.visibility === 'hidden' || cs.display === 'none') return null;
          const vw = window.innerWidth, vh = window.innerHeight;
          const x0 = Math.max(r.left, 0), x1 = Math.min(r.right, vw);
          const y0 = Math.max(r.top, 0), y1 = Math.min(r.bottom, vh);
          if (x1 <= x0 || y1 <= y0) return null;   // tamamen ekran dışında
          const x = (x0 + x1) / 2, y = (y0 + y1) / 2;
          const hit = document.elementFromPoint(x, y);
          return {{x: x, y: y, w: r.width, h: r.height,
                   text: (el.textContent||'').trim().slice(0,40),
                   disabled: !!el.disabled,
                   // Üstünü başka bir şey kapatıyorsa tık ona gider.
                   ortu: (hit && (hit === el || el.contains(hit) || hit.contains(el)))
                         ? null : (hit ? (hit.id || hit.className || hit.tagName) : 'yok')}};
        }})()""")

    async def click(self, selector: str, label: str = ""):
        b = await self.box(selector)
        if b is None:
            self.log.append(f"  ✘ TIKLANAMADI {label or selector} — öğe yok/görünmez")
            return False
        if b.get("disabled"):
            self.log.append(f"  ✘ TIKLANAMADI {label or selector} — düğme pasif")
            return False
        if b.get("ortu"):
            self.log.append(f"  ✘ TIKLANAMADI {label or selector} — üstünü "
                            f"{b['ortu']!r} kapatıyor")
            return False
        for kind in ("mousePressed", "mouseReleased"):
            await self.call("Input.dispatchMouseEvent", type=kind,
                            x=b["x"], y=b["y"], button="left", clickCount=1)
        self.log.append(f"  ✔ tık  {label or selector}  ({b['text']!r})")
        return True

    async def hover(self, selector: str, label: str = ""):
        b = await self.box(selector)
        if b is None:
            self.log.append(f"  ✘ HOVER OLMADI {label or selector}")
            return False
        await self.call("Input.dispatchMouseEvent", type="mouseMoved",
                        x=b["x"], y=b["y"])
        self.log.append(f"  ✔ hover {label or selector}")
        return True

    async def type(self, selector: str, text: str):
        if not await self.click(selector, "girdi kutusu"):
            return False
        await self.call("Input.insertText", text=text)
        got = await self.js(f"document.querySelector({selector!r}).value")
        ok = got == text
        self.log.append(f"  {'✔' if ok else '✘'} yazdı {text[:44]!r}")
        return ok

    async def press(self, key: str, code: str, vk: int):
        for kind in ("keyDown", "keyUp"):
            await self.call("Input.dispatchKeyEvent", type=kind, key=key,
                            code=code, windowsVirtualKeyCode=vk,
                            nativeVirtualKeyCode=vk)

    async def select(self, selector: str, value: str):
        ok = await self.js(f"""(() => {{
          const s = document.querySelector({selector!r});
          if (!s) return false;
          const o = [...s.options].find(o => o.value === {value!r});
          if (!o) return false;
          s.value = {value!r};
          s.dispatchEvent(new Event('change', {{bubbles: true}}));
          return true;
        }})()""")
        self.log.append(f"  {'✔' if ok else '✘'} seçti {selector} = {value}")
        return ok

    async def shot(self, name: str):
        r = await self.call("Page.captureScreenshot", format="png")
        out = HERE / f"{name}.png"
        out.write_bytes(base64.b64decode(r["data"]))
        return out

    async def wait_for(self, expr: str, timeout: float = 200, every: float = 2):
        """Bir koşul doğru olana kadar bekle. Zaman aşımı bir hata değil,
        ölçülecek bir sonuç: neyin ne kadar sürdüğü sunumda önemli."""
        t0 = time.time()
        while time.time() - t0 < timeout:
            if await self.js(expr):
                return round(time.time() - t0, 1)
            await asyncio.sleep(every)
        return None


async def open_page(url: str) -> tuple[Page, object]:
    for _ in range(40):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/json", timeout=1) as r:
                tabs = json.load(r)
            page = next(t for t in tabs if t["type"] == "page")
            ws = await websockets.connect(page["webSocketDebuggerUrl"],
                                          max_size=60 * 1024 * 1024)
            p = Page(ws)
            await p.call("Page.enable")
            await p.call("Runtime.enable")
            await p.goto(url, wait=3)
            return p, ws
        except Exception:
            await asyncio.sleep(0.5)
    raise RuntimeError("Chrome'a bağlanılamadı")


def launch(profile: str):
    return subprocess.Popen(
        [CHROME, f"--remote-debugging-port={PORT}", f"--user-data-dir={profile}",
         "--headless=new", "--disable-gpu", "--no-sandbox", "--no-first-run",
         "--window-size=1800,1150", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
