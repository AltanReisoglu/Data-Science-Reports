"""Zincir gerçekten adım adım mı doluyor? Sayfa açılır açılmaz ölç."""
import asyncio, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import cdp

RUN = sys.argv[1] if len(sys.argv) > 1 else "chat-0006"

async def main():
    proc = cdp.launch(tempfile.mkdtemp())
    # `open_page` 3 sn bekliyor; yürüyüş 10 adım × ~260 ms = 2,6 sn sürüyor,
    # yani ilk bakış her zaman geç kalıyordu. Beklemeden bağlan.
    p, ws = await cdp.open_page("about:blank")
    try:
        await p.call("Page.navigate", url=f"http://127.0.0.1:8000/akis?run={RUN}")
        print(f"=== {RUN} · zincir dolarken ===", flush=True)
        for i in range(14):
            await asyncio.sleep(0.4)
            r = await p.js("(() => { const n = document.querySelectorAll('.canvas [data-node]');"
                           " if (!n.length) return null;"
                           " return [n.length,"
                           "   document.querySelectorAll('.canvas [data-node].is-pending').length,"
                           "   document.querySelectorAll('.canvas [data-node].is-live').length,"
                           "   document.querySelectorAll('.canvas [data-edge].is-live').length]; })()")
            if r:
                print(f"  +{0.4*(i+1):4.1f}sn  kutu {r[0]}  sönük {r[1]}  yanan {r[2]}"
                      f"  yanan-ok {r[3]}", flush=True)
        for l in p.log: print(l, flush=True)
    finally:
        await ws.close(); proc.terminate()

asyncio.run(main())
