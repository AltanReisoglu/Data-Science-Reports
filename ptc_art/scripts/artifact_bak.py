"""Bir artifact'in KÜNYESİNİ ve İÇİNİ tek komutta göster.

Depodaki baytlar çoğunlukla ikili (parquet, pdf, png, tar) — `curl | json.tool`
işe yaramıyor, `cat` ekranı bozuyor. Bu betik içerik tipine bakıp doğru
biçimde açıyor.

Kullanım:
    python scripts/artifact_bak.py art_79f039144ecc
    python scripts/artifact_bak.py departman_ozet.parquet
    python scripts/artifact_bak.py rapor.pdf@onaylanmis
    python scripts/artifact_bak.py art_79f0 --kaydet /tmp/x.parquet
    python scripts/artifact_bak.py --liste

Ön koşul:  kubectl port-forward svc/artifact-service 8080:8080
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
from pathlib import Path

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(KOK / ".env")

import requests  # noqa: E402

from grounded_assistant.agent.graph import _kapsam_jetonu  # noqa: E402

ADRES = os.environ.get("ARTIFACT_SERVICE_URL", "http://127.0.0.1:8080").rstrip("/")
#: Jeton kapsamı bir workflow kimliği; okuma için hangisi olduğu önemsiz.
KAPSAM = os.environ.get("BAKIS_KAPSAMI", "bakis")


def _basliklar() -> dict:
    jeton = _kapsam_jetonu(KAPSAM)
    if not jeton:
        sys.exit("Kapsam jetonu üretilemedi — imza anahtarı okunamıyor "
                 "(cluster'daki ptc-scope-signing Secret'ı).")
    return {"X-Scope-Token": jeton}


def _al(yol: str, **kw):
    y = requests.get(ADRES + yol, headers=_basliklar(), timeout=15, **kw)
    if y.status_code == 404:
        sys.exit(f"bulunamadı: {yol}")
    y.raise_for_status()
    return y


def liste(limit: int = 50) -> None:
    for k in _al(f"/artifacts?limit={limit}").json():
        print(f"{k['artifact_id']}  {k['name'][:38]:<38} "
              f"{(k.get('content_type') or '?')[:28]:<28} "
              f"{k.get('size_bytes') or 0:>9} B")


def _cozumle(hedef: str) -> dict:
    """Kısaltılmış id, tam id ya da ad(@alias) → künye."""
    if hedef.startswith("art_"):
        tam = hedef
        if len(hedef) < 16:  # kısaltma verilmiş, listeden tamamla
            adaylar = [k["artifact_id"] for k in _al("/artifacts?limit=500").json()
                       if k["artifact_id"].startswith(hedef)]
            if not adaylar:
                sys.exit(f"{hedef} ile başlayan artifact yok")
            if len(adaylar) > 1:
                sys.exit(f"{hedef} birden çok kayda uyuyor: {adaylar}")
            tam = adaylar[0]
        return _al(f"/artifacts/{tam}/metadata").json()
    # ad ya da ad@alias — /uri künyeyi verir, çıplak uç BAYT verir
    return _al(f"/artifacts/by-name/{hedef}/uri").json()


def _goster(ad: str, tip: str, ham: bytes) -> None:
    tip = (tip or "").lower()
    if "parquet" in tip or ad.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(io.BytesIO(ham))
        print(f"satır {len(df)} · sütun {list(df.columns)}\n")
        print(df.head(30).to_string())
        if len(df) > 30:
            print(f"\n… {len(df) - 30} satır daha")
        return
    if "json" in tip or ad.endswith(".json"):
        print(json.dumps(json.loads(ham.decode("utf-8")), indent=2,
                         ensure_ascii=False)[:4000])
        return
    if "csv" in tip or ad.endswith(".csv"):
        metin = ham.decode("utf-8", "replace").splitlines()
        print("\n".join(metin[:30]))
        if len(metin) > 30:
            print(f"… {len(metin) - 30} satır daha")
        return
    if tip.startswith("text/") or ad.endswith((".md", ".txt", ".log")):
        print(ham.decode("utf-8", "replace")[:4000])
        return
    if ad.endswith(".tar") or "tar" in tip:
        import tarfile
        with tarfile.open(fileobj=io.BytesIO(ham)) as t:
            print("DİZİN artifact'i — tar içeriği:")
            for u in t.getmembers():
                print(f"  {u.size:>8} B  {u.name}")
        return
    # ikili: pdf, png, …
    hedef = Path("/tmp") / ad
    hedef.write_bytes(ham)
    print(f"ikili içerik ({tip or 'bilinmiyor'}) — {len(ham)} bayt\n"
          f"kaydedildi: {hedef}\n"
          f"aç: xdg-open {hedef}")


def bak(hedef: str, kaydet: str | None, yalniz_kunye: bool) -> None:
    kunye = _cozumle(hedef)
    aid = kunye["artifact_id"]
    tam = _al(f"/artifacts/{aid}/metadata").json()

    print("── KÜNYE " + "─" * 52)
    for alan in ("artifact_id", "name", "type", "content_type", "size_bytes",
                 "content_hash", "workflow_id", "run_id", "node_id",
                 "alias", "created_at", "parents"):
        if alan in tam and tam[alan] not in (None, [], {}):
            print(f"{alan:<14} {tam[alan]}")

    soy = _al(f"/artifacts/{aid}/lineage").json()
    ata = [d for d in soy["nodes"] if d.get("yon") == "ata"]
    print(f"{'soy':<14} {len(ata)} ebeveyn"
          + (" — " + ", ".join(d["name"] for d in ata[:5]) if ata else ""))

    ham = _al(f"/artifacts/{aid}").content
    if kaydet:
        Path(kaydet).write_bytes(ham)
        print(f"\nkaydedildi: {kaydet} ({len(ham)} bayt)")
        return
    if yalniz_kunye:
        return
    print("\n── İÇERİK " + "─" * 51)
    _goster(tam["name"], tam.get("content_type") or "", ham)


def main() -> None:
    a = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    a.add_argument("hedef", nargs="?", help="artifact_id | ad | ad@alias")
    a.add_argument("--liste", action="store_true", help="depodaki kayıtlar")
    a.add_argument("--kaydet", help="ham baytları bu dosyaya yaz")
    a.add_argument("--kunye", action="store_true", help="içeriği basma")
    n = a.parse_args()
    if n.liste or not n.hedef:
        liste()
        return
    bak(n.hedef, n.kaydet, n.kunye)


if __name__ == "__main__":
    main()
