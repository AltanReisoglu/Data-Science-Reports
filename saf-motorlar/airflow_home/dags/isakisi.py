#!/usr/bin/env python3
"""
isakisi.py — Dört saf POC'nin ORTAK İŞİ. Hiçbir orkestrasyon içermez.

Amaç: motorları kıyaslarken "iş" değişken olmasın. Herkes AYNI beş adımı
AYNI bağımlılıkla koşturacak; fark yalnız MOTORDAN gelsin.

    cek ─────► tara ──┐
                       ├──► esle ──► rapor
    test ─────────────┘

Buradaki fonksiyonlar SAF: yan etkisi yok, deterministik, birbirini tanımıyor.
Veri akışını (kimin çıktısı kime girdi olur) her motor KENDİ yöntemiyle çözecek —
asıl kıyas noktalarından biri bu.

HATA ENJEKSİYONU: `SAF_HATA` ortam değişkeni bir adım adı içeriyorsa o adım
patlar. `SAF_HATA="tara"` → geçici (ilk denemede), `SAF_HATA="tara!"` → kalıcı.
Böylece her motora AYNI hatayı verip tepkisini ölçebiliyoruz.
"""
from __future__ import annotations

import hashlib
import os
import re
import time

# Deneme sayacı: süreç-içi. Motorlar retry'ı farklı yerde yaptığı için
# "kaçıncı deneme" bilgisi motordan gelmiyor — bunu biz sayıyoruz.
_DENEME: dict[str, int] = {}


def _hata_kontrol(adim: str):
    """SAF_HATA ayarlıysa bu adımı patlat. Kalıcı ise her denemede."""
    ayar = os.environ.get("SAF_HATA", "")
    if not ayar:
        return
    kalici = ayar.endswith("!")
    hedef = ayar.rstrip("!")
    if hedef != adim:
        return
    _DENEME[adim] = _DENEME.get(adim, 0) + 1
    if kalici or _DENEME[adim] == 1:
        raise RuntimeError(
            f"{'kalıcı' if kalici else 'geçici'} hata: {adim} "
            f"(deneme {_DENEME[adim]})")


# ───────────────────────── beş adım ─────────────────────────
# Not: gerçek IO yok — kaynak metin deterministik olarak üretiliyor.
# Amaç motoru ölçmek, dosya okumak değil.

def _kaynak(path: str) -> str:
    tohum = hashlib.sha1(path.encode()).hexdigest()
    satirlar = [f"# {path}", "import os, sys"]
    for i in range(600):
        satirlar.append(f"def f_{i}(x):  # {tohum[i % 40]}")
        satirlar.append(f"    return x + {i}")
    satirlar += [
        "def login(user, password, mfa_token=None):",
        '    """BUG BURADA: mfa_token None ise kontrol atlanıyor."""',
        "    if mfa_token is None:",
        "        return True",
        "    return verify(user, password, mfa_token)",
    ]
    return "\n".join(satirlar)


def cek(path: str = "auth/login.py") -> dict:
    """Kaynağı oku, ölç. Ham metni DÖNDÜRMEZ — referans döndürür."""
    _hata_kontrol("cek")
    t = _kaynak(path)
    return {"path": path, "satir": len(t.splitlines()), "bayt": len(t),
            "sha1": hashlib.sha1(t.encode()).hexdigest()[:12], "_ham": len(t)}


def test(suite: str = "auth") -> dict:
    """Test paketini koştur, çıktıyı yapılandırılmış veriye çevir."""
    _hata_kontrol("test")
    hatalar = [
        {"test": "tests/test_auth.py::test_mfa", "sebep": "MFA baypas edilebiliyor"},
        {"test": "tests/test_auth.py::test_session", "sebep": "oturum süresi dolmuyor"},
    ]
    return {"suite": suite, "toplam": 240, "gecen": 238, "kalan": len(hatalar),
            "hatalar": hatalar}


def tara(_cek: dict, desen: str = "mfa_token") -> dict:
    """Kaynakta desen ara. GİRDİSİ cek()'in çıktısı — veri akışı buradan geçiyor."""
    _hata_kontrol("tara")
    if not isinstance(_cek, dict) or "path" not in _cek:
        raise ValueError(f"tara(): cek() çıktısı bekleniyordu, gelen: {type(_cek)}")
    t = _kaynak(_cek["path"])
    bulunan = [{"satir": i, "metin": ln.strip()[:90]}
               for i, ln in enumerate(t.splitlines(), 1)
               if re.search(desen, ln, re.I)]
    return {"desen": desen, "adet": len(bulunan), "eslesme": bulunan[:20],
            "path": _cek["path"]}


def esle(_tara: dict, _test: dict) -> dict:
    """İKİ girdiyi birleştir (join düğümü) — test hatalarını kod bulgularıyla eşle."""
    _hata_kontrol("esle")
    for ad, v, anahtar in (("tara", _tara, "eslesme"), ("test", _test, "hatalar")):
        if not isinstance(v, dict) or anahtar not in v:
            raise ValueError(f"esle(): {ad}() çıktısı eksik/bozuk: {type(v)}")
    korele = []
    for h in _test["hatalar"]:
        kelimeler = re.findall(r"[A-Za-zğüşıöçĞÜŞİÖÇ]{4,}", h["sebep"])
        kanit = next((m for m in _tara["eslesme"]
                      if any(k.lower() in m["metin"].lower() for k in kelimeler)), None)
        korele.append({"test": h["test"], "sebep": h["sebep"],
                       "kanit": kanit["metin"] if kanit else None,
                       "satir": kanit["satir"] if kanit else None})
    return {"korelasyon": korele,
            "kanitli": sum(1 for c in korele if c["kanit"]),
            "kanitsiz": sum(1 for c in korele if not c["kanit"])}


def rapor(_esle: dict, _tara: dict, _test: dict, baslik: str = "Denetim") -> dict:
    """ÜÇ girdiden markdown rapor üret."""
    _hata_kontrol("rapor")
    for ad, v, anahtar in (("esle", _esle, "korelasyon"), ("tara", _tara, "adet"),
                           ("test", _test, "toplam")):
        if not isinstance(v, dict) or anahtar not in v:
            raise ValueError(f"rapor(): {ad}() çıktısı eksik/bozuk: {type(v)}")
    satirlar = [f"# {baslik}", "",
                f"- Desen eşleşmesi: **{_tara['adet']}**",
                f"- Koşan test: **{_test['toplam']}**",
                f"- Kanıtlı hata: **{_esle['kanitli']}**", ""]
    for c in _esle["korelasyon"]:
        satirlar.append(f"## {c['test']}")
        satirlar.append(f"- Sebep: {c['sebep']}")
        satirlar.append("- Kod kanıtı: " +
                        (f"satır {c['satir']} → `{c['kanit']}`" if c["kanit"]
                         else "_bulunamadı_"))
    md = "\n".join(satirlar)
    return {"md": md, "uzunluk": len(md), "bolum": md.count("## "),
            "desen_adedi": _tara["adet"], "test_adedi": _test["toplam"]}


# ───────────────────── graf tanımı (motorlara veri olarak) ─────────────────────
# Motorlar bunu KENDİ yöntemleriyle hayata geçirecek. Ortak bir "board" DEĞİL —
# sadece "iş neydi" sorusunun tek yerde durması için.

GRAF = {
    "cek":   {"fn": cek,   "bagimli": []},
    "test":  {"fn": test,  "bagimli": []},
    "tara":  {"fn": tara,  "bagimli": ["cek"]},
    "esle":  {"fn": esle,  "bagimli": ["tara", "test"]},
    "rapor": {"fn": rapor, "bagimli": ["esle", "tara", "test"]},
}

SIRA = ["cek", "test", "tara", "esle", "rapor"]


def dogrula(sonuc: dict) -> tuple[bool, str]:
    """Akış DOĞRU sonuç üretti mi? (motorlar bunu ortak kıstas olarak kullanır)"""
    r = sonuc.get("rapor")
    if not isinstance(r, dict):
        return False, "rapor üretilmedi"
    if r.get("desen_adedi", 0) < 1:
        return False, f"desen adedi {r.get('desen_adedi')} — veri akışı KOPUK"
    if r.get("test_adedi", 0) != 240:
        return False, f"test adedi {r.get('test_adedi')} — veri akışı KOPUK"
    return True, f"{r['desen_adedi']} eşleşme · {r['test_adedi']} test · {r['uzunluk']}B"


if __name__ == "__main__":
    t0 = time.time()
    s = {}
    s["cek"] = cek()
    s["test"] = test()
    s["tara"] = tara(s["cek"])
    s["esle"] = esle(s["tara"], s["test"])
    s["rapor"] = rapor(s["esle"], s["tara"], s["test"])
    ok, not_ = dogrula(s)
    print(f"{'✓' if ok else '✗'} {not_} · {round(time.time()-t0, 3)} sn")
