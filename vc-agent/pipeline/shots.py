"""Kamera kareleri: çekimi OpenClaw yapar, dosyayı biz adlandırır ve biz sunarız.

### Neden bu bölüşüm

`/openclaw fotoğrafımı çek` cümlesi bugün de çalışıyor — OpenClaw'ın ajanının
kabuğu var ve bu makinede `exec` politikası `mode=full · ask=off`. Ama o yolda
dosyanın **nereye** düştüğüne model karar veriyor, ve modelin seçtiği bir yolu
sonradan bulup ekrana basmak iki kötü seçenek bırakıyor: ya çıktıyı ayrıştırıp
yol tahmin edeceksin, ya da bir dizini tarayacaksın.

Bu yüzden ayrımı tersine çeviriyoruz. **Kimliği ve yolu biz üretiyoruz**, komutu
tam metin olarak biz yazıyoruz, modele kalan tek şey onu çalıştırmak. `schedule`
alt komutunda verilen kararın aynısı: soru sormak için düz cümle doğru şekil, ama
sonunda *var olması gereken* bir şey için yanlış şekil.

### Yol asla dışarıdan gelmiyor

Sunucu tarafında tek bir dizin var (`~/.vcagent/shots`) ve dosyalar yalnız
**kimlikle** isteniyor. Kimlik bir uuid4 hex'i ve regex'e uyuyor; istekten gelen
metin hiçbir zaman bir yol parçası olarak kullanılmıyor. Üstüne `resolve()`
sonrası dizin kontrolü var — kuşak ve pantolon askısı, çünkü bir web sunucusunda
dosya sunan kodun tek ciddi arızası yol kaçışıdır.

### Saklama

Kamera karesi kişisel veri. Varsayılan olarak son `KEEP` kare tutuluyor, gerisi
her çekimde siliniyor; `clear()` hepsini siler. Bu bir denetim kaydı değil,
geçici bir tampon — ve öyle davranıyor.
"""

from __future__ import annotations

import re
import shutil
import uuid
from pathlib import Path
from typing import Any

import config

SHOTS: Path = config.STATE / "shots"

# uuid4().hex — 32 onaltılık karakter, başka hiçbir şey.
ID_RE = re.compile(r"^[0-9a-f]{32}$")

# Kaç kare tutulacak. Kamera karesi kişisel veri; bu bir tampon, arşiv değil.
KEEP = 5

# Çekim komutu. Tek kare, sabit çözünürlük, üzerine yaz. `-nostdin` OpenClaw'ın
# ajanı bunu bir kabukta koştururken terminali kilitlemesin diye.
CAPTURE = (
    "ffmpeg -hide_banner -loglevel error -nostdin "
    "-f v4l2 -video_size {size} -i {device} -frames:v 1 -y {path}"
)

DEFAULT_DEVICE = "/dev/video0"
DEFAULT_SIZE = "640x480"


class ShotError(ValueError):
    """İstenen kare adlandırılamıyor ya da bulunamıyor."""


def _dir() -> Path:
    SHOTS.mkdir(parents=True, exist_ok=True)
    return SHOTS


def new_id() -> str:
    return uuid.uuid4().hex


def path_for(shot_id: str) -> Path:
    """Kimlikten dosya yolu. Kimlik regex'e uymuyorsa yol hiç kurulmuyor.

    Sıra önemli: önce biçim doğrulanıyor, *sonra* birleştirme yapılıyor. Ters
    sırada `..` içeren bir kimlik önce bir yol üretir, ve o yolu sonradan
    reddetmek çok daha kolay yanlış yazılır.
    """
    if not ID_RE.match(shot_id or ""):
        raise ShotError("geçersiz kare kimliği")
    base = _dir().resolve()
    target = (base / f"{shot_id}.jpg").resolve()
    if not target.is_relative_to(base):
        raise ShotError("kare kimliği dizinin dışına çıkıyor")
    return target


def command(shot_id: str, *, device: str = DEFAULT_DEVICE,
            size: str = DEFAULT_SIZE) -> str:
    """OpenClaw'a gönderilecek tam komut metni."""
    return CAPTURE.format(size=size, device=device, path=path_for(shot_id))


def sentence(shot_id: str, *, device: str = DEFAULT_DEVICE,
             size: str = DEFAULT_SIZE) -> str:
    """Ajana gidecek cümle: komutu ver, başka bir şey isteme.

    “Fotoğrafımı çek” demiyoruz. Tam komutu veriyoruz, çünkü onaylanan şeyle
    çalışan şeyin aynı olması gerekiyor — onay kartında bu metin görünüyor.
    """
    return (
        "Şu komutu olduğu gibi çalıştır ve yalnız çıktısını bildir, "
        f"başka hiçbir şey yapma:\n{command(shot_id, device=device, size=size)}"
    )


def exists(shot_id: str) -> bool:
    try:
        return path_for(shot_id).is_file()
    except ShotError:
        return False


def read(shot_id: str) -> bytes:
    target = path_for(shot_id)
    if not target.is_file():
        raise ShotError("kare bulunamadı")
    return target.read_bytes()


def recent(limit: int = KEEP) -> list[dict[str, Any]]:
    """En yeniden eskiye, kimlik ve boyutla."""
    if not SHOTS.exists():
        return []
    items = sorted(SHOTS.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    return [
        {"id": p.stem, "bytes": p.stat().st_size, "at": int(p.stat().st_mtime)}
        for p in items[:limit]
    ]


def prune(keep: int = KEEP) -> int:
    """Son `keep` kareyi bırak, gerisini sil. Silinen sayısını döndürür."""
    if not SHOTS.exists():
        return 0
    items = sorted(SHOTS.glob("*.jpg"), key=lambda p: p.stat().st_mtime, reverse=True)
    dropped = 0
    for old in items[keep:]:
        try:
            old.unlink()
            dropped += 1
        except OSError:
            pass
    return dropped


def clear() -> int:
    """Hepsini sil. Kullanıcı “kareleri sil” dediğinde çağrılıyor."""
    if not SHOTS.exists():
        return 0
    count = len(list(SHOTS.glob("*.jpg")))
    shutil.rmtree(SHOTS, ignore_errors=True)
    return count


def local_capture(shot_id: str, *, device: str = DEFAULT_DEVICE,
                  size: str = DEFAULT_SIZE, timeout: float = 20.0) -> dict[str, Any]:
    """Yedek yol: ffmpeg'i doğrudan biz koşturuyoruz.

    OpenClaw Gateway kapalıyken ya da kotası dolmuşken kare yine de alınabilsin
    diye. Aynı dizine, aynı adlandırmayla yazıyor — sunma tarafı ikisini
    ayırt etmiyor, ve ayırt etmemeli.
    """
    import subprocess

    target = path_for(shot_id)
    _dir()
    proc = subprocess.run(
        command(shot_id, device=device, size=size).split(),
        capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0 or not target.is_file():
        return {"ok": False, "id": shot_id,
                "error": (proc.stderr or "ffmpeg kare üretmedi").strip()[:300]}
    prune()
    return {"ok": True, "id": shot_id, "bytes": target.stat().st_size, "by": "local"}


__all__ = [
    "CAPTURE", "DEFAULT_DEVICE", "DEFAULT_SIZE", "ID_RE", "KEEP", "SHOTS",
    "ShotError", "clear", "command", "exists", "local_capture", "new_id",
    "path_for", "prune", "read", "recent", "sentence",
]
