"""Artifact serileştirme — Parquet/Arrow, ve pickle'a kapalı kapı.

## Format seçimi (araştırma dokümanı §4.1)

| Format | Ne için |
|---|---|
| **Parquet** | Varsayılan. En küçük, şemalı, sütunlu — dataframe'in evi |
| **Arrow IPC** | Hızlı devir gerekiyorsa; tüm Arrow tiplerini korur |
| JSON | Küçük sözlük/liste ve metadata |
| metin / ham bayt | Rapor, log, resim |

## pickle neden burada YOK

pickle'ın deserialization'ı tasarım gereği KOD ÇALIŞTIRIR (CWE-502). Bizde bu
teorik bir risk değil: artifact'i yazan taraf **LLM'in ürettiği koddur**. Bir
sonraki adımın onu pickle ile geri yüklemesi, "LLM'in ürettiği veriyi kod olarak
çalıştır" demektir.

Bu yüzden iki taraflı savunma var:
  - **Yazarken**: pickle bir çıktı formatı olarak hiç sunulmuyor.
  - **Okurken**: hem content_type kontrol ediliyor, hem de baytların başına
    bakılıp pickle imzası taşıyanlar reddediliyor — yani depoya başka bir yoldan
    pickle girmiş olsa bile buradan geçemez.

Dagster'ın *varsayılan* IO manager'ının pickle yazdığını hatırlayın: "hazır olanı
kullan" yolu doğrudan bu riskin içine giriyor. Format açıkça seçilmeli.
"""

from __future__ import annotations

import json
from typing import Any

PARQUET = "application/vnd.apache.parquet"
ARROW = "application/vnd.apache.arrow.file"
JSON = "application/json"
TEXT = "text/plain"
BINARY = "application/octet-stream"

#: Dosya uzantısından content_type. `/output` süpürmesi için: LLM sıradan kod
#: yazıp `df.to_csv("/output/rapor.csv")` dediğinde, dosyanın ne olduğunu
#: uzantısından anlamak zorundayız — elimizde başka ipucu yok.
#:
#: `.csv` bilerek TEXT'e eşleniyor (`text/csv` değil): böylece geri okunduğunda
#: `str` olarak gelir ve çağıran doğrudan parse edebilir. Tanımadığımız her şey
#: BINARY — PDF, PNG, zip hepsi olduğu gibi saklanır, çözülmeye çalışılmaz.
_UZANTI_TIPLERI = {
    ".parquet": "application/vnd.apache.parquet",
    ".arrow": "application/vnd.apache.arrow.file",
    ".json": "application/json",
    ".csv": "text/csv",
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".html": "text/html",
}


def content_type_for_filename(filename: str) -> str:
    """Uzantıdan content_type; bilinmiyorsa ham bayt."""
    nokta = filename.rfind(".")
    uzanti = filename[nokta:].lower() if nokta > 0 else ""
    return _UZANTI_TIPLERI.get(uzanti, BINARY)


#: `pickle.dumps` protokol 2-5 çıktısı `\x80` + protokol numarası ile başlar.
_PICKLE_MAGIC = tuple(bytes([0x80, p]) for p in range(2, 6))

#: Adında pickle geçen her content_type reddedilir (pkl, x-pickle, ...).
_YASAKLI_ANAHTAR_KELIMELER = ("pickle", "pkl")


class UnsafeArtifact(Exception):
    """Güvensiz bir serileştirme biçimi tespit edildi (CWE-502)."""


def _pickle_mi(data: bytes, content_type: str) -> str | None:
    """Reddetme gerekçesini döner; güvenliyse None."""
    dusuk = content_type.lower()
    for kelime in _YASAKLI_ANAHTAR_KELIMELER:
        if kelime in dusuk:
            return f"content_type '{content_type}' pickle'a işaret ediyor"
    if data[:2] in _PICKLE_MAGIC:
        return "baytlar pickle imzasıyla başlıyor (\\x80 + protokol)"
    return None


def guvenlik_kontrolu(data: bytes, content_type: str) -> None:
    """Pickle'ı hem etiketinden hem imzasından yakalar. Depoya giren ve depodan
    çıkan HER artifact bu kapıdan geçer."""
    gerekce = _pickle_mi(data, content_type)
    if gerekce is not None:
        raise UnsafeArtifact(
            f"Artifact reddedildi — {gerekce}. pickle deserialization kod "
            "çalıştırır (CWE-502) ve bu artifact'i LLM'in ürettiği kod yazmış "
            "olabilir. Dataframe için Parquet, ham veri için JSON/metin kullanın."
        )


def serialize(value: Any, prefer: str = PARQUET) -> tuple[bytes, str]:
    """Bir Python değerini (bayt, content_type) ikilisine çevirir.

    DataFrame ise `prefer` (Parquet ya da Arrow IPC) uygulanır; diğer tipler
    kendi doğal formatlarına gider.
    """
    if isinstance(value, bytes):
        return value, BINARY
    if isinstance(value, str):
        return value.encode("utf-8"), TEXT

    if _dataframe_mi(value):
        return _dataframe_serialize(value, prefer)

    # Kalanlar: dict, list, sayı, bool, None → JSON.
    # Bilerek `default=str` YOK — sessizce string'e düşen bir nesne, tip
    # bilgisinin kaybolduğu yerdir (araştırma §4.3). Serileştirilemiyorsa
    # hata versin, çağıran açıkça karar versin.
    return json.dumps(value, ensure_ascii=False).encode("utf-8"), JSON


def deserialize(data: bytes, content_type: str) -> Any:
    """Baytları geri çevirir. Pickle imzalı içerik BURADAN GEÇEMEZ."""
    guvenlik_kontrolu(data, content_type)

    if content_type == PARQUET:
        import io  # noqa: PLC0415

        import pandas as pd  # noqa: PLC0415

        return pd.read_parquet(io.BytesIO(data))
    if content_type == ARROW:
        import io  # noqa: PLC0415

        import pyarrow.ipc as ipc  # noqa: PLC0415

        with ipc.open_file(io.BytesIO(data)) as reader:
            return reader.read_pandas()
    if content_type == JSON:
        return json.loads(data.decode("utf-8"))
    # `text/*` — yalnızca `text/plain` değil.
    #
    # NEDEN (2026-09-04'te bulunan gerçek tutarsızlık): `.csv` eskiden
    # `text/plain`e eşleniyordu, bu yüzden `df.to_csv("/output/x.csv")` ile
    # süpürülen bir dosya `system.Artifact` oluyordu; aynı veriyi
    # `put_artifact(df, ...)` ile yazınca `system.Dataset` oluyordu. AYNI veri,
    # iki farklı tip — sırf hangi yoldan geçtiğine göre.
    #
    # Düzeltme `.csv`yi `text/csv`ye taşımaktı; ama o zaman burada TEXT
    # eşitliği tutmuyor ve içerik metin yerine ham bayt olarak dönüyordu.
    # Aile kontrolü ikisini birden çözüyor.
    if content_type.startswith("text/"):
        return data.decode("utf-8")
    return data


def _dataframe_mi(value: Any) -> bool:
    """pandas'ı import ETMEDEN dataframe tespiti — bu modülü metadata
    katmanının pandas'a bağımlı hale getirmemek için.

    MRO taranıyor, tek bir modül yoluna bakılmıyor: pandas 2.x sınıfı
    `pandas.core.frame.DataFrame`, pandas 3.x ise `pandas.DataFrame` olarak
    raporluyor (3.0.5 ile doğrulandı). Sabit yola bakan bir kontrol sürüm
    yükseltmesinde sessizce bozulur — dataframe JSON'a düşer ve
    "not JSON serializable" ile patlar. Alt sınıflar da böylece kapsanıyor."""
    return any(
        cls.__module__.split(".")[0] == "pandas" and cls.__qualname__ == "DataFrame"
        for cls in type(value).__mro__
    )


def _dataframe_serialize(df: Any, prefer: str) -> tuple[bytes, str]:
    import io  # noqa: PLC0415

    buf = io.BytesIO()
    if prefer == ARROW:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.ipc as ipc  # noqa: PLC0415

        tablo = pa.Table.from_pandas(df)
        with ipc.new_file(buf, tablo.schema) as writer:
            writer.write_table(tablo)
        return buf.getvalue(), ARROW

    df.to_parquet(buf)  # motor: pyarrow (imajda kurulu)
    return buf.getvalue(), PARQUET


# ---------------------------------------------------------------------------
# KFP tip sözlüğü (2026-09-04, OpenShift hizalaması)
#
# Burada duruyor çünkü bu dosya sandbox imajına AYNEN kopyalanıyor — yani
# sandbox ile servis tipleri aynı kurala göre çıkarıyor. `service.py`'ye
# koysaydık sandbox'tan erişilemezdi ve iki taraf ayrışırdı.
# ---------------------------------------------------------------------------

TIPLER = (
    "system.Artifact",   # taban tip
    "system.Dataset",
    "system.Model",
    "system.Metrics",
    "system.HTML",
    "system.Markdown",
)
VARSAYILAN_TIP = "system.Artifact"

_TIP_ESLEME = {
    PARQUET: "system.Dataset",
    ARROW: "system.Dataset",
    "text/csv": "system.Dataset",
    "text/html": "system.HTML",
    "text/markdown": "system.Markdown",
}


def tip_cikar(content_type: str, deger=None) -> str:
    """İçerikten KFP tipini tahmin eder.

    NEDEN OTOMATİK: LLM'in her `put_artifact` çağrısında tip yazmasını beklemek,
    tam da unutulacak türden bir yük. Açıkça verilirse o kazanır.

    NEDEN `deger` PARAMETRESİ VAR: bir sayısal sözlüğün "metrik" olduğu ancak
    NESNEYE bakarak anlaşılır — serileştirildikten sonra o da sadece
    `application/json`. Bu yüzden çıkarımın değerli hâli SANDBOX tarafında,
    nesne hâlâ elde iken çalışıyor. Servis yalnızca baytı gördüğü için
    `deger=None` ile çağırıyor ve content_type'a düşüyor.
    """
    if content_type in _TIP_ESLEME:
        return _TIP_ESLEME[content_type]
    if isinstance(deger, dict) and deger and all(
        isinstance(v, (int, float)) and not isinstance(v, bool) for v in deger.values()
    ):
        return "system.Metrics"
    return VARSAYILAN_TIP
