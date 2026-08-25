"""
KATEGORİLER — zihniyetlerin paylaştığı ortak mantık.

Kullanıcının tasarım şartı buydu: *"frameworkler ortak olabilir, sadece
mentalitesi farklı olanlar farklı strateji olarak değerlendirilsin."*

On yedi zihniyetin çoğu aynı mekanizmayı kullanıyor ve yalnızca BİR KARARDA
ayrışıyor. Örnek: `arize-control`, `agentscope-grace`, `hermes-no-pressure`,
`claude-advisory` ve `agentbudget-dollar` — beşi de aynı sayacı tutuyor. Fark
sadece **sayaç dolduğunda ne olduğu**: sert kes · lütuf turu ver · uyarma ·
geri sayımı göster · rezerv ayır.

O yüzden ortak mantık burada, taban sınıflarda. Her strateji dosyası yalnızca
kendi ayırt edici kararını yazıyor — böylece dosyayı açan biri "bu kaynağın
farkı ne" sorusunun cevabını beş satırda görüyor.

Beş kategori, `docs/zihniyetler.md`'deki seviye merdiveniyle aynı:

    budget    SAYAC     bir sayi tut, asinca ne yapacagina karar ver
    window    PENCERE   son N olayi tut, karsilastir
    evidence  DUNYA     modelin disindan kanit al
    shape     SEKIL     donguya izin verme
    decision  KARAR     hangi eylem / hangi esik
"""

from .budget import BudgetStrategy      # noqa: F401
from .decision import DecisionStrategy  # noqa: F401
from .evidence import EvidenceStrategy  # noqa: F401
from .shape import ShapeStrategy        # noqa: F401
from .window import WindowStrategy      # noqa: F401

KATEGORI = {
    "budget": ("SAYAC", "bir sayi tut, asinca ne yapacagina karar ver"),
    "window": ("PENCERE", "son N olayi tut, karsilastir"),
    "evidence": ("DUNYA", "modelin disindan kanit al"),
    "shape": ("SEKIL", "donguye izin verme"),
    "decision": ("KARAR", "hangi eylem / hangi esik"),
    "-": ("TABAN", "kontrol yok"),
}
