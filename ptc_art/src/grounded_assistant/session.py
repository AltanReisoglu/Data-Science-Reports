"""Oturum kimliği — artifact kapsamının ve konuşma hafızasının ortak anahtarı.

## Neden ayrı bir modül

Bu fonksiyonu hem `web/app.py` hem `cli.py` kullanıyor. `web/app.py` zaten
`cli.py`'den import ediyor (`_build_answer`), dolayısıyla ters yönde bir import
döngü yaratırdı. İkisinin de altında duran nötr bir yer gerekiyordu.

## Ne işe yarıyor

`session_id` TEK bir değer ama İKİ yere birden gidiyor:

    session_id ──> thread_id    (konuşma hafızası — checkpointer)
               └─> workflow_id  (artifact kapsamı — MinIO + kayıt defteri)

2026-09-04 öncesinde her bağlantıda/çağrıda `uuid4()` ile yeniden üretiliyordu.
Konuşma hafızasının gitmesi beklenen bir şeydi; asıl sorun ikincisiydi:
artifact'ler depoda SAĞ KALIYOR ama onları gösteren `workflow_id` bir daha asla
üretilmediği için **erişilemez** hale geliyorlardı — kalıcı bir depoya yazıp
okuma anahtarını çöpe atmak.
"""

from __future__ import annotations

import re
import uuid

#: Kabul edilen biçim — UUID.
#:
#: DAR OLMASI ŞART: bu değer `workflow_id` olarak artifact deposuna gidiyor ve
#: orada S3 anahtarının bir parçası oluyor (`{owner}/{workflow}/{node}/{run}/…`).
#: Çağıranın verdiği serbest metni oraya geçirmek, anahtar düzenini ona
#: yazdırmak olurdu. Serviste ayrıca isim doğrulaması var ama savunma kimliği
#: ilk kabul eden yerde de olmalı.
_OTURUM_BICIMI = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


def oturum_kimligi(disaridan: str | None) -> str:
    """Verilen kimliği kabul eder, yoksa/bozuksa yenisini üretir.

    Anthropic'in `container.id` modeli: kimliği istemci saklar (web'de
    `localStorage`, CLI'de `--session`), yeniden bağlanınca geri gönderir.
    Göndermezse temiz oturum açılır — **kalıcılık opt-in kalıyor**, kimse
    istemeden başkasının oturumuna düşmüyor.

    SINIR — bu bir kimlik doğrulama DEĞİL: kimliği bilen herkes o oturumun
    artifact'lerini okuyabilir. Uuid tahmin edilemez olduğu için PoC'de yeterli,
    üretimde gerçek auth'a bağlanmalı (karar dokümanı §2.1).
    """
    if disaridan and _OTURUM_BICIMI.match(disaridan):
        return disaridan.lower()
    return str(uuid.uuid4())
