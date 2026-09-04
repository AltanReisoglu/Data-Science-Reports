"""Kapsam jetonu — sandbox'ın hangi workflow adına konuştuğunu kanıtlar.

## Neden gerekli

`ArtifactService` her çağrıda "bu artifact çağıranın workflow'una ait mi" diye
bakıyor. Ama kapsamı çağıranın KENDİSİ bildiriyorsa bu kontrol dekoratiftir:
sandbox'ta çalışan kod LLM tarafından yazılmıştır ve `workflow_id="wf_baskasi"`
yazmasını engelleyen hiçbir şey yoktur.

Araştırma dokümanı §6.1'in tespiti tam buydu: paylaşılan bir depo, Cilium'un
GÖREMEDİĞİ bir çalıştırma-arası kanal açar; yetkilendirme ağ katmanında değil,
uygulama katmanında kurulmak zorundadır. Bu modül o kurulumun temeli.

## Nasıl çalışıyor

`sandbox_runner` (laptop'ta, sandbox'ın erişemediği yerde) her çalıştırma için
bir jeton **imzalar** ve Job'un ortam değişkenine koyar. Sandbox jetonu her
artifact çağrısında geri gönderir; Tool Gateway **imzayı doğrulayıp** kapsamı
jetondan okur — çağıranın söylediğinden değil.

Ortak sır bir Kubernetes Secret'ında; gateway'e mount edilir, runner cluster
API'sinden okur. Sandbox'ın eline yalnızca İMZALANMIŞ jeton geçer, sır geçmez.

**Sandbox jetonu okuyabilir** (kendi ortam değişkeni), ama başka bir workflow
için geçerli bir jeton **üretemez** — imza anahtarı onda yok. Capability
modelinin kuralı: yetki taklit edilemez bir referansla taşınır, ve sahibi onu
zayıflatabilir, güçlendiremez.

## Sınırı açıkça yazalım

Jeton, aynı sandbox içinde çalışan koda karşı bir sır değildir. Koruduğu şey
**workflow'lar arası** sınırdır, sandbox içi değil. Sandbox içinde zaten tek bir
güven alanı var (kodun tamamını aynı LLM yazdı).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

_ALG = hashlib.sha256

#: Jeton ömrü. Bir sandbox çalıştırması `activeDeadlineSeconds: 30` ile sınırlı;
#: 15 dakika fazlasıyla geniş ama saat kaymalarına dayanıklı.
VARSAYILAN_OMUR_SANIYE = 900


class InvalidScopeToken(Exception):
    """Jeton bozuk, imzası tutmuyor ya da süresi dolmuş."""


@dataclass(frozen=True)
class Scope:
    """Bir sandbox çalıştırmasının doğrulanmış kimliği."""

    workflow_id: str
    run_id: str
    owner: str
    node_id: str | None = None


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def issue_token(
    secret: str,
    scope: Scope,
    *,
    ttl_seconds: int = VARSAYILAN_OMUR_SANIYE,
    now: datetime | None = None,
) -> str:
    """`sandbox_runner` çağırır — Job'un ortamına konacak jetonu üretir."""
    su_an = now or datetime.now(UTC)
    payload = {
        "workflow_id": scope.workflow_id,
        "run_id": scope.run_id,
        "owner": scope.owner,
        "node_id": scope.node_id,
        "exp": (su_an + timedelta(seconds=ttl_seconds)).timestamp(),
    }
    govde = _b64(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    return f"{govde}.{_b64(_imza(secret, govde))}"


def verify_token(secret: str, token: str, *, now: datetime | None = None) -> Scope:
    """Tool Gateway çağırır — kapsamı jetondan okur, çağıranın iddiasından değil."""
    try:
        govde, imza = token.split(".", 1)
        payload = json.loads(_unb64(govde))
    except (ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise InvalidScopeToken("Jeton biçimi bozuk") from exc

    # compare_digest: imzayı bayt bayt karşılaştırmanın zamanlaması üzerinden
    # sızdırmamak için.
    if not hmac.compare_digest(_b64(_imza(secret, govde)), imza):
        raise InvalidScopeToken("Jeton imzası doğrulanamadı")

    su_an = (now or datetime.now(UTC)).timestamp()
    if float(payload.get("exp", 0)) < su_an:
        raise InvalidScopeToken("Jetonun süresi dolmuş")

    return Scope(
        workflow_id=payload["workflow_id"],
        run_id=payload["run_id"],
        owner=payload["owner"],
        node_id=payload.get("node_id"),
    )


def _imza(secret: str, govde: str) -> bytes:
    return hmac.new(secret.encode(), govde.encode(), _ALG).digest()
