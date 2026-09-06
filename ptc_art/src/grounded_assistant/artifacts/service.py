"""Artifact Service — metadata + nesne deposu + serileştirmeyi birleştiren katman.

`services/artifact_service/app.py` bunu HTTP olarak dışarı açar. (2026-09-04'e
kadar Tool Gateway MCP tool'ları olarak açıyordu; baytlar base64'e çevrilip MCP
gövdesinde taşındığı için hem %33 şişiyor hem iki uçta belleğe alınıyordu. O
gün alınan kararla artifact işi gateway'den ayrıldı ve taşıma akışlı HTTP'ye
geçti — gateway artık yalnızca tool proxy'si.)

Sandbox'ın gördüğü tek şey opak `artifact_id`'lerdir; bucket, anahtar düzeni ve
kimlik bilgisi buradan dışarı çıkmaz (capability / unforgeable reference
deseni).

## Burada uygulanan dört kontrol

1. **Kapsam** — `artifact_id` çağıranın workflow'una ait mi? Ağ politikası bu
   akışı GÖREMEZ (paylaşılan depo, NIST'in "covert storage channel" tanımı),
   yetkilendirmenin yapılabileceği tek yer burası.
2. **İsim** — yol geçişi yok. `artifact_save("/etc/shadow")` sınıfı.
3. **Boyut** — üst sınır.
4. **Format** — pickle hem etiketinden hem bayt imzasından reddedilir.

## Dedup

Aynı içerik (aynı `content_hash`) bu workflow'da zaten varsa bayt YENİDEN
YÜKLENMEZ; yeni kayıt var olan `storage_uri`'yi gösterir. Metaflow'un
content-addressed datastore'unun küçük hâli. Kimlik yine yeni: artifact'ler
değişmez, "aynı içerik" ile "aynı artifact" farklı şeylerdir.
"""

from __future__ import annotations

import hashlib
import os
import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from grounded_assistant.artifacts.metadata import ArtifactMeta, MetadataStore
from grounded_assistant.artifacts.serialize import (
    PARQUET,
    TIPLER,
    VARSAYILAN_TIP,
    deserialize,
    guvenlik_kontrolu,
    serialize,
    tip_cikar,
    uzanti_icin,
)
from grounded_assistant.artifacts.store import ObjectStore

#: Harf/rakam/nokta/tire/alt tire. Bilerek DAR: `/`, `..`, boşluk, mutlak yol yok.
#: Nokta serbest çünkü node'lar arası konvansiyon böyle: "extract.tickets".
_GECERLI_ISIM = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


VARSAYILAN_BOYUT_SINIRI = 100 * 1024 * 1024  # 100 MiB

#: Bucket içindeki kök yol — KFP'nin `pipeline_root`'unun karşılığı.
#:
#: OpenShift/KFP mentalitesinde "bucket'ın neresine yazılacağı" KOD değil
#: YAPILANDIRMA: KFP bunu üç düzeyde ayarlatıyor (deployment ConfigMap'i,
#: pipeline başına, çalıştırma başına). Bizde ilk ikisi var — ortam
#: değişkeniyle dağıtım varsayılanı, ve `create*(root=...)` ile çağrı başına.
#:
#: Boş bırakılırsa anahtar eskisi gibi `{owner}/...` ile başlar; yani var olan
#: kurulumlar etkilenmez.
VARSAYILAN_KOK = os.environ.get("PTC_ARTIFACT_ROOT", "").strip("/")

class ScopeViolation(Exception):
    """Başka bir workflow'un artifact'ine erişim denendi."""


class InvalidArtifactName(Exception):
    """İsim yol geçişi içeriyor ya da biçime uymuyor."""


class ArtifactTooLarge(Exception):
    """Boyut sınırı aşıldı."""


@dataclass
class ArtifactService:
    metadata: MetadataStore
    objects: ObjectStore
    size_limit: int = VARSAYILAN_BOYUT_SINIRI

    # -- yazma ------------------------------------------------------------

    def create(
        self,
        value: Any,
        *,
        name: str,
        workflow_id: str,
        run_id: str,
        owner: str,
        node_id: str | None = None,
        parents: tuple[str, ...] = (),
        ttl_seconds: int | None = None,
        prefer: str = PARQUET,
        artifact_type: str | None = None,
        user_metadata: dict | None = None,
        root: str | None = None,
    ) -> ArtifactMeta:
        """Python değerinden artifact üretir; serileştirmeyi bu taraf yapar."""
        data, content_type = serialize(value, prefer=prefer)
        return self.create_bytes(
            data, content_type, name=name, workflow_id=workflow_id, run_id=run_id,
            owner=owner, node_id=node_id, parents=parents, ttl_seconds=ttl_seconds,
            artifact_type=artifact_type or tip_cikar(content_type, value),
            user_metadata=user_metadata, root=root,
        )

    def create_bytes(
        self,
        data: bytes,
        content_type: str,
        *,
        name: str,
        workflow_id: str,
        run_id: str,
        owner: str,
        node_id: str | None = None,
        parents: tuple[str, ...] = (),
        ttl_seconds: int | None = None,
        artifact_type: str | None = None,
        user_metadata: dict | None = None,
        root: str | None = None,
    ) -> ArtifactMeta:
        """Hazır baytlardan artifact üretir — **gateway'in kullandığı yol**.

        Serileştirme sandbox içinde yapılır (dataframe MCP'den nesne olarak
        geçemez), gateway'e base64 bayt gelir. pickle kontrolü burada TEKRAR
        uygulanır: baytları gönderen taraf LLM'in ürettiği koddur.
        """
        _isim_dogrula(name)
        guvenlik_kontrolu(data, content_type)

        if len(data) > self.size_limit:
            raise ArtifactTooLarge(
                f"{len(data)} bayt > {self.size_limit} sınırı. Veriyi sandbox içinde "
                "süzüp özetleyin — PTC'nin bütün amacı bu."
            )

        content_hash = "sha256:" + hashlib.sha256(data).hexdigest()
        artifact_id = "art_" + uuid.uuid4().hex[:12]

        # Dedup: aynı içerik bu workflow'da varsa baytı tekrar yükleme.
        ikiz = self.metadata.find_by_hash(workflow_id, content_hash)
        if ikiz is not None:
            storage_uri = ikiz.storage_uri
        else:
            key = self._anahtar(
                owner, workflow_id, node_id, run_id, artifact_id, content_type,
                VARSAYILAN_KOK if root is None else _kok_dogrula(root.strip("/")),
            )
            storage_uri = self.objects.put(key, data, content_type)

        return self.metadata.create(
            ArtifactMeta(
                artifact_id=artifact_id,
                name=name,
                workflow_id=workflow_id,
                run_id=run_id,
                node_id=node_id,
                content_hash=content_hash,
                content_type=content_type,
                size_bytes=len(data),
                storage_uri=storage_uri,
                parents=parents,
                owner=owner,
                created_at=datetime.now(UTC),
                ttl_seconds=ttl_seconds,
                artifact_type=_tip_dogrula(artifact_type or tip_cikar(content_type)),
                user_metadata=user_metadata or {},
            )
        )

    def delete(self, artifact_id: str, *, owner: str) -> bool:
        """ÖNCE bayt, SONRA metadata.

        Ters sıra (MLflow'un sorunu) görünmez ama ücreti işleyen yetim blob
        bırakır. Bu sıra ise en kötü ihtimalle SARKAN REFERANS bırakır — o
        tespit edilebilir ve onarılabilir.

        Dedup nedeniyle aynı baytı birden çok kayıt gösteriyor olabilir; o
        durumda bayt SİLİNMEZ, yalnızca bu kaydın referansı düşer.
        """
        meta = self._yetkili_meta(artifact_id, owner)
        paylasan = [
            m
            for m in self.metadata.list_for_owner(owner, limit=10_000)
            if m.storage_uri == meta.storage_uri and m.artifact_id != artifact_id
        ]
        if not paylasan:
            self.objects.delete(_anahtar_ayikla(meta.storage_uri))
        return self.metadata.delete(artifact_id)

    # -- okuma ------------------------------------------------------------

    def get(self, artifact_id: str, *, owner: str) -> Any:
        meta = self._yetkili_meta(artifact_id, owner)
        return self._yukle(meta)

    def get_by_name(self, name: str, *, owner: str,
                    workflow_id: str | None = None) -> Any | None:
        """İsimle en yeni sürüm — çalıştırmalar arası devrin ana yolu."""
        meta = self.metadata.latest_by_name_for_owner(owner, name, workflow_id)
        return None if meta is None else self._yukle(meta)

    def get_bytes(
        self, *, owner: str, workflow_id: str | None = None,
        artifact_id: str | None = None, name: str | None = None,
    ) -> tuple[bytes, ArtifactMeta] | None:
        """Ham baytlar + metadata — **gateway'in kullandığı yol**.

        `artifact_id` verilirse kimlikle, `name` verilirse o isimdeki EN YENİ
        sürümle çözer. Çözümleme sandbox tarafında yapılır, gateway baytı
        olduğu gibi geçirir.
        """
        if artifact_id is not None:
            meta = self._yetkili_meta(artifact_id, owner)
        elif name is not None:
            meta = self.metadata.latest_by_name_for_owner(owner, name, workflow_id)
            if meta is None:
                return None
        else:
            raise ValueError("artifact_id ya da name verilmeli")
        return self.objects.get(_anahtar_ayikla(meta.storage_uri)), meta

    def resolve(
        self, *, owner: str, workflow_id: str | None = None,
        artifact_id: str | None = None, name: str | None = None,
        strict: bool = False,
    ) -> ArtifactMeta | None:
        """Kimlik ya da isimden metadata çözer — baytlara DOKUNMADAN.

        HTTP katmanı önce bunu çağırır: 404'ü ve başlıkları baytları hiç
        okumadan üretebilmek için.
        """
        if artifact_id is not None:
            return self._yetkili_meta(artifact_id, owner)
        if name is not None:
            if strict:
                if not workflow_id:
                    return None
                return self.metadata.latest_in_workflow(owner, workflow_id, name)
            return self.metadata.latest_by_name_for_owner(owner, name, workflow_id)
        raise ValueError("artifact_id ya da name verilmeli")

    def iter_bytes(self, meta: ArtifactMeta, chunk_size: int = 1024 * 1024):
        """Çözülmüş bir metadata'nın baytlarını parça parça verir.

        `resolve` ile birlikte `get_bytes`'ın akışlı karşılığı: 100 MiB'lik bir
        artifact indirilirken servis belleği sabit kalır.
        """
        return self.objects.iter_get(_anahtar_ayikla(meta.storage_uri), chunk_size)

    def create_stream(
        self,
        fileobj,
        length: int,
        content_type: str,
        *,
        name: str,
        workflow_id: str,
        run_id: str,
        owner: str,
        content_hash: str,
        node_id: str | None = None,
        parents: tuple[str, ...] = (),
        ttl_seconds: int | None = None,
        artifact_type: str | None = None,
        user_metadata: dict | None = None,
        root: str | None = None,
    ) -> ArtifactMeta:
        """`create_bytes`'ın akışlı hâli — baytlar dosya nesnesinde, bellekte değil.

        Çağıran (HTTP katmanı) gövdeyi zaten parça parça okurken üç şeyi
        yapmış olmalı: pickle imzası kontrolü, boyut sınırı, ve sha256'yı
        akış boyunca biriktirmek. Bu yüzden `content_hash` DIŞARIDAN geliyor
        ve `guvenlik_kontrolu` burada tekrar çalıştırılmıyor — tüm baytı
        yeniden okumak, akışlı olmanın bütün anlamını götürürdü.

        İsim ve content_type doğrulaması yine BURADA: onlar ucuz ve bu sınıf
        tek yetkilendirme noktası.
        """
        _isim_dogrula(name)
        if length > self.size_limit:
            raise ArtifactTooLarge(f"{length} bayt > {self.size_limit} sınırı.")

        artifact_id = "art_" + uuid.uuid4().hex[:12]

        ikiz = self.metadata.find_by_hash(workflow_id, content_hash)
        if ikiz is not None:
            storage_uri = ikiz.storage_uri
        else:
            key = self._anahtar(
                owner, workflow_id, node_id, run_id, artifact_id, content_type,
                VARSAYILAN_KOK if root is None else _kok_dogrula(root.strip("/")),
            )
            storage_uri = self.objects.put_stream(key, fileobj, length, content_type)

        return self.metadata.create(
            ArtifactMeta(
                artifact_id=artifact_id,
                name=name,
                workflow_id=workflow_id,
                run_id=run_id,
                node_id=node_id,
                content_hash=content_hash,
                content_type=content_type,
                size_bytes=length,
                storage_uri=storage_uri,
                parents=parents,
                owner=owner,
                created_at=datetime.now(UTC),
                ttl_seconds=ttl_seconds,
                artifact_type=_tip_dogrula(artifact_type or tip_cikar(content_type)),
                user_metadata=user_metadata or {},
            )
        )

    def metadata_of(self, artifact_id: str, *, owner: str) -> ArtifactMeta:
        return self._yetkili_meta(artifact_id, owner)

    def list(self, *, owner: str, limit: int = 200) -> list[ArtifactMeta]:
        """Tenant'ta ne var — manifestin kaynağı, workflow'lar arası."""
        return self.metadata.list_for_owner(owner, limit=limit)

    def lineage(self, artifact_id: str, *, owner: str, limit: int = 1000) -> dict:
        """Bir artifact'in soy ağacı: yukarı ATALAR, aşağı ÜRÜNLER.

        NEDEN BELLEKTE: `parents` bir JSON metin sütunu, yani "beni ebeveyn
        gösterenler" sorgusu SQL'de `LIKE '%id%'` olurdu — indekssiz, ve
        `art_abc` ile `art_abcdef` gibi önek çakışmalarına açık. Bir
        workflow'un artifact sayısı üç haneli bile değil; grafiği tek
        listeden kurmak hem doğru hem yeterince hızlı. Postgres'e geçince
        kenarları ayrı tabloya alma seçeneği açık kalıyor.

        Kapsam: YALNIZCA bu workflow. `_yetkili_meta` kökü zaten doğruluyor;
        kenarlar da aynı listeden kurulduğu için başka bir workflow'a sızma
        yolu yok.
        """
        kok = self._yetkili_meta(artifact_id, owner)
        hepsi = {m.artifact_id: m for m in
                 self.metadata.list_for_owner(owner, limit=limit)}
        hepsi.setdefault(kok.artifact_id, kok)

        # Kenarlar: yalnızca İKİ UCU DA bu workflow'da olanlar. Süresi dolup
        # silinmiş bir ebeveyne işaret eden kenar çizilmez — düğümsüz kenar
        # grafikte kopuk bir ok olurdu.
        cocuklar: dict[str, list[str]] = {}
        ebeveynler: dict[str, list[str]] = {}
        for m in hepsi.values():
            for ebeveyn in m.parents:
                if ebeveyn in hepsi:
                    cocuklar.setdefault(ebeveyn, []).append(m.artifact_id)
                    ebeveynler.setdefault(m.artifact_id, []).append(ebeveyn)

        def gez(baslangic: str, komsuluk: dict[str, list[str]]) -> dict[str, int]:
            """BFS — derinlik döner. Döngü olsa bile `gorulen` sonsuza sokmaz."""
            gorulen: dict[str, int] = {}
            sira = [(baslangic, 0)]
            while sira:
                dugum, d = sira.pop(0)
                for komsu in komsuluk.get(dugum, ()):
                    if komsu not in gorulen and komsu != baslangic:
                        gorulen[komsu] = d + 1
                        sira.append((komsu, d + 1))
            return gorulen

        atalar = gez(kok.artifact_id, ebeveynler)
        urunler = gez(kok.artifact_id, cocuklar)

        dugumler = [_soy_dugumu(kok, 0, "kok")]
        for kimlik, d in sorted(atalar.items(), key=lambda kv: kv[1]):
            dugumler.append(_soy_dugumu(hepsi[kimlik], -d, "ata"))
        for kimlik, d in sorted(urunler.items(), key=lambda kv: kv[1]):
            dugumler.append(_soy_dugumu(hepsi[kimlik], d, "urun"))

        ilgili = {d["artifact_id"] for d in dugumler}
        kenarlar = [
            {"from": e, "to": c}
            for e, cs in sorted(cocuklar.items())
            for c in sorted(cs)
            if e in ilgili and c in ilgili
        ]
        return {"root": kok.artifact_id, "nodes": dugumler, "edges": kenarlar}

    # -- iç ---------------------------------------------------------------

    def _yetkili_meta(self, artifact_id: str, owner: str) -> ArtifactMeta:
        """Yetki sınırı **tenant** (2026-09-06'da workflow'dan genişletildi).

        KFP'de izolasyon sınırı namespace: bütün çalıştırmalar aynı
        `pipeline_root` altına yazar ve bir bileşen başka bir run'ın çıktısını
        okuyabilir. Çalıştırma başına mühürlemek bizim eklediğimiz bir şeydi;
        istenen davranış "başka workflow artifact'ini görebilsin ve
        kullanabilsin" olduğu için sınır oraya taşındı.

        Yerleştirme hâlâ çalıştırma başına: anahtar yolu
        `{owner}/{workflow}/{node}/{run}/...` — yani kim ne zaman üretti bilgisi
        kaybolmuyor, yalnızca okuma kapısı açılıyor.
        """
        meta = self.metadata.get(artifact_id)
        if meta is None or meta.owner != owner:
            # Var-ama-yetkisiz ile hiç-yok ayrımı BİLEREK yapılmıyor: aksi halde
            # çağıran, başka workflow'larda hangi id'lerin var olduğunu
            # deneyerek öğrenebilirdi.
            raise ScopeViolation(
                f"{artifact_id} bu tenant'ta bulunamadı ({owner})."
            )
        return meta

    def _yukle(self, meta: ArtifactMeta) -> Any:
        data = self.objects.get(_anahtar_ayikla(meta.storage_uri))
        return deserialize(data, meta.content_type)

    @staticmethod
    def _anahtar(
        owner: str,
        workflow_id: str,
        node_id: str | None,
        run_id: str,
        artifact_id: str,
        content_type: str,
        kok: str = "",
    ) -> str:
        """`{owner}/{workflow}/{node}/{run}/{artifact_id}{uzantı}`

        Argo'nun anahtarı `{{workflow.uid}}` ile parametreleme pratiğinin
        karşılığı; eşzamanlı çalıştırmalar çakışmaz. Kullanıcı/workflow'a çapalı
        olması, durable deponun uçucu bir anahtara bağlı kalmamasını sağlar.
        """
        govde = (
            f"{owner}/{workflow_id}/{node_id or '_'}/{run_id}/"
            f"{artifact_id}{uzanti_icin(content_type)}"
        )
        return f"{kok}/{govde}" if kok else govde


def _tip_dogrula(tip: str) -> str:
    """Bilinmeyen tipi sessizce taban tipe düşürür.

    Reddetmiyoruz: tip bir GÜVENLİK kontrolü değil, bir etiket. Yanlış etiket
    yüzünden kullanıcının çıktısını kaybetmek orantısız olurdu.
    """
    return tip if tip in TIPLER else VARSAYILAN_TIP


#: Depo kökü: `/` ile ayrılmış, harf/rakamla başlayan segmentler. `..` ve mutlak
#: yol yok. `_GECERLI_ISIM`'in çok segmentli hâli.
_GECERLI_KOK = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}(/[A-Za-z0-9][A-Za-z0-9._-]{0,63}){0,4}$")


def _kok_dogrula(kok: str) -> str:
    """Depo kökünü doğrular — boşsa boş döner.

    NEDEN (2026-09-06'da fark edildi): `X-Artifact-Root` başlığı sandbox'a
    açık uçtan noktada hiç doğrulanmadan kabul ediliyordu. Sandbox'ın kendi
    istemcisi bu başlığı GÖNDERMİYOR, ama imajda `requests` var — LLM'in
    yazdığı kod ham bir POST atıp kökü kendisi seçebilirdi.

    Bu, mimarinin ana iddiasını deliyordu: "baytın NEREYE yazılacağını servis
    belirler, çağıran değil". Anahtarın geri kalanı (`owner/workflow/node/run`)
    zaten jetondan geliyordu; kök tek boşluktu.

    Sızma sınırlıydı — başka bir workflow'un artifact'i OKUNAMAZ (kayıt defteri
    `workflow_id` kontrol ediyor) ve var olan bir nesne EZİLEMEZ (`artifact_id`
    sunucuda üretiliyor). Ama paylaşılan bir bucket'ta (ODF/OpenShift AI'da
    olağan durum) başka bir prefix'e nesne serpmek mümkündü.

    KFP'de `pipeline_root` da pipeline'ı TANIMLAYAN kişinin ayarı, adımın
    kodunun değil. Doğru hizalama bu.
    """
    if not kok:
        return ""
    if not _GECERLI_KOK.match(kok):
        raise InvalidArtifactName(
            f"Geçersiz depo kökü: {kok!r}. İzin verilen: harf/rakam ile başlayan, "
            "'/' ile ayrılmış en fazla 5 segment; '..' ve mutlak yol yok."
        )
    return kok


def _isim_dogrula(name: str) -> None:
    if not _GECERLI_ISIM.match(name or ""):
        raise InvalidArtifactName(
            f"Geçersiz artifact adı: {name!r}. İzin verilen: harf/rakam ile başlayan, "
            "harf, rakam, nokta, tire ve alt çizgi (en fazla 128). Yol ayıracı ve "
            "'..' kabul edilmez — anahtarı üreten taraf servistir, çağıran değil."
        )


def _anahtar_ayikla(storage_uri: str) -> str:
    """`s3://bucket/a/b/c` → `a/b/c`"""
    return storage_uri.split("/", 3)[3]


def _soy_dugumu(m: ArtifactMeta, derinlik: int, yon: str) -> dict:
    """Soy grafiği düğümü. `depth` işaretli: negatif = ata, pozitif = ürün."""
    return {
        "artifact_id": m.artifact_id,
        "name": m.name,
        "artifact_type": m.artifact_type,
        "content_type": m.content_type,
        "size_bytes": m.size_bytes,
        "created_at": m.created_at.isoformat(),
        "node_id": m.node_id,
        "run_id": m.run_id,
        "depth": derinlik,
        "yon": yon,
    }
