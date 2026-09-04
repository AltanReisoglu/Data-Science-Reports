# Error Recovery ve Artifact Persistence — Ön İnceleme

İki yeni konunun mevcut PoC koduna göre incelenmesi. **Bu bir karar dokümanı değil**
— ne var, ne eksik, hangi tasarım gerilimleri çıkıyor onu tespit ediyor. Hiçbir kod
değiştirilmedi.

İncelenen kod: `sandbox_image/entrypoint.py`, `src/grounded_assistant/ptc/sandbox_runner.py`,
`src/grounded_assistant/agent/graph.py`, `src/grounded_assistant/models.py`,
`k8s/sandbox/job-template.yaml`.

---

## 1. Error Recovery (stack trace → LLM → düzeltilmiş kod)

### 1.1 Şu an ne oluyor

Boru hattı zaten var ama **sinyal fakir**. Hata şu yoldan geçiyor:

```
entrypoint.py:128     except Exception as exc:
                          print(json.dumps({"status": "error", "message": str(exc)}))
        ↓
sandbox_runner.py     error_message alanına yazılır
        ↓
graph.py              return f"Sandbox çalıştırması başarısız oldu.{detail}
                              Tahmini bir değer üretme."
```

Yani LLM'e ulaşan şey **yalnızca `str(exc)`**.

### 1.2 Tespit edilen üç boşluk

**a) Traceback hiç yakalanmıyor.**
`str(exc)` sadece mesajı verir: `name 'tickets' is not defined`. Hangi satır, hangi
çağrı zinciri — hiçbiri yok. Oysa self-repair'in çalışması için gereken asıl bilgi
tam olarak bu.

İyi haber: temel hazır. Kod `exec(compile(code, CODE_PATH, "exec"), ...)` ile
derleniyor — yani `CODE_PATH` gerçek bir dosya adı olarak geçiyor ve
`traceback.format_exc()` **doğru satır numaralarını** üretebilecek durumda. Şu an
sadece çağrılmıyor.

**b) Prompt tam ters yönde yönlendiriyor.**
Dönen mesaj *"Tahmini bir değer üretme"* diyor — yani modeli **durmaya** teşvik
ediyor. Düzeltip tekrar denemesini söyleyen hiçbir ifade yok. Self-repair için bu
metnin niyeti tersine çevrilmeli.

**c) Hata sınıflandırması yok — ve burada gerçek bir çelişki var.**

`run_ptc_code`'un üç ayrı başarısızlık dalı var ama hepsi aynı sayaca yazılıyor:

| Durum | Tekrar denemek mantıklı mı |
|---|---|
| `SandboxRunStatus.ERROR` (kod hatası: NameError, TypeError, KeyError…) | **Evet** — düzeltilebilir |
| `SandboxRunStatus.DENIED_ACTION` (ağ engeli) | **Hayır** — 2026-09-01'de bunu bilerek engelledik |
| `SandboxRunStatus.TIMEOUT` | Duruma göre |

### 1.3 Ana gerilim: retry sınırı ile self-repair karşı karşıya

`MAX_SANDBOX_RUNS_PER_TURN = 2` sabiti, agent'ın **ağ seviyesinde reddedilen** bir
hedefi farklı URL'lerle tekrar tekrar denemesini durdurmak için konuldu. Ama
`trace.sandbox_run_count()` çalıştırmaları **sebebine bakmadan** sayıyor.

Sonuç: self-repair için kalan bütçe tam olarak **1 deneme**.
```
1. çalıştırma → NameError
2. çalıştırma → düzeltilmiş kod    ← sınır doldu
3. çalıştırma → reddedilir (ikinci bir düzeltme şansı yok)
```

Gerçek bir kod hatasında ilk düzeltmenin tutmaması sık görülür (LLM yanlış varsayımla
düzeltir, ikinci traceback asıl sorunu gösterir). Yani bugünkü sınır self-repair'i
pratikte tek atışa indiriyor.

**Çözüm yönü (uygulanmadı, sadece tespit):** sayaç *sebep-farkında* olmalı — ağ
engeli KESİN kalmalı (0 tekrar), kod hatası ayrı ve daha geniş bir bütçe almalı.
İki farklı davranışı tek sayaçla yönetmek mümkün değil.

### 1.4 Gözden kaçmaması gereken güvenlik notu

Traceback'i LLM'e vermek, sandbox'ın **iç durumunu** dışarı taşır: dosya yolları,
değişken adları, yüklü modül isimleri, bazen değişken değerleri. Bu bir sızıntı
kanalı değil (LLM zaten kodu kendisi yazdı) ama traceback'in *hangi kısmının*
aktarılacağı bilinçli seçilmeli — özellikle `fetch_url` gibi tool'ların iç hata
mesajları onaylı hedeflerin adreslerini içerebilir.

---

## 2. Artifact Persistence (dosya, dataframe, ara çıktı)

### 2.1 Şu an kalıcılık için hiçbir mekanizma yok — üstelik bu kasıtlı

Üç ayrı yerde engelleniyor:

**a) Yazılabilir disk yok.**
Pod'a mount edilen tek volume, kodun geldiği ConfigMap:
```yaml
volumes:
  - name: code
    configMap:
      name: ptc-code-{run_id}
```
Kubernetes'te **ConfigMap volume'ları salt-okunurdur**. Sandbox kodu `/sandbox`
altına yazamaz. Başka bir volume (PVC, emptyDir) tanımlı değil.

**b) Sonuç yolu yalnızca JSON metni taşıyor.**
```
entrypoint.py       json.dumps({"status": "success", "result": value})
sandbox_runner.py   result_text = str(parsed.get("result"))
```
İki kısıt birden: değer **JSON'a serileştirilebilir olmalı** ve sonunda **string'e
çevriliyor**. Bir `pandas.DataFrame` bu yoldan geçemez — `json.dumps` hata verir.

**c) Pod'un ömrü buna izin vermiyor.**
```yaml
activeDeadlineSeconds: 30
ttlSecondsAfterFinished: 300
backoffLimit: 0
```
Her çalıştırma yeni bir pod; iş bitince ConfigMap+Job siliniyor.

### 2.2 Ana gerilim: bu, PoC'nin tezinin bir parçasıyla çelişiyor

Sunumun ve dokümanların açıkça savunduğu bir iddia var:

> *"Sandbox pod'u doğar, en fazla 30sn çalışır, biter bitmez ConfigMap+Job açıkça
> silinir — **hiçbir zaman kalıcı iz bırakmaz**."*

Artifact persistence tam olarak bunun tersini ister. Bu bir engel değil ama
**sessizce geçilemeyecek bir takas**: "iz bırakmaz" iddiası ya nitelenmeli
("çalıştırma ortamı iz bırakmaz, üretilen veri onaylı depoda saklanır") ya da
bırakılmalı.

### 2.3 Daha önemlisi: yeni bir kör nokta yaratıyor

Bu, 2 Eylül'de bulduğumuz pod-içi `localhost` bulgusuyla **aynı sınıfta** bir sorun.

Paylaşılan bir artifact deposu, iki çalıştırma arasında **Cilium'un görmediği bir
veri yolu** açar:
```
Çalıştırma A  ──yazar──►  [artifact deposu]  ──okur──►  Çalıştırma B
```
Cilium ağ katmanında çalışır. İki çalıştırma birbirine paket göndermiyor, **depo
üzerinden** haberleşiyor. Egress politikası bu akışı ne görür ne engeller.

Bugünkü modelde bu kanal **yapısal olarak yok** (her run sıfırdan, paylaşılan hiçbir
şey yok). Artifact persistence eklendiği anda oluşur ve erişim kontrolü ayrı bir
katmanda (depo tarafında) kurulmak zorunda kalır.

### 2.4 Dört olası yön (değerlendirme, öneri değil)

| Yaklaşım | Nasıl | Güvenlik modeline etkisi |
|---|---|---|
| **PVC mount** | Pod'a yazılabilir kalıcı volume | En basit. Ama izolasyon garantisi düşer, GC/kota gerekir, çok-çalıştırma erişim kontrolü yok |
| **Tool Gateway üzerinden nesne deposu** (MinIO/S3) | `put_artifact()` / `get_artifact()` yeni tool'lar olarak | **Mevcut modelle en tutarlısı.** Sandbox yine sadece Tool Gateway'e çıkar, Cilium politikası hiç değişmez; erişim kontrolü gateway'de, zaten allowlist'in olduğu yerde |
| **ConfigMap/Secret** | Küçük çıktılar için | 1 MiB sınırı var (bu sınır zaten *kodun* boyutunu da kısıtlıyor). DataFrame için uygun değil |
| **Agent'ın belleğinde taşıma** | Çalıştırmalar arası veriyi agent tutar | Depo hiç yok, kör nokta oluşmaz. Ama boyut LLM bağlamıyla sınırlı — dataframe için gerçekçi değil |

İkinci seçenek dikkat çekici: artifact'i **onaylı kanaldan** geçirmek, projenin
kurucu ilkesinin ("erişim değil, kanal onaylanır") artifact'lere de uygulanması
demek. Yeni bir egress kuralı gerektirmiyor, çünkü sandbox yine yalnızca Tool
Gateway ile konuşuyor.

### 2.5 Serileştirme ayrı bir iş kalemi

DataFrame desteği yalnızca "depo" sorunu değil. Bugünkü `json.dumps` yolu tip
bilgisini de kaybediyor (`str(...)` ile string'e düşürülüyor). Kalıcılık eklenirse
formatın ayrıca kararlaştırılması gerekir — Parquet/Arrow (tip korur, sütunlu) vs.
CSV (basit, tip kaybeder) vs. pickle (**güvensiz** — LLM'in ürettiği veriyi pickle
ile geri yüklemek uzaktan kod çalıştırma yolu açar, elenmeli).

---

## 3. İki konunun birbirine değdiği nokta

Bağımsız görünüyorlar ama bir yerde kesişiyorlar: **self-repair, artifact'leri
tekrar üretme maliyetini artırır.**

Çok adımlı bir workflow'un 4. adımında hata alıp düzeltilmiş kodu baştan
çalıştırmak, 1-3. adımların çıktısını da yeniden üretmek demek — artifact
persistence yoksa her düzeltme denemesi tüm zinciri tekrar koşturur. Bu hem
~7.21 sn'lik pod maliyetini hem de tool çağrılarını katlar.

Yani ikinci konu, birincisinin pratikte kullanılabilir olmasını kolaylaştırıyor.

---

## 4. Özet — cevaplanması gereken sorular

**Error recovery için:**
1. `traceback.format_exc()` çıktısının ne kadarı LLM'e verilecek (tamamı mı, son N
   frame mi)?
2. Retry sayacı sebep-farkında hale getirilecek mi? Ağ engeli 0, kod hatası N —
   N kaç?
3. Düzeltme denemeleri `Trace`'e nasıl kaydedilecek (aynı turun parçası mı, ayrı mı)?

**Artifact persistence için:**
1. "Hiçbir zaman kalıcı iz bırakmaz" iddiası korunacak mı, nitelenecek mi?
2. Depo hangi katmanda — PVC mi, Tool Gateway arkasında nesne deposu mu?
3. Çalıştırmalar arası erişim kontrolü kim yapacak? (Cilium yapamaz — §2.3)
4. Artifact'lerin ömrü/GC politikası ne olacak?
5. DataFrame için serileştirme formatı? (pickle elenmeli)
