# 24 — PoC promptları: zamanlayıcı, takım, Docker, OpenClaw

*Hepsi 19–20 Ağustos 2026'da bu makinede **koşturuldu**. Sayılar o koşulardan.
Koşturulmayan tek satır yok; bir şey denenmediyse öyle yazıyor.*

---

## §0 · Her seferinde önce bu

```
Reset chat
```

Ölçüldü: 15 turluk bir oturumda ajan `search_docs`'u **hiç çağırmıyor** —
bağlamdan cevaplayıp `memory_search` ile idare ediyor, tur 5,8 saniyede bitiyor
ve şeritte anlatacak bir şey kalmıyor. Kirli oturum demoyu sessizce boşaltıyor.

Ve **koşular bellekte**: sunucuyu yeniden başlatırsan hepsi siliniyor,
`Akış ↗` düğmesi yine pasif oluyor.

---

## §1 · Zamanlayıcı — üç adımda, ve ilki başarısız

**① Önce yanlış söz dizimiyle sor.** Reddedilmesi anlatının parçası:

```
/openclaw schedule her sabah 09:00 | son taramayı özetle
```

> `'her sabah 09:00' anlaşılmadı. Üç biçim var: 'her gün 09:00' · '30dk' · '20dk sonra'`

**Çeviremediğinde sormuyor, sözdizimini yazıp reddediyor.** Zamanlamada tahmin
etmek pahalı: yanlış okunan bir cümle her gün yanlış saatte koşan bir iş demek.

**② Doğrusu — ve kapı tutuyor:**

```
/openclaw schedule her gün 09:00 | son taramanın özetini çıkar
```

> `cron.add OpenClaw'da bir şey değiştirir. Approve request ee353e6a3b0a…`

**③ Onayla, sonra satırı tekrarla.** Onay o turu geri getirmiyor — grant
tüketiliyor ve çağrı yeniden yapılıyor:

```
/openclaw cron.list
```

Ölçüldü: **2 iş → 3 iş**, yenisi `{"kind":"cron","expr":"0 9 * * *","tz":"Europe/Istanbul"}`.

**Akış ekranında:** her ikisi de kayıt bırakıyor — `cron-0001` **error**
(ayrıştırma), `cron-0002` **blocked** (kapı). Grafın üst bandı **boş**, ve o
boşluk doğru olanı söylüyor: zamanlamada AutoGen'in hiçbir parçası koşmuyor.

---

## §2 · Takım — beş tip, ikisini koştur

Seçici + `Takımla sor`. Kadro **Planner · Researcher · Critic**; Researcher'da
`search_docs` + `scan_facts`, Critic'te `search_docs`, Planner **tool'suz**.

```
onay kapısı neden runtime seviyesinde olmalı, kısa değerlendir
```

Ölçülen iki koşu:

| tip | LLM | token | süre | tool | not |
|---|---:|---:|---:|---:|---|
| `selector` | 14 | 23.095 | 81,7 sn | 8 | model her turda seçiyor |
| `swarm` | 6 | **16.888** | **28,3 sn** | 14 | devir tur 4'te |

**Sunumda söylenecek:** *"Aynı görev, aynı ajanlar, tek değişen sırayı kimin
belirlediği."* Ve grafta iş bölümü görünüyor — Planner'ın tool kutusu **yok**,
Researcher ile Critic'inki var.

> **Uyarı:** takım turu uzun. Süre daralırsa bu adımı at, sayıyı söyle.

---

## §3 · Docker terminali — reddet, sonra onayla

```
82, 64 ve 91 sayılarının medyanını ve varyansını Python ile hesapla
```

Kapı tutuyor ve **gerekçeyi kendisi yazıyor**:

> *"Modelin yazdığı Python kodu çalıştırılacak. Kod izole bir Docker
> konteynerinde koşuyor, ama konteynerin ağ erişimi var — AutoGen'in yürütücüsü
> ağ izolasyonu için bir parametre sunmuyor."*

**Demonun en önemli anı burada:** ajan **çökmedi**. Hesabı elle yapıp cevabı
yine verdi ve onayın beklediğini söyledi. Kapı bir istisna fırlatmıyor, bir
**cevap** üretiyor.

Sohbette iki düğme çıkıyor: **`Approve and retry`** ve **`Deny`**. Onayla:

* terminal **2,0 saniyede** açılıyor
* içinde `$ python /workspace/tmp.py` ve kodun kendisi
* `body.term-open` → sol sütun genişliyor

Yedek sorular (ikisi de ölçüldü, ikisi de kodu tetikledi):

```
97 ile 143 sayılarının EBOB'unu Python yazarak bul
şu üç sayının standart sapmasını hesapla: 82, 64, 91
```

---

## §4 · OpenClaw flex — niş yüzeyleri açığa çıkarmak

Söz dizimi üç yola ayrılıyor: `methods` yerelde cevaplanıyor · bir **metot adı**
Gateway'e gidiyor · bir **cümle** OpenClaw'ın kendi ajanına gidiyor (ve kapı onu
tutuyor).

### Onay gerektirmeyenler — `read` katmanı

```
/openclaw methods
/openclaw commands.list
/openclaw audit.list
/openclaw cron.list
/openclaw models.list
/openclaw sessions.list
/openclaw channels.status
```

Ölçülen sonuçlar ve hangi niş slaytı açtıkları:

| komut | sonuç | açtığı konu |
|---|---|---|
| `commands.list` | **89 komut** | `/steer` · `/btw` · `/goal` · `/export-trajectory` — koşan tura müdahalenin dört yolu |
| `audit.list` | **100 olay** + devam imleci | **iki kayıt hattı** — KKB için en önemli slayt |
| `cron.list` | 3 iş, cron ifadeleriyle | zamanlama yığını · cron'un OR tuzağı |
| `models.list` | sağlayıcı + bağlam penceresi | **model failover** |
| `methods` | read / write / forbidden sınıflaması | yetki metot kapsamıyla başlar |

**Flex cümlesi:** *"Bu 89 komutun dördü koşan bir turun içine müdahale ediyor —
`/steer` yön veriyor, `/btw` bağlamı kirletmeden soruyor. AutoGen'de bir turun
içine girmenin hiçbir yolu yok."*

### Kapının tuttuğu — güven modelinin kanıtı

```
/openclaw list
```

> *"Bu satır OpenClaw'ın kendi ajanına gidiyor. O ajanın kabuk erişimi var ve
> şu an onay sormadan çalıştırıyor (**exec: mode=full, ask=off**); bizim kapımız
> içeride ne yapacağını görmez."*

**Bu, sunumun en iyi 20 saniyesi.** "OpenClaw'ın güven modeli bizim kurumumuz
için yanlış" cümlesini biz kurmuyoruz — sistem kendi kuruyor, gerekçesiyle.

### Denenmedi

`node.list` **0 düğüm** döndürdü (bu makinede bağlı cihaz yok), `usage.status`
sağlayıcı listesi boş. İkisi de çalışıyor ama gösterecek verileri yok — demoda
açma.

---

## §5 · Sıra

1. `Reset chat`
2. **§3 Docker** — kapı reddediyor, ajan çökmüyor, onayla, terminal açılıyor
3. **§4 OpenClaw** — `commands.list` ve `audit.list`, sonra `/openclaw list` ile
   kapının tuttuğunu göster
4. **§1 Zamanlayıcı** — yanlış söz dizimi → doğrusu → onay → `cron.list`
5. **§2 Takım** — vakit varsa; yoksa tablodaki sayıyı söyle

Docker önce, çünkü onay kapısını en iyi o gösteriyor ve perdenin tezi o.
