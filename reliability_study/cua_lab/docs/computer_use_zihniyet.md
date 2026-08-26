# Faz 0 — Computer-Use Zihniyet Çalışması

Tarih: 2026-08-24 · Yöntem: kaynak kodun doğrudan okunması (klonlama yok, hedefli `raw` çekimi)

Kod yazmadan önce computer-use framework'lerinin nasıl düşündüğünü çıkarmak. Üç soru:
**eylem uzayı ne**, **ekran nasıl temsil ediliyor**, **döngü nasıl sonlanıyor**.

---

## 1 · Anthropic referans döngüsü `[K]`

`anthropics/claude-quickstarts` → `computer-use-demo/computer_use_demo/loop.py` (388 satır)

### Eylem uzayı — `computer_20251124`, 17 eylem

```
key · type · mouse_move · left_click · left_click_drag · right_click · middle_click
double_click · screenshot · cursor_position · left_mouse_down · left_mouse_up
scroll · hold_key · wait · triple_click · zoom
```

`scroll` ayrıca yön alıyor: `up | down | left | right`.

Üç sürüm var ve birikimli: `20241022` (10 eylem) → `20250124` (+6: fare basılı tutma,
kaydırma, tuş basılı tutma, **`wait`**, üçlü tıklama) → `20251124` (+`zoom`).

`wait`'in ayrı bir eylem olması dikkate değer: model "bekle" diyebiliyor. Bizim durgunluk
dedektörümüz için tuzak — meşru bir `wait` ekranı değiştirmez ve durgunluk sayılmamalı.

### Ekran temsili — ham piksel

Ekran görüntüsü doğrudan görüntü bloğu olarak gidiyor. Erişilebilirlik ağacı, DOM ya da
öğe numaralandırması yok. Model piksel koordinatıyla tıklıyor.

### Döngü sonlanması — **hiçbir kontrol yok**

```python
while True:
    ...
    if not tool_result_content:
        return messages
```

Tek çıkış: modelin araç çağırmayı bırakması. **Tur sayacı yok, döngü tespiti yok,
token/süre/maliyet bütçesi yok.** Model sonsuza kadar araç çağırırsa döngü sonsuza kadar döner.

Bu, `harness_kontrolleri.md`'deki desenin en saf hâli: satıcının kendi referans
uygulamasında bile ana döngü sınırsız. Projemizin varlık sebebi tam olarak bu boşluk.

### Bağlam yönetimi — iki mekanizma, biri diğerini iptal ediyor

**Görüntü budama** (`_maybe_filter_to_n_most_recent_images`): ekran görüntüleri
*"konuşma ilerledikçe değeri azalan"* varsayımıyla son N tanesi hariç siliniyor.
Silme **öbek öbek** yapılıyor (`images_to_remove -= images_to_remove % min_removal_threshold`)
ki prompt cache mümkün olduğunca az kırılsın.

Ama prompt caching açıkken budama **tamamen kapatılıyor**:

```python
# Because cached reads are 10% of the price, we don't think it's
# ever sensible to break the cache by truncating images
only_n_most_recent_images = 0
```

Yani **caching ile budama çelişiyor** ve caching kazanıyor. Bizim bütçe stratejilerimiz
için doğrudan sonuç: token maliyetini düşürmenin iki yolu birbirini iptal edebiliyor,
hangisinin kazanacağı bir tasarım kararı.

### Araç kümesi hatası yayılımı

Bir araç kümesinin (toolset) üyelerinden biri başarısız olursa kalan üye çağrıları
**yürütülmeden** `"Not executed: an earlier computer action in this turn failed."` ile
cevaplanıyor. Sıralı yürütme, ilk hatada durma. Paralel araç çağrısı varsayımını kıran
bir tasarım — bizim olay akışımızda da aynı ayrım gerekiyor.

---

## 2 · cua — set-of-marks ile ekran temsili `[D]`

`trycua/cua` → `libs/python/som`

Ham piksel yerine **öğe tabanlı** temsil: YOLO ile ikon tespiti + EasyOCR ile metin tanıma,
sonuç numaralandırılmış anotasyonlar. Model piksel koordinatı yerine "3 numaralı öğe" diyor.

**Bedeli:** YOLO + EasyOCR = `torch` + `transformers`. `cua-agent` paketinin 63 bağımlılığı
ve birkaç GB'lık kurulumu buradan geliyor. Her adımda bir vision pipeline koşuyor.

**Bizim için anlamı:** set-of-marks, durgunluk dedektörünü sağlamlaştırırdı — ekran hash'i
yerine *öğe kümesi* hash'i yanıp sönen imleçten etkilenmez. Ama maliyeti ağır. Faz 4'te
toleranslı piksel hash'i ile başlayıp, gerekirse bu yola geçmek üzere not düşülüyor.

---

## 3 · OpenAdapt — karşı-tez `[D]`

`OpenAdaptAI/OpenAdapt`

Kendi tanımı:

> *"Gösterilen bir görevi, tarayıcı/Windows/macOS/Linux/RDP/Citrix için **incelenebilir,
> deterministik bir programa** derler. **Sağlıklı koşumlar hiçbir üretken model çağrısı
> yapmaz.** Sonuç bildirilmeden önce beyan edilen sonucu canlı duruma karşı doğrular.
> Gerekli kanıt eksikse veya canlı durumla çelişiyorsa koşum incelemeye durur."*

Bu bir guardrail değil, **guardrail'e olan ihtiyacı ortadan kaldıran bir mimari**.
Döngüye girme riskinin kökten çözümü: modeli döngüden çıkarmak.

Üç ayrı fikir taşıyor:
1. **Gösterimden derleme** — bir kez göster, deterministik programa çevir
2. **Sağlıklı yolda model yok** — model yalnızca sapma olduğunda devreye giriyor
3. **Sonucu canlı duruma karşı doğrula** — `sde_offer_loop`'un "doğrulama kapılı durma"
   ilkesinin daha sert hâli; kanıt yoksa koşum durur, "başarılı" denmez

---

## 4 · Tasarımımıza etkisi

| Bulgu | Karar |
|---|---|
| Referans döngü sınırsız | Bizim `loop.py` baştan strateji kancalı; `none` stratejisi bu sınırsız hâli taban çizgisi olarak koruyor |
| 17 eylemlik uzay, `wait` dahil | `SandboxBackend` bu uzayı karşılıyor; `wait` durgunluk sayacından muaf tutulacak |
| Ham piksel temsili | Faz 4'te toleranslı gri-tonlama hash'i; set-of-marks maliyeti nedeniyle ertelendi |
| Caching ↔ budama çelişkisi | Bütçe stratejileri hangisini seçtiğini açıkça raporlayacak |
| Araç kümesi ilk hatada durur | Olay akışında "yürütülmedi" ayrı bir gözlem türü olacak |

---

## 5 · Bildirilmesi gereken SOTA zihniyet

Kullanıcının şartı: *"internette farklı sota bir mentalite görürsen bunu bana belirtirsin."*

**OpenAdapt'ın "sağlıklı koşumda model çağrısı yok" yaklaşımı, strateji listemizde
karşılığı olmayan bir zihniyet.** On yedi stratejinin tamamı "model döngüde, biz onu
sınırlıyoruz" varsayımı üstüne kurulu. OpenAdapt bu varsayımı reddediyor.

Uygulanabilir bir indirgemesi var ve 18. strateji olabilir:

> **`replay-first`** — başarılı bir yörünge kaydedilir; sonraki koşumlarda önce
> deterministik olarak tekrar oynatılır, yalnızca ekran beklenenden saparsa modele sorulur.
> Sağlıklı koşumda sıfır model çağrısı, dolayısıyla sıfır döngü riski.

Bu eklenmedi — kullanıcının kararı bekleniyor.
