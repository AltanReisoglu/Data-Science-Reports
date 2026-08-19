# Tarayıcı sürücüsü — demoyu insan gibi koştur

`pipeline/tests/` altındaki 484 test model çağırmadan koşuyor ve CI'da her
değişiklikte çalışıyor. Buradakiler **öyle değil**: canlı bir sunucu, gerçek
bir Chrome ve (bazıları için) gerçek bir model istiyorlar. O yüzden ayrı bir
klasörde ve `test_` önekleri yok — pytest onları toplamıyor.

Neden var: `--dump-dom` ve `--screenshot` sayfanın *sonucunu* gösteriyor ama
oraya nasıl gelindiğini göstermiyor. Bir düğmenin gerçekten tıklanabilir
olduğunu, girdinin odağı aldığını ve cevabın ekrana düştüğünü ancak olay
göndererek öğrenirsin. Sunumdan önce öğrenilecek en pahalı şey, bir düğmenin
pasif olduğu.

## Kullanım

```bash
# 1 · sunucu ayakta olmalı
cd pipeline && VC_ALLOW_CODE_EXEC=1 python server.py

# 2 · zincir gerçekten adım adım mı doluyor
python pipeline/tests/drive/walk.py chat-0006
```

## `cdp.py` — sürücü

`Input.dispatchMouseEvent` gerçek koordinata gerçek tık gönderiyor,
`Input.insertText` metni kutuya yazıyor. `Runtime.evaluate` yalnız **okumak**
için: bir düğmeye `el.click()` demek, o düğmenin görünür ve tıklanabilir
olduğunu doğrulamıyor — ki asıl soru o.

`box()` öğe yoksa, sıfır boyutluysa ya da `visibility:hidden` ise `None`
dönüyor, ve `click()` bunu bir hata olarak raporluyor. Sessizce başarılı
görünen bir tık, hiç tıklamamaktan kötüdür.

## Bu sürücünün yakaladıkları

Hepsi ölçüldü, hiçbiri koddan okunarak bulunmadı:

* **`Akış ↗` taze sayfada pasif.** Kayıtlı tur yoksa basılamıyor. Demoda sıra
  zorunlu: önce soru, sonra akış.
* **Zincir doğru dolıyor.** +0,4 sn'de 4 sönük / 1 yanan, +0,8'de bir *ok*
  yanıyor, +4,8'de hepsi parlak ve ışık sönmüş.
* **Kutu-içi penceresi gerçek fareyle açılıyor** — `Analyst` ve `Kapı` ikisi de.
* **MAF düğmesi rozeti değiştiriyor**: `AutoGen` → `MAF`.
* **Demo sorusu yanlıştı.** `docs/23 §4 Durak 2`'ye bak: eski soru 39 adım ve
  altı tool çağrısı üretiyor, ekranda anlatılan on aşama hiç çıkmıyor.

## Sürücünün kendi öğrettiği

Üç hata yaptım ve üçü de **uygulamada değil, ölçümdeydi** — not olarak duruyor:

1. `#stop` DOM'dan silinmiyor, `hidden` oluyor. `!getElementById('stop')`
   hiçbir zaman doğru olmuyordu ve bekleme her seferinde zaman aşımına düştü.
2. `Runtime.evaluate` düz bir **ifade** alıyor; üst seviye `await` orada
   geçersiz. Sessizce `undefined` döndü, akış ekranı `run=None` ile açıldı.
3. `js()` istisnayı yutuyordu. Takım beklemesi 200 saniye "zaman aşımı" dedi,
   oysa koşu 110 saniyede bitmişti — sürücü uygulamayı değil kendini ölçtü.

Ortak ders: bir ölçüm aracının sessiz kalması, ölçtüğü şeyin sağlam olduğu
anlamına gelmiyor. `js()` artık `exceptionDetails` okuyor ve bağırıyor.
