# Staj Defteri

Günlük çalışma kaydı. Her gün için bir dosya.

---

## Dosya düzeni

```
staj-defteri/
├── README.md          ← bu dosya: şablon ve kurallar
├── 2026-08-03.md      ← günlük kayıtlar
├── 2026-08-04.md
├── 2026-08-05.md
└── haftalik/
    └── 2026-H31.md    ← haftalık özet (opsiyonel)
```

**İsimlendirme:** `YYYY-AA-GG.md`. Böyle yazınca dosyalar kendiliğinden tarih sırasına diziliyor. `3agustos.md` gibi yazma — sıralama bozulur.

---

## Günlük şablon

Yeni gün açarken aşağıyı kopyala:

```markdown
# GG Ay YYYY — <Haftanın günü>

**Staj günü:** N
**Bugünün hedefi:** <tek cümle — akşam "oldu/olmadı" diye cevaplanabilir olmalı>

---

## Yapılanlar

### 1. <İş kaleminin adı>

**Ne:** <tek cümle — ne yapıldı>
**Nasıl:** <adım adım: hangi araç, hangi sıra, hangi karar. En önemli kısım burası.>
**Çıktı:** <somut: dosya adı, satır sayısı, ölçüm, ekran çıktısı>
**Neden böyle:** <alternatif neydi, niye bunu seçtim — varsa>

### 2. <İkinci iş kalemi>

**Ne:**
**Nasıl:**
**Çıktı:**

---

## Öğrendiklerim

1. **<Kavram>** — <tek paragraf, kendi cümlelerinle>
2. **<Kavram>** — <...>

---

## Takıldığım yer ve çözümü

**Sorun:**
**Neden oldu:**
**Nasıl çözdüm:**
**Bir dahakine:**

---

## Yarına kalanlar

- [ ]
- [ ]

---

## Kaynaklar

-
```

---

## Yazma kuralları

**1. Saat tutma — yöntem yaz.**
Saat aralığı kimseye bir şey anlatmıyor; *nasıl* yaptığın anlatıyor. Her iş kalemi için "nasıl" satırını doldur: hangi aracı kullandın, hangi sırayla gittin, yolda hangi kararı verdin.

❌ "14:00–16:00 Rapor üzerinde çalıştım."
⚠️ "Rapora 11. bölümü ekledim — 977 satır."
✅ "Rapora 11. bölümü ekledim. Önce iki kaynak listesini `grep` ile taradım (2.289 satırlık dosyayı okumadan tema çıkarmak için), çıkan başlıkları 8 katmana ayırdım, doğrulanması gereken 4 iddia için web araması yaptım, sonra tek dosyada yazdım — 977 satır. Bitince `grep` tabanlı bir doğrulama komutuyla tüm iç linkleri ve tablo kolon sayılarını kontrol ettim."

Üçüncüsü hem ne yaptığını hem **nasıl düşündüğünü** gösteriyor. Mentörün görmek istediği bu.

**2. Ne yaptığını değil, neyi neden yaptığını yaz.**
Defteri okuyan kişi (mentör, okul) senin gün sonunda ne *ürettiğini* değil, ne *öğrendiğini* görmek istiyor. "Dosya yazdım" bir çıktı; "tool çıktısının neden `user` rolünde döndüğünü anladım" bir kazanım.

**3. Takıldığın yeri mutlaka yaz.**
Defterin en değerli kısmı burası. Sorunsuz geçen gün, ya bir şey öğrenmediğin ya da yazmadığın gündür. Hata + çözüm ikilisi, staj değerlendirmesinde en çok puan getiren bölüm.

**4. O gün yaz.**
Ertesi gün yazılan defter uydurmaya dönüşür. Saat aralıkları ve "neden şöyle yaptım" bilgisi 24 saat içinde buharlaşıyor.

**5. Kaynak bırak.**
Okuduğun makale, paper, dokümantasyon — linkini koy. Hem raporunda kaynakça lazım olacak, hem "araştırma yaptım" iddian kanıtlı olur.

**6. Kısa tut.**
Bir gün = yarım sayfa yeter. Uzun defter tutulmuyor; tutulmayan defter işe yaramıyor.

---

## Ne yazılmaz

| Yazma | Neden |
|---|---|
| Şirket içi gizli veri, müşteri adı, kimlik bilgisi | Defter okulla paylaşılıyor |
| API anahtarı, şifre, token | Asla — dosya git'e girerse sızar |
| "Bugün bir şey yapmadım" | Beklendi/araştırdı/kurulum yaptı da bir iştir, öyle yaz |
| Kopyala-yapıştır dokümantasyon | Kendi cümlelerinle özetle, yoksa öğrenmemişsin demektir |

---

## Haftalık özet (opsiyonel ama tavsiye)

Cuma günü `haftalik/YYYY-Hnn.md` aç ve 5 günü tek sayfaya indir:

```markdown
# 2026 — 31. Hafta (3–7 Ağustos)

**Haftanın çıktısı:** <somut teslim>
**En çok zaman alan iş:**
**En çok öğrendiğim şey:**
**Gelecek haftanın hedefi:**
```

Staj sonu raporunu yazarken bu beş satır, beş günlük defteri tekrar okumaktan hızlı oluyor.
