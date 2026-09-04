# Quickstart: Web Arayüzü + Canlı PTC İzleme Paneli (Faz 4)

Uygulama kodu değil, uçtan uca doğrulama rehberidir.

## Ön koşullar

- Faz 1/2'nin çalışır durumda olması: `.env`, `kind` cluster + Cilium + Tool Gateway +
  sandbox image (Faz 2 quickstart.md'sindeki kurulum).
- `pip install fastapi "uvicorn[standard]"` (bu fazın tek yeni bağımlılığı).

## Çalıştırma

```bash
uvicorn grounded_assistant.web.app:app --reload
```

Tarayıcıda `http://localhost:8000` açılır.

## Doğrulama senaryoları

### Senaryo 1 — Temel soru-cevap (US1, P1)

Tarayıcıda: "Uzaktan çalışma politikamız nedir?"

**Beklenen**: Yanıt metni, grounded durumu ve kaynaklar ekranda görünür — CLI'de aynı
soru sorulduğunda alınan sonuçla tutarlı.

### Senaryo 2 — Zemin bulunamadığında açık uyarı (US1)

Tarayıcıda: hiçbir kaynağın bilgi içermediği bir soru (ör. "Marsta ofisimiz var mı?").

**Beklenen**: "Bu soruyla ilgili hiçbir erişim yolunda veri bulunamadı" gibi açık bir
mesaj — tahmini bir yanıt YOK.

### Senaryo 3 — PTC yaşam döngüsünü canlı izleme (US2, P2 — bu fazın asıl testi)

Tarayıcıda: "4 kaynaktaki tüm dokümanları tara, VPN konusunu geçen kaç tanesi var?"

**Beklenen**: Sol-alt panelde, yanıt gelmeden ÖNCE şu sırayla akış görünür:
`configmap_created` → `job_created` (kodun kendisiyle) → birden fazla `tool_call` →
`final`. Panelin dolması, nihai `answer` mesajından ÖNCE tamamlanmış olmalı.

### Senaryo 4 — Engellenen bir eylem panelde görünür (US2)

Backend'e (geçici bir test amaçlı) `sandbox_test_fixtures/escape_attempt.py`'yi
çalıştıracak bir soru/tetikleyici verilir (bkz. Faz 2 quickstart.md Senaryo 2).

**Beklenen**: Panelde `denied_action` satırı (hedef + verdict) görünür, nihai durum
`status=denied_action`'dır.

### Senaryo 5 — PTC kullanılmayan bir soruda panel boş kalmaz (US2)

Tarayıcıda: doğrudan bir canlı-sistem sorgusu tetikleyen basit bir soru (ör. "Şu an
açık kritik ticket sayısı kaç?").

**Beklenen**: Panelde hiç `ptc_event` gelmez; arayüz bunu "bu soru için sandbox
kullanılmadı" şeklinde açıkça gösterir (sessizce boş kalmaz — FR-006).

### Senaryo 6 — İki sekme birbirine karışmaz (US3, P3)

İki farklı tarayıcı sekmesinden aynı anda iki farklı soru sorulur (biri PTC tetikleyen,
biri tetiklemeyen).

**Beklenen**: Her sekme yalnızca kendi sorgusunun yanıtını/panelini gösterir.
