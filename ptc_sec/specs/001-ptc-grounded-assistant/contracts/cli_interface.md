# Contract: CLI Arayüzü

`grounded_assistant.cli` (typer tabanlı) komut satırı arayüzü.

## Komut: `ask`

```text
$ assistant ask "Uzaktan çalışma politikamız nedir?"
```

**Çıktı formatı** (insan-okunabilir, stdout):

```text
Yanıt:
<asistanın metni>

Kaynaklar: kurumsal bilgi bankası (wiki, policy)
```

Veri bulunamadığında (FR-007):

```text
Yanıt:
Bu soruyla ilgili hiçbir erişim yolunda (bilgi bankası, canlı sistem) veri bulunamadı.

Kaynaklar: (yok)
```

Kısmi hata durumunda (FR-010), kaynaklar satırına eksik/başarısız kaynak eklenir:

```text
Kaynaklar: kurumsal bilgi bankası (wiki) — [destek talebi arşivi: erişilemedi]
```

## Opsiyonel bayrak: `--trace`

Ham izlenebilirlik kaydını (Answer.source_refs, her KnowledgeBaseSource'un status'ü,
her LiveToolCall'ın durumu) JSON olarak ek çıktı verir — denetim/test amaçlı
(spec.md SC-005).

```text
$ assistant ask "..." --trace
```

## Çıkış kodları

| Kod | Anlam |
|---|---|
| 0 | Yanıt üretildi (grounded veya "bulunamadı" — ikisi de başarılı bir çalıştırma sayılır) |
| 1 | Beklenmeyen hata (ör. mock MCP sunucusuna hiç bağlanılamadı) |
