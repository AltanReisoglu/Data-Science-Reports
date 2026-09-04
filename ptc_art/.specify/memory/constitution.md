<!--
Sync Impact Report
- Version change: (none, template) → 1.0.0 (initial ratification)
- Modified principles: N/A — first concrete constitution; all template placeholders replaced
- Added sections: Core Principles (I–V), Security & Data Handling Constraints,
  Development Workflow, Governance
- Removed sections: none
- Templates requiring updates:
  ✅ .specify/templates/plan-template.md — Constitution Check gate is generic
     ("[Gates determined based on constitution file]"); derives from this file
     at plan time, no direct edit needed.
  ✅ .specify/templates/spec-template.md — no hardcoded principle references;
     unaffected.
  ✅ .specify/templates/tasks-template.md — no hardcoded principle references;
     unaffected.
  ✅ specs/001-ptc-grounded-assistant/spec.md — already aligns with Principles
     I–III (grounding, approved channels, traceability) and already defers the
     stack decision per Principle IV; no changes required.
- Follow-up TODOs: none
-->

# PTC Sec Constitution

## Core Principles

### I. Zemine Dayalılık — Uydurma Yok (Grounding-First, No Fabrication)

Her yanıttaki her olgusal iddia, projenin tanımlı erişim yollarından (kurumsal
bilgi bankası, canlı sistemler, hafıza) fiilen çağrılmış bir araçtan dönen
veriye izlenebilir olmalıdır. Hiçbir erişim yolu ilgili veri döndürmediğinde,
sistem bunu açıkça belirtmeli; zemin bulunmayan bir iddia asla üretilmemelidir.

**Rationale**: Bu, depodaki tüm PoC'lerin (bkz. `specs/001-ptc-grounded-assistant`)
çekirdek değer önermesidir — kaynağa dayanmayan bir iddia, kurumsal bir ortamda
güven kaybına yol açar ve PoC'nin varlık nedenini geçersiz kılar.

### II. Yalnızca Onaylı Kanal (Approved-Channel-Only Access / PTC Egress Disiplini)

Agent kodu, herhangi bir dış sisteme, veri kaynağına veya canlı araca YALNIZCA
`docs/topic_is_this.md` ve depodaki PTC (Programmatic Tool Calling) egress-policy
araştırma dokümanlarında tarif edilen, açıkça onaylanmış tool/skill kanalları
üzerinden erişebilir. Agent çalışma zamanı kodundan doğrudan/keyfi ağ erişimi —
prototipleme kolaylığı gerekçesiyle bile — YASAKTIR; ihtiyaç varsa kanalı
onaylatmak gerekir, kanalı atlamak değil.

**Rationale**: Bu, deponun tüm var oluş amacı olan güvenlik modelidir; bir PoC
içinde bile bu modeli atlamak, PoC'nin kendi tezini geçersiz kılar.

### III. İzlenebilirlik ve Denetlenebilirlik (Traceability & Auditability)

Üretilen her yanıt, hangi erişim yolu/yollarının ve hangi kaynak(lar)ın katkı
sağladığını, bağımsız bir gözden geçirenin sonradan denetleyebileceği şekilde
kaydetmelidir. Kısmi hatalar (ör. paralel kaynaklardan birinin hata vermesi)
sessizce yutulmamalı, açıkça yüzeye çıkarılmalıdır.

**Rationale**: "0 sonuç her zaman bir açıklama borçludur" — bulunamayan ya da
eksik veri, nerede arandığı belirtilmeden kabul edilemez bir yanıttır.

### IV. Teknoloji Kararları Kullanıcıya Aittir (Explicit User Sign-off on Stack Choices)

Protokol, framework, kütüphane ve mimari seçimleri, bir plana veya koda
yazılmadan önce proje sahibiyle açıkça teyit edilmelidir; agent kendi başına
bir teknoloji yığınına karar vermemelidir. Spesifikasyon aşamasında makul,
teknoloji-agnostik varsayımlar kabul edilebilir; ancak somut bir yığına
bağlanmak, planlama aşamasında proje sahibinin açık kararını gerektirir.

**Rationale**: Proje sahibinin doğrudan talimatı ("kafana göre kod yazma");
PoC'nin teknik kimliği kasıtlı olarak sahibi tarafından belirlenir, agent
tarafından varsayılmaz.

### V. Basitlik — PoC Kapsam Disiplini (Simplicity, PoC-Scope Discipline)

Yalnızca aktif spec'in kullanıcı hikayelerinin gerektirdiği kadarı inşa edilir.
Spec açıkça talep etmediği sürece spekülatif soyutlama, prodüksiyon sertleştirmesi
(çok-kiracılı yetkilendirme, yatay ölçekleme vb.) veya gelecekteki varsayımsal
ihtiyaçlar için tasarım yapılmaz.

**Rationale**: Bu bir kanıt (PoC) deposudur; erken soyutlama, çekirdek tezi
(PTC onaylı-kanal disiplini + zemine dayalılık) doğrulamayı yavaşlatır ve
gereksiz karmaşıklık ekler.

## Security & Data Handling Constraints

- Her tool/skill entegrasyonu varsayılan olarak dış erişimi reddetmeli (default-deny)
  ve yalnızca onaylı hedefleri izin listesine almalıdır — bu, depodaki PTC
  araştırma dokümanlarında tarif edilen eBPF/Cilium enforcement modelini yansıtır.
- Kullanıcıya özel kalıcı hafıza, tercih ve etkileşim bağlamıyla sınırlıdır;
  hassas kişisel veya kurumsal gizli verinin hafızada saklanması, ayrıca
  belgelenmiş açık bir karar gerektirir.
- Her araç çağrısı ve sonucu — kaynak, zaman damgası, başarı/hata durumu dahil —
  Principle III'ün gerektirdiği izlenebilirliği desteklecek ayrıntıda kaydedilmelidir.

## Development Workflow

- Özellikler Spec Kit akışından geçer: `/speckit-specify` → (opsiyonel
  `/speckit-clarify`) → `/speckit-plan` → `/speckit-tasks` → `/speckit-implement`.
- `/speckit-plan`, proje sahibinin açık onayı olmadan bir Technical Context
  (dil, framework, bağımlılıklar) sonuçlandıramaz — bu, Principle IV'ü bir
  süreç kapısı olarak uygular.
- `plan.md` içindeki Constitution Check, Phase 0 araştırmasına geçmeden önce
  yukarıdaki beş ilkenin her birini doğrulamalıdır; bir ihlal varsa Complexity
  Tracking'de gerekçelendirilmeli ya da plan buna göre revize edilmelidir.

## Governance

Bu anayasa, bu depo için geçerli ad hoc uygulamaların üzerindedir; bununla
çelişen herhangi bir spec/plan, ya buna uyacak şekilde düzeltilmeli ya da
planın Complexity Tracking bölümünde sapma gerekçelendirilmelidir.

Değişiklikler `/speckit-constitution` üzerinden önerilir, geri yazılmadan önce
proje sahibinin açık onayını gerektirir ve semantic versioning kurallarına göre
sürüm artırılır: MAJOR (bir ilke geriye dönük uyumsuz şekilde kaldırılır/yeniden
tanımlanır), MINOR (yeni ilke eklenir veya mevcut bir ilke maddi olarak genişletilir),
PATCH (yalnızca ifade/açıklık düzeltmeleri). Uyum denetiminin birincil noktası
`/speckit-plan`'ın Constitution Check kapısıdır.

**Version**: 1.0.0 | **Ratified**: 2026-08-27 | **Last Amended**: 2026-08-27
