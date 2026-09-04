# Arşiv — Egress Policy (önceki case)

Bu dizin, projenin **önceki** case'ine ait dokümanları tutuyor:

> PTC: egress policy (eBPF / Cilium) — only via approved tool channels

2026-09-03'te arşivlendi. Sebep: aktif case artık **artifact persistence**
(`docs/topic_is_this.md`), ve bu 16 doküman kod yazarken gürültü yapıyordu.

**Hiçbiri silinmedi, sadece taşındı.** Egress tarafına dönmek gerekirse hepsi burada.

## İçerik

| Dosya | Ne |
|---|---|
| `PTC_egress_policy_eBPF_Cilium.md` | Ana araştırma dokümanı |
| `..._addendum.md`, `..._Technical_Reference.md` | Ek + teknik referans |
| `PTC_Egress_Policy_Implementation_Walkthrough.md` | Uygulama adımları |
| `PTC_Egress_Policy_OpenAI_Incident.md` | Vaka analizi |
| `PTC_Ingress_Policy_Implementation_Plan.md` | Ingress tarafı planı |
| `PTC_Calisma_Sureci_Kubernetes_Cilium.md` | Bir çalıştırmada K8s+Cilium süreci |
| `PTC_Faz2_Dosya_Rehberi.md` | Faz 2 dosya rehberi (egress tezi merkezli) |
| `PTC_Daha_Hafif_Alternatifler_Arastirmasi.md` | K8s Job maliyetine alternatifler |
| `PTC_OpenShift_Uyumluluk_Arastirmasi.md` | CNI değişimi / OpenShift |
| `PTC_Dokumantasyon_Rehberi.md`, `PTC_Tum_Dokumanlar_Ozeti.md` | Egress dönemi doküman indeksi/özeti |
| `PTC_Live_Demo_Script.md` | Sunum demo script'i |
| `onayli-kanal-slaytlar.html`, `Onayli_Kanal_Sunum.pdf` | Sunum |
| `topic_is_this.md.eski` | Önceki case tanımı |

## Arşivlenmeyenler ve sebebi

Bunlar egress'ten bahsetse de **kökte kaldı**, çünkü hâlâ gerekli:

- `PTC_Komut_Referansi.md`, `PTC_Calisma_Komutlari.md` — cluster'ı kurma/çalıştırma
  komutları; sandbox'ı koşturmak için lazım
- `PTC_Kubernetes_Yapisi.md` — kaynak yapısı referansı; MinIO/volume eklerken lazım
- `k8s/policies/*.ciliumnetworkpolicy.yaml` — **kod**. Üstelik artifact deposu
  eklenirse sandbox egress kuralının düzenlenmesi gerekebilir (2026-09-03'te
  kullanıcı egress politikasının değiştirilebilir olduğunu belirtti)
