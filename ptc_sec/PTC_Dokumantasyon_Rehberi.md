# PTC Dokümantasyon Rehberi — Hepsi Bir Arada

Bu proje, tek bir tez etrafında ("Sandbox/agent ortamlarının dış ağ erişimi eBPF/Cilium
ile merkezi olarak kontrol edilir") 13 ayrı Markdown dokümanı + bir slayt destesi + bir
PDF biriktirdi. Bu rehber, hangi dokümanın NE İÇİN var olduğunu ve hangi sırayla
okunması gerektiğini özetliyor — kaybolmamak için.

## Önce bunu okuyun — sunum

- **`Onayli_Kanal_Sunum.pdf`** (28 sayfa) / **`onayli-kanal-slaytlar.html`** (kaynağı) —
  projenin TAMAMININ, bir seyirciye anlatılacak şekilde özetlendiği yer. Foundational
  kavramlardan (eBPF/Cilium/Kubernetes) başlayıp, bulunan zafiyet+düzeltmeye, maliyet
  optimizasyonu alternatiflerine, canlı demo planına kadar hepsi burada.

## Kavramsal temel — konuyu ilk anlarken yazılanlar (kronolojik olarak en eski)

1. **`docs/topic_is_this.md`** — projenin tek cümlelik tezi, her şeyin kök nedeni.
2. **`PTC_egress_policy_eBPF_Cilium.md`** — "egress policy" kavramının kendisinin
   ilk, en temel açıklaması ("Konunun özü" ile başlıyor).
3. **`PTC_Egress_Policy_OpenAI_Incident.md`** — konuyu somutlaştıran, gerçek bir
   olay/örnek üzerinden anlatan doküman — "destekleyici servisin de kısıtlı olması
   gerekir" ilkesinin (Tool Gateway'in kendi egress'i) kaynağı burası.
4. **`PTC_egress_policy_eBPF_Cilium_addendum.md`** — "sadece onaylı tool/API
   kanallarına erişim" ifadesindeki "tool" kelimesinin tam olarak neyi kapsadığını
   netleştiren ek doküman.
5. **`PTC_Egress_Policy_Cilium_eBPF_Technical_Reference.md`** — yukarıdaki 3
   kavramsal dokümanın ÜZERİNE kurulu, somut teknik referans (onların yerine
   geçmiyor, üzerine ekleniyor).
6. **`PTC_Faz2_Dosya_Rehberi.md`** — Faz 2'nin (Cilium/eBPF sandbox) hangi
   dosyasının ne işe yaradığının rehberi.

## Uygulama detayı — bu oturumda yazılanlar (en güncel, en somut)

7. **`PTC_Egress_Policy_Implementation_Walkthrough.md`** — projenin İKİ KATMANLI
   savunmasının (sandbox-egress + tool-gateway-egress) baştan sona nasıl çalıştığı,
   artı bu oturumda bulunan **paylaşılan-IP zafiyeti + SNI düzeltmesi** ve **Hubble
   akış tamponu doluluğu** bulgularının canlı kanıtlarıyla anlatımı. **En önemli tek
   doküman** — deck'in düzyazı hâli gibi düşünülebilir.
8. **`PTC_Kubernetes_Yapisi.md`** — cluster/node seviyesinden tek tek kaynaklara
   (Deployment, Service, Job/ConfigMap, 3 CiliumNetworkPolicy) kadar tüm yapının
   envanteri.
9. **`PTC_Calisma_Sureci_Kubernetes_Cilium.md`** — bir PTC çalıştırması sırasında
   Kubernetes'in VE Cilium'un (özellikle `cilium-agent`'ın CNI akışının) tam olarak
   hangi sırayla, ne zaman devreye girdiğinin zaman çizelgesi.
10. **`PTC_Ingress_Policy_Implementation_Plan.md`** — CiliumNetworkPolicy ingress
    kuralı vs Cilium Ingress Controller/Gateway API ayrımı, ve `tool-gateway-ingress`
    politikasının (henüz uygulanmamış, hazır) uygulama planı.

## Operasyonel referans — çalıştırırken/hata ayıklarken

11. **`PTC_Komut_Referansi.md`** — sıfırdan kurulumdan Hubble tampon sıfırlamaya
    kadar tüm komutlar, tek yerde.
12. **`PTC_Live_Demo_Script.md`** — canlı demo'nun 5. adımı (prompt-injection
    senaryosu) için tam çalıştırma script'i — canlı test edilmiş, bir hatası
    bulunup düzeltilmiş.

## Araştırma — "daha iyisi var mı" soruları

13. **`PTC_Daha_Hafif_Alternatifler_Arastirmasi.md`** — ~7sn'lik PTC maliyetine
    alternatifler: `SandboxWarmPool` (Kubernetes-native, en olgun), Firecracker
    snapshot/restore (en radikal, milisaniyeler), WebAssembly/WASI (farklı bir
    paradigma). Deck'in 24-26. slaytlarının kaynağı.
14. **`PTC_OpenShift_Uyumluluk_Arastirmasi.md`** — ekip OpenShift kullandığı için:
    Cilium OpenShift'e kurulur mu (evet, ama CNI GÖÇÜ riskli), OpenShift'in native
    `EgressFirewall`'ı ile karşılaştırma, ve "Cilium'da kalma" kararının gerekçesi
    (SNI/`serverNames` koruması EgressFirewall'da yok).

## Önerilen okuma sırası — role göre

- **Sunum yapacaksanız:** Sadece PDF yeterli.
- **Kodu anlamak isteyen biriyseniz:** #7 (Walkthrough) → #9 (Çalışma Süreci) → #8
  (Kubernetes Yapısı).
- **Demo'yu çalıştıracaksanız:** #11 (Komut Referansı) → #12 (Demo Script).
- **"Neden böyle tasarladık" sorusuna cevap arıyorsanız:** #1-6 (kavramsal temel),
  kronolojik sırayla.
- **"Daha iyisi/başka platformda olur mu" diye merak ediyorsanız:** #13-14.
