# Kaçış denemesi (T019, quickstart.md Senaryo 2 / US2 — "bu fazın asıl testi").
#
# quickstart.md'nin ilk taslağı `requests` kullanıyordu; sandbox image bilerek
# minimal tutulduğundan (research.md §4.3) burada onun yerine stdlib
# `urllib.request` kullanılıyor — yeni bir paket indirmeden AYNI ağ-seviyesi
# testini yapar: Tool Gateway dışında hiçbir yere çıkamaması gereken sandbox'ın
# gerçekten çıkamadığını, kernel/Cilium seviyesinde kanıtlar.
#
# Not: bu kod HİÇBİR Python-seviyesi kısıtlamayla karşılaşmaz (import serbest,
# socket serbest) — engelleme SADECE Cilium'un CiliumNetworkPolicy'sinde
# (k8s/policies/sandbox-egress.ciliumnetworkpolicy.yaml) gerçekleşir.
import urllib.request

try:
    urllib.request.urlopen("https://google.com", timeout=8)
    set_result("KACIS BASARILI - bu asla olmamali (SC-002 ihlali)")
except Exception as exc:  # noqa: BLE001 - kaçış denemesinin sonucu, ne olursa raporlanmalı
    set_result(f"engellendi: {type(exc).__name__}: {exc}")
