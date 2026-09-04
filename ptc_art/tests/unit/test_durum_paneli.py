"""Durum paneli veri toplayıcısı (2026-09-04).

Panelin tezi: Hubble ağ akışlarını, OpenShift Console pod'ları, MinIO Console
bucket'ı, MLMD UI artifact soyunu gösteriyor — ama bizim yapımızda bu dördü
birbirine bağlı ve tek ekranda gösteren hazır bir şey yok.

Sınanan davranış, panelin en kritik özelliği: **üç toplayıcı bağımsız**.
Cluster kapalıyken sayfanın tamamen kararması, "bu bölüm okunamadı" demekten
kötüdür.
"""

from __future__ import annotations

from grounded_assistant.web import durum


def test_kubernetes_kapaliyken_hata_doner_patlamaz(monkeypatch):
    def patla():
        raise RuntimeError("cluster yok")

    monkeypatch.setattr(durum, "_core", patla)
    sonuc = durum.podlar()
    assert "error" in sonuc and "cluster yok" in sonuc["error"]


def test_artifact_servisi_tanimsizsa_yol_gosteren_hata(monkeypatch):
    monkeypatch.delenv("ARTIFACT_SERVICE_URL", raising=False)
    sonuc = durum.artifactler("wf", lambda w: "jeton")
    assert "ARTIFACT_SERVICE_URL" in sonuc["error"]
    assert "port-forward" in sonuc["error"]  # ne yapılacağını söylüyor


def test_workflow_yoksa_hata_degil_bilgi(monkeypatch):
    """Oturum henüz açılmamışsa bu bir HATA değil — normal ilk hâl."""
    monkeypatch.setenv("ARTIFACT_SERVICE_URL", "http://ornek:8080")
    sonuc = durum.artifactler(None, lambda w: "jeton")
    assert "error" not in sonuc and sonuc["kayitlar"] == []


def test_jeton_uretilemezse_acik_hata(monkeypatch):
    monkeypatch.setenv("ARTIFACT_SERVICE_URL", "http://ornek:8080")
    sonuc = durum.artifactler("wf", lambda w: None)
    assert "Kapsam jetonu" in sonuc["error"]


def test_bilesenlerin_yetenekleri_dogru_isaretli():
    """Panelin anlattığı asıl şey bu: hangi pod neye erişebiliyor."""
    esleme = {b["ad"]: b for b in durum.BILESENLER}
    assert esleme["tool-gateway"]["internet"] is True
    assert esleme["tool-gateway"]["depo"] is False
    assert esleme["artifact-service"]["internet"] is False
    assert esleme["artifact-service"]["depo"] is True


def test_yas_bicimi_okunabilir():
    from datetime import UTC, datetime, timedelta

    simdi = datetime.now(UTC)
    assert durum._yas(None) == "—"
    assert durum._yas(simdi - timedelta(seconds=30)).endswith("sn")
    assert durum._yas(simdi - timedelta(minutes=10)).endswith("dk")
    assert durum._yas(simdi - timedelta(hours=5)).endswith("sa")
    assert durum._yas(simdi - timedelta(days=3)).endswith("gün")
