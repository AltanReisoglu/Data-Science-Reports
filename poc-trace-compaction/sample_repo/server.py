"""Basit HTTP sunucu iskeleti."""
import config


def start():
    """Sunucuyu config'teki HOST:PORT üzerinde başlatır."""
    print(f"Sunucu başlıyor: {config.HOST}:{config.PORT}")
    print(f"Debug: {config.DEBUG}, timeout: {config.TIMEOUT}s")
    # ... gerçek bind burada olurdu
    return (config.HOST, config.PORT)


def health():
    """Sağlık kontrolü — sunucunun dinlediği portu döndürür."""
    return {"status": "ok", "port": config.PORT}
