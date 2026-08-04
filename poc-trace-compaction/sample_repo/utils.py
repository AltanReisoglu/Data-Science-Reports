"""Yardımcı fonksiyonlar."""


def format_addr(host, port):
    """host ve port'u tek bir adres string'ine çevirir."""
    return f"{host}:{port}"


def is_valid_port(port):
    """Port geçerli aralıkta mı (1-65535)."""
    return isinstance(port, int) and 1 <= port <= 65535
