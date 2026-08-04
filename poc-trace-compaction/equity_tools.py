"""
equity_tools.py — HF dataset (sccaglayanworkacc/equity-research-agentic-eval)
case'lerini çalıştırmak için SİMÜLE edilmiş finansal tool seti.

Amaç: trace compaction sistemini bu domain'e UYGUN hâle getirmek. Mock veri
gerçekçi ama canned (API/anahtar gerekmez); ajan orkestrasyonu ve trace GERÇEK,
compaction katmanı bu gerçek trace üzerinde çalışır.

ledger.py'nin GENEL sözleşmesini kullanır (TOOL_META): her tool kendi
  {cat, resource, ttl}
bilgisini bildirir → aynı ledger hem dosya hem ticker kaynağına çalışır.
ek-a mentalitesi: spesifik isim, "ne zaman çağrılacağını" söyleyen description.
"""
from __future__ import annotations


# --- mock veri (gerçekçi, canned) — ticker başına ----------------------------
_DATA = {
    "XOM": {
        "company_info": "Exxon Mobil Corp (XOM) — Energy / Integrated Oil & Gas. "
                        "Merkez: Irving, TX. Çalışan ~62,000. Piyasa değeri $470B. Beta 0.92.",
        "income": ("Gelir tablosu (yıllık, $B):\n"
                   "  revenue: 2023=344.6  2022=413.7  2021=285.6\n"
                   "  net_income: 2023=36.0  2022=55.7  2021=23.0\n"
                   "  EPS: 2023=8.89  2022=13.26 · gross_margin: 31%"),
        "ratios": ("Temel oranlar: P/E=13.2 · P/B=1.9 · ROE=17.6% · "
                   "debt/equity=0.20 · current_ratio=1.48 · dividend_yield=3.4%"),
        "technical": ("Teknik: RSI(14)=54.2 · MACD=+0.8 (bullish) · "
                      "SMA50=108.4 · SMA200=104.1 · 52w=95.8–120.7"),
        "analyst": ("Analist: Buy=12 · Hold=8 · Sell=1 · consensus=Moderate Buy · "
                    "avg_target=$128.5"),
        "price": "XOM son fiyat: $112.30 (piyasa açık, +0.8% gün içi)",
        "history": ("1y fiyat serisi (aylık kapanış): 98.1, 101.4, 105.9, 110.2, "
                    "108.7, 112.0, 115.3, 118.9, 116.4, 113.2, 110.8, 112.3"),
        "fundamentals": ("Temeller: market_cap=$470B · enterprise_value=$495B · "
                         "revenue_ttm=$338B · fcf=$33B · shares_out=4.0B"),
        "news": ("Son haberler (XOM): Q3 kâr beklentiyi aştı · Guyana üretimi rekor · "
                 "düşük-karbon yatırımı 2027'ye $17B · Arkansas lityum projesi onaylandı · "
                 "SEC iklim açıklama kuralına uyum planı yayınlandı."),
    }
}


def _pad(core: str, kind: str) -> str:
    """Gerçek finansal API çıktısı büyüktür (çok satır kalem). Mock'u o boyuta getir."""
    extra = "\n".join(f"  {kind}_line_{i}: {round(1000 + i * 13.7, 2)} "
                      f"(fy{2019 + i % 5}, seg={i % 4})" for i in range(30))
    return core + "\n" + extra


def _get(ticker, key):
    # boş/geçersiz ticker → GERÇEK hata (status=error) → hata-zinciri reachable
    if not ticker or not str(ticker).strip():
        raise ValueError("ticker zorunlu ve boş olamaz — geçerli bir sembol ver (ör. XOM)")
    t = _DATA.get(str(ticker).upper())
    if not t:
        # net sinyal: model başka tool denemesin, tek seferde dursun
        return (f"HATA: '{ticker}' bu demoda desteklenmiyor. Mevcut ticker: "
                f"{', '.join(_DATA)}. Başka finansal tool ÇAĞIRMA; kullanıcıya "
                f"bu sembol için veri olmadığını söyle.")
    # fiyat/haber kısa kalır; tablo/oran/geçmiş büyük (gerçekçi)
    if key in ("income", "ratios", "technical", "history", "fundamentals"):
        return _pad(t[key], key)
    return t[key]


# --- tool implementasyonları (mock) -----------------------------------------

def get_company_info(ticker: str) -> str: return _get(ticker, "company_info")
def get_income_statements(ticker: str, freq: str = "annual") -> str: return _get(ticker, "income")
def get_key_financial_ratios(ticker: str) -> str: return _get(ticker, "ratios")
def get_technical_indicators(ticker: str) -> str: return _get(ticker, "technical")
def get_analyst_recommendations(ticker: str) -> str: return _get(ticker, "analyst")
def get_stock_price(ticker: str) -> str: return _get(ticker, "price")
def get_historical_prices(ticker: str, period: str = "1y") -> str: return _get(ticker, "history")
def get_stock_fundamentals(ticker: str) -> str: return _get(ticker, "fundamentals")
def get_company_news(ticker: str) -> str: return _get(ticker, "news")
def web_search(query: str) -> str:
    return ("Arama sonuçları: XOM düşük-karbon 2027'ye $17B · CCS projeleri · "
            "Arkansas lityum girişimi · Q3 kâr beklentiyi aştı · Guyana ramp devam.")
def visualize_data(instruction: str = "", data: str = "") -> str:
    return f"Grafik üretildi: {instruction or 'finansal özet'} — kaydedildi chart_xom.png"


DISPATCH = {
    "get_company_info": get_company_info, "get_income_statements": get_income_statements,
    "get_key_financial_ratios": get_key_financial_ratios,
    "get_technical_indicators": get_technical_indicators,
    "get_analyst_recommendations": get_analyst_recommendations,
    "get_stock_price": get_stock_price, "get_historical_prices": get_historical_prices,
    "get_stock_fundamentals": get_stock_fundamentals, "get_company_news": get_company_news,
    "web_search": web_search, "visualize_data": visualize_data,
}


# --- GENEL ledger sözleşmesi (ledger.py bunu kullanır) ----------------------
# Her tool: kategori + kaynak anahtarı (+ volatilse ttl). Ledger domain-bağımsız
# çalışır: finansal fetch'ler ticker kaynağını okur; fiyat/teknik ZAMANLA bayatlar.
_tk = lambda a: a.get("ticker", "?")
# verbatim=True: çıktı kritik metrik taşır (P/E, revenue, hedef) → katlanırken korunur
TOOL_META = {
    "get_company_info": {"cat": "read", "resource": _tk},
    "get_income_statements": {"cat": "read", "resource": _tk, "verbatim": True},
    "get_key_financial_ratios": {"cat": "read", "resource": _tk, "verbatim": True},
    "get_stock_fundamentals": {"cat": "read", "resource": _tk, "verbatim": True},
    "get_analyst_recommendations": {"cat": "read", "resource": _tk, "verbatim": True},
    "get_historical_prices": {"cat": "read", "resource": _tk},
    "get_company_news": {"cat": "read", "resource": _tk, "ttl": 6},           # haber zamanla eskir
    "get_technical_indicators": {"cat": "read", "resource": _tk, "ttl": 4},   # volatil
    "get_stock_price": {"cat": "read", "resource": _tk, "ttl": 1},            # çok volatil
    "web_search": {"cat": "search", "resource": lambda a: a.get("query", "")},
    "visualize_data": {"cat": "write", "resource": lambda a: "chart"},        # act (sentez)
}


# --- OpenAI-uyumlu şemalar (ek-a mentalitesi) -------------------------------
def _schema(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}

_TICKER = {"ticker": {"type": "string", "description": "Hisse sembolü, ör. 'XOM'"}}

SCHEMAS = [
    _schema("get_company_info", "Şirketin iş profili/sektör/özet bilgisini döndürür. "
            "Bir hisseyi analiz etmeye başlarken önce bunu çağır.", _TICKER, ["ticker"]),
    _schema("get_income_statements", "Gelir tablosunu (revenue, net income, EPS) döndürür. "
            "Gelir/kârlılık trendi gerektiğinde çağır.",
            {**_TICKER, "freq": {"type": "string", "description": "'annual' veya 'quarterly'"}}, ["ticker"]),
    _schema("get_key_financial_ratios", "P/E, ROE, borç/özkaynak gibi temel oranları döndürür. "
            "Değerleme sorulduğunda çağır.", _TICKER, ["ticker"]),
    _schema("get_technical_indicators", "RSI, MACD, hareketli ortalamaları döndürür. "
            "Teknik analiz gerektiğinde çağır (veri zamanla bayatlar).", _TICKER, ["ticker"]),
    _schema("get_analyst_recommendations", "Analist alım/satım dağılımı ve hedef fiyatı döndürür.",
            _TICKER, ["ticker"]),
    _schema("get_stock_price", "Anlık hisse fiyatını döndürür (çok volatil — kısa süre sonra bayat).",
            _TICKER, ["ticker"]),
    _schema("get_historical_prices", "Geçmiş fiyat serisini döndürür.",
            {**_TICKER, "period": {"type": "string", "description": "ör. '1y', '6mo'"}}, ["ticker"]),
    _schema("get_stock_fundamentals", "Piyasa değeri, FCF, hisse sayısı gibi temelleri döndürür.",
            _TICKER, ["ticker"]),
    _schema("get_company_news", "Şirkete özel SON haberleri döndürür (yapılandırılmış). "
            "Belirli bir şirketin güncel gelişmeleri sorulduğunda web_search yerine bunu çağır "
            "(haber zamanla eskir).", _TICKER, ["ticker"]),
    _schema("web_search", "İnternette GENEL/sektörel arama yapar. Tek şirkete bağlı olmayan "
            "makro/sektör bilgisi gerektiğinde çağır (şirket haberi için get_company_news).",
            {"query": {"type": "string", "description": "Arama sorgusu"}}, ["query"]),
    _schema("visualize_data", "Toplanan finansal veriden bir özet grafik üretir. "
            "Tüm veri toplandıktan SONRA, sentez adımında çağır.",
            {"instruction": {"type": "string", "description": "Grafik açıklaması"},
             "data": {"type": "string", "description": "Grafiklenecek veri özeti"}}, ["instruction"]),
]

# CWL delimiter (§5.2): ajanın çalışmasını expl/act episode'larına bölmesi için.
# Faz 6 (bağımlılık-farkında episode eviction) ancak ajan bunu çağırırsa devreye girer.
from tools import SCHEMAS as _FILE_SCHEMAS
_DELIMITER = next(s for s in _FILE_SCHEMAS if s["function"]["name"] == "delimiter")
SCHEMAS.append(_DELIMITER)
