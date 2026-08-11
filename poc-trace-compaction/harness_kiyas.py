"""
harness_kiyas.py — AYNI tool trace'ini altı farklı compaction mantığından geçirir.

Soru şu: bir ajanın tool geçmişi bağlama sığmadığında, hangi coding agent onu
NASIL küçültür ve geriye NE bırakır? Burada tek bir sohbetin gerçek trace'i
alınıp altı mantıktan ayrı ayrı geçirilir; her tool için ÖNCE (ham çıktı) ve
SONRA (o mantığın context'te bıraktığı metin) kırpılmadan çıkarılır.

  cwl          — bu POC'nin kendi mantığı (deterministik kademeli eviction)
  hermes       — dedup → tip-farkında özet → arg-kırp → basınç demotion
  opencode     — canlı spill (diske dök) + backward-prune
  openclaw     — grupla → parçala → LLM chunk-özeti
  codex        — ortadan-kesme + model-turn windowing (handoff özeti)
  claude_code  — microcompaction (diske dök + referans) + auto-compaction

Beş harness mantığı `demo-brain-agent/compaction.py`'den geliyor — orada zaten
gerçek sistemlerin davranışı taklit edilmiş durumda, burada yeniden yazmıyoruz.

EŞLEŞTİRME: her tool mesajının `tool_call_id`'si var ve stratejiler mesajı
değiştirirken bu alanı koruyor (`_View.replaced` kopyalıyor). SONRA'yı bu id ile
buluyoruz — aynı tool iki kez çağrıldığında ada göre eşleştirme yanlış satırı
gösterirdi. Id ortadan kalkmışsa o birim o mantıkta **düşürülmüş** demektir.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from config import estimate_tokens

_DBA = Path(__file__).resolve().parent.parent / "demo-brain-agent"
if str(_DBA) not in sys.path:
    sys.path.insert(0, str(_DBA))

import compaction as CP   # noqa: E402  (yol yukarıda kuruluyor)

# Panelde soldan sağa bu sırayla sekme olur. "none" yok — sıkıştırmayan bir
# mantığın ÖNCE/SONRA'sı aynı olurdu, sekme açmaya değmez.
HARNESSLER = ("hermes", "opencode", "openclaw", "codex", "claude_code")


# ───────────────────────────── tool'u insan diline çevir ─────────────────────

def _ilk_cumle(s: str, n: int = 90) -> str:
    s = (s or "").strip().replace("\n", " ")
    for nokta in (". ", " — ", "; "):
        if nokta in s:
            s = s.split(nokta, 1)[0]
            break
    return s[:n]


def tool_sozlugu(schemas) -> dict:
    """SCHEMAS'tan tool adı → insan okunur açıklama."""
    d = {}
    for s in schemas or []:
        f = s.get("function") or {}
        if f.get("name"):
            d[f["name"]] = _ilk_cumle(f.get("description", ""))
    return d


def arg_metni(args: dict, n: int = 60) -> str:
    """{'ticker':'XOM'} → 'ticker=XOM' (kısa, okunur)."""
    if not args:
        return ""
    p = []
    for k, v in args.items():
        sv = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
        p.append(f"{k}={sv}" if len(str(sv)) <= 24 else f"{k}={str(sv)[:21]}…")
    return ", ".join(p)[:n]


def ilk_satir(s: str, n: int = 76) -> str:
    """Ham çıktının ilk anlamlı satırı — listede 'ne döndü' sütunu için."""
    for ln in (s or "").splitlines():
        if ln.strip():
            return ln.strip()[:n]
    return ""


# ───────────────────────────── mesaj köprüsü ─────────────────────────────────

def mesajlar(ag) -> list[dict]:
    """Ajanın HAM mesaj dizisi + her tool mesajına `name` eklenmiş kopyası.

    `ag.messages` sıkıştırılmamış kaynaktır (compaction yalnız render anında
    uygulanıyor) — yani harness mantıkları ajanın gördüğü ham bağlamı görür.
    `name` alanı lazım çünkü hermes/opencode özetleri tool ADINA göre üretiyor;
    OpenAI şemasında tool mesajında ad yok, yalnız tool_call_id var.
    """
    out = []
    for m in (ag.messages or []):
        if m.get("role") == "tool":
            seq = ag._call_seq.get(m.get("tool_call_id"))
            ev = ag.trace.by_seq(seq) if seq is not None else None
            ad = (ev.payload.get("name") if ev else "") or "tool"
            out.append({**m, "name": ad})
        else:
            out.append(dict(m))
    return out


def _tool_satirlari(ag, sozluk) -> list[dict]:
    """Her tool çağrısının kimliği + ham çıktısı (mantıktan bağımsız kısım)."""
    satir = []
    for tcid, seq in ag._call_seq.items():
        ev = ag.trace.by_seq(seq)
        if ev is None or ev.type != "tool":
            continue
        ad = ev.payload.get("name", "?")
        ham = str(ev.payload.get("output", ""))
        satir.append({
            "id": tcid, "seq": ev.seq, "ad": ad,
            "aciklama": sozluk.get(ad, ""),
            "arg": arg_metni(ev.payload.get("args") or {}),
            "durum": ev.status,
            "ozet_satir": ilk_satir(ham),
            "once": ham, "once_tok": estimate_tokens(ham),
        })
    satir.sort(key=lambda r: r["seq"])
    return satir


def _tok(s: str) -> int:
    """Token tahmini. config.estimate_tokens boş metne 1 döndürüyor (max(1,…));
    burada 0 lazım — düşürülmüş bir birim '1 token' yer kaplıyormuş gibi
    görünmemeli."""
    return estimate_tokens(s) if s else 0


def _oran(once: int, sonra: int) -> float:
    return round((1 - sonra / once) * 100, 1) if once else 0.0


# ───────────────────────────── mantık 0: bu POC (CWL) ────────────────────────

def _cwl(ag, sozluk, satirlar, msgs) -> dict:
    """POC'nin kendi compaction'ı: olayın kaderi zaten trace'te yazılı.

    ÖLÇEK BİRLİĞİ: toplam `once`/`sonra` burada da CP.total_tokens ile ölçülüyor.
    Ajanın kendi `raw_token_cost`/`rendered_token_cost`'u farklı bir tahminci
    kullanıyor; onu kullansak CWL sekmesi diğer beşiyle kıyaslanamayan bir sayı
    gösterirdi (ölçüldü: 3.396 vs 2.578 — aynı bağlam, iki farklı cetvel).
    """
    toollar = []
    for r in satirlar:
        ev = ag.trace.by_seq(r["seq"])
        if ev.cleared:
            kader, sonra = "silindi", "[silindi] " + ev.clear_note
        elif ev.evicted and ev.summary is not None:
            kader, sonra = "özet", ev.summary.render()
        else:
            kader, sonra = "tam", r["once"]
        st = _tok(sonra)
        toollar.append({**r, "kader": kader, "sonra": sonra, "sonra_tok": st,
                        "pct": _oran(r["once_tok"], st),
                        "neden": ev.neden or ("dokunulmadı — koruma penceresi ya da bütçe içinde"
                                              if kader == "tam" else "")})
    once = CP.total_tokens(msgs)
    sonra = CP.total_tokens(ag._render_messages(msgs))
    return {"ad": "cwl", "baslik": "CWL · bu POC", "ekol": "deterministik", "llm": False,
            "ozet": "Kademeli eviction: dedup → bayat → hata-zinciri → keşif katlama → "
                    "kategori → CWL episode → acil (en büyük önce). Koruma penceresi "
                    "son 3 birime dokunmaz; öğrenilen dersler playbook'a taşınır.",
            "once": once, "sonra": sonra, "pct": _oran(once, sonra),
            "tetiklendi": ag.metrics["compaction_passes"] > 0,
            "log": ag.compactor.log[-14:], "toollar": toollar, "ek_mesajlar": []}


# ───────────────────────────── mantık 1-5: harness'ler ───────────────────────

def _harness(ad: str, msgs: list[dict], satirlar: list[dict], budget: int) -> dict:
    bilgi = CP.STRATEGY_INFO.get(ad, {})
    try:
        r = CP.compact(ad, msgs, budget=budget)
    except Exception as e:                       # bir mantık patlarsa diğerleri sürsün
        return {"ad": ad, "baslik": ad, "ekol": bilgi.get("ekol", "—"),
                "llm": bilgi.get("llm", False), "ozet": bilgi.get("ozet", ""),
                "once": 0, "sonra": 0, "pct": 0.0, "tetiklendi": False,
                "log": [f"HATA: {type(e).__name__}: {e}"], "toollar": [], "ek_mesajlar": []}

    sonra_by_id, ek = {}, []
    for m in r.messages:
        v = CP._View(m)
        if v.role == "tool" and v.tool_call_id:
            sonra_by_id[v.tool_call_id] = v.content

    # Bu mantığın ÜRETTİĞİ yeni metin (handoff özeti, birleşik özet…): girişte
    # olmayan içerik. Birleştiren stratejilerde tool'ların gittiği yer burasıdır,
    # bu yüzden ayrıca gösteriliyor — yoksa "düştü" yanıltıcı olurdu.
    giris_metin = {(m.get("content") or "") for m in msgs}
    for m in r.messages:
        v = CP._View(m)
        if v.role == "tool" and v.tool_call_id:
            continue
        if v.content and v.content not in giris_metin:
            ek.append({"rol": v.role, "metin": v.content,
                       "tok": CP.est(v.content)})

    toollar = []
    for s in satirlar:
        if s["id"] in sonra_by_id:
            sonra = sonra_by_id[s["id"]]
            kader = "tam" if sonra == s["once"] else "özet"
        else:
            sonra = ""
            kader = "birleşti" if ek else "düştü"
        st = _tok(sonra)
        toollar.append({**s, "kader": kader, "sonra": sonra, "sonra_tok": st,
                        "pct": _oran(s["once_tok"], st),
                        "neden": _neden(ad, kader)})

    return {"ad": ad, "baslik": ad, "ekol": bilgi.get("ekol", "—"),
            "llm": bilgi.get("llm", False), "ozet": bilgi.get("ozet", ""),
            "once": r.before, "sonra": r.after, "pct": round(r.pct, 1),
            "tetiklendi": r.triggered, "log": r.log[-14:],
            "toollar": toollar, "ek_mesajlar": ek}


_NEDEN = {
    "hermes":      {"özet": "tip-farkında tek satır özet (Pass 2) ya da dedup",
                    "düştü": "basınç altında en eski birim demote edildi (Pass 4)"},
    "opencode":    {"özet": "eşik üstü çıktı diske döküldü → context'e önizleme + referans",
                    "düştü": "backward-prune: koruma penceresi dışında kaldı"},
    "openclaw":    {"özet": "grubun LLM chunk-özetine indi",
                    "birleşti": "birden çok tool tek LLM özetinde birleşti"},
    "codex":       {"özet": "truncate_middle — baş ve son tutuldu, orta atıldı",
                    "birleşti": "model-turn windowing: handoff özetine girdi, yeni pencere açıldı"},
    "claude_code": {"özet": "microcompaction — çıktı diske döküldü, yerine referans kaldı",
                    "birleşti": "auto-compaction: konuşma özetine girdi"},
}


def _neden(ad: str, kader: str) -> str:
    if kader == "tam":
        return "dokunulmadı — koruma penceresi ya da bütçe içinde"
    return _NEDEN.get(ad, {}).get(kader, "")


# ───────────────────────────── dışa açık giriş ───────────────────────────────

def kiyasla(ag, budget: int | None = None) -> dict:
    """Altı mantığın tamamını koştur ve panele hazır JSON üret."""
    sozluk = tool_sozlugu(getattr(ag, "schemas", None))
    satirlar = _tool_satirlari(ag, sozluk)
    msgs = mesajlar(ag)
    bt = int(budget or ag.compactor.budget)

    mantiklar = [_cwl(ag, sozluk, satirlar, msgs)]
    for ad in HARNESSLER:
        mantiklar.append(_harness(ad, msgs, satirlar, bt))
    return {"budget": bt, "tool_sayisi": len(satirlar),
            "ham": CP.total_tokens(msgs), "mantiklar": mantiklar}
