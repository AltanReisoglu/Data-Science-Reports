#!/usr/bin/env python3
"""
gosterim.py — Her tool-trace stratejisinin NE YAPTIĞINI tam metinle gösterir.

POC'lardaki sorun: 100K token'lık sentetik bloblar okunmuyordu. Burada iz
KÜÇÜK ama GERÇEKÇİ tutuldu — her mesajın içeriğini olduğu gibi görebilirsin,
kısaltma yok. Bütçe de küçük seçildi ki stratejiler gerçekten tetiklensin.

    python3 gosterim.py              # tüm stratejiler
    python3 gosterim.py hermes       # tek strateji
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compaction as CP

BUDGET = 400          # küçük bütçe → stratejiler tetiklensin
W = 96                # satır genişliği


# ─────────────────────── ÖRNEK İZ (küçük ama gerçekçi) ───────────────────────

LOGIN_PY = """import hashlib, secrets, time
from db import users, sessions
from mfa import verify_totp, send_sms_code

SESSION_TTL = 3600
MAX_ATTEMPTS = 5


def _hash(pw: str, salt: str) -> str:
    return hashlib.sha256((salt + pw).encode()).hexdigest()


def verify_password(user: str, password: str) -> bool:
    row = users.get(user)
    if not row:
        return False
    return _hash(password, row['salt']) == row['pw_hash']


def verify_mfa(user: str, token: str) -> bool:
    row = users.get(user)
    if row.get('mfa_type') == 'totp':
        return verify_totp(row['totp_secret'], token)
    return row.get('sms_code') == token


def create_session(user: str) -> str:
    sid = secrets.token_hex(16)
    sessions[sid] = {'user': user, 'exp': time.time() + SESSION_TTL}
    return sid


def login(user, password, mfa_token=None):
    \"\"\"Kullanıcı girişi. Şifre + (varsa) MFA doğrular, oturum açar.\"\"\"
    row = users.get(user)
    if not row:
        return {'ok': False, 'error': 'no_such_user'}
    if row.get('attempts', 0) >= MAX_ATTEMPTS:
        return {'ok': False, 'error': 'locked'}
    if not verify_password(user, password):
        row['attempts'] = row.get('attempts', 0) + 1
        return {'ok': False, 'error': 'bad_credentials'}
    if mfa_token is None:
        return {'ok': True, 'user': user, 'sid': create_session(user)}   # <<< BUG
    if not verify_mfa(user, mfa_token):
        return {'ok': False, 'error': 'bad_mfa'}
    row['attempts'] = 0
    return {'ok': True, 'user': user, 'mfa': True, 'sid': create_session(user)}"""

TEST_OUT = """============================= test session starts ==============================
platform linux -- Python 3.12.3, pytest-8.2.0
rootdir: /srv/app, configfile: pyproject.toml
collected 12 items

tests/test_auth.py::test_login_ok PASSED                                 [  8%]
tests/test_auth.py::test_bad_password PASSED                             [ 16%]
tests/test_auth.py::test_no_such_user PASSED                             [ 25%]
tests/test_auth.py::test_lockout_after_5 PASSED                          [ 33%]
tests/test_auth.py::test_mfa_required FAILED                             [ 41%]
tests/test_auth.py::test_mfa_totp_ok PASSED                              [ 50%]
tests/test_auth.py::test_mfa_sms_ok PASSED                               [ 58%]
tests/test_auth.py::test_session_created PASSED                          [ 66%]
tests/test_auth.py::test_session_expiry FAILED                           [ 75%]
tests/test_auth.py::test_session_refresh PASSED                          [ 83%]
tests/test_auth.py::test_logout PASSED                                   [ 91%]
tests/test_auth.py::test_concurrent_login PASSED                         [100%]

=================================== FAILURES ===================================
______________________________ test_mfa_required _______________________________
    def test_mfa_required():
        u = make_user('ali', mfa_type='totp')
        r = login('ali', 'dogru-sifre')          # mfa_token verilmedi
>       assert r['ok'] is False
E       AssertionError: MFA baypas edilebiliyor (mfa_token=None ile giriş başarılı)

========================= 2 failed, 10 passed in 2.4s =========================="""

GREP_OUT = """auth/login.py:37: def login(user, password, mfa_token=None):
auth/login.py:47:     if mfa_token is None:
auth/login.py:50:     if not verify_mfa(user, mfa_token):
auth/session.py:12: def session_from_login(sid, user):
auth/session.py:44: def refresh_login_token(sid):
auth/session.py:61: LOGIN_AUDIT = 'auth.login'
api/routes.py:12:   @app.post('/login')
api/routes.py:13:   def login_route(): return login(**request.json)
api/routes.py:28:   @app.post('/login/mfa')
api/routes.py:29:   def login_mfa_route(): return login(**request.json)
tests/test_auth.py:7:  from auth.login import login
tests/test_auth.py:19: def test_login_ok():
tests/test_auth.py:31: def test_mfa_required():
db/models.py:88:    last_login = Column(DateTime)
db/models.py:89:    login_attempts = Column(Integer, default=0)"""


def trace():
    """Her seferinde AYNI iz — stratejiler adil karşılaştırılsın."""
    return [
        {"role": "system", "content": "Sen bir kod-denetim ajanısın.", "name": "", "tool_calls": []},
        {"role": "user", "content": "auth/login.py'daki MFA hatasını bul ve testlerle doğrula.",
         "name": "", "tool_calls": []},
        {"role": "assistant", "content": "", "name": "",
         "tool_calls": [{"name": "read_file", "args": {"path": "auth/login.py"}}]},
        {"role": "tool", "content": LOGIN_PY, "name": "read_file", "tool_calls": []},
        {"role": "assistant", "content": "", "name": "",
         "tool_calls": [{"name": "grep", "args": {"pattern": "login"}}]},
        {"role": "tool", "content": GREP_OUT, "name": "grep", "tool_calls": []},
        {"role": "assistant", "content": "", "name": "",
         "tool_calls": [{"name": "run_tests", "args": {"suite": "auth"}}]},
        {"role": "tool", "content": TEST_OUT, "name": "run_tests", "tool_calls": []},
        # AYNI dosyayı ikinci kez okuma → Hermes'in dedup'ı burada görünür
        {"role": "assistant", "content": "", "name": "",
         "tool_calls": [{"name": "read_file", "args": {"path": "auth/login.py"}}]},
        {"role": "tool", "content": LOGIN_PY, "name": "read_file", "tool_calls": []},
        {"role": "assistant", "content": "login() içinde mfa_token None ise kontrol atlanıyor.",
         "name": "", "tool_calls": []},
    ]


# ─────────────────────── gösterim yardımcıları ───────────────────────

def box(txt: str, pad: str = "  │ ") -> str:
    """Metni olduğu gibi, satır satır çerçeve içinde göster (KISALTMA YOK)."""
    if not txt:
        return pad + "(boş)"
    return "\n".join(pad + ln for ln in txt.split("\n"))


def label(v) -> str:
    if v.role == "tool":
        return f"tool[{v.tool_name}]"
    if v.role == "assistant" and v.tool_calls:
        names = ", ".join(tc.get("name", "?") for tc in v.tool_calls)
        return f"assistant → çağrı: {names}"
    return v.role


def show(strategy: str):
    before = trace()
    res = CP.compact(strategy, before, budget=BUDGET)
    bv, av = CP.views(before), CP.views(res.messages)

    info = CP.STRATEGY_INFO[strategy]
    print("\n" + "═" * W)
    print(f"  {strategy.upper()}  —  {info['ozet']}")
    print(f"  ekol: {info['ekol']} · LLM: {'evet' if info['llm'] else 'hayır'} · bütçe: {BUDGET} token")
    print("═" * W)
    print(f"  {res.summary_line()}")
    print(f"  mesaj sayısı: {len(before)} → {len(res.messages)}"
          + ("   (silme yok)" if len(before) == len(res.messages) else "   (mesaj birleşti/düştü)"))

    print("\n  STRATEJİ NE YAPTI:")
    for ln in res.log:
        print(f"    {ln}")

    print("\n" + "─" * W)
    print("  MESAJ MESAJ — ÖNCE ▸ SONRA        (içerikler TAM, kısaltılmadı)")
    print("─" * W)

    n = max(len(bv), len(av))
    for i in range(n):
        b = bv[i] if i < len(bv) else None
        a = av[i] if i < len(av) else None
        bt = b.tokens if b else 0
        at = a.tokens if a else 0

        if b is None:
            print(f"\n  #{i}  (yoktu) ▸ {label(a)}  {at}t   [YENİ]")
            print(box(a.content))
            continue
        if a is None:
            print(f"\n  #{i}  {label(b)}  {bt}t ▸ (düştü)   [KALDIRILDI]")
            print(box(b.content))
            continue

        same = (b.content == a.content)
        mark = "AYNEN KALDI" if same else f"DEĞİŞTİ  {bt}t → {at}t"
        print(f"\n  #{i}  {label(b)}   [{mark}]")
        if same:
            if b.content:
                print(box(b.content, "  ┊ "))
            elif b.tool_calls:
                print(f"  ┊ (tool çağrısı, içerik yok)")
        else:
            print("  ÖNCE:")
            print(box(b.content, "  │ "))
            print("  SONRA:")
            print(box(a.content, "  ▶ "))


def main():
    which = sys.argv[1:] or list(CP.STRATEGIES)
    t = trace()
    print("=" * W)
    print("  TOOL-TRACE GÖSTERİMİ — küçük, okunabilir, KISALTILMAMIŞ iz")
    print("=" * W)
    print(f"  Başlangıç: {len(t)} mesaj, {CP.total_tokens(t)} token · hedef bütçe: {BUDGET}")
    print("  İzde bilerek var olanlar:")
    print("    • AYNI dosyanın 2 kez okunması  → dedup'ı görmek için")
    print("    • 3 farklı tool (read_file/grep/run_tests) → tip-farkında özeti görmek için")
    print("    • her tool çağrısının bir sonucu → çift bütünlüğünü izlemek için")
    for s in which:
        if s not in CP.STRATEGIES:
            print(f"\n[!] bilinmeyen strateji: {s}")
            continue
        show(s)
    print("\n" + "=" * W)


if __name__ == "__main__":
    main()
