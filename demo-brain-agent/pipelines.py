#!/usr/bin/env python3
"""
pipelines.py — Kurulan iş akışlarını (DAG) SAKLA ve geri getir.

Kullanıcı belirli bir otomasyon istediğinde ajan bir graf kurar. O graf tek
kullanımlık olmamalı: kaydedilsin, sayfadan görülsün, tekrar çalıştırılabilsin.

Her kayıt: hedef, paket, motor, düğümler (fn + argüman + bağımlılık + sonuç),
zaman, süre ve özet metrikler. Basit JSON dosyaları — incelemesi kolay.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

STORE = Path(__file__).resolve().parent / "pipelines_store"
STORE.mkdir(exist_ok=True)


def save(goal: str, pack: str, backend: str, tasks: list, stats: dict) -> str:
    """Kurulan grafı kaydet. Dönüş: pipeline id."""
    pid = f"p_{int(time.time()*1000)%100000000:08d}"
    nodes = [{
        "id": t["id"], "title": t["title"], "kind": t["kind"], "fn": t.get("fn"),
        "args": t.get("fn_args") or {}, "parents": t.get("parents") or [],
        "status": t["status"], "priority": t.get("priority", 5),
        "created_by": t.get("created_by", "agent"),
        "result": _short(t.get("result")),
    } for t in tasks]
    doc = {"id": pid, "goal": goal, "pack": pack, "backend": backend,
           "at": int(time.time()), "nodes": nodes, "stats": stats}
    (STORE / f"{pid}.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1),
                                       encoding="utf-8")
    return pid


def _short(result, cap: int = 220):
    if not result:
        return ""
    s = str(result)
    try:
        d = json.loads(s)
        if isinstance(d, dict):
            d = {k: v for k, v in d.items() if k not in ("_raw_chars", "rows", "text")}
            s = json.dumps(d, ensure_ascii=False)
    except Exception:
        pass
    return s[:cap] + ("…" if len(s) > cap else "")


def listing(limit: int = 50) -> list:
    """Kayıtlı akışların özeti (yeniden eskiye)."""
    out = []
    for f in sorted(STORE.glob("p_*.json"), reverse=True)[:limit]:
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        out.append({"id": d["id"], "goal": d["goal"], "pack": d["pack"],
                    "backend": d["backend"], "at": d["at"],
                    "n": len(d["nodes"]), "stats": d.get("stats", {})})
    return out


def load(pid: str) -> dict | None:
    f = STORE / f"{pid}.json"
    if not f.exists():
        return None
    return json.loads(f.read_text(encoding="utf-8"))


def layers(nodes: list) -> list:
    """Düğümleri bağımlılık derinliğine göre KATMANLARA ayır (görsel çizim için)."""
    by_id = {n["id"]: n for n in nodes}
    depth: dict[str, int] = {}

    def d(nid, seen=()):
        if nid in depth:
            return depth[nid]
        if nid in seen:                      # döngü koruması
            return 0
        n = by_id.get(nid)
        if not n or not n["parents"]:
            depth[nid] = 0
            return 0
        depth[nid] = 1 + max((d(p, seen + (nid,)) for p in n["parents"] if p in by_id),
                             default=-1)
        return depth[nid]

    for n in nodes:
        d(n["id"])
    out: list = []
    for nid, lvl in depth.items():
        while len(out) <= lvl:
            out.append([])
        out[lvl].append(by_id[nid])
    return out


if __name__ == "__main__":
    ls = listing()
    print(f"{len(ls)} kayıtlı akış:\n")
    for p in ls:
        print(f"  {p['id']}  [{p['pack']}/{p['backend']}]  {p['n']} düğüm  {p['goal'][:56]}")
    if ls:
        d = load(ls[0]["id"])
        print(f"\n── {d['id']} katmanları ──")
        for i, lay in enumerate(layers(d["nodes"])):
            print(f"  katman {i}: " + " | ".join(
                (n["fn"] or "AJAN") + f"({n['id']})" for n in lay))
