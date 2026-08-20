"""MAF'ın birincil kaynaklarını tek dosyaya indirir — `docs/05` ile aynı disiplin.

`docs/05` AutoGen'in kullanıcı kılavuzunu **birebir** taşıyor, çünkü bir iddiayı
satır numarasıyla gösterebilmek, onu özetleyip "böyle diyor" demekten farklı bir
şey. MAF için aynı şeyi yapmak iki depo gerektiriyor, ve bunu bilmek başlı başına
bir bulgu:

* **Kullanıcı kılavuzu** `microsoft/agent-framework` deposunda **yok**. Learn'de
  yayımlanıyor ve kaynağı `MicrosoftDocs/semantic-kernel-docs` içinde
  `agent-framework/` klasöründe duruyor — yani MAF'ın dokümantasyonu hâlâ Semantic
  Kernel'in doküman deposunda yaşıyor.
* **Tasarım kayıtları (ADR)** ise kod deposunda. AutoGen'in hiç yayımlamadığı bir
  şey: kararın *neden* öyle alındığı, reddedilen alternatiflerle birlikte.

İki çıktı, iki farklı soru:

| çıktı | kaynak | ne cevaplıyor |
|---|---|---|
| `20-maf-user-guide.md` | Learn kılavuzu | **nasıl kullanılır** |
| `21-maf-tasarim-kararlari.md` | ADR'ler | **neden böyle** |

### Tek editoryal müdahale: dil bölgeleri

Learn sayfaları `::: zone pivot="programming-language-csharp"` bloklarıyla C#,
Python ve Go'yu aynı dosyada taşıyor. Python dışı bölgeler **çıkarıldı** ve her
çıkarma yerinde bir işaretle duruyor — satır numarası vermek istiyorsak metnin
tek dilde ve sabit olması gerekiyor. Bunun dışında tek karakter değiştirilmedi.

Kaynakların ikisi de MIT, Microsoft Corporation. SHA'lar dosya başlığına
yazılıyor: belge yeniden üretilebilir olmalı.

Kullanım:  `python docs/tools/fetch_maf_docs.py [--work DIZIN]`
"""

from __future__ import annotations

import argparse
import io
import json
import re
import tarfile
import urllib.request
from datetime import date
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent

# Learn kılavuzunun **tamamı**, soldaki gezinme ağacındaki sırayla.
#
# `integrations/` ve `support/` bir tur dışarıda bırakılmıştı — "sağlayıcı başına
# tekrar eden bağlantı ayrıntısı" diye. Bu yanlış bir kesimdi: `support/upgrade/`
# altında **2026'nın kırıcı değişiklikleri** duruyor, ki MAF'ın hızının bedelini
# ölçen tek birincil kaynak o; ve `integrations/by-component/middleware/purview.md`
# yönetişim tarafını anlatıyor — bir bankaya sunum yapılırken atlanacak son şey.
#
# Sıra bilinçli: yeni bölümler **sona** ekleniyor, böylece `22-maf-turkce.md`'nin
# `20:satır` atıfları kaymıyor. Bir belgenin satır numarası verdiği anda o
# numaralar arayüz olur.
GUIDE_SECTIONS = (
    "overview", "get-started", "concepts", "agents",
    "workflows", "journey", "migration-guide", "hosting",
    "integrations", "support",
)

ZONE_OPEN = re.compile(r'^::: *zone +pivot="([^"]+)"\s*$')
ZONE_END = re.compile(r"^::: *zone-end\s*$")
FRONTMATTER = re.compile(r"\A---\n.*?\n---\n", re.S)


def head_sha(repo: str) -> str:
    with urllib.request.urlopen(
        f"https://api.github.com/repos/{repo}/commits/main", timeout=30
    ) as r:
        return json.load(r)["sha"]


def tarball(repo: str, sha: str, work: Path) -> Path:
    """Depoyu bir kez indirip açar; ikinci koşuda yeniden indirmez."""
    out = work / f"{repo.split('/')[-1]}-{sha[:12]}"
    if out.exists():
        return out
    with urllib.request.urlopen(
        f"https://codeload.github.com/{repo}/tar.gz/{sha}", timeout=600
    ) as r:
        blob = r.read()
    with tarfile.open(fileobj=io.BytesIO(blob), mode="r:gz") as tf:
        tf.extractall(work, filter="data")
    (work / f"{repo.split('/')[-1]}-{sha}").rename(out)
    return out


def strip_zones(text: str) -> tuple[str, int]:
    """Python dışı `::: zone` bloklarını at, yerine tek satırlık iz bırak.

    İz önemli: okuyan kişi metnin kesildiğini görmeli, yoksa belge kaynağı
    olduğundan daha eksik sanılır ve "burada yazmıyor" denir.
    """
    keep: list[str] = []
    dropped = 0
    on = True
    for line in text.split("\n"):
        if m := ZONE_OPEN.match(line):
            on = "python" in m.group(1)
            if not on:
                keep.append(f"> *[{m.group(1)} bölgesi çıkarıldı]*")
            continue
        if ZONE_END.match(line):
            on = True
            continue
        if on:
            keep.append(line)
        else:
            dropped += 1
    return "\n".join(keep), dropped


def anchor(title: str) -> str:
    slug = re.sub(r"[^a-z0-9\s-]", "", title.lower()).strip()
    return re.sub(r"\s+", "-", slug)


def collect(files: list[Path], root: Path, *, zones: bool) -> tuple[list[dict], int]:
    """Dosyaları oku, başlığı frontmatter'dan al, gövdeyi hazırla."""
    out: list[dict] = []
    dropped_total = 0
    for path in files:
        raw = path.read_text(encoding="utf-8")
        title = ""
        if m := re.search(r"^title: *(.+)$", raw[:2000], re.M):
            title = m.group(1).strip().strip("\"'")
        body = FRONTMATTER.sub("", raw, count=1)
        if zones:
            body, dropped = strip_zones(body)
            dropped_total += dropped
        if not title:
            m = re.search(r"^#\s+(.+)$", body, re.M)
            title = m.group(1).strip() if m else path.stem
        out.append({
            "title": title,
            "rel": str(path.relative_to(root)),
            "body": body.strip("\n"),
        })
    return out, dropped_total


def render(header: str, pages: list[dict]) -> str:
    """Tek dosya: başlık, içindekiler, sonra her sayfa kendi yol işaretiyle."""
    seen: dict[str, int] = {}
    for p in pages:
        a = anchor(p["title"])
        seen[a] = seen.get(a, 0) + 1
        p["anchor"] = a if seen[a] == 1 else f"{a}-{seen[a] - 1}"

    toc = "\n".join(
        f"- [{p['title']}](#{p['anchor']}) · `{p['rel']}`" for p in pages
    )
    parts = [header, "\n## İçindekiler\n", toc, "\n---\n"]
    for p in pages:
        parts.append(f"\n# {p['title']}\n")
        parts.append(f"*`{p['rel']}`*\n")
        parts.append(p["body"])
        parts.append("\n\n---\n")
    return "\n".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/tmp/maf-docs")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    today = date.today().isoformat()

    # ---------------------------------------------------------- kullanım kılavuzu
    sk_sha = head_sha("MicrosoftDocs/semantic-kernel-docs")
    sk = tarball("MicrosoftDocs/semantic-kernel-docs", sk_sha, work)
    af_root = sk / "agent-framework"
    guide_files: list[Path] = []
    for section in GUIDE_SECTIONS:
        guide_files += sorted((af_root / section).rglob("*.md"))
    pages, dropped = collect(guide_files, af_root, zones=True)

    header = (
        "# 20 — Microsoft Agent Framework: kullanıcı kılavuzu (tam metin)\n\n"
        f"*Kaynak: [MicrosoftDocs/semantic-kernel-docs]"
        f"(https://github.com/MicrosoftDocs/semantic-kernel-docs/tree/{sk_sha}/agent-framework) · "
        f"`agent-framework/` · çekildi {today} · commit `{sk_sha[:12]}` · "
        f"{len(pages)} sayfa*\n\n"
        "Bu belge **birebir kopyadır, özet değildir.** Sayfalar rendered HTML'den\n"
        "değil, Learn'in kendi Markdown kaynaklarından alındı.\n\n"
        "**MAF'ın kullanıcı kılavuzu kod deposunda değil.** `microsoft/agent-framework`\n"
        "içinde `docs/` var ama kılavuz yok; Learn'de yayımlanan metnin kaynağı hâlâ\n"
        "**Semantic Kernel'in doküman deposunda** duruyor. Bu, iki ürünün birleşmesinin\n"
        "belgede henüz tamamlanmadığını gösteriyor.\n\n"
        "**Tek müdahale:** Learn sayfaları C#, Python ve Go'yu `::: zone pivot`\n"
        f"bloklarıyla aynı dosyada taşıyor. Python dışı bölgeler çıkarıldı ({dropped}\n"
        "satır) ve her çıkarma yerinde `[… bölgesi çıkarıldı]` işaretiyle duruyor.\n"
        "Bunun dışında tek karakter değiştirilmedi.\n\n"
        "Telif MIT, Microsoft Corporation. Bizim yorumumuz ve ölçümlerimiz için\n"
        "[22-maf-turkce.md](22-maf-turkce.md) — burada tek satır bile bize ait değil.\n\n"
        "---\n"
    )
    guide_out = DOCS / "20-maf-user-guide.md"
    guide_out.write_text(render(header, pages), encoding="utf-8")

    # ------------------------------------------------------------ tasarım kayıtları
    maf_sha = head_sha("microsoft/agent-framework")
    maf = tarball("microsoft/agent-framework", maf_sha, work)
    adr_root = maf / "docs"
    adr_files = [
        p for p in sorted((adr_root / "decisions").glob("*.md"))
        if "template" not in p.name and p.name != "README.md"
    ]
    adr_files += sorted((adr_root / "features").rglob("*.md"))
    adr_files += sorted((adr_root / "specs").glob("*.md"))
    adr_files = [p for p in adr_files if "template" not in p.name]
    adr_files.append(adr_root / "FAQS.md")
    adr_pages, _ = collect([p for p in adr_files if p.exists()], adr_root, zones=False)

    # Depo belgeleri: kılavuzda olmayan ama iddia dayanağı olan dosyalar. En
    # önemlisi `PACKAGE_STATUS.md` — 36 paketin kaçının gerçekten `released`
    # olduğunu yalnız burası yazıyor, ve bu bir olgunluk sorusunun cevabı.
    repo_files = [
        maf / "python" / "README.md",
        maf / "python" / "PACKAGE_STATUS.md",
        maf / "python" / "packages" / "core" / "README.md",
        maf / "python" / "packages" / "orchestrations" / "README.md",
        maf / "python" / "packages" / "tools" / "README.md",
        maf / "python" / "samples" / "autogen-migration" / "README.md",
        maf / "python" / "samples" / "02-agents" / "harness" / "README.md",
        maf / "python" / "samples" / "02-agents" / "harness" / "build_your_own_claw" / "README.md",
        maf / "python" / "samples" / "02-agents" / "skills" / "README.md",
        maf / "python" / "samples" / "02-agents" / "security" / "README.md",
        maf / "python" / "samples" / "03-workflows" / "README.md",
    ]
    repo_pages, _ = collect([p for p in repo_files if p.exists()], maf, zones=False)
    adr_pages += repo_pages

    adr_header = (
        "# 21 — Microsoft Agent Framework: tasarım kararları (tam metin)\n\n"
        f"*Kaynak: [microsoft/agent-framework]"
        f"(https://github.com/microsoft/agent-framework/tree/{maf_sha}/docs) · "
        f"`docs/` + seçili `python/` belgeleri · çekildi {today} · "
        f"commit `{maf_sha[:12]}` · {len(adr_pages)} belge*\n\n"
        "**AutoGen'de bunun karşılığı yok.** AutoGen \"nasıl yapılır\"ı anlattı;\n"
        "MAF ayrıca **neden böyle yapıldığını** yayımlıyor — değerlendirilen ve\n"
        "*reddedilen* alternatiflerle birlikte. Bir çerçeveyi değerlendirirken en\n"
        "çok işe yarayan malzeme bu: API'nin ne olduğunu okumak kolay, hangi\n"
        "tercihin hangi bedelle alındığını okumak zordur.\n\n"
        "Bizim için doğrudan karşılığı olan kararlar — onay kapısı (`0006`),\n"
        "bağlam sıkıştırma (`0019`), beceri tasarımı (`0021`), prompt enjeksiyon\n"
        "savunması (`0024`) — [22-maf-turkce.md](22-maf-turkce.md)'de tek tek\n"
        "bizim uygulamamızla yan yana konuyor.\n\n"
        "Sonda depo belgeleri var: `PACKAGE_STATUS.md` (36 paketin kaçı gerçekten\n"
        "`released`), AutoGen göç örnekleri, ve *build your own claw* — Microsoft'un\n"
        "kendi harness örneği, ki bir **kişisel finans / yatırım asistanı** ve\n"
        "`valuation` + `risk-scoring` becerileriyle geliyor.\n\n"
        "Telif MIT, Microsoft Corporation. Birebir kopya.\n\n"
        "---\n"
    )
    adr_out = DOCS / "21-maf-tasarim-kararlari.md"
    adr_out.write_text(render(adr_header, adr_pages), encoding="utf-8")

    for path in (guide_out, adr_out):
        n = len(path.read_text(encoding="utf-8").split("\n"))
        print(f"{path.name}  ·  {n} satır  ·  {path.stat().st_size / 1024:.0f} KB")
    print(f"sk-docs {sk_sha[:12]} · agent-framework {maf_sha[:12]}")


if __name__ == "__main__":
    main()
