"""LangGraph ajan grafiği — erişim yolu tool'larını `trace`'e bağlı olarak kurar.

.env okuma (load_dotenv) burada değil, cli.py'de (uygulama giriş noktası) yapılır;
bu modül os.environ'un zaten dolu olduğunu varsayar.
"""

from __future__ import annotations

import os
from collections.abc import Callable

from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from grounded_assistant.access_paths import live_systems
from grounded_assistant.agent import artifact_context, tool_policy
from grounded_assistant.models import SandboxRunStatus
from grounded_assistant.ptc.sandbox_runner import run_sandbox
from grounded_assistant.trace import Trace

# Altan'ın kararı (2026-09-01): bir hedef engellendiğinde (ör. onaylı olmayan bir
# domain), agent sınırsız sayıda YENİ kod deneyip her seferinde ayrı bir
# ConfigMap+Job+Pod yaratabiliyordu (ör. http://altan.com denenip engellenince,
# hemen ardından https://alta.com denemesi — her biri ayrı ~7sn'lik bir K8s
# çalıştırması). Bu, hem maliyeti hem PTC panelindeki gürültüyü katlıyor.
# Bir turda (bir kullanıcı sorusunda) en fazla bu kadar sandbox çalıştırmasına
# izin verilir; aşılırsa `run_ptc_code` YENİ bir pod yaratmadan reddeder.
MAX_SANDBOX_RUNS_PER_TURN = 2


_SYSTEM_PROMPT = (
    "Kurumsal bir asistansın. Veriye/canlı sistemlere TEK erişim yolun "
    "run_ptc_code — yazdığın kod içinden tool proxy'lerini çağırırsın.\n\n"
    "Kritik kural (bulunan gerçek bir sorun, 2026-08-30): bir tool çağrısının "
    "BAŞARILI olması, cevabındaki HER iddianın o çağrının çıktısından geldiği "
    "anlamına gelmez. Yalnızca tool'un GERÇEKTEN döndürdüğü veriye dayanarak "
    "yanıt ver; tool çıktısının ötesine kendi bilgin/tahminlerinle geçip bunu "
    "tool sonucuymuş gibi sunma.\n\n"
    "Özellikle: web_search bir arama motoru sonucudur — yalnızca başlık, url "
    "ve kısa bir snippet döndürür; hedef sayfanın TAM içeriğini ASLA getirmez "
    "(böyle bir tool sistemde yok). Bir kullanıcı belirli bir URL verip "
    "'bunu oku/incele' dediğinde, o sayfayı GERÇEKTEN okuyamazsın — yalnızca "
    "web_search'ün döndürdüğü kısa snippet'e erişebilirsin. Bu durumda "
    "'sayfayı inceledim/okudum' gibi ifadeler KULLANMA; snippet'in ötesinde "
    "bir ayrıntı biliyorsan bile bunu tool'dan gelmiş gibi sunma — açıkça "
    "'yalnızca bir arama snippet'i görebiliyorum, sayfanın tam içeriğine "
    "erişimim yok' de.\n\n"
    f"Önemli bir sınır: bir soruda en fazla {MAX_SANDBOX_RUNS_PER_TURN} kez "
    "run_ptc_code çalıştırabilirsin. Bir hedef ağ seviyesinde engellendiyse "
    "(denied_action / 'Sandbox ... engellendi' mesajı), bu KESİN bir karar — "
    "farklı bir URL/domain/şema (http yerine https gibi) deneyerek bunu "
    "AŞMAYA ÇALIŞMA, bu sadece yeni bir engellemeye yol açar. Sınıra "
    "ulaştığında elindeki bilgiyle yanıt ver, tahmini değer üretme."
)


def _build_model() -> ChatOpenAI:
    return ChatOpenAI(
        base_url=os.environ["LLM_BASE_URL"],
        api_key=os.environ["LLM_API_KEY"],
        model=os.environ["LLM_MODEL_NAME"],
    )


def _make_ptc_tool(
    trace: Trace,
    on_ptc_event: Callable[[dict], None] | None = None,
    workflow_id: str | None = None,
):
    """T014 — Faz 2. Kod, ayrı ve Cilium ile ağ-kısıtlı bir Kubernetes Job'unda
    çalışır (sandbox_runner.run_sandbox); FR-011: başarısız bir çalıştırmadan
    tahmini değer üretilmez, modele açıkça 'üretme' talimatı döner. tool_calls
    T015 gereği Trace'e beslenir (aynı LiveToolCall/record_tool_call, Faz 1'deki
    canlı-sistem yoluyla birebir aynı mekanizma).

    `on_ptc_event` (Faz 4, T006): verilirse `run_sandbox`'a olduğu gibi iletilir
    — web arayüzünün sol-alt canlı paneli bunu kullanır (contracts/
    websocket_protocol.md). CLI hiç geçmez (`None`), davranışı değişmez."""

    @tool
    def run_ptc_code(code: str) -> str:
        """LLM'in ürettiği Python kodunu, Tool Gateway dışında hiçbir yere
        çıkamayan (Cilium/eBPF ile ağ seviyesinde kısıtlı) ayrı bir Kubernetes
        pod'unda çalıştırır. Bu, veriye/canlı sistemlere erişmenin TEK yoludur —
        search_knowledge_base(query), get_ticket_status(ticket_id),
        count_open_tickets() (DİKKAT: bir SAYI sözlüğü döner —
        {"open_count": int, "as_of": str} — ticket LİSTESİ DEĞİL; `len()` ile
        çağırma, asıl sayı için `["open_count"]` alanını oku),
        create_support_ticket(title, description),
        search_employee_directory(query), web_search(query), calculator(expression),
        fetch_url(url, method="GET") (verilen URL'ye ham bir HTTP isteği atar,
        AŞAMA AŞAMA teşhis döner — dns/tcp_connect/timeout/complete — yalnızca
        onaylı hedeflere ulaşır, başkasına giden istek ağ seviyesinde engellenir),
        resolve_dns(hostname) (yalnızca isim çözer, BAĞLANMAZ — DNS serbest
        olduğu için hemen hemen her isim çözülür, bu HEDEFE ULAŞILABİLECEĞİ
        anlamına gelmez), check_connectivity(host, port) (ham bir TCP bağlantısı
        dener, HTTP değil — hangi aşamada durduğunu bildirir)
        fonksiyonları kodun çalıştığı ortamda ZATEN HAZIR (global) tanımlı —
        bunları HİÇBİR import satırı YAZMADAN, doğrudan normal bir Python
        fonksiyonuymuş gibi çağır (ör. `count_open_tickets()["open_count"]`), ve nihai sonucu
        set_result(deger) ile bildir. Birden fazla tool çağrısını tek turda,
        programatik olarak (döngü/koşul ile) sıralamak istediğinde de bunu
        kullan.

        KALICI VERİ — `/output/` dizini. Sandbox pod'u her çalıştırmada
        SIFIRDAN doğar; değişkenler bir sonraki çağrıya TAŞINMAZ. Kalması
        gereken her şeyi `/output/` altına DOSYA olarak yaz.

        Özel bir API YOK. Düpedüz Python yaz:

            df.to_parquet("/output/satislar.parquet")   # saklanır
            df = pd.read_parquet("/output/satislar.parquet")  # geri okunur
            json.dump(ozet, open("/output/ozet.json", "w"))
            plt.savefig("/output/dagilim.png")

        Çalışma bitince `/output/` üst düzeyindeki her dosya (ve her dizin)
        otomatik olarak kalıcı depoya konur. Hata alsan bile o ana kadar
        yazdıkların kurtarılır.

        DAHA ÖNCE ÜRETİLENLERİ KULLANMAK — dosya sistemi deponun görünümüdür.
        `/output/` altındaki bir dosyayı okumak istediğinde, o dosya bu pod'da
        fiziksel olarak olmasa bile depodan otomatik iner. Yani:

            os.listdir("/output")              # depoda ne varsa listeler
            os.path.exists("/output/x.parquet")  # depodakini de sayar
            pd.read_parquet("/output/x.parquet") # yoksa indirir, sonra okur

        Kullanıcı önceki bir sonuca atıf yapıyorsa ("az önceki tabloyu", "onu
        departmana göre grupla") veriyi YENİDEN ÜRETME — önce `/output`'a bak,
        oradan oku.

        İKİ AYRI KÖK var, karıştırma:
        - `/output/` — YALNIZCA BU OTURUMUN çıktıları. "Az önce", "demin",
          "senin ürettiğin" dendiğinde burası.
        - `/artifacts/<workflow_id>/` — BAŞKA çalıştırmaların çıktıları.
          `os.listdir("/artifacts")` hangi çalıştırmalar var onu listeler.
          Bunlar bu oturumun işi DEĞİL; kullanıcı açıkça istemedikçe
          kullanma, ve asla kendi çıktın gibi sunma.

        Aradığın dosya `/output`'ta yoksa üretmen gerekiyor demektir —
        `/artifacts` altındaki benzer adlı bir dosyayı onun yerine koyma.

        PAHALI İŞİ TEKRARLAMA: 5'ten fazla tool çağrısı içeren ya da döngüyle
        veri toplayan bir bloktan önce çıktısı var mı diye bak:

            if os.path.exists("/output/acik_ticketlar.parquet"):
                df = pd.read_parquet("/output/acik_ticketlar.parquet")
            else:
                df = ticket_taramasi()
                df.to_parquet("/output/acik_ticketlar.parquet")

        Kodun hata verip düzeltilmiş hâliyle yeniden çalıştırılması sık olur;
        bu kalıbı kullanırsan o pahalı iş tekrarlanmaz.

        DOSYA ADI ÖNEMLİ: uzantı içeriğin tipini belirliyor (`.parquet`, `.csv`,
        `.json`, `.png`, `.pdf`, `.xlsx`, `.md`). Uzantısız ad verme.

        Küçük skaler sonuçlar (tek sayı, kısa metin) için dosya yazma —
        onları doğrudan set_result ile döndür.

        BELGE ÜRETİMİ — kullanıcı rapor/grafik isterse gerçekten üret:
        - matplotlib kurulu (backend Agg): `plt.savefig("/output/dagilim.png")`
          ya da çok sayfalı PDF için `PdfPages("/output/rapor.pdf")`
        - reportlab kurulu: metin+tablo içeren düzgün PDF raporlar için
        - openpyxl kurulu: `df.to_excel("/output/ozet.xlsx")`

        `/output/` altına bir DİZİN de bırakabilirsin (çok dosyalı model,
        varlıklarıyla birlikte HTML rapor). Tek bir artifact olarak saklanır ve
        sonraki çalıştırmada `/output/<dizin>/<dosya>` yolunu okuyunca
        kendiliğinden geri açılır.
        Panelde PNG ve PDF önizlemesi var, yani ürettiğin belge gerçekten
        görüntülenebilir."""
        if trace.sandbox_run_count() >= MAX_SANDBOX_RUNS_PER_TURN:
            # Altan'ın kararı (2026-09-01): agent, engellenen bir hedefi farklı
            # bir URL/şema ile tekrar tekrar deneyip her seferinde yeni bir
            # ConfigMap+Job+Pod (~7sn) yaratabiliyordu. Sınıra ulaşılınca YENİ
            # bir pod hiç yaratılmadan (run_sandbox çağrılmadan) reddedilir.
            return (
                f"Bu soruda zaten {MAX_SANDBOX_RUNS_PER_TURN} kez run_ptc_code "
                "çalıştırıldı — sınıra ulaşıldı, YENİ bir sandbox çalıştırılmadı. "
                "Farklı bir URL/domain/şema deneyerek tekrar çağırma; elindeki "
                "bilgiyle yanıt ver, tahmini değer üretme."
            )
        run = run_sandbox(code, on_event=on_ptc_event, workflow_id=workflow_id)
        trace.record_sandbox_run(run)  # SC-003: çalıştırmanın kendisi (T017)
        for call in run.tool_calls:
            trace.record_tool_call(call)  # T015: çalıştırma içindeki her tool çağrısı
        for action in run.denied_actions:
            trace.record_denied_action(action)  # T021, SC-002: engellenen erişim girişimi
        for olay in run.artifacts:
            # 2026-09-03: bu çalıştırma neyi üretti / neyi yeniden üretmeden okudu.
            trace.record_artifact_event(olay)
        if run.status is SandboxRunStatus.SUCCESS:
            return run.result_text
        if run.status is SandboxRunStatus.TIMEOUT:
            return "Sandbox çalıştırması zaman aşımına uğradı. Tahmini bir değer üretme."
        if run.status is SandboxRunStatus.DENIED_ACTION:
            return (
                "Sandbox, onaylı Tool Gateway dışında bir hedefe erişmeye çalıştı; "
                "bu ağ seviyesinde (Cilium) engellendi. Tahmini bir değer üretme."
            )
        detail = f" Hata: {run.error_message}" if run.error_message else ""
        return f"Sandbox çalıştırması başarısız oldu.{detail} Tahmini bir değer üretme."

    return run_ptc_code


def _build_tools(
    trace: Trace,
    on_ptc_event: Callable[[dict], None] | None = None,
    workflow_id: str | None = None,
) -> list:
    """Altan'ın kararı (2026-08-30): agent'a artık YALNIZCA `run_ptc_code`
    bağlanıyor — knowledge_base/live_systems'in doğrudan-tool-çağırma yolları
    (T016/T022) kaldırıldı, ki her soru istisnasız PTC sandbox'ından geçsin
    ("herşeye ptc ile yanıt versin"). Bu bir yetenek kaybı DEĞİL: entrypoint.py'nin
    ALLOWED_TOOLS'u zaten search_knowledge_base/get_ticket_status/vb. TÜMÜNÜ
    Tool Gateway üzerinden sandbox koduna proxy'liyor — LLM artık bunlara
    doğrudan değil, her zaman `run_ptc_code` içinden yazdığı kodla ulaşıyor.
    Sonuç: her etkileşim PTC panelinde (configmap/job/tool_call/final) görünür
    oluyor; bedeli, basit sorularda bile bir K8s Job'unun ayağa kalkması kadar
    gecikme (demo/gözlemlenebilirlik için kabul edilen takas)."""
    return [_make_ptc_tool(trace, on_ptc_event, workflow_id)]


def build_checkpointer():
    """Workflow state'i (konuşma + grafiğin nerede kaldığı) tutan katman.

    ## ÇAĞIRAN TARAFIN DİKKAT ETMESİ GEREKEN ŞEY

    Bu fonksiyon **çalışan bir event loop içinden** çağrılmalı — async saver'lar
    kurulurken `asyncio.get_running_loop()` istiyor. Loop yoksa `InMemorySaver`'a
    düşer ve kalıcılık sessizce kaybolur.

    `build_agent` bunu KENDİSİ çağıramaz: o, bilerek loop'suz bir thread'de
    çalışıyor (`live_systems.get_live_system_tools()` içinde `asyncio.run()` var,
    o da çalışan bir loop'un içinden çağrılamaz). Bu yüzden çağıran, önce burayı
    loop içinde çağırıp sonucu `build_agent`'a **geçiriyor**.

    ## Neden async saver şart (2026-09-04'te bulunan gerçek hata)

    Hem web hem CLI `agent.ainvoke()` yoluna giriyor. Senkron `SqliteSaver` o
    yolda açık hata veriyor: *"The SqliteSaver does not support async methods."*
    Bu, 2026-08-28'de `LiveSystemTraceMiddleware`'de yaşananın aynı sınıfı —
    yalnızca senkron tarafı uygulamak, async yolda çalışmıyor.

    İlk testlerim bunu KAÇIRDI çünkü saver'ı doğrudan (senkron `put`/`get_tuple`
    ile) sınıyorlardı, `ainvoke` üzerinden değil. Test artık async yolu da
    geçiyor.

    ## Arka uç seçimi config, kod değil

        PTC_CHECKPOINT_DSN  → PostgreSQL   (üretim)
        PTC_CHECKPOINT_DB   → SQLite dosya (varsayılan duruş, metadata ile aynı)
        ikisi de yoksa      → InMemorySaver (test / tek atış)
    """
    dsn = os.environ.get("PTC_CHECKPOINT_DSN")
    if dsn:
        from langgraph.checkpoint.postgres.aio import (  # noqa: PLC0415
            AsyncPostgresSaver,
        )

        saver = AsyncPostgresSaver.from_conn_string(dsn).__aenter__()
        return saver

    yol = os.environ.get("PTC_CHECKPOINT_DB")
    if yol:
        import aiosqlite  # noqa: PLC0415
        from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver  # noqa: PLC0415

        # Bağlantı TEMBEL: `aiosqlite.connect()` henüz açmıyor, saver ilk
        # kullanımda `setup()` içinden başlatıyor. Böylece burada `await`
        # etmemiz gerekmiyor ve fonksiyon senkron kalabiliyor.
        return AsyncSqliteSaver(aiosqlite.connect(yol))

    return InMemorySaver()


def build_agent(
    trace: Trace,
    on_ptc_event: Callable[[dict], None] | None = None,
    workflow_id: str | None = None,
    checkpointer=None,
):
    """`on_ptc_event` (Faz 4, T006): PTC sandbox'ının canlı olaylarını almak
    isteyen çağıranlar (web arayüzü) için — CLI bunu hiç vermez.

    `workflow_id` (2026-09-03): artifact deposunun kapsam anahtarı; oturum
    kimliği buraya geçer. Verilmezse sandbox artifact API'sini hiç görmez."""
    tools = _build_tools(trace, on_ptc_event, workflow_id)
    tool_policy.assert_known_tools(tools)  # Principle II savunma hattı (bkz. tool_policy.py)

    # Artifact manifesti (2026-09-04): saklı artifact'lerin İSİMLERİ her model
    # çağrısından önce context'e giriyor. Google ADK'nın `LoadArtifactsTool`
    # deseni — "isim ucuz, içerik pahalı". Öncesinde keşif yumuşak garantiydi:
    # model `list_artifacts()`'i çağırmayı unutursa depodaki veriyi yeniden
    # üretiyordu. Servis adresi tanımlı değilse middleware sessizce devre dışı.
    middleware = [
        tool_policy.build_middleware(),
        live_systems.LiveSystemTraceMiddleware(trace),
        artifact_context.ArtifactContextMiddleware(
            workflow_id, lambda: _kapsam_jetonu(workflow_id)
        ),
    ]

    return create_agent(
        model=_build_model(),
        tools=tools,
        system_prompt=_SYSTEM_PROMPT,
        middleware=middleware,
        # `checkpointer` DIŞARIDAN geliyor — bkz. `build_checkpointer`
        # başlığındaki gerekçe (async saver'lar loop içinde kurulmak zorunda,
        # bu fonksiyon ise loop'suz bir thread'de çalışıyor). Verilmezse
        # kalıcılık YOK: bu, test/tek-atış bağlamları için bilinçli bir
        # varsayılan, üretim yolu her zaman açıkça geçiriyor.
        checkpointer=checkpointer or InMemorySaver(),
    )


def _kapsam_jetonu(workflow_id: str | None) -> str | None:
    """Ajanın artifact servisine konuşurken kullandığı kapsam jetonu.

    Sandbox'ınkiyle AYNI mekanizma ve aynı imza anahtarı; farkı `run_id`'nin
    olmaması — ajan bir çalıştırma değil, listeleme yapıyor. Anahtar cluster'daki
    Secret'ta; okuyamazsak None döner ve manifest enjeksiyonu sessizce kapanır.
    """
    if not workflow_id:
        return None
    try:
        from kubernetes import client, config  # noqa: PLC0415

        from grounded_assistant.artifacts.scope import Scope, issue_token  # noqa: PLC0415
        from grounded_assistant.ptc import sandbox_runner  # noqa: PLC0415

        config.load_kube_config()
        anahtar = sandbox_runner._read_signing_key(client.CoreV1Api())
        if not anahtar:
            return None
        return issue_token(
            anahtar,
            Scope(workflow_id=workflow_id, run_id="agent", owner="ptc", node_id=None),
        )
    except Exception:  # noqa: BLE001 — jeton üretilemezse manifest yok, tur sürer
        return None


async def invoke_and_resolve(agent, message: str, thread_id: str) -> dict:
    """Ajanı çalıştırır; bir interrupt dönerse tool_policy.auto_resolve ile
    (insan olmadan) anında karar verip devam ettirir.

    `ainvoke` (senkron `invoke` DEĞİL) kullanılıyor — bulunan gerçek bir hata
    (2026-08-28, Faz 4'ün web testleriyle ortaya çıktı, Faz 1'den beri
    varmış): `live_systems`'in MCP-adapted tool'ları (langchain_mcp_adapters)
    yalnızca async çağrılabiliyor; `agent.invoke()`'un senkron tool-çalıştırma
    yolu bunlarda `NotImplementedError: StructuredTool does not support sync
    invocation` fırlatıyordu. `ainvoke`, `BaseTool._arun`'un varsayılanı
    sayesinde (`langchain_core/tools/base.py:932` —
    `run_in_executor(None, self._run, ...)`) senkron tool'ları (`run_ptc_code`,
    `search_knowledge_base`) da otomatik bir thread'de çalıştırarak doğru
    ele alıyor — yani her iki tool türü için de doğru olan tek yol bu."""
    config = {"configurable": {"thread_id": thread_id}}
    messages = {"messages": [{"role": "user", "content": message}]}
    result = await agent.ainvoke(messages, config=config)

    while "__interrupt__" in result:
        hitl_request = result["__interrupt__"][0].value
        command = tool_policy.auto_resolve(hitl_request)
        result = await agent.ainvoke(command, config=config)

    return result
