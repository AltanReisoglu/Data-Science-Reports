// Faz 4 — düz JS, build aracı/framework yok (Principle V) — ama artık CDN
// üzerinden 2 gerçek kütüphane var: marked.js (yanıt metnini markdown olarak
// render eder) ve highlight.js (PTC panelindeki Python kodunu renklendirir).
// Tek WebSocket, çift yönlü (contracts/websocket_protocol.md).

// Kalıcı oturum kimliği (2026-09-04). Bu satırlar olmadan sunucu her bağlantıda
// yeni bir uuid üretiyordu ve o uuid iki yere birden gidiyordu: konuşma
// hafızasının anahtarı (thread_id) VE artifact'lerin kapsamı (workflow_id).
// Sonuç: sayfa yenilendiğinde artifact'ler MinIO'da sağ kalıyor ama onları
// gösteren workflow_id bir daha üretilemediği için ERİŞİLEMEZ hale geliyordu —
// kalıcı bir depoya yazıp okuma anahtarını çöpe atmak.
//
// Anthropic'in `container.id` modeli: kimliği istemci saklar, yeniden bağlanınca
// geri gönderir. Saklamazsa (gizli sekme, temizlenmiş depo) sunucu temiz bir
// oturum açar — kalıcılık opt-in kalıyor.
function oturumKimligi() {
  try {
    let id = localStorage.getItem("ptc_session_id");
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem("ptc_session_id", id);
    }
    return id;
  } catch (e) {
    return null; // localStorage kapalı — sunucu yeni oturum açsın
  }
}

const _oturum = oturumKimligi();
const ws = new WebSocket(
  `ws://${location.host}/ws${_oturum ? `?session=${encodeURIComponent(_oturum)}` : ""}`,
);

const form = document.getElementById("question-form");
const input = document.getElementById("question-input");
const chat = document.getElementById("chat");
const ptcLog = document.getElementById("ptc-panel-log");
const submitButton = document.getElementById("submit-button");
const buttonLabel = submitButton.querySelector(".button-label");
const spinner = submitButton.querySelector(".spinner");
const connectionDot = document.getElementById("connection-dot");
const exampleChips = document.getElementById("example-chips");
const ptcPanel = document.getElementById("ptc-panel");
const ptcPanelHeader = document.getElementById("ptc-panel-header");
const ptcPanelToggle = document.getElementById("ptc-panel-toggle");

// Aşama başına küçük bir görsel ipucu — harici ikon fontu yok, sadece unicode.
const STAGE_ICONS = {
  configmap_created: "📦",
  job_created: "⚙️",
  tool_call: "🔧",
  denied_action: "⛔",
  final: "🏁",
};

// FR-006: bir soru PTC kullanmadan yanıtlanırsa panel sessizce boş kalmamalı.
let sawPtcEventForCurrentQuestion = false;
let panelCleared = false;
// Sohbetteki, henüz yanıtı gelmemiş "düşünüyor" balonu — answer/error gelince
// bu balonun İÇERİĞİ değişir (yeni bir balon eklenmez), böylece sıradaki
// soru-cevap çifti, öncekini SİLMEDEN sohbete eklenir.
let currentThinkingBubble = null;

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  input.disabled = isLoading;
  spinner.hidden = !isLoading;
  buttonLabel.hidden = isLoading;
}

function setConnectionStatus(status) {
  connectionDot.className = "connection-dot " + status; // "connected" | "disconnected"
}

function scrollChatToEnd(el) {
  el.scrollIntoView({ behavior: "smooth", block: "end" });
}

function appendUserMessage(text) {
  const div = document.createElement("div");
  div.className = "msg-user";
  div.textContent = text; // kullanıcı girdisi — her zaman textContent
  chat.appendChild(div);
  scrollChatToEnd(div);
}

function appendThinkingBubble() {
  const div = document.createElement("div");
  div.className = "msg-assistant thinking";
  const dots = document.createElement("span");
  dots.className = "thinking-dots";
  dots.innerHTML = "<span></span><span></span><span></span>"; // statik, veri içermiyor
  div.appendChild(dots);
  chat.appendChild(div);
  scrollChatToEnd(div);
  return div;
}

function renderAssistantMessage(bubble, msg) {
  bubble.className = "msg-assistant";
  bubble.innerHTML = ""; // "düşünüyor" içeriğini temizle

  const header = document.createElement("div");
  header.className = "msg-header";
  const badge = document.createElement("span");
  badge.className = "badge " + (msg.grounded ? "grounded" : "not-grounded");
  badge.textContent = msg.grounded ? "✓ Zemine dayalı" : "✕ Zemine dayalı DEĞİL";
  header.appendChild(badge);
  bubble.appendChild(header);

  const textDiv = document.createElement("div");
  textDiv.className = "msg-text";
  // marked.parse: LLM'in yanıtı genelde markdown (kalın, madde işaretleri vb.)
  // içeriyor — düz metin olarak göstermek yıldız işaretlerini olduğu gibi
  // basardı. Not: içerik sanitize edilmiyor (bu dosyadaki TEK innerHTML kullanımı
  // budur) — bu PoC yalnızca localhost'ta, kimliksiz, tek kullanıcılı (spec.md
  // Assumptions); metin de kendi LLM'imizden geliyor, keyfi bir kullanıcıdan
  // değil. Kamuya açık bir dağıtımda DOMPurify gibi bir sanitizer olmadan
  // kullanılmamalı.
  textDiv.innerHTML = marked.parse(msg.text);
  bubble.appendChild(textDiv);

  const meta = document.createElement("div");
  meta.className = "msg-meta";

  const sourcesRow = document.createElement("div");
  sourcesRow.className = "meta-row";
  const sourcesLabel = document.createElement("span");
  sourcesLabel.className = "meta-label";
  sourcesLabel.textContent = "Kaynaklar";
  const sourcesValue = document.createElement("span");
  sourcesValue.className = "meta-value";
  sourcesValue.textContent = msg.source_refs.length ? msg.source_refs.join(", ") : "(yok)";
  sourcesRow.append(sourcesLabel, sourcesValue);
  meta.appendChild(sourcesRow);

  if (msg.partial_failure_notes.length) {
    const warnRow = document.createElement("div");
    warnRow.className = "meta-row warning";
    const warnLabel = document.createElement("span");
    warnLabel.className = "meta-label";
    warnLabel.textContent = "Kısmi hatalar";
    const warnValue = document.createElement("span");
    warnValue.className = "meta-value";
    warnValue.textContent = msg.partial_failure_notes.join("; ");
    warnRow.append(warnLabel, warnValue);
    meta.appendChild(warnRow);
  }

  bubble.appendChild(meta);
  scrollChatToEnd(bubble);
}

function clearPanelHintOnce() {
  // İlk gerçek olay geldiğinde, sayfa yüklendiğindeki ipucu satırını temizle.
  if (!panelCleared) {
    ptcLog.innerHTML = "";
    panelCleared = true;
  }
}

function appendPtcLine(text, cssClass) {
  clearPanelHintOnce();
  const div = document.createElement("div");
  div.className = "line" + (cssClass ? " " + cssClass : "");
  div.textContent = text;
  ptcLog.appendChild(div);
  ptcLog.scrollTop = ptcLog.scrollHeight; // her zaman en alta kaydır (terminal gibi)
}

function appendPtcCodeBlock(code) {
  clearPanelHintOnce();
  const pre = document.createElement("pre");
  const codeEl = document.createElement("code");
  codeEl.className = "language-python";
  codeEl.textContent = code;
  pre.appendChild(codeEl);
  ptcLog.appendChild(pre);
  hljs.highlightElement(codeEl); // highlight.js — Python sözdizimi renklendirme
  ptcLog.scrollTop = ptcLog.scrollHeight;
}

// T016 — ptc_event mesajlarını stage'e göre biçimlendirip panele ekler.
function handlePtcEvent(msg) {
  sawPtcEventForCurrentQuestion = true;
  const icon = STAGE_ICONS[msg.stage] || "•";
  switch (msg.stage) {
    case "configmap_created":
      appendPtcLine(`${icon} [${msg.run_id}] ConfigMap oluşturuldu`, "info");
      break;
    case "job_created":
      appendPtcLine(`${icon} [${msg.run_id}] Job oluşturuldu — çalıştırılan kod:`, "info");
      appendPtcCodeBlock(msg.code);
      break;
    case "tool_call":
      appendPtcLine(`  ${icon} ${msg.tool_name}(${JSON.stringify(msg.arguments)}): ${msg.status}`);
      break;
    case "denied_action":
      appendPtcLine(
        `  ${icon} ENGELLENDİ [${msg.source_pod}]: ${msg.attempted_destination} (${msg.verdict})`,
        "denied",
      );
      if (msg.raw_flow) {
        appendPtcLine(`      hubble: ${msg.raw_flow}`, "denied-raw");
      }
      break;
    case "final": {
      const text = msg.status === "success" ? msg.result_text : msg.error_message;
      const detail = text ? ` — ${text}` : "";
      appendPtcLine(
        `${icon} [${msg.run_id}] Bitti — durum: ${msg.status}${detail}`,
        msg.status === "success" ? "final-success" : "final-error",
      );
      break;
    }
    default:
      appendPtcLine(JSON.stringify(msg), "info");
  }
}

function handleAnswer(msg) {
  setLoading(false);

  // FR-006: hiç ptc_event gelmediyse (doğrudan tool-calling ya da hiç tool
  // kullanılmadıysa) panel sessizce boş kalmasın.
  if (!sawPtcEventForCurrentQuestion) {
    appendPtcLine("(Bu soru için sandbox kullanılmadı.)", "info");
  }

  if (currentThinkingBubble) {
    renderAssistantMessage(currentThinkingBubble, msg);
    currentThinkingBubble = null;
  }
}

function submitQuestion(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  exampleChips.hidden = true; // ilk soru gönderilince örnekler yerini sohbete bıraksın
  sawPtcEventForCurrentQuestion = false;
  setLoading(true);
  appendUserMessage(trimmed);
  currentThinkingBubble = appendThinkingBubble();
  ws.send(JSON.stringify({ type: "question", text: trimmed }));
  input.value = "";
}

ws.addEventListener("open", () => setConnectionStatus("connected"));
ws.addEventListener("close", () => setConnectionStatus("disconnected"));
ws.addEventListener("error", () => setConnectionStatus("disconnected"));

// Kaçış demosu bu projede UI'dan KALDIRILDI (2026-09-06): egress-policy
// PoC'undan (ptc_sec) kalmaydı; buranın konusu artifact kalıcılığı. Sunucu
// tarafındaki `demo_escape` mesajı duruyor — zararsız, ama tetikleyen bir
// düğme yok. Sonuç yine de gelirse panele yazılıyor.
function handleDemoResult(msg) {
  const ok = msg.denied_count > 0;
  appendPtcLine(
    ok
      ? `✅ Demo tamamlandı — ${msg.denied_count} engelleme kaydı (durum: ${msg.status})`
      : `⚠️ Demo tamamlandı ama HİÇ engelleme kaydı yok (durum: ${msg.status}) — beklenmeyen!`,
    ok ? "final-success" : "final-error",
  );
}

ws.addEventListener("message", (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "ptc_event") {
    handlePtcEvent(msg);
  } else if (msg.type === "answer") {
    handleAnswer(msg);
  } else if (msg.type === "demo_result") {
    handleDemoResult(msg);
  } else if (msg.type === "error") {
    setLoading(false);
    appendPtcLine(`Hata: ${msg.message}`, "denied");
    if (currentThinkingBubble) {
      currentThinkingBubble.className = "msg-assistant";
      currentThinkingBubble.textContent = `Hata: ${msg.message}`;
      currentThinkingBubble = null;
    }
  }
});

// T011 — soru gönderimi, tek WebSocket üzerinden (research.md §3).
form.addEventListener("submit", (event) => {
  event.preventDefault();
  submitQuestion(input.value);
});

// Örnek soru çipleri — tıklanınca doğrudan gönderilir (boş durumda hızlı deneme).
exampleChips.querySelectorAll(".chip").forEach((chip) => {
  chip.addEventListener("click", () => submitQuestion(chip.dataset.question));
});

// PTC panelini daraltma/genişletme (macOS pencere davranışı gibi).
ptcPanelHeader.addEventListener("click", () => {
  const collapsed = ptcPanel.classList.toggle("collapsed");
  ptcPanelToggle.textContent = collapsed ? "▢" : "‒";
});
