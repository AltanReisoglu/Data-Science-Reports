// Faz 4 — düz JS, build aracı yok, harici hiçbir kaynak yok (Principle V).
// Tek WebSocket, çift yönlü (contracts/websocket_protocol.md).

const ws = new WebSocket(`ws://${location.host}/ws`);

const form = document.getElementById("question-form");
const input = document.getElementById("question-input");
const answerArea = document.getElementById("answer-area");
const answerText = document.getElementById("answer-text");
const groundedBadge = document.getElementById("grounded-badge");
const sourceRefsEl = document.getElementById("source-refs");
const partialFailures = document.getElementById("partial-failures");
const partialFailuresText = document.getElementById("partial-failures-text");
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

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  input.disabled = isLoading;
  spinner.hidden = !isLoading;
  buttonLabel.hidden = isLoading;
}

function setConnectionStatus(status) {
  connectionDot.className = "connection-dot " + status; // "connected" | "disconnected"
}

function appendPtcLine(text, cssClass) {
  // İlk gerçek olay geldiğinde, sayfa yüklendiğindeki ipucu satırını temizle.
  if (!panelCleared) {
    ptcLog.innerHTML = "";
    panelCleared = true;
  }
  const div = document.createElement("div");
  div.className = "line" + (cssClass ? " " + cssClass : "");
  div.textContent = text;
  ptcLog.appendChild(div);
  ptcLog.scrollTop = ptcLog.scrollHeight; // her zaman en alta kaydır (terminal gibi)
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
      appendPtcLine(msg.code);
      break;
    case "tool_call":
      appendPtcLine(`  ${icon} ${msg.tool_name}(${JSON.stringify(msg.arguments)}): ${msg.status}`);
      break;
    case "denied_action":
      appendPtcLine(`  ${icon} ENGELLENDİ: ${msg.attempted_destination} (${msg.verdict})`, "denied");
      break;
    case "final":
      appendPtcLine(
        `${icon} [${msg.run_id}] Bitti — durum: ${msg.status}`,
        msg.status === "success" ? "final-success" : "final-error",
      );
      break;
    default:
      appendPtcLine(JSON.stringify(msg), "info");
  }
}

function handleAnswer(msg) {
  setLoading(false);
  exampleChips.hidden = true; // ilk yanıttan sonra örnekler yerini gerçek sohbete bıraksın

  // FR-006: hiç ptc_event gelmediyse (doğrudan tool-calling ya da hiç tool
  // kullanılmadıysa) panel sessizce boş kalmasın.
  if (!sawPtcEventForCurrentQuestion) {
    appendPtcLine("(Bu soru için sandbox kullanılmadı.)", "info");
  }

  answerArea.hidden = false;
  answerText.textContent = msg.text;
  groundedBadge.textContent = msg.grounded ? "✓ Zemine dayalı" : "✕ Zemine dayalı DEĞİL";
  groundedBadge.className = "badge " + (msg.grounded ? "grounded" : "not-grounded");
  sourceRefsEl.textContent = msg.source_refs.length ? msg.source_refs.join(", ") : "(yok)";

  if (msg.partial_failure_notes.length) {
    partialFailures.hidden = false;
    partialFailuresText.textContent = msg.partial_failure_notes.join("; ");
  } else {
    partialFailures.hidden = true;
  }
}

function submitQuestion(text) {
  const trimmed = text.trim();
  if (!trimmed) return;
  sawPtcEventForCurrentQuestion = false;
  setLoading(true);
  ws.send(JSON.stringify({ type: "question", text: trimmed }));
  input.value = "";
}

ws.addEventListener("open", () => setConnectionStatus("connected"));
ws.addEventListener("close", () => setConnectionStatus("disconnected"));
ws.addEventListener("error", () => setConnectionStatus("disconnected"));

ws.addEventListener("message", (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === "ptc_event") {
    handlePtcEvent(msg);
  } else if (msg.type === "answer") {
    handleAnswer(msg);
  } else if (msg.type === "error") {
    setLoading(false);
    appendPtcLine(`Hata: ${msg.message}`, "denied");
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
