/* Local chat UI for the prompt-synthesis pipeline.
   No build step and no dependencies - served straight from webapp/static. */

const state = {
  conversations: [],
  currentId: null,
  conversation: null,
  messages: [],
  usage: null,
  busy: false,
};

const el = (id) => document.getElementById(id);
const transcript = el("transcript");

/* ------------------------------------------------------------------ api */

async function api(path, options = {}) {
  const response = await fetch(`/api${path}`, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || `Request failed (${response.status})`);
  }
  return response.json();
}

function applyState(payload) {
  state.conversation = payload.conversation;
  state.messages = payload.messages || [];
  state.usage = payload.usage || null;
  if (payload.conversation) state.currentId = payload.conversation.id;
  renderConversation();
  loadConversations();
}

/* ------------------------------------------------------------------ markdown
   LLM output is untrusted text: escape first, then apply a small subset of
   markdown to the escaped string. Nothing here ever inserts raw model output. */

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

function renderMarkdown(source) {
  const blocks = [];
  // Pull fenced code out first so its contents are never treated as markdown.
  let text = escapeHtml(source).replace(/```(\w*)\n([\s\S]*?)```/g, (_match, lang, code) => {
    blocks.push(`<pre><code data-lang="${lang}">${code.replace(/\n$/, "")}</code></pre>`);
    return ` BLOCK${blocks.length - 1} `;
  });

  const inline = (line) =>
    line
      .replace(/`([^`]+)`/g, "<code>$1</code>")
      .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
      .replace(/(^|[\s(])\*([^*\n]+)\*/g, "$1<em>$2</em>");

  const html = [];
  let list = null; // 'ul' | 'ol' | null

  const closeList = () => {
    if (list) { html.push(`</${list}>`); list = null; }
  };

  for (const rawLine of text.split("\n")) {
    const line = rawLine.trimEnd();
    const placeholder = line.match(/^ BLOCK(\d+) $/);
    if (placeholder) { closeList(); html.push(blocks[Number(placeholder[1])]); continue; }
    if (!line.trim()) { closeList(); continue; }

    const heading = line.match(/^(#{1,4})\s+(.*)$/);
    if (heading) {
      closeList();
      const level = Math.min(heading[1].length + 1, 4);
      html.push(`<h${level}>${inline(heading[2])}</h${level}>`);
      continue;
    }
    const ordered = line.match(/^\s*(\d+)[.)]\s+(.*)$/);
    if (ordered) {
      if (list !== "ol") { closeList(); html.push("<ol>"); list = "ol"; }
      html.push(`<li>${inline(ordered[2])}</li>`);
      continue;
    }
    const bullet = line.match(/^\s*[-*+]\s+(.*)$/);
    if (bullet) {
      if (list !== "ul") { closeList(); html.push("<ul>"); list = "ul"; }
      html.push(`<li>${inline(bullet[1])}</li>`);
      continue;
    }
    closeList();
    html.push(`<p>${inline(line)}</p>`);
  }
  closeList();
  return html.join("");
}

/* ------------------------------------------------------------------ sidebar */

function relativeTime(iso) {
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

async function loadConversations() {
  const { conversations } = await api("/conversations");
  state.conversations = conversations;
  renderSidebar();
}

function renderSidebar() {
  const list = el("conversation-list");
  list.innerHTML = "";
  if (!state.conversations.length) {
    list.innerHTML = `<p class="empty" style="margin:24px 8px;font-size:12px">No conversations yet.</p>`;
    return;
  }
  for (const conversation of state.conversations) {
    const node = document.createElement("div");
    node.className = `conversation${conversation.id === state.currentId ? " active" : ""}`;
    node.innerHTML = `
      <div class="conversation-name"></div>
      <div class="conversation-meta">
        <span>${relativeTime(conversation.updated_at)}</span>
        <span>|</span>
        <span>${conversation.stage === "complete" ? "done" : conversation.stage}</span>
      </div>`;
    node.querySelector(".conversation-name").textContent = conversation.title;
    node.onclick = () => selectConversation(conversation.id);
    list.appendChild(node);
  }
}

/* ------------------------------------------------------------------ transcript */

function card({ label, body, className = "", open = false }) {
  const node = document.createElement("div");
  node.className = `card ${className}${open ? " open" : ""}`;
  node.innerHTML = `
    <div class="card-head"><span class="chev">&#9654;</span><span class="card-label"></span></div>
    <div class="card-body md"></div>`;
  node.querySelector(".card-label").textContent = label;
  node.querySelector(".card-body").innerHTML = body;
  node.querySelector(".card-head").onclick = () => node.classList.toggle("open");
  return node;
}

function renderQuestions(message) {
  const answered = message.kind === "questions_answered";
  const questions = (message.payload && message.payload.questions) || [];
  // Answers are stored back onto the card when it is superseded, so reopening a
  // conversation shows the picks rather than a row of blank radios.
  const submitted = new Map(
    ((message.payload && message.payload.answers) || []).map((answer) => [answer.question, answer]),
  );
  const node = document.createElement("div");
  node.className = `msg questions${answered ? " answered" : ""}`;

  const intro = document.createElement("p");
  intro.className = "questions-intro";
  intro.textContent = answered
    ? "Answered."
    : `${message.content} (round ${message.payload.round} of ${message.payload.max_rounds})`;
  node.appendChild(intro);

  questions.forEach((question, index) => {
    const block = document.createElement("div");
    block.className = "question";
    const title = document.createElement("div");
    title.className = "question-text";
    title.textContent = question.question;
    if (question.multi_select) {
      const hint = document.createElement("span");
      hint.className = "question-hint";
      hint.textContent = "choose any that apply";
      title.appendChild(hint);
    }
    block.appendChild(title);

    const answer = submitted.get(question.question);
    const chosen = new Set((answer && answer.selected) || []);

    const options = document.createElement("div");
    options.className = "options";
    const inputType = question.multi_select ? "checkbox" : "radio";

    question.options.forEach((option) => {
      const label = document.createElement("label");
      label.className = `option${chosen.has(option) ? " checked" : ""}`;
      const input = document.createElement("input");
      input.type = inputType;
      input.name = `q${index}`;
      input.value = option;
      input.checked = chosen.has(option);
      input.disabled = answered;
      input.onchange = () => {
        options.querySelectorAll(".option").forEach((row) => {
          row.classList.toggle("checked", row.querySelector("input").checked);
        });
      };
      const span = document.createElement("span");
      span.textContent = option;
      label.append(input, span);
      options.appendChild(label);
    });
    block.appendChild(options);

    // Always present, whatever options the model produced - this is the "Other" box.
    const other = document.createElement("textarea");
    other.className = "other-box";
    other.rows = 1;
    other.placeholder = "Other - write your own answer";
    other.disabled = answered;
    other.dataset.other = String(index);
    if (answer && answer.other) other.value = answer.other;
    block.appendChild(other);

    node.appendChild(block);
  });

  if (!answered) {
    const actions = document.createElement("div");
    actions.className = "questions-actions";

    const submit = document.createElement("button");
    submit.className = "btn-small";
    submit.textContent = "Submit answers";
    submit.onclick = () => submitAnswers(node, questions);

    const skip = document.createElement("button");
    skip.className = "btn-small";
    skip.textContent = "Skip - use your best judgment";
    skip.onclick = () => runBusy(() => api(`/conversations/${state.currentId}/skip`, { method: "POST" }));

    actions.append(submit, skip);
    node.appendChild(actions);
  }
  return node;
}

function submitAnswers(node, questions) {
  const answers = questions.map((question, index) => {
    const selected = [...node.querySelectorAll(`input[name="q${index}"]:checked`)].map((input) => input.value);
    const other = node.querySelector(`textarea[data-other="${index}"]`).value.trim();
    return { question: question.question, selected, other };
  });
  return runBusy(() =>
    api(`/conversations/${state.currentId}/answers`, {
      method: "POST",
      body: JSON.stringify({ answers }),
    }),
  );
}

function renderFinal(message) {
  const node = card({
    label: "Final prompt",
    body: renderMarkdown(message.content),
    className: "card-final",
    open: true,
  });
  const actions = document.createElement("div");
  actions.className = "final-actions";

  const copy = document.createElement("button");
  copy.className = "btn-small";
  copy.textContent = "Copy prompt";
  copy.onclick = async () => {
    await navigator.clipboard.writeText(message.content);
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy prompt"), 1400);
  };

  const download = document.createElement("a");
  download.className = "btn-small";
  download.textContent = "Download .md";
  download.href = `/api/conversations/${state.currentId}/output`;
  download.style.textDecoration = "none";

  const path = document.createElement("span");
  path.className = "saved-path";
  const saved = (message.payload && message.payload.output_path) || "";
  path.textContent = saved ? `Saved to ${saved.split("/").slice(-3).join("/")}` : "";
  path.title = saved;

  actions.append(copy, download, path);
  node.querySelector(".card-body").appendChild(actions);
  return node;
}

function renderConversation() {
  const conversation = state.conversation;
  el("conversation-title").textContent = conversation ? conversation.title : "New chat";
  el("conversation-id").textContent = conversation ? conversation.id : "";
  el("usage").textContent =
    state.usage && state.usage.calls
      ? `${state.usage.calls} calls | ${state.usage.total_tokens.toLocaleString()} tokens | ${state.usage.latency_seconds.toFixed(1)}s`
      : "";

  if (conversation) {
    el("target-model").value = conversation.target_model || "";
    el("audience").value = conversation.audience || "";
    el("model-notes").value = conversation.model_notes || "";
    el("humanize").checked = Boolean(conversation.humanize);
    el("max-rounds").value = conversation.max_clarify_rounds;
  }

  transcript.innerHTML = "";
  if (!state.messages.length) {
    const empty = document.createElement("div");
    empty.className = "empty";
    empty.innerHTML = `Describe what you want a prompt for.<br />It gets built, clarified with you, planned, optimized${
      conversation && conversation.target_model ? ", and tuned for " + escapeHtml(conversation.target_model) : ""
    }, then saved under <code>outputs/${conversation ? conversation.id : ""}/</code>.`;
    transcript.appendChild(empty);
    return;
  }

  for (const message of state.messages) {
    let node;
    if (message.role === "user") {
      node = document.createElement("div");
      node.className = "msg msg-user";
      const bubble = document.createElement("div");
      bubble.className = "bubble";
      bubble.textContent = message.content;
      node.appendChild(bubble);
    } else if (message.kind === "questions" || message.kind === "questions_answered") {
      node = renderQuestions(message);
    } else if (message.kind === "final") {
      node = renderFinal(message);
      node.classList.add("msg");
    } else if (message.kind === "error") {
      node = card({ label: "Stage failed", body: escapeHtml(message.content), className: "card-error", open: true });
      node.classList.add("msg");
    } else if (message.kind === "stage") {
      node = card({ label: message.payload.label, body: renderMarkdown(message.content) });
      node.classList.add("msg");
    } else {
      node = card({ label: "Assistant", body: renderMarkdown(message.content), open: true });
      node.classList.add("msg");
    }
    transcript.appendChild(node);
  }

  if (state.busy) {
    const thinking = document.createElement("div");
    thinking.className = "thinking";
    thinking.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span><span>Running the pipeline...</span>`;
    transcript.appendChild(thinking);
  }
  transcript.scrollTop = transcript.scrollHeight;
}

/* ------------------------------------------------------------------ actions */

function setBusy(busy) {
  state.busy = busy;
  el("send").disabled = busy;
  el("composer-input").disabled = busy;
  document.querySelectorAll(".questions-actions button").forEach((button) => (button.disabled = busy));
  renderConversation();
}

async function runBusy(work) {
  if (state.busy) return;
  setBusy(true);
  try {
    applyState(await work());
  } catch (error) {
    alert(error.message);
  } finally {
    setBusy(false);
  }
}

async function selectConversation(conversationId) {
  state.currentId = conversationId;
  applyState(await api(`/conversations/${conversationId}`));
}

async function newConversation() {
  // Carry the current settings forward - they are almost always the same next time.
  const current = state.conversation || {};
  applyState(
    await api("/conversations", {
      method: "POST",
      body: JSON.stringify({
        audience: current.audience || "a general-purpose LLM",
        target_model: current.target_model || "",
        model_notes: current.model_notes || "",
        humanize: Boolean(current.humanize),
        max_clarify_rounds: current.max_clarify_rounds ?? 3,
      }),
    }),
  );
  el("composer-input").focus();
}

async function send() {
  const input = el("composer-input");
  const content = input.value.trim();
  if (!content) return;
  if (!state.currentId) await newConversation();
  input.value = "";
  input.style.height = "auto";
  await runBusy(() =>
    api(`/conversations/${state.currentId}/messages`, {
      method: "POST",
      body: JSON.stringify({ content }),
    }),
  );
}

let settingsTimer = null;
function saveSettings() {
  if (!state.currentId) return;
  clearTimeout(settingsTimer);
  settingsTimer = setTimeout(async () => {
    const payload = {
      target_model: el("target-model").value,
      audience: el("audience").value,
      model_notes: el("model-notes").value,
      humanize: el("humanize").checked,
      max_clarify_rounds: Number(el("max-rounds").value),
    };
    const result = await api(`/conversations/${state.currentId}`, {
      method: "PATCH",
      body: JSON.stringify(payload),
    });
    state.conversation = result.conversation;
  }, 400);
}

/* ------------------------------------------------------------------ wiring */

el("new-chat").onclick = newConversation;
el("send").onclick = send;
el("toggle-settings").onclick = () => {
  const panel = el("settings");
  panel.hidden = !panel.hidden;
};
["target-model", "audience", "model-notes", "humanize", "max-rounds"].forEach((id) => {
  el(id).addEventListener("change", saveSettings);
  el(id).addEventListener("input", saveSettings);
});

const composer = el("composer-input");
composer.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    send();
  }
});
composer.addEventListener("input", () => {
  composer.style.height = "auto";
  composer.style.height = `${Math.min(composer.scrollHeight, 200)}px`;
});

(async function init() {
  await loadConversations();
  if (state.conversations.length) {
    await selectConversation(state.conversations[0].id);
  } else {
    await newConversation();
  }
  composer.focus();
})();
