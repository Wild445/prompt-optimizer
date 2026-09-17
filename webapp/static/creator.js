/* The prompt-creation workspace: the six-stage synthesis chat.
   Shared helpers live in common.js; the shell that switches between this
   workspace and the optimizer lives in main.js. */

import { api, card, el, escapeHtml, formatCost, readNdjson, relativeTime, renderMarkdown } from "./common.js";

  const state = {
    conversations: [],
    currentId: null,
    conversation: null,
    messages: [],
    usage: null,
    busy: false,
    stages: [],
    stagesError: null,
    expandedStages: new Set(),
    steps: [],
  };

  /* Nodes owned by the in-flight run. They are appended straight to the transcript
     instead of going through renderConversation(), which would wipe them on every
     token; the closing `state` event re-renders everything from the server. */
  const live = { node: null, body: null, stage: null, text: "", thinking: null };

  const transcript = el("transcript");

  function applyState(payload) {
    state.conversation = payload.conversation;
    state.messages = payload.messages || [];
    state.usage = payload.usage || null;
    state.steps = payload.steps || [];
    if (payload.conversation) state.currentId = payload.conversation.id;
    renderConversation();
    loadConversations();
  }

  /* ------------------------------------------------------------------ sidebar */
  
  async function loadConversations() {
    const { conversations } = await api("/conversations");
    state.conversations = conversations;
    renderSidebar();
  }
  
  async function loadStages() {
    // A stale cached bundle or an old server is the usual cause of failure here;
    // surface it in the picker instead of aborting startup.
    try {
      const { stages } = await api("/stages");
      state.stages = stages;
    } catch (error) {
      state.stages = [];
      state.stagesError = error.message;
    }
  }
  
  function renderSidebar() {
    const list = el("conversation-list");
    const count = el("conversation-count");
    list.innerHTML = "";
    if (count) {
      const total = state.conversations.length;
      count.textContent = total ? `${total} chat${total === 1 ? "" : "s"}` : "";
    }
    if (!state.conversations.length) {
      list.innerHTML = `<p class="empty" style="margin:24px 8px;font-size:12px">No conversations yet.</p>`;
      return;
    }
    for (const conversation of state.conversations) {
      const node = document.createElement("div");
      node.className = `conversation${conversation.id === state.currentId ? " active" : ""}`;
      node.innerHTML = `
        <div class="conversation-main">
          <div class="conversation-name"></div>
          <div class="conversation-meta">
            <span>${relativeTime(conversation.updated_at)}</span>
            <span>|</span>
            <span>${conversation.stage === "complete" ? "done" : conversation.stage}</span>
            <span>|</span>
            <span>${formatCost(conversation.estimated_cost)}</span>
          </div>
        </div>
        <button type="button" class="conversation-delete" title="Delete chat" aria-label="Delete chat">&#10005;</button>`;
      node.querySelector(".conversation-name").textContent = conversation.title;
      node.onclick = () => selectConversation(conversation.id);
      node.querySelector(".conversation-delete").onclick = (event) => {
        // Without this the row's own click handler would re-open what we just deleted.
        event.stopPropagation();
        deleteConversation(conversation);
      };
      list.appendChild(node);
    }
  }
  
  async function deleteConversation(conversation) {
    if (state.busy) return;
    if (!window.confirm(`Delete "${conversation.title}"? Its messages and usage history are removed from the database.`)) {
      return;
    }
    try {
      await api(`/conversations/${conversation.id}`, { method: "DELETE" });
    } catch (error) {
      alert(error.message);
      await loadConversations();
      return;
    }
    state.conversations = state.conversations.filter((item) => item.id !== conversation.id);
    if (state.currentId !== conversation.id) {
      renderSidebar();
      return;
    }
    // The open chat just disappeared: fall back to the newest one, or a blank chat.
    state.currentId = null;
    state.conversation = null;
    state.messages = [];
    state.usage = null;
    if (state.conversations.length) {
      await selectConversation(state.conversations[0].id);
    } else {
      await newConversation();
    }
  }
  
  /* ------------------------------------------------------------------ transcript */
  
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
      skip.onclick = () => {
        lockQuestions(node);
        runStream(`/conversations/${state.currentId}/skip/stream`);
      };
  
      actions.append(submit, skip);
      node.appendChild(actions);
    }
    return node;
  }
  
  function lockQuestions(node) {
    // The card is superseded server-side; grey it out now so it can't be answered twice.
    node.classList.add("answered");
    node.querySelectorAll("input, textarea, button").forEach((field) => (field.disabled = true));
  }
  
  function submitAnswers(node, questions) {
    const answers = questions.map((question, index) => {
      const selected = [...node.querySelectorAll(`input[name="q${index}"]:checked`)].map((input) => input.value);
      const other = node.querySelector(`textarea[data-other="${index}"]`).value.trim();
      return { question: question.question, selected, other };
    });
    lockQuestions(node);
    return runStream(`/conversations/${state.currentId}/answers/stream`, { answers });
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
        ? `${state.usage.calls} calls | ${state.usage.total_tokens.toLocaleString()} tokens | ${state.usage.latency_seconds.toFixed(1)}s | ${formatCost(state.usage.estimated_cost)}`
        : "";
  
    if (conversation) {
      el("target-model").value = conversation.target_model || "";
      el("audience").value = conversation.audience || "";
      el("model-notes").value = conversation.model_notes || "";
      el("humanize").checked = Boolean(conversation.humanize);
      el("max-rounds").value = conversation.max_clarify_rounds;
    }
  
    renderStagePicker();
    renderProgress();
  
    transcript.innerHTML = "";
    if (!state.messages.length) {
      const empty = document.createElement("div");
      empty.className = "empty";
      const startStage = (conversation && conversation.start_stage) || "build_prompt";
      const startInfo = state.stages.find((stage) => stage.key === startStage);
      const startLabel = startStage === "build_prompt" ? "Describe what you want a prompt for" : `Paste in your ${(startInfo && startInfo.title.toLowerCase()) || "input"}`;
      empty.innerHTML = `${startLabel}.<br />It gets built, clarified with you, planned, optimized${
        conversation && conversation.target_model ? ", and tuned for " + escapeHtml(conversation.target_model) : ""
      }, then saved under <code>outputs/${conversation ? conversation.id : ""}/</code>.${
        startStage !== "build_prompt" ? `<br />Starting from step: <strong>${escapeHtml((startInfo && startInfo.title) || startStage)}</strong> - pick a different step in the sidebar first if that's wrong.` : ""
      }`;
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
  
    transcript.scrollTop = transcript.scrollHeight;
  }
  
  /* ------------------------------------------------------------------ progress panel */
  
  const STEP_NOTES = { done: "done", active: "running", pending: "waiting" };
  
  function renderProgress() {
    const list = el("progress-list");
    const status = el("progress-status");
    if (!list) return;
    list.innerHTML = "";
  
    const steps = state.steps || [];
    if (!steps.length) {
      const note = document.createElement("li");
      note.className = "progress-empty";
      note.textContent = "Send a message to start the pipeline.";
      list.appendChild(note);
      status.textContent = "Idle";
      status.className = "progress-status";
      return;
    }
  
    const done = steps.filter((step) => step.status === "done").length;
    const finished = done === steps.length;
    status.textContent = finished ? "All done" : `${done} of ${steps.length}`;
    status.className = `progress-status${finished ? " ok" : ""}`;
  
    const waitingOnUser =
      !state.busy && state.conversation && state.conversation.stage === "clarifying";
  
    steps.forEach((step, index) => {
      const item = document.createElement("li");
      item.className = `progress-step ${step.status}`;
  
      const icon = document.createElement("span");
      icon.className = "step-icon";
      if (step.status === "done") icon.textContent = "\u2713";
      else if (step.status === "active") icon.innerHTML = `<span class="step-spinner"></span>`;
      else icon.textContent = String(index + 1);
  
      const text = document.createElement("span");
      text.className = "step-text";
      const title = document.createElement("span");
      title.className = "step-title";
      title.textContent = step.title;
      const note = document.createElement("span");
      note.className = "step-note";
      note.textContent =
        step.status === "active" && waitingOnUser ? "needs your answer" : STEP_NOTES[step.status];
      text.append(title, note);
  
      item.append(icon, text);
      list.appendChild(item);
    });
  }
  
  /* ------------------------------------------------------------------ stage picker */
  
  function renderStagePicker() {
    const list = el("stage-list");
    if (!list) return;
    list.innerHTML = "";
    if (!state.stages.length) {
      const note = document.createElement("p");
      note.className = "stage-empty";
      note.textContent = state.stagesError
        ? `Could not load steps: ${state.stagesError}`
        : "Loading steps\u2026";
      list.appendChild(note);
      return;
    }
    const conversation = state.conversation;
    const selected = (conversation && conversation.start_stage) || "build_prompt";
    const locked = Boolean(conversation) && conversation.stage !== "awaiting_idea";
  
    for (const stageInfo of state.stages) {
      const isSelected = stageInfo.key === selected;
      const isExpanded = state.expandedStages.has(stageInfo.key);
      const item = document.createElement("div");
      item.className = `stage-item${isSelected ? " selected" : ""}${isExpanded ? " expanded" : ""}${locked ? " locked" : ""}`;
  
      const row = document.createElement("div");
      row.className = "stage-item-row";
  
      const radio = document.createElement("input");
      radio.type = "radio";
      radio.name = "start-stage";
      radio.value = stageInfo.key;
      radio.checked = isSelected;
      radio.disabled = locked;
      radio.onclick = (event) => event.stopPropagation();
      radio.onchange = () => setStartStage(stageInfo.key);
  
      const title = document.createElement("span");
      title.className = "stage-item-title";
      title.textContent = stageInfo.title;
  
      const toggle = document.createElement("span");
      toggle.className = "stage-item-toggle";
      toggle.textContent = "\u25B6";
  
      row.append(radio, title, toggle);
      row.onclick = (event) => {
        if (event.target === radio) return;
        item.classList.toggle("expanded");
        if (item.classList.contains("expanded")) state.expandedStages.add(stageInfo.key);
        else state.expandedStages.delete(stageInfo.key);
      };
  
      const desc = document.createElement("div");
      desc.className = "stage-item-desc";
      desc.textContent = stageInfo.description;
  
      item.append(row, desc);
      list.appendChild(item);
    }
  }
  
  async function setStartStage(startStage) {
    if (!state.currentId) return;
    try {
      const result = await api(`/conversations/${state.currentId}`, {
        method: "PATCH",
        body: JSON.stringify({ start_stage: startStage }),
      });
      state.conversation = result.conversation;
      renderStagePicker();
    } catch (error) {
      alert(error.message);
      renderStagePicker();
    }
  }
  
  /* ------------------------------------------------------------------ actions */
  
  function setBusy(busy) {
    state.busy = busy;
    el("send").disabled = busy;
    el("composer-input").disabled = busy;
    document.querySelectorAll(".questions-actions button").forEach((button) => (button.disabled = busy));
    renderProgress();
  }
  
  /* ------------------------------------------------------------------ live run */
  
  function scrollToBottom(force = false) {
    // Don't yank the view back down if the user has scrolled up to read something.
    const nearBottom = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 140;
    if (force || nearBottom) transcript.scrollTop = transcript.scrollHeight;
  }
  
  function appendUserBubble(text) {
    const node = document.createElement("div");
    node.className = "msg msg-user";
    const bubble = document.createElement("div");
    bubble.className = "bubble";
    bubble.textContent = text;
    node.appendChild(bubble);
    transcript.appendChild(node);
    scrollToBottom(true);
  }
  
  function showThinking(label) {
    hideThinking();
    const node = document.createElement("div");
    node.className = "thinking";
    node.innerHTML = `<span class="dot"></span><span class="dot"></span><span class="dot"></span><span class="thinking-label"></span>`;
    node.querySelector(".thinking-label").textContent = label;
    transcript.appendChild(node);
    live.thinking = node;
    scrollToBottom(true);
  }
  
  function hideThinking() {
    if (live.thinking) live.thinking.remove();
    live.thinking = null;
  }
  
  function stageTitle(key) {
    const step = (state.steps || []).find((item) => item.key === key);
    const info = state.stages.find((item) => item.key === key);
    return (step && step.title) || (info && info.title) || key;
  }
  
  function startLiveCard(stage) {
    hideThinking();
    const node = card({ label: `${stageTitle(stage)} - writing...`, body: "", className: "card-streaming", open: true });
    node.classList.add("msg");
    // Deltas arrive mid-word and mid-syntax, so they are shown as plain text and
    // only re-rendered as markdown once the stage completes.
    node.querySelector(".card-body").classList.add("stream-text");
    transcript.appendChild(node);
    live.node = node;
    live.body = node.querySelector(".card-body");
    live.stage = stage;
    live.text = "";
    scrollToBottom(true);
  }
  
  function appendDelta(event) {
    if (live.stage !== event.stage) startLiveCard(event.stage);
    live.text += event.text;
    live.body.textContent = live.text;
    scrollToBottom();
  }
  
  function finishLiveCard() {
    hideThinking();
    if (!live.node) return;
    live.body.classList.remove("stream-text");
    live.body.innerHTML = renderMarkdown(live.text);
    live.node.querySelector(".card-label").textContent = stageTitle(live.stage);
    live.node.classList.remove("card-streaming", "open");
    live.node = null;
    live.body = null;
    live.stage = null;
    live.text = "";
  }
  
  function clearLive() {
    hideThinking();
    live.node = null;
    live.body = null;
    live.stage = null;
    live.text = "";
  }
  
  function handleEvent(event) {
    if (event.steps) {
      state.steps = event.steps;
      renderProgress();
    }
    if (event.type === "stage_start") showThinking(`${event.title}...`);
    else if (event.type === "delta") appendDelta(event);
    else if (event.type === "stage_done") finishLiveCard();
    else if (event.type === "error") {
      clearLive();
      alert(event.detail);
    } else if (event.type === "state") {
      clearLive();
      applyState(event);
    }
  }
  
  async function runStream(path, body, echo) {
    if (state.busy) return;
    setBusy(true);
    if (echo) appendUserBubble(echo);
    showThinking("Starting...");
    try {
      await readNdjson(path, body, handleEvent);
    } catch (error) {
      clearLive();
      alert(error.message);
      // The run may have advanced server-side before the connection broke.
      if (state.currentId) applyState(await api(`/conversations/${state.currentId}`));
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
          start_stage: current.start_stage || "build_prompt",
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
    await runStream(`/conversations/${state.currentId}/messages/stream`, { content }, content);
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

  let attached = false;

  function attach() {
    if (attached) return;
    attached = true;

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
  }

  /* Called by main.js the first time this workspace is shown. Loading is deferred
     so opening the app straight into the optimizer costs nothing here. */
  export async function init() {
    attach();
    if (state.currentId) return;
    await loadStages();
    await loadConversations();
    if (state.conversations.length) {
      await selectConversation(state.conversations[0].id);
    } else {
      await newConversation();
    }
    el("composer-input").focus();
  }
