/* The prompt-optimization workspace: build an LLM-as-a-judge from your own
   observations, grade a fixed set of test cases with it, and loop the prompt
   until they pass.

   Unlike the creation workspace there is no free-text composer: every step owns
   its own form, rendered from the message kind the server stored, so reopening a
   session lands the user back on exactly the step they left. */

import { api, card, el, escapeHtml, formatCost, node, readNdjson, relativeTime, renameInline, renderMarkdown, truncated } from "./common.js";

const EMPTY_SCORE = { total: 0, passed: 0, failed: 0, errored: 0 };

const state = {
  list: [],
  currentId: null,
  optimization: null,
  messages: [],
  criteria: [],
  cases: [],
  results: [],
  scoreboard: { ...EMPTY_SCORE },
  promptVersions: [],
  usage: null,
  steps: [],
  stepInfo: [],
  templateKinds: ["jinja2", "langchain"],
  columnHelp: "",
  expandedSteps: new Set(),
  busy: false,
  /* Set while a run is streaming so the panel counts up from the events rather
     than from the last server snapshot. Cleared when the run settles. */
  liveScore: null,
  liveCases: [],
};

const live = { node: null, body: null, step: null, text: "", thinking: null, runCard: null };

const transcript = () => el("opt-transcript");

const STEP_NOTES = { done: "done", active: "running", pending: "waiting" };

const TEMPLATE_LABELS = { jinja2: "Jinja2 — {{ variable }}", langchain: "LangChain — {variable}" };

/* ------------------------------------------------------------------ state */

function applyState(payload) {
  state.optimization = payload.optimization;
  state.messages = payload.messages || [];
  state.criteria = payload.criteria || [];
  state.cases = payload.cases || [];
  state.results = payload.results || [];
  state.scoreboard = payload.scoreboard || { ...EMPTY_SCORE };
  state.promptVersions = payload.prompt_versions || [];
  state.usage = payload.usage || null;
  state.steps = payload.steps || [];
  if (payload.template_kinds) state.templateKinds = payload.template_kinds;
  if (payload.optimization) state.currentId = payload.optimization.id;
  state.liveScore = null;
  renderSession();
  loadList();
}

async function loadList() {
  const { optimizations } = await api("/optimizations");
  state.list = optimizations;
  renderSidebar();
}

async function loadStepInfo() {
  try {
    const info = await api("/optimizer/steps");
    state.stepInfo = info.steps || [];
    state.templateKinds = info.template_kinds || state.templateKinds;
    state.columnHelp = info.column_help || "";
  } catch (error) {
    state.stepInfo = [];
    state.columnHelp = error.message;
  }
}

/* ------------------------------------------------------------------ sidebar */

/* A rename in progress pins the list, for the reason given in creator.js. */
let renaming = false;
let sidebarStale = false;

function renderSidebar() {
  if (renaming) { sidebarStale = true; return; }
  const list = el("optimization-list");
  const count = el("optimization-count");
  if (!list) return;
  list.innerHTML = "";
  if (count) {
    const total = state.list.length;
    count.textContent = total ? `${total} run${total === 1 ? "" : "s"}` : "";
  }
  if (!state.list.length) {
    list.innerHTML = `<p class="empty" style="margin:16px 8px;font-size:12px">No optimizations yet.</p>`;
    return;
  }
  for (const item of state.list) {
    const row = node("div", { class: `conversation${item.id === state.currentId ? " active" : ""}` });
    row.innerHTML = `
      <div class="conversation-main">
        <div class="conversation-name"></div>
        <div class="conversation-meta">
          <span>${relativeTime(item.updated_at)}</span>
          <span>|</span>
          <span>${item.iteration ? `iteration ${item.iteration}` : stageLabel(item.stage)}</span>
          <span>|</span>
          <span>${formatCost(item.estimated_cost)}</span>
        </div>
      </div>
      <button type="button" class="conversation-rename" title="Rename optimization" aria-label="Rename optimization">&#9998;</button>
      <button type="button" class="conversation-delete" title="Delete optimization" aria-label="Delete optimization">&#10005;</button>`;
    const name = row.querySelector(".conversation-name");
    name.textContent = item.title;
    row.onclick = () => select(item.id);
    const startRename = (event) => {
      // The pencil opens the editor without also opening the session.
      event.stopPropagation();
      renaming = true;
      renameInline(name, {
        value: item.title,
        onSave: (title) => rename(item.id, title),
        onEnd: () => {
          renaming = false;
          if (sidebarStale) { sidebarStale = false; renderSidebar(); }
        },
      });
    };
    row.querySelector(".conversation-rename").onclick = startRename;
    name.ondblclick = startRename;
    row.querySelector(".conversation-delete").onclick = (event) => {
      // Without this the row's own click handler would re-open what we just deleted.
      event.stopPropagation();
      remove(item);
    };
    list.appendChild(row);
  }
}

function stageLabel(stage) {
  return String(stage || "").replace(/_/g, " ");
}

function renderStepGuide() {
  const list = el("optimizer-step-list");
  if (!list) return;
  list.innerHTML = "";
  if (!state.stepInfo.length) {
    list.appendChild(node("p", { class: "stage-empty", text: "Loading steps…" }));
    return;
  }
  for (const info of state.stepInfo) {
    const expanded = state.expandedSteps.has(info.key);
    const item = node("div", { class: `stage-item${expanded ? " expanded" : ""}` });
    const row = node(
      "div",
      { class: "stage-item-row" },
      node("span", { class: "stage-item-title", text: info.title }),
      node("span", { class: "stage-item-toggle", text: "▶" }),
    );
    row.onclick = () => {
      item.classList.toggle("expanded");
      if (item.classList.contains("expanded")) state.expandedSteps.add(info.key);
      else state.expandedSteps.delete(info.key);
    };
    item.append(row, node("div", { class: "stage-item-desc", text: info.description }));
    list.appendChild(item);
  }
}

/* ------------------------------------------------------------------ transcript */

function renderSession() {
  const optimization = state.optimization;
  el("optimization-title").textContent = optimization ? optimization.title : "New optimization";
  el("optimization-id").textContent = optimization ? optimization.id : "";
  el("optimization-usage").textContent =
    state.usage && state.usage.calls
      ? `${state.usage.calls} calls | ${state.usage.total_tokens.toLocaleString()} tokens | ${state.usage.latency_seconds.toFixed(1)}s | ${formatCost(state.usage.estimated_cost)}`
      : "";

  renderStepGuide();
  renderProgress();

  const view = transcript();
  view.innerHTML = "";
  for (const message of state.messages) {
    const rendered = renderMessage(message);
    if (rendered) view.appendChild(rendered);
  }
  const pending = pendingForm();
  if (pending) view.appendChild(pending);
  view.scrollTop = view.scrollHeight;
}

function pendingForm() {
  /* The only step with no stored card behind it: nothing has happened yet, so
     there is nothing for the server to have written into the transcript. */
  const optimization = state.optimization;
  if (!optimization || optimization.stage !== "awaiting_prompt") return null;
  return promptForm();
}

function renderMessage(message) {
  const kinds = {
    prompt: () => userBubble(message.content),
    text: () => (message.role === "user" ? userBubble(message.content) : assistantCard(message)),
    dataset: () => userBubble(message.content),
    review: () => userBubble(message.content),
    approval: () => userBubble(message.content),
    step: () => stepCard(message),
    error: () => withClass(card({ label: "Step failed", body: escapeHtml(message.content), className: "card-error", open: true })),
    observations_form: () => observationsForm(message),
    observations_form_done: () => null,
    criteria_form: () => criteriaForm(message, false),
    criteria_form_done: () => criteriaForm(message, true),
    criteria_added: () => criteriaAddedCard(message),
    dataset_form: () => datasetForm(message, false),
    dataset_form_done: () => datasetForm(message, true),
    run_form: () => runForm(message, false),
    run_form_done: () => null,
    results_form: () => resultsForm(message, false),
    results_form_done: () => resultsSummary(message),
    changes_form: () => changesForm(message, false),
    changes_form_done: () => changesForm(message, true),
    iteration_form: () => iterationForm(message, false),
    iteration_form_done: () => iterationForm(message, true),
    complete: () => withClass(card({ label: "Optimization complete", body: renderMarkdown(message.content), className: "card-final", open: true })),
  };
  const build = kinds[message.kind];
  return build ? build() : assistantCard(message);
}

function withClass(element) {
  element.classList.add("msg");
  return element;
}

function userBubble(text) {
  return node("div", { class: "msg msg-user" }, node("div", { class: "bubble", text }));
}

function assistantCard(message) {
  return withClass(card({ label: "Assistant", body: renderMarkdown(message.content), open: true }));
}

function stepCard(message) {
  const payload = message.payload || {};
  const element = withClass(card({ label: payload.label || "Step", body: renderMarkdown(message.content) }));
  const body = element.querySelector(".card-body");

  if (payload.changes && payload.changes.length) {
    body.appendChild(node("h4", { text: "Prescribed changes" }));
    const list = node("ol", { class: "change-list" });
    for (const change of payload.changes) {
      list.appendChild(
        node(
          "li",
          {},
          node("div", { class: "change-what", text: change.change || "" }),
          node("div", {
            class: "change-where",
            text: `${(change.criteria || []).join(", ") || "from your remark"} → ${change.prompt_section || "(missing)"}`,
          }),
          change.evidence ? node("div", { class: "change-evidence", text: change.evidence }) : null,
        ),
      );
    }
    body.appendChild(list);
  }
  if (payload.step === "revise") {
    body.appendChild(copyRow(message.content, `Prompt v${payload.version || "?"}`));
  }
  return element;
}

function copyRow(text, label) {
  const copy = node("button", { class: "btn-small", text: "Copy" });
  copy.onclick = async () => {
    await navigator.clipboard.writeText(text);
    copy.textContent = "Copied";
    setTimeout(() => (copy.textContent = "Copy"), 1400);
  };
  const download = node("a", {
    class: "btn-small",
    text: "Download .md",
    href: `/api/optimizations/${state.currentId}/prompt`,
    style: "text-decoration:none",
  });
  return node("div", { class: "final-actions" }, copy, download, node("span", { class: "saved-path", text: label || "" }));
}

/* ------------------------------------------------------------------ step 1: the prompt */

function promptForm() {
  const optimization = state.optimization || {};
  const box = node("div", { class: "msg form-card" });
  box.append(
    node("h3", { text: "The prompt you want to optimize" }),
    node("p", {
      class: "form-hint",
      text:
        "Paste the final prompt as it runs today, template variables and all." +
        " Everything after this — the judge, the test run, the revisions — is built around it.",
    }),
  );

  const area = node("textarea", {
    class: "form-area",
    rows: 10,
    placeholder: "Paste the prompt here…",
    id: "opt-prompt-input",
  });
  box.appendChild(area);

  const select = node("select", { class: "form-select", id: "opt-template-kind" });
  for (const kind of state.templateKinds) {
    select.appendChild(
      node("option", { value: kind, text: TEMPLATE_LABELS[kind] || kind, selected: optimization.template_kind === kind }),
    );
  }
  box.appendChild(
    node(
      "label",
      { class: "form-row" },
      node("span", { text: "Prompt template" }),
      select,
      node("small", { text: "How the variables in the prompt are written. Values come from the test-case file later." }),
    ),
  );

  const submit = node("button", { class: "btn-primary form-submit", text: "Build the judge scaffold" });
  submit.onclick = () => {
    const prompt = area.value.trim();
    if (!prompt) {
      area.focus();
      return;
    }
    run(`/optimizations/${state.currentId}/prompt/stream`, { prompt, template_kind: select.value });
  };
  box.appendChild(submit);
  return box;
}

/* ------------------------------------------------------------------ step 3: observations */

function observationsForm(message) {
  const payload = message.payload || {};
  const box = node("div", { class: "msg form-card" });
  box.append(
    node("h3", { text: "What is the current prompt getting wrong?" }),
    node("p", { class: "form-hint", text: message.content }),
  );
  if (payload.template_variables && payload.template_variables.length) {
    box.appendChild(
      node("p", {
        class: "form-note",
        text: `Template variables found in the prompt: ${payload.template_variables.join(", ")}`,
      }),
    );
  }
  const area = node("textarea", {
    class: "form-area",
    rows: 7,
    placeholder:
      "One problem per line, e.g.\nIt answers in paragraphs instead of bullets\nIt invents figures that aren't in the input\nIt ignores the region variable",
  });
  const submit = node("button", { class: "btn-primary form-submit", text: "Draft the success criteria" });
  submit.onclick = () => {
    const observations = area.value.trim();
    if (!observations) {
      area.focus();
      return;
    }
    run(`/optimizations/${state.currentId}/observations/stream`, { observations });
  };
  box.append(area, submit);
  return box;
}

/* ------------------------------------------------------------------ step 4: criteria review */

function criteriaForm(message, done) {
  const payload = message.payload || {};
  const criteria = payload.criteria || [];
  const box = node("div", { class: `msg form-card${done ? " form-done" : ""}` });
  box.append(
    node("h3", { text: done ? "Success criteria (submitted)" : "Review the success criteria" }),
    node("p", { class: "form-hint", text: message.content }),
  );
  if (payload.error) {
    box.appendChild(node("p", { class: "form-warning", text: `Drafter output could not be read: ${payload.error}` }));
  }

  const list = node("div", { class: "criteria-list" });
  const rows = [];

  const addRow = (criterion) => {
    const row = node("div", { class: "criterion" });
    const title = node("input", {
      class: "criterion-title",
      type: "text",
      value: criterion.title || "",
      placeholder: "Short name for the check",
      disabled: done,
    });
    const description = node("textarea", {
      class: "criterion-desc",
      rows: 2,
      placeholder: "What makes a response pass this check",
      disabled: done,
    });
    description.value = criterion.description || "";
    const badge = node("span", { class: `criterion-origin origin-${criterion.origin || "agent"}`, text: criterion.origin || "agent" });
    const drop = node("button", { class: "criterion-drop", title: "Remove this criterion", text: "✕", disabled: done });
    const entry = { title, description, removed: false };
    drop.onclick = () => {
      entry.removed = !entry.removed;
      row.classList.toggle("removed", entry.removed);
      drop.textContent = entry.removed ? "↺" : "✕";
      drop.title = entry.removed ? "Restore this criterion" : "Remove this criterion";
    };
    row.append(
      node("div", { class: "criterion-head" }, node("span", { class: "criterion-id", text: criterion.id || "" }), title, badge, drop),
      description,
    );
    rows.push(entry);
    list.appendChild(row);
  };

  criteria.forEach(addRow);
  box.appendChild(list);

  if (done) return box;

  const addButton = node("button", { class: "btn-small", text: "+ Add a criterion" });
  addButton.onclick = () => addRow({ title: "", description: "", origin: "user" });

  const extras = node("textarea", {
    class: "form-area",
    rows: 3,
    placeholder: "Or type extra criteria here, one per line — they reach the judge exactly as typed.",
  });

  const submit = node("button", { class: "btn-primary form-submit", text: "Build the LLM-as-a-judge" });
  submit.onclick = () => {
    const kept = rows
      .filter((entry) => !entry.removed && (entry.title.value.trim() || entry.description.value.trim()))
      .map((entry) => ({
        title: entry.title.value.trim(),
        description: entry.description.value.trim() || entry.title.value.trim(),
        origin: "user",
      }));
    if (!kept.length && !extras.value.trim()) {
      alert("The judge needs at least one success criterion.");
      return;
    }
    run(`/optimizations/${state.currentId}/criteria/stream`, { criteria: kept, additions: extras.value });
  };

  box.append(node("div", { class: "form-actions" }, addButton), extras, submit);
  return box;
}

function criteriaAddedCard(message) {
  const payload = message.payload || {};
  const body = (payload.added || [])
    .map((criterion) => `- **${escapeHtml(criterion.title || "")}** — ${escapeHtml(criterion.description || "")}`)
    .join("\n");
  return withClass(
    card({ label: message.content, body: renderMarkdown(body), className: "card-added", open: true }),
  );
}

/* ------------------------------------------------------------------ step 6: dataset */

function datasetForm(message, done) {
  const payload = message.payload || {};
  const box = node("div", { class: `msg form-card${done ? " form-done" : ""}` });
  box.append(
    node("h3", { text: done ? "Test cases (uploaded)" : "Upload the test cases" }),
    node("p", { class: "form-hint", text: message.content }),
  );

  const columns = node(
    "div",
    { class: "column-spec" },
    node("code", { text: "message_id" }),
    node("span", { text: "a stable id per test case — keep it the same across iterations" }),
    node("code", { text: "input_payload" }),
    node("span", { text: "the input the prompt is run against" }),
    node("code", { text: "other_input_params" }),
    node("span", {
      text: "the template variables, as JSON or key=value lines. Leave blank if the prompt has none.",
    }),
  );
  box.appendChild(columns);

  if (payload.template_variables && payload.template_variables.length) {
    box.appendChild(
      node("p", {
        class: "form-note",
        text: `This prompt needs: ${payload.template_variables.join(", ")} — supply them in other_input_params.`,
      }),
    );
  }

  if (done) {
    box.appendChild(
      node("p", {
        class: "form-note",
        text: payload.filename ? `${payload.case_count || 0} test case(s) from ${payload.filename}.` : "Uploaded.",
      }),
    );
    return box;
  }

  const file = node("input", { class: "form-file", type: "file", accept: ".xlsx,.xlsm,.csv,.tsv" });
  const status = node("p", { class: "form-note", text: "" });
  const upload = node("button", { class: "btn-primary form-submit", text: "Upload test cases" });
  upload.onclick = async () => {
    if (!file.files || !file.files.length) {
      file.click();
      return;
    }
    upload.disabled = true;
    status.className = "form-note";
    status.textContent = "Reading the file…";
    try {
      applyState(await uploadDataset(file.files[0]));
    } catch (error) {
      status.className = "form-warning";
      status.textContent = error.message;
      upload.disabled = false;
    }
  };

  box.append(
    node(
      "div",
      { class: "form-actions" },
      file,
      node("a", {
        class: "btn-small",
        text: "Download the template",
        href: "/api/optimizer/dataset-template",
        style: "text-decoration:none",
      }),
    ),
    status,
    upload,
  );
  return box;
}

async function uploadDataset(fileObject) {
  const form = new FormData();
  form.append("file", fileObject);
  // Not the json api() helper: multipart needs the browser to set its own
  // Content-Type, boundary included.
  const response = await fetch(`/api/optimizations/${state.currentId}/dataset`, { method: "POST", body: form });
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || `Upload failed (${response.status})`);
  }
  return response.json();
}

/* ------------------------------------------------------------------ step 7-8: the run */

function runForm(message, done) {
  const payload = message.payload || {};
  const box = node("div", { class: `msg form-card${done ? " form-done" : ""}` });
  box.append(node("h3", { text: "Run the test cases" }), node("p", { class: "form-hint", text: message.content }));
  if (payload.missing_variables && payload.missing_variables.length) {
    box.appendChild(
      node("p", {
        class: "form-warning",
        text: `No row supplies a value for: ${payload.missing_variables.join(", ")}. Those cases may fail to render.`,
      }),
    );
  }
  if (!done) {
    const go = node("button", { class: "btn-primary form-submit", text: `Run all ${payload.case_count || state.cases.length} test cases` });
    go.onclick = () => run(`/optimizations/${state.currentId}/run/stream`, {});
    box.appendChild(go);
  }
  return box;
}

/* ------------------------------------------------------------------ step 8-9: results table */

function verdictPill(result) {
  if (result.status === "error") return node("span", { class: "pill pill-error", text: "error" });
  return result.verdict
    ? node("span", { class: "pill pill-pass", text: "✓ pass" })
    : node("span", { class: "pill pill-fail", text: "✗ fail" });
}

function failureText(result) {
  if (result.status === "error") return result.error || "";
  return (result.failures || [])
    .map((failure) => `${failure.id || "?"}: ${failure.why || ""}${failure.prompt_section ? ` [${failure.prompt_section}]` : ""}`)
    .join("\n");
}

function resultsForm(message, done) {
  const payload = message.payload || {};
  const box = node("div", { class: "msg form-card wide-card" });
  box.append(
    node("h3", { text: `Iteration ${payload.iteration || state.optimization.iteration} results` }),
    node("p", { class: "form-hint", text: message.content }),
  );

  const flags = new Map();
  const table = node("table", { class: "results-table" });
  table.innerHTML = `
    <thead><tr>
      <th>message_id</th><th>Input</th><th>Response</th><th>Judge</th>
      <th>Failed criteria</th><th>Why</th><th>Your take</th>
    </tr></thead>`;
  const tbody = node("tbody");

  const submit = node("button", { class: "btn-primary form-submit", text: "Continue — summarize and revise" });
  const counter = node("p", { class: "form-note" });

  const refreshCounter = () => {
    const flagged = [...flags.values()].filter((entry) => entry.flagged);
    const unexplained = flagged.filter((entry) => !entry.reason.value.trim());
    counter.className = unexplained.length ? "form-warning" : "form-note";
    counter.textContent = unexplained.length
      ? `${unexplained.length} flagged test case(s) still need a reason before you can continue.`
      : flagged.length
        ? `${flagged.length} verdict(s) flagged as wrong. Everything else counts as agreed.`
        : "Nothing flagged — every verdict counts as agreed. Flag only the ones you disagree with.";
    submit.disabled = Boolean(unexplained.length);
  };

  for (const result of state.results) {
    const row = node("tr", { class: result.verdict ? "row-pass" : "row-fail" });
    const reason = node("textarea", {
      class: "row-reason",
      rows: 2,
      placeholder: "Why is this verdict wrong?",
      hidden: true,
    });
    reason.value = result.user_reason || "";
    const flag = node("button", {
      class: "btn-flag",
      text: "Flag as wrong",
      title: "Only click this if you disagree with the judge",
    });
    const entry = { flagged: !result.user_agrees, reason, message_id: result.message_id };
    const paint = () => {
      row.classList.toggle("row-flagged", entry.flagged);
      reason.hidden = !entry.flagged;
      flag.textContent = entry.flagged ? "Flagged — undo" : "Flag as wrong";
      flag.classList.toggle("flagged", entry.flagged);
      refreshCounter();
    };
    flag.onclick = () => {
      entry.flagged = !entry.flagged;
      paint();
      if (entry.flagged) reason.focus();
    };
    reason.oninput = refreshCounter;
    flags.set(result.message_id, entry);

    row.append(
      node("td", { class: "cell-id", text: result.message_id }),
      node("td", {}, truncated(result.input_payload, 120)),
      node("td", {}, truncated(result.response, 180)),
      node("td", { class: "cell-verdict" }, verdictPill(result)),
      node("td", { class: "cell-criteria", text: (result.failed_criteria || []).join(", ") }),
      node("td", {}, truncated(failureText(result), 160)),
      node("td", { class: "cell-take" }, flag, reason),
    );
    tbody.appendChild(row);
    paint();
  }
  table.appendChild(tbody);
  box.appendChild(node("div", { class: "table-scroll" }, table));

  const saved = node("a", {
    class: "btn-small",
    text: "Download this table (.csv)",
    href: `/api/optimizations/${state.currentId}/results.csv`,
    style: "text-decoration:none",
  });
  const path = node("span", { class: "saved-path", title: payload.output_path || "" });
  path.textContent = payload.output_path ? `Saved to ${payload.output_path.split("/").slice(-3).join("/")}` : "";
  box.appendChild(node("div", { class: "form-actions" }, saved, path));

  submit.onclick = () => {
    const feedback = [...flags.values()].map((entry) => ({
      message_id: entry.message_id,
      agrees: !entry.flagged,
      reason: entry.flagged ? entry.reason.value.trim() : "",
    }));
    run(`/optimizations/${state.currentId}/review/stream`, { feedback });
  };

  box.append(counter, submit);
  refreshCounter();
  return box;
}

function resultsSummary(message) {
  const payload = message.payload || {};
  const board = payload.scoreboard || EMPTY_SCORE;
  return withClass(
    card({
      label: `Iteration ${payload.iteration || "?"} — reviewed (${board.passed}/${board.total} passed)`,
      body: renderMarkdown(
        `${message.content}\n\nThe full table is saved at \`${payload.output_path || "outputs/optimizations/"}\`.`,
      ),
    }),
  );
}

/* ------------------------------------------------------------------ step 10: approve the changes */

/* One approve/reject row. Everything starts approved, so rejecting is the
   deliberate act and a user who just presses Continue gets what the analyst
   proposed — the same default the results table uses for the judge's verdicts. */
function proposalRow(parts, index, done, kept) {
  const rejected = kept ? !kept.has(index) : false;
  const row = node("div", { class: `proposal${rejected ? " rejected" : ""}` });
  const toggle = node("button", { class: "btn-approve", disabled: done });
  const entry = { index, rejected };

  const paint = () => {
    row.classList.toggle("rejected", entry.rejected);
    toggle.textContent = entry.rejected ? "Rejected" : "Approved";
    toggle.title = entry.rejected ? "Approve this change after all" : "Do not apply this change";
    toggle.classList.toggle("approved", !entry.rejected);
  };
  toggle.onclick = () => {
    entry.rejected = !entry.rejected;
    paint();
    if (parts.onChange) parts.onChange();
  };
  paint();

  row.append(
    node(
      "div",
      { class: "proposal-head" },
      node("span", { class: "proposal-index", text: String(index + 1) }),
      node("div", { class: "proposal-what", text: parts.what }),
      toggle,
    ),
  );
  if (parts.where) row.appendChild(node("div", { class: "change-where", text: parts.where }));
  if (parts.evidence) row.appendChild(node("div", { class: "change-evidence", text: parts.evidence }));
  return { row, entry };
}

function changesForm(message, done) {
  const payload = message.payload || {};
  const changes = payload.changes || [];
  const newCriteria = payload.new_criteria || [];
  /* Only present once submitted: the positions the user kept. */
  const keptChanges = done ? new Set(payload.approved_changes || []) : null;
  const keptCriteria = done ? new Set(payload.approved_criteria || []) : null;

  const box = node("div", { class: `msg form-card${done ? " form-done" : ""}` });
  box.append(
    node("h3", { text: done ? "Proposed changes (reviewed)" : "Approve the proposed changes" }),
    node("p", { class: "form-hint", text: message.content }),
  );
  if (payload.summary) box.appendChild(node("p", { class: "form-note", text: payload.summary }));

  const counter = node("p", { class: "form-note" });
  const submit = node("button", { class: "btn-primary form-submit", text: "Apply the approved changes" });
  const changeEntries = [];
  const criteriaEntries = [];

  const refreshCounter = () => {
    const keptCount = changeEntries.filter((entry) => !entry.rejected).length;
    const keptCriteriaCount = criteriaEntries.filter((entry) => !entry.rejected).length;
    submit.textContent = keptCount
      ? `Apply ${keptCount} approved change${keptCount === 1 ? "" : "s"}`
      : "Continue without changing the prompt";
    counter.className = keptCount ? "form-note" : "form-warning";
    counter.textContent = keptCount
      ? `${keptCount} of ${changes.length} change(s) approved` +
        (newCriteria.length ? `, ${keptCriteriaCount} of ${newCriteria.length} new criteria approved.` : ".")
      : "Every change is rejected — the prompt will stay exactly as it is.";
  };

  if (changes.length) {
    const list = node("div", { class: "proposal-list" });
    changes.forEach((change, index) => {
      const { row, entry } = proposalRow(
        {
          what: change.change || "",
          where: `${(change.criteria || []).join(", ") || "from your remark"} → ${change.prompt_section || "(missing)"}`,
          evidence: change.evidence || "",
          onChange: refreshCounter,
        },
        index,
        done,
        keptChanges,
      );
      changeEntries.push(entry);
      list.appendChild(row);
    });
    box.append(node("h4", { text: "Changes to the prompt" }), list);
  }

  if (newCriteria.length) {
    const list = node("div", { class: "proposal-list" });
    newCriteria.forEach((criterion, index) => {
      const { row, entry } = proposalRow(
        {
          what: criterion.title || "",
          where: criterion.description || "",
          onChange: refreshCounter,
        },
        index,
        done,
        keptCriteria,
      );
      criteriaEntries.push(entry);
      list.appendChild(row);
    });
    box.append(
      node("h4", { text: "New success criteria for the judge" }),
      node("p", {
        class: "form-note",
        text: "These come from remarks no existing criterion covered. Approved ones are scored from the next run on.",
      }),
      list,
    );
  }

  if (done) return box;

  submit.onclick = () => {
    const approvedChanges = changeEntries.filter((entry) => !entry.rejected).map((entry) => entry.index);
    const approvedCriteria = criteriaEntries.filter((entry) => !entry.rejected).map((entry) => entry.index);
    if (!approvedChanges.length && changes.length) {
      const confirmed = window.confirm(
        "You rejected every proposed change. The prompt stays at its current version. Continue?",
      );
      if (!confirmed) return;
    }
    run(`/optimizations/${state.currentId}/changes/stream`, {
      approved_changes: approvedChanges,
      approved_criteria: approvedCriteria,
    });
  };

  box.append(counter, submit);
  refreshCounter();
  return box;
}

/* ------------------------------------------------------------------ step 12: iterate */

function iterationForm(message, done) {
  const payload = message.payload || {};
  const box = node("div", { class: `msg form-card${done ? " form-done" : ""}` });
  const heading =
    payload.changed === false
      ? `Prompt v${payload.version || "?"} is unchanged`
      : `Prompt v${payload.version || "?"} is ready`;
  box.append(node("h3", { text: heading }), node("p", { class: "form-hint", text: message.content }));

  if (payload.new_criteria && payload.new_criteria.length) {
    const list = node("ul", { class: "added-criteria" });
    for (const criterion of payload.new_criteria) {
      list.appendChild(node("li", {}, node("strong", { text: criterion.title || "" }), node("span", { text: ` — ${criterion.description || ""}` })));
    }
    box.append(node("p", { class: "form-note", text: "New success criteria added to the judge for the next run:" }), list);
  }

  if (!done) {
    const again = node("button", { class: "btn-primary form-submit", text: `Run iteration ${payload.next_iteration || "?"}` });
    again.onclick = () => run(`/optimizations/${state.currentId}/run/stream`, {});
    const stop = node("button", { class: "btn-small", text: "Stop here" });
    stop.onclick = async () => {
      applyState(await api(`/optimizations/${state.currentId}/stop`, { method: "POST" }));
    };
    box.append(again, node("div", { class: "form-actions" }, stop));
  }
  return box;
}

/* ------------------------------------------------------------------ progress panel */

function renderProgress() {
  const list = el("optimizer-progress-list");
  const status = el("optimizer-progress-status");
  if (!list) return;
  list.innerHTML = "";

  renderScoreboard();

  const steps = state.steps || [];
  if (!steps.length) {
    list.appendChild(node("li", { class: "progress-empty", text: "Paste a prompt to start the loop." }));
    status.textContent = "Idle";
    status.className = "progress-status";
    return;
  }
  const done = steps.filter((step) => step.status === "done").length;
  const iteration = state.optimization ? state.optimization.iteration : 0;
  status.textContent = iteration ? `iteration ${iteration}` : `${done} of ${steps.length}`;
  status.className = "progress-status";

  steps.forEach((step, index) => {
    const item = node("li", { class: `progress-step ${step.status}` });
    const icon = node("span", { class: "step-icon" });
    if (step.status === "done") icon.textContent = "✓";
    else if (step.status === "active") icon.innerHTML = `<span class="step-spinner"></span>`;
    else icon.textContent = String(index + 1);
    item.append(
      icon,
      node(
        "span",
        { class: "step-text" },
        node("span", { class: "step-title", text: step.title }),
        node("span", { class: "step-note", text: step.waiting ? "needs you" : STEP_NOTES[step.status] }),
      ),
    );
    list.appendChild(item);
  });
}

function renderScoreboard() {
  const panel = el("scoreboard");
  if (!panel) return;
  const board = state.liveScore || state.scoreboard || EMPTY_SCORE;
  const total = board.total || (state.liveScore ? state.liveScore.expected : 0) || state.cases.length;
  panel.innerHTML = "";
  if (!total) {
    panel.appendChild(node("p", { class: "progress-empty", text: "No test cases have been run yet." }));
    return;
  }
  const graded = board.passed + board.failed + board.errored;
  panel.append(
    node(
      "div",
      { class: "score-row score-pass" },
      node("span", { class: "score-label", text: `Success (${board.passed}/${total} test cases)` }),
      node("span", { class: "score-mark", text: "✓" }),
    ),
    node(
      "div",
      { class: "score-row score-fail" },
      node("span", { class: "score-label", text: `Failed (${board.failed}/${total} test cases)` }),
      node("span", { class: "score-mark", text: "✗" }),
    ),
  );
  if (board.errored) {
    panel.appendChild(
      node(
        "div",
        { class: "score-row score-error" },
        node("span", { class: "score-label", text: `Errors (${board.errored}/${total} test cases)` }),
        node("span", { class: "score-mark", text: "!" }),
      ),
    );
  }
  const bar = node("div", { class: "score-bar" });
  bar.append(
    node("span", { class: "bar-pass", style: `width:${(board.passed / total) * 100}%` }),
    node("span", { class: "bar-fail", style: `width:${(board.failed / total) * 100}%` }),
    node("span", { class: "bar-error", style: `width:${(board.errored / total) * 100}%` }),
  );
  panel.append(bar, node("p", { class: "score-note", text: `${graded} of ${total} graded` }));
}

/* ------------------------------------------------------------------ live run */

function scrollToBottom(force = false) {
  // Don't yank the view back down if the user has scrolled up to read something.
  const view = transcript();
  const nearBottom = view.scrollHeight - view.scrollTop - view.clientHeight < 140;
  if (force || nearBottom) view.scrollTop = view.scrollHeight;
}

function showThinking(label) {
  hideThinking();
  const element = node(
    "div",
    { class: "thinking" },
    node("span", { class: "dot" }),
    node("span", { class: "dot" }),
    node("span", { class: "dot" }),
    node("span", { class: "thinking-label", text: label }),
  );
  transcript().appendChild(element);
  live.thinking = element;
  scrollToBottom(true);
}

function hideThinking() {
  if (live.thinking) live.thinking.remove();
  live.thinking = null;
}

function stepTitle(key) {
  const step = (state.steps || []).find((item) => item.key === key);
  const info = state.stepInfo.find((item) => item.key === key);
  return (step && step.title) || (info && info.title) || key;
}

function startLiveCard(step) {
  hideThinking();
  const element = withClass(card({ label: `${stepTitle(step)} - writing...`, body: "", className: "card-streaming", open: true }));
  // Deltas arrive mid-word and mid-syntax, so they are shown as plain text and
  // only re-rendered as markdown once the step completes.
  element.querySelector(".card-body").classList.add("stream-text");
  transcript().appendChild(element);
  live.node = element;
  live.body = element.querySelector(".card-body");
  live.step = step;
  live.text = "";
  scrollToBottom(true);
}

function finishLiveCard() {
  hideThinking();
  if (!live.node) return;
  live.body.classList.remove("stream-text");
  live.body.innerHTML = renderMarkdown(live.text);
  live.node.querySelector(".card-label").textContent = stepTitle(live.step);
  live.node.classList.remove("card-streaming", "open");
  clearLive();
}

function clearLive() {
  hideThinking();
  live.node = null;
  live.body = null;
  live.step = null;
  live.text = "";
  live.runCard = null;
}

function startRunCard(total) {
  hideThinking();
  const element = node("div", { class: "msg run-live" });
  element.append(node("div", { class: "run-live-head", text: `Running ${total} test cases…` }));
  const rows = node("div", { class: "run-live-rows" });
  element.appendChild(rows);
  transcript().appendChild(element);
  live.runCard = { node: element, rows, head: element.querySelector(".run-live-head"), total };
  scrollToBottom(true);
}

function appendRunRow(event) {
  if (!live.runCard) startRunCard(event.scoreboard.total || state.cases.length);
  const item = event.case;
  const board = event.scoreboard;
  const status = item.status === "error" ? "error" : item.passed ? "pass" : "fail";
  live.runCard.rows.appendChild(
    node(
      "div",
      { class: `run-live-row run-${status}` },
      node("span", { class: "run-id", text: item.message_id }),
      node("span", { class: "run-mark", text: status === "pass" ? "✓" : status === "fail" ? "✗" : "!" }),
      node("span", {
        class: "run-detail",
        text: item.error || (item.failed_criteria || []).join(", ") || "all criteria passed",
      }),
    ),
  );
  live.runCard.head.textContent =
    `Running test cases… ${board.passed} passed, ${board.failed} failed of ${live.runCard.total}`;
  scrollToBottom();
}

function handleEvent(event) {
  if (event.steps) {
    state.steps = event.steps;
    renderProgress();
  }
  if (event.type === "step_start") {
    if (event.step === "respond") state.liveScore = { ...EMPTY_SCORE, expected: event.total || state.cases.length };
    showThinking(`${event.title}…`);
  } else if (event.type === "case_start") {
    state.liveScore = { ...EMPTY_SCORE, expected: event.total };
    startRunCard(event.total);
    renderScoreboard();
  } else if (event.type === "case_done") {
    state.liveScore = { ...event.scoreboard, expected: (state.liveScore && state.liveScore.expected) || event.scoreboard.total };
    appendRunRow(event);
    renderScoreboard();
  } else if (event.type === "delta") {
    if (live.step !== event.step) startLiveCard(event.step);
    live.text += event.text;
    live.body.textContent = live.text;
    scrollToBottom();
  } else if (event.type === "step_done") {
    if (event.scoreboard) state.liveScore = { ...event.scoreboard, expected: event.scoreboard.total };
    if (live.node) finishLiveCard();
    else hideThinking();
  } else if (event.type === "error") {
    clearLive();
    alert(event.detail);
  } else if (event.type === "state") {
    clearLive();
    applyState(event);
  }
}

function setBusy(busy) {
  state.busy = busy;
  document.querySelectorAll("#opt-transcript button, #opt-transcript input, #opt-transcript textarea, #opt-transcript select")
    .forEach((field) => (field.disabled = busy));
}

async function run(path, body) {
  if (state.busy) return;
  setBusy(true);
  showThinking("Starting…");
  try {
    await readNdjson(path, body, handleEvent);
  } catch (error) {
    clearLive();
    alert(error.message);
    // The run may have advanced server-side before the connection broke.
    if (state.currentId) applyState(await api(`/optimizations/${state.currentId}`));
  } finally {
    setBusy(false);
  }
}

/* ------------------------------------------------------------------ actions */

async function select(optimizationId) {
  state.currentId = optimizationId;
  applyState(await api(`/optimizations/${optimizationId}`));
}

async function create() {
  const previous = state.optimization || {};
  applyState(
    await api("/optimizations", {
      method: "POST",
      body: JSON.stringify({ template_kind: previous.template_kind || state.templateKinds[0] }),
    }),
  );
}

async function rename(optimizationId, title) {
  await api(`/optimizations/${optimizationId}`, { method: "PATCH", body: JSON.stringify({ title }) });
  const row = state.list.find((item) => item.id === optimizationId);
  if (row) row.title = title;
  if (state.optimization && state.optimization.id === optimizationId) {
    state.optimization.title = title;
    el("optimization-title").textContent = title;
  }
  await loadList();
}

async function remove(item) {
  if (state.busy) return;
  if (!window.confirm(`Delete "${item.title}"? Its criteria, test cases, and graded runs are removed from the database.`)) {
    return;
  }
  try {
    await api(`/optimizations/${item.id}`, { method: "DELETE" });
  } catch (error) {
    alert(error.message);
    await loadList();
    return;
  }
  state.list = state.list.filter((row) => row.id !== item.id);
  if (state.currentId !== item.id) {
    renderSidebar();
    return;
  }
  // The open session just disappeared: fall back to the newest one, or a blank one.
  state.currentId = null;
  state.optimization = null;
  state.messages = [];
  if (state.list.length) await select(state.list[0].id);
  else await create();
}

/* ------------------------------------------------------------------ wiring */

let attached = false;

function attach() {
  if (attached) return;
  attached = true;
  el("new-optimization").onclick = create;

  // The open session is renamed from its own heading too, so nobody has to find
  // the matching sidebar row first.
  const heading = el("optimization-title");
  heading.classList.add("renamable");
  heading.title = "Click to rename";
  heading.onclick = () => {
    if (!state.currentId) return;
    renameInline(heading, {
      value: (state.optimization && state.optimization.title) || "",
      onSave: (title) => rename(state.currentId, title),
    });
  };
}

/* Called by main.js the first time this workspace is shown. Loading is deferred
   so opening the app straight into the chat pipeline costs nothing here. */
export async function init() {
  attach();
  if (state.currentId) return;
  await loadStepInfo();
  await loadList();
  if (state.list.length) await select(state.list[0].id);
  else await create();
}
