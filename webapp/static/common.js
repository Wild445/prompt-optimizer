/* Helpers shared by both workspaces (chat pipeline and prompt optimizer).
   No build step and no dependencies - served straight from webapp/static. */

export const el = (id) => document.getElementById(id);

/* ------------------------------------------------------------------ api */

export async function api(path, options = {}) {
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

/* Reads an NDJSON progress stream: one JSON event per line, handed to `onEvent`
   as it lands so the user sees the pipeline working instead of a frozen page. */
export async function readNdjson(path, body, onEvent) {
  const response = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  if (!response.ok || !response.body) {
    const detail = await response.json().catch(() => ({ detail: response.statusText }));
    throw new Error(detail.detail || `Request failed (${response.status})`);
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split("\n");
    // A chunk can split a line in half; the tail waits for the next read.
    buffer = lines.pop();
    for (const line of lines) if (line.trim()) onEvent(JSON.parse(line));
  }
  if (buffer.trim()) onEvent(JSON.parse(buffer));
}

/* ------------------------------------------------------------------ markdown
   LLM output is untrusted text: escape first, then apply a small subset of
   markdown to the escaped string. Nothing here ever inserts raw model output. */

export function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
}

export function renderMarkdown(source) {
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

/* ------------------------------------------------------------------ widgets */

export function card({ label, body, className = "", open = false }) {
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

/* Turn a label into an inline text box so a session can be renamed wherever its
   name is shown - the sidebar row or the topbar. Enter or blur commits, Escape
   reverts; an empty or unchanged name is a no-op. The new name is painted
   optimistically and rolled back if `onSave` rejects, so the rename feels
   instant without lying about what the server stored. */
export function renameInline(label, { value, onSave, onEnd, maxLength = 120 } = {}) {
  if (label.dataset.editing === "1") return null;
  const original = value === undefined ? label.textContent : String(value);

  const input = node("input", { type: "text", class: "rename-input", maxlength: String(maxLength) });
  input.value = original;
  label.dataset.editing = "1";
  label.hidden = true;
  label.after(input);
  input.focus();
  input.select();

  let settled = false;
  const finish = async (commit) => {
    if (settled) return;
    settled = true;
    const next = input.value.trim();
    input.remove();
    label.hidden = false;
    delete label.dataset.editing;
    // Before onSave, so the list this label lives in is free to re-render again
    // on the reload that a successful save kicks off.
    if (onEnd) onEnd();
    if (!commit || !next || next === original) return;
    label.textContent = next;
    try {
      await onSave(next);
    } catch (error) {
      label.textContent = original;
      alert(error.message);
    }
  };

  input.onkeydown = (event) => {
    // Sidebar rows and the composer listen for keys too; editing owns them here.
    event.stopPropagation();
    if (event.key === "Enter") { event.preventDefault(); finish(true); }
    else if (event.key === "Escape") { event.preventDefault(); finish(false); }
  };
  input.onblur = () => finish(true);
  // Clicking inside the box must not fall through to the row's "open this chat".
  input.onclick = (event) => event.stopPropagation();
  input.ondblclick = (event) => event.stopPropagation();
  return input;
}

/* Format an estimated-cost float (USD) for display; under a cent still shows
   as non-zero so cheap runs don't all read as "$0.00". */
export function formatCost(amount) {
  const value = Number(amount) || 0;
  if (value === 0) return "$0.00";
  if (value < 0.01) return `$${value.toFixed(4)}`;
  return `$${value.toFixed(2)}`;
}

export function relativeTime(iso) {
  const seconds = Math.floor((Date.now() - new Date(iso).getTime()) / 1000);
  if (seconds < 60) return "just now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ago`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h ago`;
  return `${Math.floor(seconds / 86400)}d ago`;
}

/* Build an element in one call: `node("div", {class: "x"}, "text", childNode)`. */
export function node(tag, attributes = {}, ...children) {
  const element = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    if (key === "class") element.className = value;
    else if (key === "text") element.textContent = value;
    else if (key === "html") element.innerHTML = value;
    else if (key.startsWith("on")) element[key.toLowerCase()] = value;
    else if (value === true) element.setAttribute(key, "");
    else element.setAttribute(key, value);
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    element.append(child);
  }
  return element;
}

/* Collapse long cell text to a clickable one-liner; the full text stays in the
   DOM so nothing is lost when the user expands it. */
export function truncated(text, limit = 140) {
  const full = String(text || "");
  if (full.length <= limit) return node("span", { class: "cell-text", text: full });
  const span = node("span", { class: "cell-text clipped", text: `${full.slice(0, limit)}…`, title: "Click to expand" });
  span.onclick = () => {
    const open = span.classList.toggle("clipped");
    span.textContent = open ? `${full.slice(0, limit)}…` : full;
  };
  return span;
}
