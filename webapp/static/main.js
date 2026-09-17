/* Shell for the two workspaces.

   The sidebar heads are switches, not disclosures: exactly one workspace is open
   at a time, and its section is the only one showing a history - the other
   workspace's chats stay out of the way until it is the one being used.

   What collapses inside the open section is the step picker alone. Each section
   remembers its own picker, so folding the workflow away hands that room to the
   history list below it instead of hiding it too. */

import * as creator from "./creator.js";
import * as optimizer from "./optimizer.js";

const WORKSPACES = {
  creator: { init: creator.init, title: "Prompt Creation" },
  optimizer: { init: optimizer.init, title: "Prompt Optimization" },
};

const STORAGE_KEY = "promptCreator.workspace";
const PICKER_KEY = "promptCreator.pickerCollapsed";

let current = null;

function panels(name) {
  return document.querySelectorAll(`[data-workspace="${name}"]`);
}

export async function activate(name) {
  if (!WORKSPACES[name] || name === current) return;
  current = name;
  document.body.dataset.workspace = name;
  for (const key of Object.keys(WORKSPACES)) {
    const open = key === name;
    panels(key).forEach((panel) => panel.classList.toggle("active", open));
    const section = document.querySelector(`.workspace-section[data-section="${key}"]`);
    if (section) {
      section.classList.toggle("active", open);
      section.querySelector(".section-head").setAttribute("aria-pressed", String(open));
    }
  }
  try {
    // Remembering the last workspace is a convenience; a browser that refuses
    // storage (private windows, blocked site data) should still switch fine.
    localStorage.setItem(STORAGE_KEY, name);
  } catch (error) {
    /* not worth reporting */
  }
  await WORKSPACES[name].init();
}

/* ------------------------------------------------------------------ step pickers

   Collapsed state is per section and survives a reload: whoever folds the
   pipeline away to browse their chats wants it folded next time too. */

function readCollapsed() {
  try {
    return new Set(JSON.parse(localStorage.getItem(PICKER_KEY) || "[]"));
  } catch (error) {
    return new Set();
  }
}

function writeCollapsed(collapsed) {
  try {
    localStorage.setItem(PICKER_KEY, JSON.stringify([...collapsed]));
  } catch (error) {
    /* the picker still collapses for this visit */
  }
}

function wirePickers() {
  const collapsed = readCollapsed();
  document.querySelectorAll(".stage-picker[data-picker]").forEach((picker) => {
    const key = picker.dataset.picker;
    const head = picker.querySelector(".stage-picker-head");
    if (!head) return;

    const apply = (isCollapsed) => {
      picker.classList.toggle("collapsed", isCollapsed);
      head.setAttribute("aria-expanded", String(!isCollapsed));
    };
    apply(collapsed.has(key));

    head.onclick = () => {
      const isCollapsed = !picker.classList.contains("collapsed");
      apply(isCollapsed);
      if (isCollapsed) collapsed.add(key);
      else collapsed.delete(key);
      writeCollapsed(collapsed);
    };
  });
}

/* ------------------------------------------------------------------ startup */

function wire() {
  document.querySelectorAll(".workspace-section").forEach((section) => {
    const name = section.dataset.section;
    section.querySelector(".section-head").onclick = () => activate(name);
  });
  wirePickers();
}

(async function start() {
  wire();
  let remembered = "creator";
  try {
    remembered = localStorage.getItem(STORAGE_KEY) || "creator";
  } catch (error) {
    /* fall back to the default */
  }
  await activate(WORKSPACES[remembered] ? remembered : "creator");
})();
