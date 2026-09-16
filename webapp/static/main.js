/* Shell for the two workspaces.

   The sidebar is an accordion: exactly one section is open, and the open section
   is the workspace shown on the right. Each workspace loads itself the first time
   it is opened, so starting in one costs nothing in the other. */

import * as creator from "./creator.js";
import * as optimizer from "./optimizer.js";

const WORKSPACES = {
  creator: { init: creator.init, title: "Prompt Creation" },
  optimizer: { init: optimizer.init, title: "Prompt Optimization" },
};

const STORAGE_KEY = "promptCreator.workspace";

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
      section.classList.toggle("expanded", open);
      section.querySelector(".section-head").setAttribute("aria-expanded", String(open));
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

function wire() {
  document.querySelectorAll(".workspace-section").forEach((section) => {
    const name = section.dataset.section;
    section.querySelector(".section-head").onclick = () => activate(name);
  });
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
