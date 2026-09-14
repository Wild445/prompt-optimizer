# NIQ Prompt Creator — Prompt Synthesis Pipeline

Six prompt-synthesis capabilities built on Azure OpenAI, using `runtime.py`'s
`.prompty` + `complete_prompt()` pattern so LLM calling (auth headers, model
resolution, logging, token/latency capture) stays centralized in one place.

## Layout

```
prompts/                         # versioned .prompty files — edit these, not Python, to change behavior
  01_prompt_builder.prompty
  02_clarify.prompty
  03_workflow_planner.prompty
  04_prompt_optimizer.prompty
  05_model_specific_prompting.prompty
  06_ai_humanizer.prompty
synthesis/
  service.py                     # one function per stage — call independently
  orchestrator.py                # run_pipeline() — chains all stages (batch/CLI)
webapp/
  app.py                         # FastAPI: conversation API + serves the UI
  runner.py                      # the same stages as a resumable state machine (chat)
  db.py                          # SQLite: conversations, messages, per-stage telemetry
  static/                        # dependency-free frontend (no build step)
chat_common/                     # your provided llm_service.py / logging.py, packaged to match runtime.py's imports
runtime.py                       # unchanged — the calling layer
run_ui.py                        # `python run_ui.py` — starts the local chat UI

data/conversations.db            # created on first run (gitignored)
outputs/<conversation_id>/       # saved prompts per conversation (gitignored)
```

## Environment variables (from `chat_common/services/llm_service.py`)

| Variable | Purpose |
|---|---|
| `cis_llm_endpoint` | Azure OpenAI endpoint |
| `cis_llm_apikey` | Azure OpenAI API key |
| `cis_llm_apiversion` | Azure OpenAI API version |
| `cis_chat_rms_consumer_id` | Value for the `X-NIQ-CIS-Consumer` header |
| `CIS_LLM_4_DOT_1_DEPLOYMENT` | Deployment name used as `model.id` in every `.prompty` file |

`pip install -r requirements.txt` first — `prompty`, `openai`, `langchain-openai`,
and `python-json-logger` aren't installed in this sandbox.

## Usage — chat UI

```bash
pip install -r requirements.txt
python run_ui.py          # http://127.0.0.1:8000
```

Needs Python 3.10+ (`chat_common/common/logging.py` uses `str | None` at runtime).

- **Left sidebar** lists every conversation, newest first; click one to switch
  back to it with its full history. **+ New chat** mints a new `conversation_id`
  and carries your current settings (target model, notes, humanizer) forward.
- **Clarification is multiple choice.** Stage 2 returns 2-5 concrete options per
  question; the UI renders radios (or checkboxes when the question is
  `multi_select`) plus an **Other** free-text box on every question. *Skip — use
  your best judgment* jumps straight to the final stages. Answers are folded back
  into the brief and the round repeats until Clarify reports `ready` or
  `max_clarify_rounds` is hit.
- **Intermediate stages** appear as collapsed cards you can expand; the final
  prompt is expanded by default with Copy and Download buttons.
- **Every finished run is saved automatically** to
  `outputs/<conversation_id>/`:

  | File | Contents |
  |---|---|
  | `final_prompt_<UTC timestamp>.md` | the prompt plus its clarified brief and plan — one per run, so reruns accumulate |
  | `latest.md` | just the newest final prompt, at a stable path for tooling |
  | `conversation.json` | the full transcript and token/latency totals |

- A stage that throws (a 400 on an unsupported parameter, say) is written into
  the transcript as an error card and the conversation is left resumable rather
  than wedged mid-run.

### API

`GET/POST /api/conversations`, `GET/PATCH/DELETE /api/conversations/{id}`,
`POST /api/conversations/{id}/messages`, `/answers`, `/skip`, and
`GET /api/conversations/{id}/output`. Interactive docs at `/api/docs`.

## Usage — independent steps (menu-style)

```python
from synthesis import service

built = await service.build_prompt("I want something that writes product descriptions.")
print(built.content)

status = await service.clarify(built.content)
# {"status": "needs_clarification",
#  "questions": [{"question": "...", "options": ["...", "..."], "multi_select": false}],
#  "clarified_summary": "..."}

plan = await service.plan_workflow(built.content)
optimized = await service.optimize_prompt(built.content)
tuned = await service.tune_for_model(
    optimized.content,
    target_model="Azure OpenAI GPT-5.4",
    model_notes=open("docs/gpt-5.4-prompting-guide.md").read(),  # see "Target-model notes"
)
humanized = await service.humanize(tuned.content)
```

## Usage — chained pipeline

```python
from synthesis.orchestrator import run_pipeline

async def ask_user(questions: list[str]) -> str:
    # wire this to your actual UI/chat turn; shown here as a stand-in
    for q in questions:
        print(q)
    return input("> ")

result = await run_pipeline(
    "I want something that writes product descriptions for retail clients.",
    target_model="Azure OpenAI GPT-5.4",
    model_notes=gpt54_prompting_guide,   # optional but strongly recommended
    humanize_output=True,
    answer_fn=ask_user,       # omit to skip clarification (max_clarify_rounds=0 does the same)
)
print(result.model_tuned_prompt)
print(result.trace)  # e.g. ["build_prompt", "clarify x1", "plan_workflow", "optimize_prompt", "tune_for_model", "humanize"]
```

## Target-model notes (stage 5)

Stage 5 adapts a prompt to one destination model. The deployment doing the
adapting cannot know the conventions of a model released after its own training
cutoff, so the prompt is written to stay generic rather than invent quirks when
it has nothing to go on. Pass the destination model's prompting guide as
`model_notes` and it is treated as ground truth; omit it and you get safe,
family-agnostic structuring only.

## Model-parameter conventions

The `.prompty` files target GPT-5-family reasoning deployments:

- **No `temperature` / `top_p`.** Reasoning deployments accept only the default
  values and 400 on anything else. Where a stage previously relied on sampling
  (the humanizer's `temperature: 0.6` for sentence variety, Clarify's `0.2` for
  determinism), that behavior is now specified in the instructions instead.
- **`max_output_tokens` raised.** The budget is shared with reasoning tokens, so
  the old 800-1500 caps risked burning the whole allowance on reasoning and
  returning empty content. Current values are ~2.5x the visible-output target.
- **`reasoning_effort` / `verbosity`** are present but commented out in each
  file: `runtime._COMPLETION_PARAM_KEYS` filters them out today (see below).

### Known runtime gap

`runtime._COMPLETION_PARAM_KEYS` allows `max_tokens`, but
`_parameters_from_prompt` emits `max_completion_tokens` (the only form the
reasoning models accept). Every `max_output_tokens` value in these files is
therefore dropped before the API call, and no output cap is applied at all.
One-line fix, left unapplied because `runtime.py` is meant to stay untouched:

```python
_COMPLETION_PARAM_KEYS = (
    "temperature", "max_tokens", "max_completion_tokens", "response_format",
    "top_p", "timeout", "reasoning_effort", "verbosity",
)
```

### Untrusted text never reaches the template

Every stage's payload is passed as `extra_messages`, not as a templated
`.prompty` input, so the `.prompty` files are system-only. This is what
`runtime.merge_history`'s docstring prescribes: PromptyChatParser splits the
rendered template on role markers, so a pasted prompt containing a line reading
`system:` — routine input for a prompt-authoring tool — would otherwise inject a
new role block. Only short trusted knobs (`audience`, `max_questions`,
`target_model`, `model_notes`) are templated.

## Adding a 7th stage later ("Smart Handoff")

Not built here per your instructions. To add it later: drop a new
`prompts/07_smart_handoff.prompty`, add one function to `synthesis/service.py`
calling `complete_prompt()` against it, and wire it into
`synthesis/orchestrator.py` if it should chain automatically — no changes to
`runtime.py` or `chat_common` needed.
