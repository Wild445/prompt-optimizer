# NIQ Prompt Creator — Prompt Synthesis & Optimization

Two workspaces over Azure OpenAI, both using `runtime.py`'s `.prompty` +
`complete_prompt()` pattern so LLM calling (auth headers, model resolution,
logging, token/latency capture) stays centralized in one place:

- **Prompt Creation** — six synthesis stages that turn a rough idea into a
  finished prompt.
- **Prompt Optimization** — build an LLM-as-a-judge from your own observations
  about how a prompt fails, grade a fixed set of test cases with it, and loop
  the prompt until they pass.

The UI's left sidebar holds one section per workspace. Each section head is a
button: clicking it opens that workflow's full view on the right, and only the
open workspace shows its history. Inside the open section the step list folds
away on its own — collapsing it hands the space to the history below, which is
the point, so the history never collapses with it. Each section remembers
whether its step list is folded.

## Layout

```
prompts/                         # versioned .prompty files — edit these, not Python, to change behavior
  01_prompt_builder.prompty      # 01-06: the synthesis pipeline
  02_clarify.prompty
  03_workflow_planner.prompty
  04_prompt_optimizer.prompty
  05_model_specific_prompting.prompty
  06_ai_humanizer.prompty
  10_judge_scaffold.prompty      # 10-15: the optimization loop
  11_criteria_drafter.prompty
  12_criteria_consolidator.prompty
  13_judge_builder.prompty
  14_failure_analyst.prompty
  15_prompt_revisor.prompty
synthesis/
  service.py                     # one function per stage — call independently
  orchestrator.py                # run_pipeline() — chains all stages (batch/CLI)
optimization/
  service.py                     # one function per optimization step, plus the two ad-hoc calls
  dataset.py                     # .xlsx/.csv test cases + Jinja/LangChain template rendering
webapp/
  app.py                         # FastAPI: both APIs + serves the UI
  runner.py                      # the synthesis stages as a resumable state machine (chat)
  optimizer.py                   # the optimization loop as a resumable state machine
  db.py                          # SQLite: conversations, messages, per-stage telemetry
  opt_db.py                      # SQLite: optimizations, criteria, test cases, graded results
  static/                        # dependency-free ES modules (no build step)
    common.js                    #   helpers shared by both workspaces
    creator.js / optimizer.js    #   one module per workspace
    main.js                      #   the shell that switches between them
chat_common/                     # your provided llm_service.py / logging.py, packaged to match runtime.py's imports
runtime.py                       # the calling layer (complete_prompt / stream_prompt / complete_messages)
run_ui.py                        # `python run_ui.py` — starts the local UI

data/conversations.db            # created on first run (gitignored)
outputs/<conversation_id>/       # saved prompts per conversation (gitignored)
outputs/optimizations/<id>/      # prompt versions, judge, criteria, graded runs (gitignored)
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

## Usage — prompt optimization

Open **Prompt Optimization** in the sidebar. The loop, in order:

1. **Paste the prompt** you want to optimize and pick its template dialect
   (Jinja2 `{{ var }}` or LangChain `{var}`). An agent drafts the skeleton of an
   LLM-as-a-judge around it.
2. **Say how it fails**, one problem per line. Another agent turns those
   observations into success criteria a judge can score `true`/`false`.
3. **Review the criteria** — edit, delete, or add your own. A consolidator merges
   both lists, stamps them `C1..Cn`, and a builder folds them into the judge
   prompt.
4. **Upload the test cases** as `.xlsx` or `.csv` (there is a *Download the
   template* button):

   | Column | Contents |
   |---|---|
   | `message_id` | a stable id per test case — keep it the same across iterations so runs can be compared |
   | `input_payload` | the input the prompt is run against |
   | `other_input_params` | the template variables, as a JSON object or `key=value` lines. Blank if the prompt has none. |

   Keeping the input fixed is the point: the only thing that changes between
   iterations is the prompt.
5. **Run.** Every case is rendered, answered by the current prompt, and scored by
   the judge. The right-hand panel counts up live — `Success (X/N)` with a green
   tick, `Failed (Y/N)` with a red cross.
6. **Review the table** in the UI. Each row shows the input, the response, the
   verdict, which criteria failed, why, and which part of the prompt the judge
   blames. Every row counts as agreed by default; press **Flag as wrong** only on
   the verdicts you disagree with. A flagged row opens a reason box and
   **Continue** stays disabled until every flagged row has one — so a
   disagreement can never be recorded without the reason that makes it useful.
7. **Analyze and revise.** An analyst reads the run plus your remarks into a list
   of changes, each tied to the criteria it fixes and the part of the prompt
   responsible. A remark no criterion covers becomes a *new* success criterion,
   added to the judge before the next run. A revisor applies the changes and
   produces the next prompt version.
8. **Run again** against the same test cases, or stop.

Everything lands in `outputs/optimizations/<id>/`:

| File | Contents |
|---|---|
| `prompt_v<n>.md` | every version of the prompt under test |
| `judge_scaffold.md` / `judge_prompt.md` | the judge skeleton and the finished judge |
| `criteria.json` | the active success criteria, with their ids |
| `iteration_<n>_results.csv` | message_id, input, params, response, verdict, failed criteria, reason, and your agreement |
| `session.json` | the whole session, including every iteration |

Test cases run `PROMPT_OPTIMIZER_CONCURRENCY` at a time (default 4). A case that
throws is recorded with `status=error` and counted separately rather than
abandoning the run.

### API

`GET/POST /api/optimizations`, `GET/PATCH/DELETE /api/optimizations/{id}`,
`POST /api/optimizations/{id}/prompt|observations|criteria|run|review` (each
`/stream`), `POST /api/optimizations/{id}/dataset` (multipart),
`POST /api/optimizations/{id}/stop`, `GET /api/optimizations/{id}/results.csv`
and `/prompt`, plus `GET /api/optimizer/steps` and
`/api/optimizer/dataset-template`.

### Trying it without an API key

```bash
PROMPT_CREATOR_FAKE_LLM=1 python run_ui.py     # both workspaces, canned responses
python scripts/test_optimizer_offline.py       # the whole loop, headless
python scripts/test_pipeline_offline.py        # the six synthesis stages, headless
```

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
