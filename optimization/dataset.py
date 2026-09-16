"""Test-case ingestion and prompt-template rendering for the optimization loop.

The optimizer keeps the *input* fixed and varies the prompt, so the test data has
to come from somewhere repeatable rather than from a chat box. That is this
module: read a three-column workbook into rows, and render the prompt under test
against each row.

Expected columns (header row, case-insensitive, order irrelevant):

``message_id``
    Stable identifier for the test case. Used to line runs up against each other
    across iterations, so it must not change between uploads.
``input_payload``
    The input the prompt under test is run against — what a real user would send.
``other_input_params``
    Values for the template variables in the prompt (the ones a live system would
    fill from an API). JSON object, or ``key=value`` pairs separated by newlines
    or semicolons. Blank when the prompt has no variables.
"""

from __future__ import annotations

import csv
import io
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

#: Template dialects offered in the UI dropdown. ``jinja2`` reads ``{{ var }}``,
#: ``langchain`` reads the f-string ``{var}`` syntax LangChain's PromptTemplate
#: uses. Everything else about the two flows is identical.
TEMPLATE_KINDS: tuple[str, ...] = ("jinja2", "langchain")

REQUIRED_COLUMNS: tuple[str, ...] = ("message_id", "input_payload")
OPTIONAL_COLUMNS: tuple[str, ...] = ("other_input_params",)

#: What the UI tells the user to upload, kept here so the copy and the parser
#: can never drift apart.
COLUMN_HELP = (
    "message_id, input_payload, other_input_params"
    " — one row per test case. other_input_params holds the template variables"
    " as JSON (or key=value lines) and may be left blank."
)


class DatasetError(ValueError):
    """A spreadsheet the user needs to fix, phrased for the user rather than the log."""


class TemplateError(ValueError):
    """The prompt could not be rendered against a row's variables."""


@dataclass(frozen=True)
class TestCase:
    """One row of the uploaded workbook."""

    message_id: str
    input_payload: str
    other_input_params: dict[str, Any]


# --------------------------------------------------------------------------- parsing


def _clean(value: Any) -> str:
    """Render a cell as text, treating the empty-ish values openpyxl returns as blank."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        # openpyxl types every unformatted number as float, which would turn the
        # message_id 12 into "12.0" and break the join against the previous run.
        return str(int(value))
    return str(value).strip()


def parse_params(raw: str) -> dict[str, Any]:
    """Read ``other_input_params`` as JSON, falling back to ``key=value`` lines.

    Both forms show up in practice: a JSON blob when the values were exported
    from the calling system, hand-typed pairs when someone filled the sheet in
    by hand. Neither is worth rejecting.
    """
    text = (raw or "").strip()
    if not text:
        return {}
    if text.startswith("{"):
        try:
            loaded = json.loads(text)
        except json.JSONDecodeError as error:
            raise DatasetError(f"other_input_params is not valid JSON: {error}") from error
        if not isinstance(loaded, dict):
            raise DatasetError("other_input_params must be a JSON object, not a list or scalar")
        return loaded

    params: dict[str, Any] = {}
    for chunk in re.split(r"[\n;]+", text):
        if not chunk.strip():
            continue
        if "=" not in chunk:
            raise DatasetError(
                f"Cannot read {chunk.strip()!r} in other_input_params."
                " Use a JSON object, or key=value pairs separated by newlines or semicolons."
            )
        key, _, value = chunk.partition("=")
        params[key.strip()] = value.strip()
    return params


def _rows_from_xlsx(data: bytes) -> list[list[Any]]:
    try:
        from openpyxl import load_workbook
    except ImportError as error:  # pragma: no cover - dependency is declared in requirements.txt
        raise DatasetError(
            "Reading .xlsx needs openpyxl. Install it with `pip install openpyxl`, or upload a .csv instead."
        ) from error
    # read_only + data_only: we want the cached values of any formulas, and no
    # styling, which on a large export is most of the parse time.
    workbook = load_workbook(io.BytesIO(data), read_only=True, data_only=True)
    try:
        return [list(row) for row in workbook.worksheets[0].iter_rows(values_only=True)]
    finally:
        workbook.close()


def _rows_from_csv(data: bytes) -> list[list[Any]]:
    # utf-8-sig: Excel's "CSV UTF-8" export writes a BOM, which would otherwise
    # become part of the first header name and hide the message_id column.
    text = data.decode("utf-8-sig", errors="replace")
    return [list(row) for row in csv.reader(io.StringIO(text))]


def parse_dataset(data: bytes, filename: str = "") -> list[TestCase]:
    """Read an uploaded workbook into test cases, or raise :class:`DatasetError`.

    Args:
        data: Raw file bytes as uploaded.
        filename: Used only to pick the reader; ``.csv`` and ``.tsv`` are read as
            delimited text, anything else as ``.xlsx``.

    Raises:
        DatasetError: Missing columns, no rows, or duplicate ``message_id``s —
            each phrased as something the user can go and fix in the sheet.
    """
    suffix = Path(filename or "").suffix.lower()
    rows = _rows_from_csv(data) if suffix in (".csv", ".tsv", ".txt") else _rows_from_xlsx(data)
    rows = [row for row in rows if any(_clean(cell) for cell in row)]
    if not rows:
        raise DatasetError("The file is empty.")

    header = [_clean(cell).lower().replace(" ", "_") for cell in rows[0]]
    missing = [column for column in REQUIRED_COLUMNS if column not in header]
    if missing:
        raise DatasetError(
            f"Missing column(s): {', '.join(missing)}. The first row must be a header of {COLUMN_HELP}"
        )
    index = {name: position for position, name in enumerate(header)}

    def cell(row: list[Any], column: str) -> str:
        position = index.get(column)
        return _clean(row[position]) if position is not None and position < len(row) else ""

    cases: list[TestCase] = []
    seen: set[str] = set()
    for number, row in enumerate(rows[1:], start=2):
        message_id = cell(row, "message_id")
        payload = cell(row, "input_payload")
        if not message_id and not payload:
            continue
        if not message_id:
            raise DatasetError(f"Row {number} has no message_id.")
        if not payload:
            raise DatasetError(f"Row {number} ({message_id}) has no input_payload.")
        if message_id in seen:
            raise DatasetError(
                f"Row {number} repeats message_id {message_id!r}."
                " Ids must be unique so runs can be compared across iterations."
            )
        seen.add(message_id)
        try:
            params = parse_params(cell(row, "other_input_params"))
        except DatasetError as error:
            raise DatasetError(f"Row {number} ({message_id}): {error}") from error
        cases.append(TestCase(message_id=message_id, input_payload=payload, other_input_params=params))

    if not cases:
        raise DatasetError("The file has a header but no test cases under it.")
    return cases


# --------------------------------------------------------------------------- rendering


def template_variables(prompt: str, template_kind: str) -> list[str]:
    """Names of the variables the prompt expects, so the UI can report them up front.

    Best-effort and regex-based rather than a real parse: it drives a hint in the
    dataset step, and a wrong guess there costs nothing — rendering is what
    actually validates.
    """
    if template_kind == "langchain":
        # Single braces only; {{escaped}} is a literal brace in f-string templates.
        found = re.findall(r"(?<!\{)\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}(?!\})", prompt or "")
    else:
        found = re.findall(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*[|}]", prompt or "")
    ordered: list[str] = []
    for name in found:
        if name not in ordered:
            ordered.append(name)
    return ordered


def render_prompt(prompt: str, template_kind: str, params: Optional[dict[str, Any]] = None) -> str:
    """Fill the prompt's template variables from one row's ``other_input_params``.

    Raises:
        TemplateError: The template is malformed, or a variable it needs has no
            value in this row. Both are user-fixable and are surfaced against the
            individual test case rather than failing the whole run.
    """
    values = params or {}
    if template_kind == "langchain":
        try:
            from langchain_core.prompts import PromptTemplate
        except ImportError as error:  # pragma: no cover - langchain-openai pulls this in
            raise TemplateError("LangChain templates need langchain-core installed.") from error
        try:
            template = PromptTemplate.from_template(prompt)
            return template.format(**{name: values.get(name, "") for name in template.input_variables})
        except Exception as error:  # noqa: BLE001 - LangChain raises several unrelated types here
            raise TemplateError(f"LangChain template error: {error}") from error

    try:
        from jinja2 import StrictUndefined, Template
    except ImportError as error:  # pragma: no cover - prompty depends on jinja2
        raise TemplateError("Jinja templates need jinja2 installed.") from error
    try:
        # StrictUndefined: a typo'd variable name should be reported, not silently
        # rendered as an empty string in every single test case.
        return Template(prompt, undefined=StrictUndefined).render(**values)
    except Exception as error:  # noqa: BLE001 - jinja2 raises several unrelated types here
        raise TemplateError(f"Jinja template error: {error}") from error


# --------------------------------------------------------------------------- sample file


SAMPLE_ROWS: tuple[tuple[str, str, str], ...] = (
    (
        "msg-001",
        "Summarise the Q3 results for the beverages category.",
        '{"region": "APAC", "reporting_period": "Q3 2026"}',
    ),
    (
        "msg-002",
        "Which three SKUs lost the most share last month?",
        '{"region": "EMEA", "reporting_period": "Q3 2026"}',
    ),
)


def sample_workbook() -> bytes:
    """Build the downloadable starter file, as .xlsx when openpyxl is available.

    Returns CSV bytes as a fallback so the "download the template" button still
    works in an install without openpyxl; :func:`parse_dataset` accepts either.
    """
    try:
        from openpyxl import Workbook
    except ImportError:
        buffer = io.StringIO()
        writer = csv.writer(buffer)
        writer.writerow([*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS])
        writer.writerows(SAMPLE_ROWS)
        return buffer.getvalue().encode("utf-8")

    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "test_cases"
    sheet.append([*REQUIRED_COLUMNS, *OPTIONAL_COLUMNS])
    for row in SAMPLE_ROWS:
        sheet.append(list(row))
    for column, width in zip("ABC", (18, 62, 46)):
        sheet.column_dimensions[column].width = width
    buffer = io.BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def sample_filename() -> str:
    """Name for the starter file, matching whichever format :func:`sample_workbook` produced."""
    try:
        import openpyxl  # noqa: F401
    except ImportError:
        return "optimizer_test_cases_template.csv"
    return "optimizer_test_cases_template.xlsx"


def to_csv(rows: Iterable[dict[str, Any]], columns: list[str]) -> str:
    """Flatten result rows into CSV text for the run's output folder."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns, extrasaction="ignore")
    writer.writeheader()
    for row in rows:
        writer.writerow({column: row.get(column, "") for column in columns})
    return buffer.getvalue()
