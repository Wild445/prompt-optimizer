"""Process-wide environment loading for the prompt-synthesis pipeline.

Import this module (or call :func:`load_env`) before anything that reads
``os.getenv`` or calls ``prompty.load``/``prompty.load_async``: Prompty
resolves ``${env:VAR}`` frontmatter while loading the file and raises
``PromptyLoadError: Environment variable 'X' not set for key 'id'`` when the
variable is missing from ``os.environ``.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent
ENV_PATH = PROJECT_ROOT / ".env"

#: Variables consumed by ``chat_common/services/llm_service.py`` and the
#: ``model`` frontmatter of every ``.prompty`` file.
REQUIRED_ENV_VARS: tuple[str, ...] = (
    "cis_llm_endpoint",
    "cis_llm_apikey",
    "cis_llm_apiversion",
    "CIS_LLM_4_DOT_1_DEPLOYMENT",
)

_loaded = False


def load_env(*, override: bool = False) -> bool:
    """Load ``<project root>/.env`` into ``os.environ`` exactly once.

    Args:
        override: When ``True``, values in ``.env`` win over variables that are
            already exported in the shell. Defaults to ``False`` so CI/prod
            environments keep control.

    Returns:
        ``True`` when a ``.env`` file was found and parsed.
    """
    global _loaded
    if _loaded:
        return ENV_PATH.is_file()
    _loaded = True
    return load_dotenv(ENV_PATH, override=override)


def missing_env_vars() -> list[str]:
    """Return the required variables that are unset or empty after loading."""
    load_env()
    return [name for name in REQUIRED_ENV_VARS if not os.getenv(name)]


# Loading on import keeps `import config` enough for scripts and tests.
load_env()