"""Test isolation for the eval lane, mirroring api/tests/conftest.py.

The live gate-replica test runs the API in-process. Without this it would read the developer's .env
(real LLM calls, free quota spent) and write `cache.sqlite` into the working directory, which the API
reloads at boot, so the next local measurement would start warm. Both are set before any test imports
the app: `ONECLICK_SQLITE` is read when `app.config` is imported, and an empty key counts as "no key"
(the router's `load_dotenv(override=False)` leaves it empty), so the engine runs rules-only.
"""

import os
import sys
import tempfile
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parents[1]
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

os.environ["ONECLICK_SQLITE"] = str(Path(tempfile.mkdtemp(prefix="oneclick-eval-tests-")) / "cache.sqlite")
os.environ["GEMINI_API_KEY"] = ""
os.environ["MISTRAL_API_KEY"] = ""
