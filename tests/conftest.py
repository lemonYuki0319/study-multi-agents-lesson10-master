"""
Shared pytest fixtures for lesson10 agent tests.

Використання:
    deepeval test run tests/
    deepeval test run tests/test_planner.py
"""

import io
import json
import sys
from pathlib import Path

import pytest

# Додаємо корінь проєкту до sys.path, щоб імпортувати agents, config тощо
sys.path.insert(0, str(Path(__file__).parent.parent))

# Windows cp1251 не підтримує emoji, які використовують агенти у print().
# Перемикаємо stdout/stderr на utf-8 для коректного виводу під час тестів.
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")


# ---------------------------------------------------------------------------
# Golden Dataset
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def golden_dataset() -> list[dict]:
    path = Path(__file__).parent / "golden_dataset.json"
    if not path.exists():
        pytest.skip(
            "golden_dataset.json не знайдено. Спочатку запустіть: python goldengen.py"
        )
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data


@pytest.fixture(scope="session")
def happy_path_examples(golden_dataset) -> list[dict]:
    return [ex for ex in golden_dataset if ex["category"] == "happy_path"]


@pytest.fixture(scope="session")
def edge_case_examples(golden_dataset) -> list[dict]:
    return [ex for ex in golden_dataset if ex["category"] == "edge_case"]


@pytest.fixture(scope="session")
def failure_case_examples(golden_dataset) -> list[dict]:
    return [ex for ex in golden_dataset if ex["category"] == "failure_case"]
