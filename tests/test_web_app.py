from __future__ import annotations

from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_web_app_renders_initial_state() -> None:
    source = Path("src/ecg_guard/web/app.py").read_text(encoding="utf-8")
    app = AppTest.from_string(source)
    app.run(timeout=15)

    assert not app.exception
    assert app.sidebar.title[0].value == "ECG Guard"
    assert len(app.warning) == 1
