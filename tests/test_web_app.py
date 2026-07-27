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
    assert app.checkbox[0].label.startswith("업로드 파일이 공개되었거나")
    assert app.button[0].label == "업로드 ECG 분석"
    assert app.button[0].disabled
