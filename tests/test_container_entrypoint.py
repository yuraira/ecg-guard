from pathlib import Path

import pytest

from ecg_guard.web import container_entrypoint
from ecg_guard.web.container_entrypoint import (
    DEFAULT_COMMAND,
    normalize_command,
    prepare_checkpoint,
)


def test_entrypoint_uses_web_default_when_no_command_is_supplied() -> None:
    assert normalize_command([]) == DEFAULT_COMMAND


def test_entrypoint_passes_normal_exec_command_through() -> None:
    command = ["ecg-guard-web", "--server.port=8501"]

    assert normalize_command(command) == command


def test_entrypoint_removes_one_legacy_shell_quote_pair() -> None:
    script = "ecg-guard-fetch-checkpoint --output /tmp/model.pt && exec ecg-guard-web"

    assert normalize_command(["/bin/sh", "-c", f"'{script}'"]) == ["/bin/sh", "-c", script]


def test_entrypoint_normalizes_a_single_legacy_command_string() -> None:
    script = "ecg-guard-fetch-checkpoint --output /tmp/model.pt && exec ecg-guard-web"

    assert normalize_command([f"/bin/sh -c '{script}'"]) == ["/bin/sh", "-c", script]


def test_entrypoint_preserves_unbalanced_shell_text() -> None:
    script = "'printf incomplete"

    assert normalize_command(["/bin/sh", "-c", script]) == ["/bin/sh", "-c", script]


def test_entrypoint_wraps_an_unprefixed_shell_script() -> None:
    script = "ecg-guard-fetch-checkpoint && exec ecg-guard-web"

    assert normalize_command([script]) == ["/bin/sh", "-c", script]


def test_entrypoint_reassembles_a_tokenized_shell_script() -> None:
    command = [
        "ecg-guard-fetch-checkpoint",
        "--output",
        "/tmp/model.pt",
        "&&",
        "exec",
        "ecg-guard-web",
        "--server.port=${PORT:-10000}",
    ]

    assert normalize_command(command) == ["/bin/sh", "-c", " ".join(command)]


def test_entrypoint_does_not_fetch_checkpoint_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_if_called(_destination: Path) -> Path:
        raise AssertionError("checkpoint fetch should be opt-in")

    monkeypatch.setattr(container_entrypoint, "download_checkpoint", fail_if_called)

    assert prepare_checkpoint({}) is None


@pytest.mark.parametrize("enabled", ["1", "true", "TRUE", "yes", "on"])
def test_entrypoint_fetches_configured_checkpoint(
    enabled: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "models" / "best_model.pt"
    requested: list[Path] = []

    def fake_download(path: Path) -> Path:
        requested.append(path)
        return path.resolve()

    monkeypatch.setattr(container_entrypoint, "download_checkpoint", fake_download)

    result = prepare_checkpoint(
        {
            "ECG_GUARD_FETCH_CHECKPOINT_AT_STARTUP": enabled,
            "ECG_GUARD_CHECKPOINT": str(destination),
        }
    )

    assert result == destination.resolve()
    assert requested == [destination]


def test_entrypoint_requires_checkpoint_path_when_fetching() -> None:
    with pytest.raises(RuntimeError, match="ECG_GUARD_CHECKPOINT is required"):
        prepare_checkpoint({"ECG_GUARD_FETCH_CHECKPOINT_AT_STARTUP": "1"})
