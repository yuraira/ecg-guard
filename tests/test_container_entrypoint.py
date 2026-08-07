from ecg_guard.web.container_entrypoint import DEFAULT_COMMAND, normalize_command


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
