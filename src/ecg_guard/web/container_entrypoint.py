"""Normalize container commands before replacing the current process."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Mapping

from ecg_guard.resources.fetch_checkpoint import download_checkpoint


DEFAULT_COMMAND = [
    "ecg-guard-web",
    "--server.address=0.0.0.0",
    "--server.port=8501",
    "--server.headless=true",
]
TRUE_VALUES = {"1", "true", "yes", "on"}
SHELL_OPERATORS = {"&&", "||", ";"}


def _strip_matching_outer_quotes(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in {"'", '"'}:
        return stripped[1:-1]
    return value


def normalize_command(arguments: Sequence[str]) -> list[str]:
    """Return a command compatible with normal and legacy Render overrides.

    Render already supplies a shell for a Docker command override. An older
    Blueprint wrapped that override in another quoted ``/bin/sh -c`` command,
    causing the complete script to be treated as one executable name. The
    image entrypoint removes only one matching quote pair from that shell
    script and otherwise passes the command through unchanged.
    """

    command = list(arguments)
    if not command:
        return DEFAULT_COMMAND.copy()

    if len(command) >= 3 and command[0] in {"/bin/sh", "sh"} and command[1] == "-c":
        command[2] = _strip_matching_outer_quotes(command[2])
        return command

    if len(command) == 1:
        single = command[0].strip()
        for shell_prefix in ("/bin/sh -c ", "sh -c "):
            if single.startswith(shell_prefix):
                return ["/bin/sh", "-c", _strip_matching_outer_quotes(single[len(shell_prefix) :])]
        if any(operator in single for operator in SHELL_OPERATORS):
            return ["/bin/sh", "-c", single]

    if any(argument in SHELL_OPERATORS for argument in command):
        return ["/bin/sh", "-c", " ".join(command)]

    return command


def prepare_checkpoint(environment: Mapping[str, str] = os.environ) -> Path | None:
    """Fetch the locked public checkpoint when ephemeral hosting requests it."""
    enabled = environment.get("ECG_GUARD_FETCH_CHECKPOINT_AT_STARTUP", "")
    if enabled.strip().lower() not in TRUE_VALUES:
        return None

    configured_path = environment.get("ECG_GUARD_CHECKPOINT", "").strip()
    if not configured_path:
        raise RuntimeError(
            "ECG_GUARD_CHECKPOINT is required when startup fetching is enabled"
        )

    destination = download_checkpoint(Path(configured_path))
    print(f"verified baseline checkpoint ready: {destination}", flush=True)
    return destination


def main() -> None:
    prepare_checkpoint()
    command = normalize_command(sys.argv[1:])
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
