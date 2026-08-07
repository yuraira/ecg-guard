"""Normalize container commands before replacing the current process."""

from __future__ import annotations

import os
import sys
from collections.abc import Sequence


DEFAULT_COMMAND = [
    "ecg-guard-web",
    "--server.address=0.0.0.0",
    "--server.port=8501",
    "--server.headless=true",
]


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

    return command


def main() -> None:
    command = normalize_command(sys.argv[1:])
    os.execvp(command[0], command)


if __name__ == "__main__":
    main()
