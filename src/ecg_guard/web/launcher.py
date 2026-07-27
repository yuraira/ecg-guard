"""Console launcher for the packaged Streamlit application."""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_checkpoint(checkpoint_path: Path, protocol_path: Path) -> str:
    """Verify the mounted checkpoint before starting the public web process."""
    if not checkpoint_path.is_file():
        raise RuntimeError("the configured ECG Guard checkpoint is missing")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    expected = str(protocol.get("checkpoint_sha256", "")).lower()
    if len(expected) != 64:
        raise RuntimeError("the inference protocol has no valid checkpoint hash")
    actual = sha256_file(checkpoint_path)
    if actual != expected:
        raise RuntimeError(
            "the configured checkpoint does not match the inference protocol"
        )
    return actual


def verify_container_checkpoint() -> str | None:
    """Apply the opt-in startup gate used by the Docker deployment."""
    enabled = os.environ.get(
        "ECG_GUARD_VERIFY_CHECKPOINT_AT_STARTUP",
        "",
    ).strip().lower()
    if enabled not in {"1", "true", "yes"}:
        return None
    checkpoint_path = Path(
        os.environ.get(
            "ECG_GUARD_CHECKPOINT",
            "/models/best_model.pt",
        )
    )
    protocol_path = (
        Path(__file__).resolve().parents[1]
        / "resources"
        / "baseline_v1_inference.json"
    )
    return verify_checkpoint(checkpoint_path, protocol_path)


def main() -> None:
    """Run the bundled app while forwarding Streamlit command-line options."""
    verify_container_checkpoint()
    app_path = Path(__file__).with_name("app.py")
    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--server.maxUploadSize=50",
        "--browser.gatherUsageStats=false",
        "--theme.primaryColor=#0F766E",
        "--theme.backgroundColor=#F7F9FC",
        "--theme.secondaryBackgroundColor=#EAF2F2",
        "--theme.textColor=#16232A",
        *sys.argv[1:],
    ]
    raise SystemExit(streamlit_cli.main())


if __name__ == "__main__":
    main()
