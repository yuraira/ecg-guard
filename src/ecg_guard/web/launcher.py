"""Console launcher for the packaged Streamlit application."""

from __future__ import annotations

import sys
from pathlib import Path

from streamlit.web import cli as streamlit_cli


def main() -> None:
    """Run the bundled app while forwarding Streamlit command-line options."""
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
