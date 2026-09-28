"""Formal MVP-10 Qt entry point for PC Torch and Atlas 310B."""

from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
from typing import Sequence

from src.config import load_config, parse_source
from src.logging_utils import configure_logging


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Person + Vehicle Qt tracking and Gallery application"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--source", help="optional camera index or local video path")
    return parser


def run(config, source_override: str | None = None) -> int:
    """Launch the single full-screen Qt application window."""

    from ui.qt.main_window import launch_qt

    return launch_qt(config, source_override)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_argument_parser().parse_args(argv)
    configure_logging()
    try:
        config = load_config(Path(args.config))
        if args.source is not None:
            config = replace(config, video=replace(config.video, source=parse_source(args.source)))
        configure_logging(config.runtime.log_level)
        return run(config)
    except KeyboardInterrupt:
        return 0
    except Exception:
        import logging

        logging.getLogger(__name__).exception("APPLICATION_ERROR")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
