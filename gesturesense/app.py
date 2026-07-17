#!/usr/bin/env python3
"""GestureSense entry point.

Parses command-line arguments, loads configuration, sets up logging and
launches the application. Kept intentionally thin — all real behaviour
lives inside the ``gesturesense`` package.

Usage::

    python app.py                          # defaults + default_config.yaml
    python app.py --config my_config.yaml  # custom configuration
    python app.py --camera 1 --debug      # quick CLI overrides
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import sys
from pathlib import Path

from gesturesense.config.settings import AppConfig, load_config
from gesturesense.utils.logging_setup import setup_logging

logger = logging.getLogger("gesturesense.app")

DEFAULT_CONFIG = Path(__file__).parent / "gesturesense" / "config" / "default_config.yaml"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Define and parse the command-line interface."""
    parser = argparse.ArgumentParser(
        prog="gesturesense",
        description="Real-time hand gesture recognition (MediaPipe + OpenCV).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help=f"Path to a YAML configuration file (default: {DEFAULT_CONFIG})",
    )
    parser.add_argument(
        "--camera",
        type=int,
        default=None,
        metavar="INDEX",
        help="Override the camera device index from the config file.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Start with the debug overlay enabled.",
    )
    parser.add_argument(
        "--fullscreen",
        action="store_true",
        help="Start in fullscreen mode.",
    )
    return parser.parse_args(argv)


def apply_cli_overrides(config: AppConfig, args: argparse.Namespace) -> AppConfig:
    """Apply CLI flags on top of the loaded configuration."""
    if args.camera is not None:
        config = dataclasses.replace(
            config, camera=dataclasses.replace(config.camera, index=args.camera)
        )
    if args.debug:
        config = dataclasses.replace(
            config, debug=dataclasses.replace(config.debug, enabled=True)
        )
    if args.fullscreen:
        config = dataclasses.replace(
            config, ui=dataclasses.replace(config.ui, fullscreen=True)
        )
    return config


def main(argv: list[str] | None = None) -> int:
    """Program entry point; returns a process exit code."""
    args = parse_args(argv)

    try:
        config = load_config(args.config)
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2
    config = apply_cli_overrides(config, args)

    setup_logging(config.logging)
    logger.info("Configuration loaded from %s", args.config)

    # Imported late so that a missing native dependency produces a friendly
    # message instead of a bare traceback before logging even exists.
    try:
        from gesturesense.core.application import Application
    except ImportError as exc:
        logger.error("Missing dependency: %s", exc)
        print(
            "\nGestureSense could not start because a dependency is missing:\n"
            f"    {exc}\n\n"
            "Install the runtime requirements first:\n"
            "    pip install -r requirements.txt\n",
            file=sys.stderr,
        )
        return 1

    return Application(config).run()


if __name__ == "__main__":
    raise SystemExit(main())
