"""Command-line entry points for first-run setup and the packaged Studio."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from . import __version__
from .config import AIsketcherConfig, load_config, save_config
from .errors import AIsketcherError


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aisketcher",
        description="Configure AIsketcher or launch its local-first Studio.",
    )
    parser.add_argument("--version", action="version", version=f"AIsketcher {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="write a versioned YAML settings ledger")
    init.add_argument(
        "--path",
        type=Path,
        help="write this file instead of the platform user config path",
    )
    init.add_argument("--force", action="store_true", help="replace an existing settings file")
    init.add_argument(
        "--preset",
        default="flux2-klein-edit@1",
        choices=(
            "flux2-klein-edit@1",
            "sdxl-canny-lite@1",
            "sdxl-canny@1",
        ),
    )
    init.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    init.add_argument("--outputs", type=int, default=4, choices=(1, 4, 8))
    init.add_argument(
        "--seed-mode", default="scout", choices=("scout", "locked", "explicit")
    )
    init.add_argument("--seed", type=int, help="optional non-negative 63-bit starting seed")
    init.add_argument("--language", default="en", choices=("en", "ko"))
    init.add_argument("--cache-dir", help="optional local model cache directory")
    init.add_argument(
        "--offline",
        action="store_true",
        help="disable model downloads while keeping Guided Sample available",
    )
    init.set_defaults(handler=_run_init)

    studio = commands.add_parser("studio", help="launch the packaged Gradio Studio locally")
    studio.add_argument("--config", type=Path, help="use an explicit YAML settings file")
    studio.add_argument("--language", choices=("en", "ko"), help="override the UI language")
    studio.add_argument("--port", type=int, help="bind a specific localhost port")
    studio.set_defaults(handler=_run_studio)
    return parser


def _run_init(args: argparse.Namespace) -> int:
    config = AIsketcherConfig(
        preset=args.preset,
        device=args.device,
        output_count=args.outputs,
        seed_mode=args.seed_mode,
        seed=args.seed,
        language=args.language,
        cache_dir=args.cache_dir,
        allow_downloads=not args.offline,
    )
    destination = save_config(config, args.path, overwrite=args.force)
    print(f"Wrote AIsketcher settings: {destination}")
    print("Next: aisketcher studio")
    return 0


def _run_studio(args: argparse.Namespace) -> int:
    config = load_config(project_path=args.config) if args.config else load_config()
    if args.language:
        values = config.to_dict()
        values["language"] = args.language
        config = AIsketcherConfig.from_mapping(values)
    return _launch_studio(config, port=args.port)


def _launch_studio(config: AIsketcherConfig, *, port: int | None = None) -> int:
    if port is not None and not 1 <= port <= 65535:
        raise ValueError("port must be between 1 and 65535")

    # Optional UI and model imports stay behind the explicit Studio command so
    # the lightweight SDK and `aisketcher init` never import Gradio or Torch.
    from . import PresetManager, Studio
    from .prompt_normalization import M2M100KoreanEnglishTranslator
    from .studio_app import AppController, build_app

    manager = PresetManager(
        cache_dir=config.cache_path,
        allow_downloads=config.allow_downloads,
    )

    def studio_factory(preset: str) -> Studio:
        return Studio.from_preset(
            preset,
            device=config.device,
            preset_manager=manager,
        )

    controller = AppController(
        studio_factory=studio_factory,
        model_installer=manager,
        prompt_translator=M2M100KoreanEnglishTranslator(
            cache_dir=str(manager.cache_dir / "translation"),
        ),
    )
    demo = build_app(
        controller,
        language=config.language,
        default_preset=config.preset,
        # The concrete 2026 default is also the implementation behind Auto.
        # Keep the first-run Simple surface friendly while still honoring an
        # explicitly configured legacy preset.
        default_simple_model=(
            "auto" if config.preset == "flux2-klein-edit@1" else config.preset
        ),
        default_output_count=config.output_count,
        default_seed_mode=str(config.seed_mode),
        default_seed=config.seed,
    )
    launch_kwargs: dict[str, Any] = dict(demo._studio_launch_kwargs)
    if port is not None:
        launch_kwargs["server_port"] = port
    try:
        demo.launch(**launch_kwargs)
    finally:
        controller.close()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """Run the CLI and turn expected user errors into concise diagnostics."""

    parser = _parser()
    args = parser.parse_args(argv)
    try:
        return int(args.handler(args))
    except (AIsketcherError, FileExistsError, OSError, ValueError) as exc:
        print(f"aisketcher: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
