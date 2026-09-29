"""Export compact inference weights or a base-bound voice adaptation delta.

Run from the repository root. All output paths must be new files.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.model_io import (build_model, export_adapter, export_runtime,
                              initialize_generator, read_config)
from modules.voice_adapter import configure_trainable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    runtime = sub.add_parser('runtime', help='Remove posterior and optionally the F0 predictor')
    runtime.add_argument('--keep-f0', action='store_true')
    adapter = sub.add_parser('adapter', help='Save only trained adapters, tied to a specific base file')
    adapter.add_argument('--base', required=True)
    adapter.add_argument('--mode', choices=['adapters', 'adapters+speaker'], default='adapters')
    for command in (runtime, adapter):
        command.add_argument('--checkpoint', required=True)
        command.add_argument('--config', required=True)
        command.add_argument('--output', required=True)
    args = parser.parse_args()
    try:
        config = read_config(args.config)
        if args.command == 'runtime':
            report = export_runtime(args.checkpoint, config, args.output, keep_f0=args.keep_f0)
        else:
            model = build_model(config)
            initialize_generator(model, args.checkpoint)
            configure_trainable(model, args.mode)
            report = export_adapter(model, args.base, config, args.output)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, 'Artifact error: {}\n'.format(exc))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
