"""Create a fresh, opt-in adapter training config. Never overwrite a config."""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.model_io import model_arguments


def derive_adapter_config(config, base, rank=16, mode='adapters', keep_f0=False):
    if type(rank) is not int or rank <= 0:
        raise ValueError('rank must be a positive integer')
    if mode not in ('adapters', 'adapters+speaker'):
        raise ValueError('mode must be adapters or adapters+speaker')
    if not Path(base).is_file():
        raise ValueError('Base generator checkpoint does not exist')
    result = copy.deepcopy(config)
    model_arguments(result)
    if not result.get('spk'):
        raise ValueError('Generate your dataset speaker mapping first')
    result['model']['adapter_rank'] = rank
    result['model']['inference_only'] = False
    if not keep_f0:
        result['model']['use_automatic_f0_prediction'] = False
    result['train']['finetune_mode'] = mode
    result['train']['init_generator'] = str(Path(base).resolve())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--rank', type=int, default=16)
    parser.add_argument('--mode', choices=['adapters', 'adapters+speaker'], default='adapters')
    parser.add_argument('--keep-f0', action='store_true')
    args = parser.parse_args()
    try:
        config = json.loads(args.source.read_text(encoding='utf-8'))
        result = derive_adapter_config(config, args.base, args.rank, args.mode, args.keep_f0)
        with args.destination.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, indent=2, ensure_ascii=False)
            stream.write('\n')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, 'Adapter configuration error: {}\n'.format(exc))
    print('Created {}. Train into a NEW experiment output directory.'.format(args.destination))
    print('Speaker IDs are unchanged. Pretrained capability and audio quality still require validation.')


if __name__ == '__main__':
    main()
