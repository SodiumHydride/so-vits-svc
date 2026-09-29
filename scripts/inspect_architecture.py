"""Count actual instantiated generator parameters; this is NOT a VRAM benchmark."""
import argparse
import copy
import json
from pathlib import Path
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modules.model_io import build_model, read_config, state_bytes
from modules.voice_adapter import configure_trainable


def inspect(config, rank=16):
    if type(rank) is not int or rank < 1:
        raise ValueError('rank must be a positive integer')
    cfg = copy.deepcopy(config)
    cfg['model']['adapter_rank'] = 0
    full = build_model(cfg)
    count = lambda model: sum(p.numel() for p in model.parameters())
    result = {'torch': str(torch.__version__), 'cuda_available': torch.cuda.is_available(),
              'test_type': 'untrained parameter counts; not a VRAM benchmark',
              'full_generator_parameters': count(full),
              'posterior_parameters': count(full.enc_q),
              'f0_predictor_parameters': count(full.f0_decoder) if hasattr(full, 'f0_decoder') else 0,
              'full_generator_state_bytes': state_bytes(full.state_dict())}
    del full
    compact = build_model(cfg, inference=True, keep_f0=False)
    result.update(compact_generator_parameters=count(compact),
                  compact_generator_state_bytes=state_bytes(compact.state_dict()))
    del compact
    cfg['model']['adapter_rank'] = rank
    adapted = build_model(cfg)
    result['adapters_only'] = configure_trainable(adapted, 'adapters')
    result['adapters_and_speaker'] = configure_trainable(adapted, 'adapters+speaker')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', type=Path)
    parser.add_argument('--rank', type=int, default=16)
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        result = inspect(read_config(args.config), args.rank)
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        parser.exit(2, 'Inspection error: {}\n'.format(exc))
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
