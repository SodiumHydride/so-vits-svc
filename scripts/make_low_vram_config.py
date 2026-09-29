"""Derive a low-VRAM experiment config without changing the model architecture.

Run after the upstream preprocessing step has generated YOUR config and spk map.
The destination must not exist. Smaller batches/windows can affect convergence;
this is not a promise of a particular VRAM requirement or audio quality.
"""
import argparse
import copy
import json
from pathlib import Path


def derive_config(config, batch_size=2, frames=256, precision='fp16'):
    result = copy.deepcopy(config)
    for section in ('train', 'data', 'model', 'spk'):
        if section not in result or not isinstance(result[section], dict):
            raise ValueError('Missing config mapping: ' + section)
    if not result['spk']:
        raise ValueError('Generate the speaker mapping before deriving this config')
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError('batch_size must be a positive integer')
    if type(frames) is not int or frames < 1:
        raise ValueError('frames must be a positive integer')
    if precision not in ('fp32', 'fp16', 'bf16'):
        raise ValueError('precision must be fp32, fp16, or bf16')
    hop = result['data']['hop_length']
    segment = result['train']['segment_size']
    if type(hop) is not int or hop < 1 or type(segment) is not int or segment < 1 or segment % hop:
        raise ValueError('segment_size must be a positive integer multiple of hop_length')
    if frames < segment // hop:
        raise ValueError('frames is shorter than the generator training segment')
    result['train'].update(batch_size=batch_size, max_speclen=frames,
                           fp16_run=precision != 'fp32',
                           half_type='bf16' if precision == 'bf16' else 'fp16',
                           all_in_mem=False, num_workers=2, max_eval_batches=4)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--frames', type=int, default=256)
    parser.add_argument('--precision', choices=['fp32', 'fp16', 'bf16'], default='fp16')
    args = parser.parse_args()
    try:
        with args.source.open(encoding='utf-8') as stream:
            config = json.load(stream)
        result = derive_config(config, args.batch_size, args.frames, args.precision)
        # Exclusive creation: never overwrite the user's config or experiment.
        with args.destination.open('x', encoding='utf-8') as stream:
            json.dump(result, stream, ensure_ascii=False, indent=2)
            stream.write('\n')
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, 'Configuration error: {}\n'.format(exc))
    print('Created {}. Model, speaker map and sampling rate are unchanged.'.format(args.destination))
    print('Experimental settings; benchmark VRAM and held-out audio before relying on them.')


if __name__ == '__main__':
    main()
