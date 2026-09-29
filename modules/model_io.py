"""Strict model initialization and small, versioned inference/adapter artifacts.

Artifacts contain tensor state dictionaries and plain metadata, never executable
pickled models. weights_only reduces pickle exposure; use trusted local files.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import torch

ADAPTER_PREFIXES = ('prior_adapter.', 'decoder_adapter.')
RUNTIME_FORMAT = 'sovits-runtime-v1'
ADAPTER_FORMAT = 'sovits-adapter-v1'


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_config(path):
    with open(path, encoding='utf-8') as stream:
        config = json.load(stream)
    model_arguments(config)
    return config


def model_arguments(config, inference=False, keep_f0=True):
    model = copy.deepcopy(config['model'])
    data, train = config['data'], config['train']
    hop, segment = data['hop_length'], train['segment_size']
    if type(hop) is not int or hop <= 0 or type(segment) is not int or segment <= 0 or segment % hop:
        raise ValueError('segment_size must be a positive multiple of hop_length')
    if math.prod(model['upsample_rates']) != hop:
        raise ValueError('Decoder upsample product must equal hop_length')
    if 'sampling_rate' in model and model['sampling_rate'] != data['sampling_rate']:
        raise ValueError('Model and data sampling rates disagree')
    model['sampling_rate'] = data['sampling_rate']
    for speaker, index in config.get('spk', {}).items():
        if type(index) is not int or not 0 <= index < model['n_speakers']:
            raise ValueError('Speaker index is outside the embedding table: ' + speaker)
    model['spec_channels'] = data['filter_length'] // 2 + 1
    model['segment_size'] = segment // hop
    model['inference_only'] = bool(inference)
    if inference and not keep_f0:
        model['use_automatic_f0_prediction'] = False
    return model


def build_model(config, inference=False, keep_f0=True):
    from models import SynthesizerInfer, SynthesizerTrn
    model_type = SynthesizerInfer if inference else SynthesizerTrn
    return model_type(**model_arguments(config, inference, keep_f0))


def read_weights(path):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError('Expected a tensor state dictionary or a model checkpoint')
    if payload.get('format') in (RUNTIME_FORMAT, ADAPTER_FORMAT):
        raise ValueError('Use the artifact loader, not training initialization, for this format')
    state = payload.get('model', payload)
    return normalize_state(state)


def normalize_state(state):
    if not isinstance(state, dict) or not state or not all(isinstance(k, str) and torch.is_tensor(v)
                                                         for k, v in state.items()):
        raise ValueError('Expected a nonempty tensor state dictionary')
    prefixed = [k.startswith('module.') for k in state]
    if any(prefixed) and not all(prefixed):
        raise ValueError('Mixed DDP and unprefixed state keys')
    return {k[7:] if all(prefixed) else k: v for k, v in state.items()}


def compatible_state(model, state, allow_new_adapters=False):
    """Validate ALL keys/shapes before mutating any model tensor."""
    expected = model.state_dict()
    allowed_drop = ('enc_q.',) if getattr(model, 'inference_only', False) else ()
    if not model.use_automatic_f0_prediction:
        allowed_drop += ('f0_decoder.',)
    kept = {k: v for k, v in state.items() if not k.startswith(allowed_drop)}
    missing = set(expected) - set(kept)
    unexpected = set(kept) - set(expected)
    adapter_keys = {k for k in expected if k.startswith(ADAPTER_PREFIXES)}
    # Only a completely absent adapter is valid for baseline initialization.
    permitted_missing = adapter_keys if allow_new_adapters and not (adapter_keys & set(kept)) else set()
    bad_missing = missing - permitted_missing
    mismatched = [k for k in set(expected) & set(kept) if expected[k].shape != kept[k].shape]
    if bad_missing or unexpected or mismatched:
        raise ValueError('Incompatible generator state: missing={}, unexpected={}, shape_mismatch={}'.format(
            sorted(bad_missing)[:8], sorted(unexpected)[:8], sorted(mismatched)[:8]))
    return kept, missing


def initialize_generator(model, checkpoint, allow_new_adapters=False):
    state = read_weights(checkpoint)
    kept, missing = compatible_state(model, state, allow_new_adapters)
    model.load_state_dict(kept, strict=not missing)
    return {'source_sha256': sha256_file(checkpoint), 'initialized_adapter_keys': sorted(missing)}


def save_exclusive(payload, destination):
    """Atomic no-overwrite publication on the destination filesystem."""
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(str(destination))
    fd, temporary = tempfile.mkstemp(prefix='.svc-', suffix='.tmp', dir=str(destination.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, destination)  # Fails atomically if another writer created it.
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def state_bytes(state):
    return sum(v.numel() * v.element_size() for v in state.values())


def export_runtime(checkpoint, config, destination, keep_f0=False):
    model = build_model(config, inference=True, keep_f0=keep_f0)
    source = read_weights(checkpoint)
    kept, _ = compatible_state(model, source)
    deployed = copy.deepcopy(config)
    deployed['train'] = {'segment_size': config['train']['segment_size']}
    deployed['data'].pop('training_files', None)
    deployed['data'].pop('validation_files', None)
    deployed['model']['use_automatic_f0_prediction'] = model.use_automatic_f0_prediction
    deployed['model']['inference_only'] = True
    state = {k: v.detach().cpu().clone() for k, v in kept.items()}
    payload = {'format': RUNTIME_FORMAT, 'config': deployed, 'model': state,
               'source_sha256': sha256_file(checkpoint)}
    save_exclusive(payload, destination)
    return {'source_generator_state_bytes': state_bytes(source), 'runtime_state_bytes': state_bytes(state),
            'removed_state_keys': len(source) - len(state), 'predict_f0': model.use_automatic_f0_prediction}


def load_runtime(path, device='cpu', dtype=None):
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != RUNTIME_FORMAT:
        raise ValueError('Not a supported SVC runtime artifact')
    model = build_model(payload['config'], inference=True)
    state = normalize_state(payload['model'])
    # Runtime artifacts must already be pruned; extra training keys are corruption.
    expected = model.state_dict()
    if set(state) != set(expected) or any(state[k].shape != expected[k].shape for k in expected):
        raise ValueError('Runtime artifact keys/shapes do not match its configuration')
    if dtype is None:
        dtype = next(v.dtype for v in state.values() if v.is_floating_point())
    model.to(dtype=dtype)
    model.load_state_dict(state, strict=True)
    return model.eval().to(device), copy.deepcopy(payload['config'])


def export_adapter(model, base_checkpoint, config, destination):
    mode = getattr(model, 'finetune_mode', 'full')
    if mode not in ('adapters', 'adapters+speaker'):
        raise ValueError('Only adapter-only training modes can be exported as a delta')
    if model_arguments(config).get('adapter_rank', 0) != model.adapter_rank:
        raise ValueError('Configuration and adapter dimensions disagree')
    base = read_weights(base_checkpoint)
    compatible_state(model, base, allow_new_adapters=True)
    keys = {name for name, p in model.named_parameters() if p.requires_grad}
    expected_keys = {k for k in model.state_dict() if k.startswith(ADAPTER_PREFIXES)
                     or (mode == 'adapters+speaker' and k.startswith('emb_g.'))}
    if keys != expected_keys:
        raise ValueError('Trainable parameter set does not match the declared adapter mode')
    state = model.state_dict()
    for name in set(state) - keys:
        if not torch.equal(state[name].detach().cpu(), base[name].to(state[name].dtype)):
            raise ValueError('Backbone changed; cannot save an adapter-only delta: ' + name)
    delta = {k: state[k].detach().cpu().clone() for k in sorted(keys)}
    payload = {'format': ADAPTER_FORMAT, 'base_sha256': sha256_file(base_checkpoint),
               'config': copy.deepcopy(config), 'finetune_mode': mode, 'delta': delta}
    save_exclusive(payload, destination)
    return {'delta_state_bytes': state_bytes(delta), 'delta_parameters': sum(v.numel() for v in delta.values())}


def load_adapter(base_checkpoint, adapter_path, inference=True, device='cpu'):
    from modules.voice_adapter import configure_trainable
    payload = torch.load(adapter_path, map_location='cpu', weights_only=True)
    if not isinstance(payload, dict) or payload.get('format') != ADAPTER_FORMAT:
        raise ValueError('Not a supported SVC adapter artifact')
    if sha256_file(base_checkpoint) != payload['base_sha256']:
        raise ValueError('Adapter was trained against a different base checkpoint')
    mode = payload['finetune_mode']
    if mode not in ('adapters', 'adapters+speaker'):
        raise ValueError('Invalid adapter mode')
    model = build_model(payload['config'], inference=inference)
    initialize_generator(model, base_checkpoint, allow_new_adapters=True)
    state = model.state_dict()
    expected = {k for k in state if k.startswith(ADAPTER_PREFIXES)
                or (mode == 'adapters+speaker' and k.startswith('emb_g.'))}
    delta = normalize_state(payload['delta'])
    if set(delta) != expected or any(delta[k].shape != state[k].shape for k in expected):
        raise ValueError('Adapter delta keys/shapes do not match its configuration')
    state.update(delta)
    model.load_state_dict(state, strict=True)
    if not inference:
        configure_trainable(model, mode)
    return model.eval().to(device), copy.deepcopy(payload['config'])
