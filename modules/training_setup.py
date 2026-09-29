"""Explicit initialization and parameter selection for fresh SVC experiments."""
from modules.model_io import initialize_generator
from modules.voice_adapter import configure_trainable


def prepare_generator(model, train_config):
    mode = getattr(train_config, 'finetune_mode', 'full')
    checkpoint = getattr(train_config, 'init_generator', None)
    if mode not in ('full', 'adapters', 'adapters+speaker'):
        raise ValueError('Unknown finetune_mode: ' + str(mode))
    if mode != 'full' and not checkpoint:
        raise ValueError('Adapter training requires train.init_generator; do not freeze a random backbone')
    if mode != 'full' and not model.adapter_rank:
        raise ValueError('Adapter training requires model.adapter_rank > 0')
    report = {}
    if checkpoint:
        report.update(initialize_generator(model, checkpoint, allow_new_adapters=bool(model.adapter_rank)))
    report.update(configure_trainable(model, mode))
    return report
