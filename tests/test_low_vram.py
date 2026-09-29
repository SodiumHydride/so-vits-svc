"""CPU regression tests for the changed logic, NOT end-to-end audio/CUDA tests.

Use AST-loaded definitions to isolate the changed data/trainer code from optional
upstream FAISS, fairseq, vocoder and plotting dependencies. File-I/O, mel
extraction and network components are explicitly stubbed where indicated.
Only Python, NumPy and PyTorch are required for this suite.
"""
import ast
import copy
import importlib.util
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runtime = load_module('tested_runtime', ROOT / 'modules/training_runtime.py')
config_tool = load_module('tested_config', ROOT / 'scripts/make_low_vram_config.py')


def definitions(path, namespace):
    source = ast.parse(path.read_text())
    source.body = [node for node in source.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    exec(compile(source, str(path), 'exec'), namespace)
    return namespace


def hparams(frames=256, volume=False):
    return SimpleNamespace(
        train=SimpleNamespace(max_speclen=frames, segment_size=16, seed=1234, vol_aug=False),
        data=SimpleNamespace(max_wav_value=32768., sampling_rate=44100,
                             filter_length=32, hop_length=4, win_length=32,
                             unit_interpolate_mode='nearest'),
        model=SimpleNamespace(vol_embedding=volume), spk={'test': 0})


def item(length, volume=True):
    ramp = torch.arange(length).float()
    return (ramp[None].repeat(3, 1), ramp.clone(), ramp[None].repeat(5, 1),
            torch.arange(length * 4).float()[None], torch.tensor([0]),
            ramp.clone(), ramp.clone() if volume else None)


def dataset_namespace():
    # Audio reading and spectral extraction are not being integration-tested here.
    import os
    return definitions(ROOT / 'data_utils.py', dict(
        os=os, random=random, np=np, torch=torch,
        load_filepaths_and_text=lambda path: [['a'], ['b']],
        load_wav_to_torch=MagicMock(), spectrogram_torch=MagicMock(),
        utils=SimpleNamespace(repeat_expand_2d=lambda tensor, frames, mode: tensor)))


class DataTests(unittest.TestCase):
    def setUp(self):
        self.ns = dataset_namespace()
        self.Loader = self.ns['TextAudioSpeakerLoader']
        self.Collate = self.ns['TextAudioCollate']

    def test_crop_respects_config_and_alignment(self):
        ds = self.Loader('unused', hparams(frames=256))
        with patch.object(random, 'randint', return_value=19):
            c, f0, spec, wave, _, uv, vol = ds.random_slice(*item(1200))
        for stream in (c[0], f0, spec[0], uv, vol):
            torch.testing.assert_close(stream, torch.arange(19, 275).float())
        torch.testing.assert_close(wave[0], torch.arange(76, 1100).float())

    def test_default_512_no_longer_hardcoded_790(self):
        ds = self.Loader('unused', hparams(frames=512))
        self.assertEqual(ds.random_slice(*item(1000))[0].shape[-1], 512)

    def test_last_possible_crop_is_reachable(self):
        ds = self.Loader('unused', hparams(frames=6))
        with patch.object(random, 'randint', side_effect=lambda low, high: high):
            result = ds.random_slice(*item(10))
        torch.testing.assert_close(result[1], torch.arange(4, 10).float())

    def test_short_and_equal_length_clips(self):
        ds = self.Loader('unused', hparams(frames=256))
        for size in (8, 256):
            self.assertEqual(ds.random_slice(*item(size))[0].shape[-1], size)

    def test_validation_crop_is_deterministic(self):
        ds = self.Loader('unused', hparams(frames=10), vol_aug=False, random_crop=False)
        with patch.object(random, 'randint', side_effect=AssertionError('random validation')):
            first = ds.random_slice(*item(30))
            second = ds.random_slice(*item(30))
        torch.testing.assert_close(first[1], torch.arange(10, 20).float())
        torch.testing.assert_close(first[1], second[1])

    def test_dataset_construction_preserves_global_rng(self):
        random.seed(81)
        state = random.getstate()
        self.Loader('unused', hparams())
        self.assertEqual(state, random.getstate())

    def test_invalid_frame_limits(self):
        for value in (0, -1, 1.5, True, 3):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.Loader('unused', hparams(value))

    def test_segment_alignment_validation(self):
        hp = hparams()
        hp.train.segment_size = 17
        with self.assertRaises(ValueError):
            self.Loader('unused', hp)

    def test_volume_none_collate(self):
        result = self.Collate()([item(8, False), item(12, False)])
        self.assertIsNone(result[-1])
        self.assertEqual(result[5].tolist(), [12, 8])
        self.assertEqual(result[0].shape, (2, 3, 12))
        self.assertTrue(torch.all(result[0][1, :, 8:] == 0))

    def test_volume_present_collate(self):
        result = self.Collate()([item(8), item(12)])
        self.assertEqual(result[-1].shape, (2, 12))
        torch.testing.assert_close(result[-1][1, :8], torch.arange(8).float())

    def test_mixed_volume_rejected_in_both_orders(self):
        for batch in ([item(8), item(12, False)], [item(8, False), item(12)]):
            with self.assertRaisesRegex(ValueError, 'every item'):
                self.Collate()(batch)

    def test_empty_batch_rejected(self):
        for batch in ([], [None]):
            with self.assertRaises(ValueError):
                self.Collate()(batch)

    def test_unaligned_features_rejected(self):
        row = list(item(8))
        row[1] = row[1][:-1]
        with self.assertRaises(ValueError):
            self.Collate()([tuple(row)])

    def test_cache_tensor_files_loaded_on_cpu(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'test'
            path.mkdir()
            wavepath = str(path / 'song.wav')
            torch.save(torch.ones(5, 8), str(path / 'song.spec.pt'))
            torch.save(torch.ones(1, 3, 8), wavepath + '.soft.pt')
            np.save(wavepath + '.f0.npy', np.stack([np.ones(8), np.ones(8)]))
            self.ns['load_wav_to_torch'].return_value = (torch.ones(32), 44100)
            ds = self.Loader('unused', hparams())
            actual = ds.get_audio(wavepath)
            self.assertTrue(all(x.device.type == 'cpu' for x in actual if x is not None))
            self.assertEqual(actual[0].shape, (3, 8))


class RuntimeTests(unittest.TestCase):
    def test_freeze_preserves_input_gradient(self):
        torch.manual_seed(42)
        net = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
        inputs = torch.randn(3, 4, requires_grad=True)
        net(inputs).square().mean().backward()
        reference = inputs.grad.clone()
        net.zero_grad(set_to_none=True)
        inputs.grad = None
        with runtime.frozen_parameters(net):
            net(inputs).square().mean().backward()
            self.assertTrue(all(p.grad is None for p in net.parameters()))
        torch.testing.assert_close(inputs.grad, reference)
        self.assertTrue(all(p.requires_grad for p in net.parameters()))

    def test_freeze_restores_preexisting_flags_on_exception(self):
        net = nn.Linear(3, 2)
        net.bias.requires_grad_(False)
        with self.assertRaises(RuntimeError):
            with runtime.frozen_parameters(net):
                raise RuntimeError('intentional')
        self.assertTrue(net.weight.requires_grad)
        self.assertFalse(net.bias.requires_grad)

    def test_freeze_keeps_training_mode(self):
        net = nn.Linear(2, 1).train()
        with runtime.frozen_parameters(net):
            self.assertTrue(net.training)

    def test_unwrap_plain_module(self):
        net = nn.Linear(2, 1)
        self.assertIs(runtime.unwrap_model(net), net)

    def test_precision_selection(self):
        self.assertEqual(runtime.resolve_precision(SimpleNamespace(fp16_run=False), 'cpu'),
                         (False, torch.float32))
        self.assertEqual(runtime.resolve_precision(SimpleNamespace(fp16_run=True, half_type='fp16'), 'cuda'),
                         (True, torch.float16))
        with self.assertRaises(ValueError):
            runtime.resolve_precision(SimpleNamespace(fp16_run=True, half_type='bad'), 'cuda')
        with self.assertRaises(ValueError):
            runtime.resolve_precision(SimpleNamespace(fp16_run=True, half_type='fp16'), 'cpu')

    def test_bf16_scaling_disabled(self):
        self.assertFalse(runtime.make_grad_scaler(True, torch.bfloat16).is_enabled())
        self.assertFalse(runtime.make_grad_scaler(False, torch.float32).is_enabled())

    def test_unsupported_bf16_rejected(self):
        with patch('torch.cuda.device'), patch('torch.cuda.is_bf16_supported', return_value=False):
            with self.assertRaisesRegex(ValueError, 'unsupported'):
                runtime.resolve_precision(SimpleNamespace(fp16_run=True, half_type='bf16'), 'cuda')

    def test_worker_seed_reproducible(self):
        with patch('torch.initial_seed', return_value=987654):
            runtime.seed_worker(0)
            first = (random.random(), np.random.rand())
            runtime.seed_worker(0)
            self.assertEqual(first, (random.random(), np.random.rand()))


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.config = {'train': {'segment_size': 10240}, 'data': {'hop_length': 512, 'sampling_rate': 44100},
                       'model': {'hidden_channels': 192}, 'spk': {'test': 0}}

    def test_config_preserves_architecture_speaker_map_and_source(self):
        original = copy.deepcopy(self.config)
        result = config_tool.derive_config(self.config)
        self.assertEqual(self.config, original)
        for section in ('model', 'spk', 'data'):
            self.assertEqual(result[section], original[section])
        self.assertEqual(result['train']['max_speclen'], 256)
        self.assertEqual(result['train']['batch_size'], 2)

    def test_fp32_preset(self):
        result = config_tool.derive_config(self.config, precision='fp32')
        self.assertFalse(result['train']['fp16_run'])

    def test_invalid_config_settings(self):
        for kwargs in ({'frames': 1}, {'batch_size': 0}, {'precision': 'fp8'}, {'frames': True}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                config_tool.derive_config(self.config, **kwargs)

    def test_empty_speaker_map_rejected(self):
        self.config['spk'] = {}
        with self.assertRaises(ValueError):
            config_tool.derive_config(self.config)

    def test_new_python_files_compile(self):
        for file in (ROOT / 'data_utils.py', ROOT / 'train.py', ROOT / 'modules/training_runtime.py',
                     ROOT / 'scripts/make_low_vram_config.py'):
            compile(file.read_text(), str(file), 'exec')



class TinyGenerator(nn.Module):
    use_automatic_f0_prediction = False

    def __init__(self):
        super().__init__()
        self.gain = nn.Parameter(torch.tensor(0.5))

    def forward(self, c, f0, uv, spec, g=None, c_lengths=None, spec_lengths=None, vol=None):
        z = c * self.gain
        wave = z[:, :1, :4].repeat_interleave(4, -1)
        zeros = torch.zeros_like(z)
        return (wave, torch.zeros(c.shape[0], dtype=torch.long), torch.ones_like(z[:, :1]),
                (z, z, z * 0.2, zeros, zeros, zeros), 0, 0, 0)

    def infer(self, c, f0, uv, g=None, vol=None):
        torch.manual_seed(123)  # Reproduce the upstream inference RNG side effect.
        return c[:, :1] * self.gain, f0


class TinyDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Conv1d(1, 2, 3, padding=1)

    def forward(self, real, fake):
        r, f = self.layer(real), self.layer(fake)
        return [r], [f], [[r]], [[f]]


def trainer_namespace():
    import os
    import time
    def slice_segments(tensor, ids, length):
        return torch.stack([row[..., int(index):int(index) + length]
                            for row, index in zip(tensor, ids)])
    def mel(tensor, *args):
        return tensor.reshape(tensor.shape[0], -1, 4).mean(-1)[:, None].repeat(1, 5, 1)
    ns = dict(torch=torch, os=os, time=time, F=F, global_step=0,
              dist=SimpleNamespace(is_initialized=lambda: False),
              unwrap_model=runtime.unwrap_model, frozen_parameters=runtime.frozen_parameters,
              cuda_autocast=runtime.cuda_autocast, utils=MagicMock(),
              commons=SimpleNamespace(slice_segments=slice_segments,
                  clip_grad_value_=lambda params, _: nn.utils.clip_grad_norm_(params, float('inf'))),
              spec_to_mel_torch=lambda tensor, *args: tensor, mel_spectrogram_torch=mel,
              discriminator_loss=lambda real, fake: (((real[0] - 1).square().mean() + fake[0].square().mean()), [], []),
              generator_loss=lambda fake: ((fake[0] - 1).square().mean(), []),
              feature_loss=lambda real, fake: (real[0][0].detach() - fake[0][0]).abs().mean(),
              kl_loss=lambda z, lq, mp, lp, mask: (z - mp).square().mean())
    return definitions(ROOT / 'train.py', ns)


def training_hparams():
    hp = hparams()
    hp.train.c_mel = hp.train.c_kl = 1.
    hp.train.log_interval = 1
    hp.train.eval_interval = 100
    hp.train.learning_rate = 1e-3
    hp.train.keep_ckpts = 0
    hp.train.max_eval_batches = 2
    hp.data.n_mel_channels = 5
    hp.data.mel_fmin, hp.data.mel_fmax = 0., 22050.
    hp.model_dir = 'test-only-not-written'
    return hp


class TrainerControlFlowTests(unittest.TestCase):
    def setUp(self):
        self.ns = trainer_namespace()
        self.hp = training_hparams()
        self.batch = (torch.randn(2, 2, 8), torch.ones(2, 8) * 220,
                      torch.randn(2, 5, 8), torch.randn(2, 1, 32),
                      torch.zeros(2, 1, dtype=torch.long), torch.full((2,), 8),
                      torch.ones(2, 8), torch.ones(2, 8))

    def test_three_optimizer_steps_with_tiny_networks(self):
        g, d = TinyGenerator(), TinyDiscriminator()
        initial_g, initial_d = g.gain.detach().clone(), d.layer.weight.detach().clone()
        opt_g, opt_d = torch.optim.AdamW(g.parameters(), lr=1e-3), torch.optim.AdamW(d.parameters(), lr=1e-3)
        self.ns['evaluate'] = MagicMock()
        from contextlib import ExitStack
        with ExitStack() as stack:
            stack.enter_context(patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor))
            for name in ('reset_peak_memory_stats', 'max_memory_allocated', 'max_memory_reserved', 'synchronize'):
                stack.enter_context(patch.object(torch.cuda, name, return_value=0))
            self.ns['train_and_evaluate'](0, 1, self.hp, [g, d], [opt_g, opt_d],
                runtime.make_grad_scaler(False, torch.float32), [[self.batch] * 3, []],
                MagicMock(), [MagicMock(), MagicMock()], False, torch.float32)
        self.assertEqual(self.ns['global_step'], 3)
        self.assertFalse(torch.equal(initial_g, g.gain.detach()))
        self.assertFalse(torch.equal(initial_d, d.layer.weight.detach()))
        self.assertTrue(all(p.requires_grad and p.grad is None for p in d.parameters()))
        self.assertTrue(all(torch.isfinite(p).all() for p in list(g.parameters()) + list(d.parameters())))
        self.ns['evaluate'].assert_called_once()

    def test_validation_limit_and_rng_restore(self):
        g = TinyGenerator().train()
        torch.manual_seed(44)
        state = torch.get_rng_state().clone()
        original_fork_rng = torch.random.fork_rng
        def cpu_fork(*args, **kwargs):
            return original_fork_rng(devices=[])
        with patch.object(torch.Tensor, 'cuda', lambda tensor, *a, **kw: tensor), \
                patch('torch.random.fork_rng', side_effect=cpu_fork):
            self.ns['evaluate'](self.hp, g, [self.batch] * 4, MagicMock())
        self.assertTrue(g.training)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))
        kwargs = self.ns['utils'].summarize.call_args.kwargs
        self.assertEqual(len(kwargs['audios']), 4)  # two predicted + two reference
        self.assertTrue(all(t.device.type == 'cpu' for t in kwargs['audios'].values()))

    def test_empty_validation_is_safe(self):
        original_fork_rng = torch.random.fork_rng
        with patch('torch.random.fork_rng', side_effect=lambda *a, **kw: original_fork_rng(devices=[])):
            self.ns['evaluate'](self.hp, TinyGenerator(), [], MagicMock())
        self.ns['utils'].summarize.assert_not_called()

    def test_real_distributed_sampler_shards_data(self):
        data = list(range(12))
        samplers = [torch.utils.data.DistributedSampler(data, num_replicas=2, rank=r, seed=18)
                    for r in range(2)]
        left, right = [list(s) for s in samplers]
        self.assertFalse(set(left) & set(right))
        self.assertEqual(set(left + right), set(data))
        samplers[0].set_epoch(1)
        self.assertNotEqual(left, list(samplers[0]))


if __name__ == '__main__':
    unittest.main()
