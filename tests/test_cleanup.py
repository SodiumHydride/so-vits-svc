"""Regression tests for utility cleanup, not model quality or GPU benchmarks.

Factories use fake constructors to check dispatch without downloading weights.
WAV I/O and plotting use real SciPy/Matplotlib. Import-boundary checks use fresh
Python processes so previously imported packages cannot hide eager dependencies.
"""
import builtins
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cluster
import utils
from modules import model_utils


def legacy_pitch(f0):
    # Arithmetic copied from utils.py blob 95b6d888..., including its high-bin behavior.
    lower = 1127 * np.log(1 + 50.0 / 700)
    upper = 1127 * np.log(1 + 1100.0 / 700)
    mel = 1127 * (1 + f0 / 700).log()
    a = 254 / (upper - lower)
    b = lower * a - 1.
    mel = torch.where(mel > 0, mel * a - b, mel)
    coarse = torch.round(mel).long()
    coarse = coarse * (coarse > 0)
    coarse = coarse + ((coarse < 1) * 1)
    coarse = coarse * (coarse < 256)
    return coarse + ((coarse >= 256) * 255)


def legacy_normalize(f0, mask, uv, random_scale=True):
    count = torch.sum(uv, dim=1, keepdim=True)
    count[count == 0] = 9999
    mean = torch.sum(f0[:, 0, :] * uv, dim=1, keepdim=True) / count
    if random_scale:
        factor = torch.Tensor(f0.shape[0], 1).uniform_(0.8, 1.2).to(f0.device)
    else:
        factor = torch.ones(f0.shape[0], 1).to(f0.device)
    return (f0 - mean.unsqueeze(-1)) * factor.unsqueeze(-1) * mask


class ImportBoundaryTests(unittest.TestCase):
    def fresh_process(self, tail):
        code = """
import importlib.abc
import sys
import torch  # Allow PyTorch's own dependency discovery before testing our modules.
import numpy
blocked = {'faiss', 'sklearn', 'librosa', 'scipy', 'matplotlib', 'fairseq'}
for name in tuple(sys.modules):
    if name.split('.')[0] in blocked:
        del sys.modules[name]
class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in blocked:
            raise ModuleNotFoundError('blocked optional import: ' + fullname, name=fullname)
sys.meta_path.insert(0, BlockOptional())
""" + tail
        process = subprocess.run([sys.executable, '-c', code], cwd=ROOT,
                                 text=True, capture_output=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stdout + process.stderr)

    def test_utils_import_without_optional_packages(self):
        self.fresh_process("\nimport utils\nassert utils.HParams(a=2).a == 2\n")

    def test_cluster_import_without_sklearn(self):
        self.fresh_process("\nimport cluster\nassert callable(cluster.get_center)\n")

    def test_tensor_helpers_without_optional_packages(self):
        self.fresh_process("""
import utils
assert utils.f0_to_coarse(torch.tensor([0., 220.])).shape == (2,)
assert utils.repeat_expand_2d(torch.ones(3, 2), 4).shape == (3, 4)
""")

    def test_retrieval_still_requires_faiss_when_called(self):
        self.fresh_process("""
import utils
try:
    utils.train_index('not-accessed')
except ModuleNotFoundError as exc:
    assert exc.name == 'faiss'
else:
    raise AssertionError('retrieval dependency was silently suppressed')
""")

    def test_cluster_load_still_requires_sklearn(self):
        self.fresh_process("""
import cluster
try:
    cluster.get_cluster_model('not-accessed.pth')
except ModuleNotFoundError as exc:
    assert exc.name == 'sklearn'
else:
    raise AssertionError('clustering dependency was silently suppressed')
""")


class PitchCompatibilityTests(unittest.TestCase):
    def test_pitch_helper_is_reexport_not_duplicate(self):
        self.assertIs(utils.f0_to_coarse, model_utils.f0_to_coarse)
        self.assertIs(utils.normalize_f0, model_utils.normalize_f0)

    def test_public_constants_preserved(self):
        self.assertEqual((utils.f0_bin, utils.f0_min, utils.f0_max), (256, 50., 1100.))

    def test_pitch_matches_legacy_across_range(self):
        for dtype in (torch.float32, torch.float64):
            x = torch.cat((torch.linspace(0., 5000., 30001, dtype=dtype),
                           torch.tensor([0., 1., 50., 220., 1100., 2000.], dtype=dtype)))
            torch.testing.assert_close(utils.f0_to_coarse(x), legacy_pitch(x), rtol=0, atol=0)

    def test_pitch_bin_edges_match(self):
        lower = 1127 * np.log(1 + 50.0 / 700)
        upper = 1127 * np.log(1 + 1100.0 / 700)
        for dtype in (torch.float32, torch.float64):
            bins = torch.arange(0., 260., dtype=dtype) + 0.5
            f0 = 700 * (torch.exp(((bins - 1) * (upper - lower) / 254 + lower) / 1127) - 1)
            f0 = torch.stack((f0 - 1e-6, f0, f0 + 1e-6))
            torch.testing.assert_close(utils.f0_to_coarse(f0), legacy_pitch(f0), rtol=0, atol=0)

    def test_normalization_finite_and_unvoiced_match(self):
        x = torch.arange(24).float().reshape(2, 1, 12)
        uv = torch.cat((torch.ones(1, 12), torch.zeros(1, 12)))
        mask = torch.ones(2, 1, 12)
        mask[:, :, -2:] = 0
        for random_scale in (False, True):
            torch.manual_seed(81)
            expected = legacy_normalize(x, mask, uv, random_scale)
            torch.manual_seed(81)
            actual = utils.normalize_f0(x, mask, uv, random_scale)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_nan_is_an_error_not_successful_process_exit(self):
        with self.assertRaisesRegex(ValueError, 'NaN'):
            utils.normalize_f0(torch.full((1, 1, 3), float('nan')),
                               torch.ones(1, 1, 3), torch.ones(1, 3), False)


class FactoryTests(unittest.TestCase):
    def test_all_thirteen_encoder_routes(self):
        expected = {
            'vec768l12': 'ContentVec768L12', 'vec256l9': 'ContentVec256L9',
            'vec256l9-onnx': 'ContentVec256L9_Onnx', 'vec256l12-onnx': 'ContentVec256L12_Onnx',
            'vec768l9-onnx': 'ContentVec768L9_Onnx', 'vec768l12-onnx': 'ContentVec768L12_Onnx',
            'hubertsoft-onnx': 'HubertSoft_Onnx', 'hubertsoft': 'HubertSoft',
            'whisper-ppg': 'WhisperPPG', 'cnhubertlarge': 'CNHubertLarge',
            'dphubert': 'DPHubert', 'whisper-ppg-large': 'WhisperPPGLarge',
            'wavlmbase+': 'WavLMBasePlus',
        }
        self.assertEqual(set(utils._SPEECH_ENCODERS), set(expected))
        for key, name in expected.items():
            with self.subTest(key=key):
                factory = MagicMock(return_value=object())
                with patch.object(utils.importlib, 'import_module', return_value=SimpleNamespace(**{name: factory})) as imp:
                    result = utils.get_speech_encoder(key, device='cpu', unused_argument=123)
                imp.assert_called_once_with('vencoder.' + name)
                factory.assert_called_once_with(device='cpu')
                self.assertIs(result, factory.return_value)

    def test_all_six_pitch_routes_and_kwargs(self):
        expected = {'pm': 'PMF0Predictor', 'crepe': 'CrepeF0Predictor', 'harvest': 'HarvestF0Predictor',
                    'dio': 'DioF0Predictor', 'rmvpe': 'RMVPEF0Predictor', 'fcpe': 'FCPEF0Predictor'}
        self.assertEqual(set(utils._F0_PREDICTORS), set(expected))
        for key, name in expected.items():
            with self.subTest(key=key):
                factory = MagicMock()
                with patch.object(utils.importlib, 'import_module', return_value=SimpleNamespace(**{name: factory})) as imp:
                    utils.get_f0_predictor(key, 512, 44100, device='cpu', threshold=0.05)
                kwargs = {'hop_length': 512, 'sampling_rate': 44100}
                if key in ('crepe', 'rmvpe', 'fcpe'):
                    kwargs.update(device='cpu', threshold=0.05)
                if key in ('rmvpe', 'fcpe'):
                    kwargs['dtype'] = torch.float32
                factory.assert_called_once_with(**kwargs)
                imp.assert_called_once_with('modules.F0Predictor.' + name)

    def test_cpu_pitch_route_does_not_require_device_kwargs(self):
        factory = MagicMock()
        with patch.object(utils.importlib, 'import_module', return_value=SimpleNamespace(PMF0Predictor=factory)):
            utils.get_f0_predictor('pm', 512, 44100)
        factory.assert_called_once_with(hop_length=512, sampling_rate=44100)

    def test_unknown_routes_import_nothing(self):
        with patch.object(utils.importlib, 'import_module') as imp:
            with self.assertRaisesRegex(Exception, 'Unknown speech encoder'):
                utils.get_speech_encoder('__not_an_encoder__')
            with self.assertRaisesRegex(Exception, 'Unknown f0 predictor'):
                utils.get_f0_predictor('__not_a_predictor__', 512, 44100)
            imp.assert_not_called()

    def test_backend_import_errors_are_not_hidden(self):
        with patch.object(utils.importlib, 'import_module', side_effect=ImportError('backend dependency missing')):
            with self.assertRaisesRegex(ImportError, 'backend dependency missing'):
                utils.get_speech_encoder('vec768l12')


class PublicUtilityTests(unittest.TestCase):
    def test_hparams_mapping_and_nested_types(self):
        hp = utils.HParams(train={'batch_size': 2}, names=['a', 'b'])
        self.assertEqual(hp.train.batch_size, 2)
        self.assertEqual(hp['names'], ['a', 'b'])
        self.assertEqual(len(hp), 2)
        self.assertIn('train', hp)
        hp['new'] = 7
        self.assertEqual(hp.get('new'), 7)
        self.assertEqual(set(hp.keys()), {'train', 'names', 'new'})
        self.assertEqual(dict(hp.items())['new'], 7)
        self.assertIn(7, list(hp.values()))
        with self.assertRaises(AttributeError):
            _ = hp.missing

    def test_inference_hparams_missing_is_none(self):
        hp = utils.InferHParams(model={'adapter_rank': 16})
        self.assertIsNone(hp.unknown)
        self.assertIsNone(hp.model.unknown)
        self.assertEqual(hp.model.adapter_rank, 16)

    def test_config_readers(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'config.json'
            p.write_text(json.dumps({'model': {'adapter_rank': 16}}))
            self.assertEqual(utils.get_hparams_from_dir(tmp).model_dir, tmp)
            self.assertEqual(utils.get_hparams_from_file(p).model.adapter_rank, 16)
            self.assertIsNone(utils.get_hparams_from_file(p, True).missing)

    def test_filelist_reading_preserved(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'list.txt'
            p.write_text('a.wav|0\nb.wav|1\n', encoding='utf-8')
            self.assertEqual(utils.load_filepaths_and_text(p), [['a.wav', '0'], ['b.wav', '1']])

    def test_real_wav_reading(self):
        from scipy.io.wavfile import write
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'audio.wav'
            samples = np.array([-32768, -1, 0, 1, 32767], dtype=np.int16)
            write(p, 44100, samples)
            audio, rate = utils.load_wav_to_torch(p)
            self.assertEqual(rate, 44100)
            torch.testing.assert_close(audio, torch.tensor(samples.astype(np.float32)))

    def test_repeat_expand_modes(self):
        x = torch.tensor([[1., 2., 3.]])
        expected = torch.tensor([[1., 1., 2., 2., 3., 3.]])
        torch.testing.assert_close(utils.repeat_expand_2d(x, 6), expected)
        torch.testing.assert_close(utils.repeat_expand_2d(x, 6, 'nearest'), expected)

    def test_volume_extractor(self):
        wave = torch.ones(1, 1024)
        actual = utils.Volume_Extractor(128).extract(wave)
        torch.testing.assert_close(actual, torch.ones(8))
        torch.testing.assert_close(utils.Volume_Extractor(128).extract(wave.numpy()), actual)

    def test_checkpoint_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
            opt_a = torch.optim.AdamW(a.parameters())
            opt_b = torch.optim.AdamW(b.parameters())
            a(torch.ones(1, 3)).sum().backward()
            opt_a.step()
            path = str(Path(tmp) / 'G_1.pth')
            utils.save_checkpoint(a, opt_a, 0.001, 3, path)
            _, _, lr, iteration = utils.load_checkpoint(path, b, opt_b)
            self.assertEqual((lr, iteration), (0.001, 3))
            for name, tensor in a.state_dict().items():
                torch.testing.assert_close(tensor, b.state_dict()[name], rtol=0, atol=0)
            self.assertTrue(opt_b.state_dict()['state'])

    def test_checkpoint_retention_preserves_zero_and_other_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            for prefix in ('G', 'D'):
                for index in (0, 1, 2, 3):
                    (Path(tmp) / f'{prefix}_{index}.pth').touch()
            (Path(tmp) / 'notes.txt').touch()
            utils.clean_checkpoints(tmp, n_ckpts_to_keep=2, sort_by_time=False)
            self.assertEqual({p.name for p in Path(tmp).iterdir()},
                             {'G_0.pth', 'G_2.pth', 'G_3.pth', 'D_0.pth', 'D_2.pth', 'D_3.pth', 'notes.txt'})

    def test_summary_routes_all_payloads(self):
        writer = MagicMock()
        utils.summarize(writer, 9, {'loss': 1}, {'hist': [2]}, {'image': 'rgb'}, {'audio': 'wave'}, 44100)
        writer.add_scalar.assert_called_once_with('loss', 1, 9)
        writer.add_histogram.assert_called_once_with('hist', [2], 9)
        writer.add_image.assert_called_once_with('image', 'rgb', 9, dataformats='HWC')
        writer.add_audio.assert_called_once_with('audio', 'wave', 9, 44100)

    def test_summary_without_payloads(self):
        writer = MagicMock()
        utils.summarize(writer, 0)
        self.assertEqual(writer.mock_calls, [])

    def test_cluster_results_preserved(self):
        km = SimpleNamespace(cluster_centers_=np.array([[1., 2.], [3., 4.]]),
                             predict=MagicMock(return_value=np.array([1, 0])))
        model = {'voice': km}
        query = np.zeros((2, 2))
        np.testing.assert_array_equal(cluster.get_cluster_result(model, query, 'voice'), [1, 0])
        np.testing.assert_array_equal(cluster.get_cluster_center_result(model, query, 'voice'), [[3, 4], [1, 2]])
        np.testing.assert_array_equal(cluster.get_center(model, [0], 'voice'), [[1, 2]])

    def test_rms_adjustment_with_fake_analysis_backend(self):
        import types
        librosa = types.ModuleType('librosa')
        librosa.feature = SimpleNamespace(rms=MagicMock(side_effect=[np.ones((1, 3), dtype=np.float32),
                                                                    np.full((1, 3), 2., dtype=np.float32)]))
        with patch.dict(sys.modules, {'librosa': librosa}):
            actual = utils.change_rms(np.zeros(6), 4, torch.ones(6) * 2, 4, 0.)
        torch.testing.assert_close(actual, torch.ones(6))

    def test_small_index_does_not_import_sklearn(self):
        import types
        faiss = types.ModuleType('faiss')
        index = MagicMock()
        index_ivf = SimpleNamespace(nprobe=0)
        faiss.index_factory = MagicMock(return_value=index)
        faiss.extract_index_ivf = MagicMock(return_value=index_ivf)
        original_import = builtins.__import__
        def reject_sklearn(name, *args, **kwargs):
            if name.startswith('sklearn'):
                raise AssertionError('small index should not load sklearn')
            return original_import(name, *args, **kwargs)
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / 'voice'
            p.mkdir()
            torch.save(torch.randn(1, 3, 100), p / 'sample.wav.soft.pt')
            with patch.dict(sys.modules, {'faiss': faiss}), patch('builtins.__import__', reject_sklearn):
                self.assertIs(utils.train_index('voice', tmp), index)
        faiss.index_factory.assert_called_once_with(3, 'IVF2,Flat')
        self.assertEqual(index_ivf.nprobe, 1)
        index.train.assert_called_once()
        index.add.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec('matplotlib'), 'Matplotlib is required for plot tests')
class PlotTests(unittest.TestCase):
    def assert_image(self, image, shape):
        self.assertEqual(image.shape, shape)
        self.assertEqual(image.dtype, np.uint8)
        self.assertTrue(image.flags.owndata)
        self.assertTrue(image.flags.c_contiguous)

    def test_all_three_public_plot_functions(self):
        import matplotlib
        with matplotlib.rc_context({'figure.dpi': 100}):
            self.assert_image(utils.plot_data_to_numpy(np.arange(8), np.arange(8) * 2), (200, 1000, 3))
            self.assert_image(utils.plot_spectrogram_to_numpy(np.zeros((5, 8))), (200, 1000, 3))
            self.assert_image(utils.plot_alignment_to_numpy(np.ones((5, 8)), 'test'), (400, 600, 3))
        import matplotlib.pyplot as plt
        self.assertEqual(plt.get_fignums(), [])

    def test_pixels_unchanged_by_next_plot(self):
        a = utils.plot_spectrogram_to_numpy(np.eye(6))
        saved = a.copy()
        _ = utils.plot_spectrogram_to_numpy(np.zeros((5, 5)))
        np.testing.assert_array_equal(a, saved)

    def test_canvas_failure_closes_figure(self):
        import matplotlib.pyplot as plt
        fig = plt.figure()
        with patch.object(fig.canvas, 'draw', side_effect=RuntimeError('draw failed')):
            with self.assertRaisesRegex(RuntimeError, 'draw failed'):
                utils._figure_to_numpy(fig, plt)
        self.assertNotIn(fig.number, plt.get_fignums())


if __name__ == '__main__':
    unittest.main()
