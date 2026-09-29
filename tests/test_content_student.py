"""CPU mechanics tests. Synthetic tensors are NOT singing-quality evidence."""
import copy
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from research.content_student import (ContentStudent, StudentConfig, align_teacher,
    frame_lengths, frame_times, pool_phoneme_spans, export_student, load_student)
from research.content_training import StudentTask, TeacherSpec, reverse_gradient

spec = importlib.util.spec_from_file_location('student_runner', ROOT / 'scripts/train_content_student.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
torch.set_num_threads(2)


def small():
    return StudentConfig(width=24, content_dim=16, performance_dim=8, blocks=3, heads=4)


def teachers():
    return [TeacherSpec('cvec', 8, '0' * 64, .02, 199.5 / 16000),
            TeacherSpec('phone', 5, '1' * 64, .02, 199.5 / 16000, kind='posterior')]


def record(samples=2640, group='song-a'):
    length = int(frame_lengths(torch.tensor([samples]))[0])
    entries = {}
    for t in teachers():
        values = torch.randn(length, t.dim)
        if t.kind == 'posterior':
            values = values.softmax(-1)
        entries[t.name] = {'features': values, 'checkpoint_sha256': t.checkpoint_sha256,
                          'hop_seconds': t.hop_seconds, 'offset_seconds': t.offset_seconds, 'kind': t.kind}
    return {'sample_rate': 16000, 'waveform': torch.randn(samples) * .1,
            'source_group': group, 'teachers': entries}


class StudentTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(10)
        self.model = ContentStudent(small()).eval()

    def test_exact_frame_lengths(self):
        values = torch.tensor([399, 400, 719, 720, 16000])
        self.assertEqual(frame_lengths(values).tolist(), [0, 1, 1, 2, 49])

    def test_physical_grid(self):
        torch.testing.assert_close(frame_times(3), torch.tensor([.01246875, .03246875, .05246875], dtype=torch.float64))

    def test_shapes_and_finite(self):
        result = self.model(torch.randn(2, 4000))
        self.assertEqual(result['content'].shape, (2, 12, 16))
        self.assertEqual(result['performance'].shape, (2, 12, 8))
        self.assertTrue(torch.isfinite(result['content']).all())

    def test_padding_does_not_change_valid_outputs(self):
        a, b = torch.randn(1360), torch.randn(2640)
        packed = torch.stack([F.pad(a, (0, b.numel() - a.numel()), value=92.), b])
        with torch.no_grad():
            alone = self.model(a[None])
            together = self.model(packed, torch.tensor([a.numel(), b.numel()]))
        n = alone['content'].shape[1]
        for key in ('content', 'performance'):
            torch.testing.assert_close(alone[key][0], together[key][0, :n], atol=2e-6, rtol=2e-5)
            self.assertEqual(int(together[key][0, n:].count_nonzero()), 0)

    def test_bad_inputs(self):
        for x in (torch.randn(399)[None], torch.ones(1, 500, dtype=torch.long), torch.randn(500), torch.zeros(0, 500)):
            with self.subTest(shape=x.shape), self.assertRaises(ValueError):
                self.model(x)
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 800), sample_rate=44100)
        with self.assertRaises(ValueError):
            self.model(torch.full((1, 800), float('nan')))
        with self.assertRaises(ValueError):
            self.model(torch.randn(1, 800), torch.tensor([800.]))

    def test_configuration_validation(self):
        for args in ({'width': 23}, {'blocks': 0}, {'sample_rate': 48000}, {'heads': True}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                StudentConfig(**args)

    def test_no_hidden_rng_reseeding(self):
        x = torch.randn(1, 800)
        state = torch.get_rng_state().clone()
        self.model(x)
        self.assertTrue(torch.equal(state, torch.get_rng_state()))

    def test_teacher_free_inference_import(self):
        code = """import sys, importlib.abc
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'fairseq','transformers','librosa','torchaudio'}:
   raise AssertionError('Unexpected runtime teacher dependency: ' + fullname)
sys.meta_path.insert(0, Block())
from research.content_student import ContentStudent, StudentConfig
import torch
torch.set_num_threads(1)
m = ContentStudent(StudentConfig(width=24,content_dim=16,performance_dim=8,blocks=1))
assert m(torch.zeros(1,800))['content'].shape == (1,2,16)
"""
        subprocess.run([sys.executable, '-c', code], cwd=ROOT, check=True, timeout=20)


class AlignmentTests(unittest.TestCase):
    def test_known_interpolation_and_extrapolation_mask(self):
        x = torch.tensor([[0., 1.], [2., 3.], [4., 5.]])
        y, mask = align_teacher(x, .02, .01, torch.tensor([-.01, .01, .02, .05, .07], dtype=torch.float64))
        self.assertEqual(mask.tolist(), [False, True, True, True, False])
        torch.testing.assert_close(y[2], torch.tensor([1., 2.]))

    def test_alignment_does_not_stretch_to_fill(self):
        x = torch.arange(10.).reshape(5, 2)
        _, mask = align_teacher(x, .02, 0., torch.arange(10).double() * .02)
        self.assertEqual(int(mask.sum()), 5)

    def test_alignment_validation(self):
        for hop in (0., float('nan'), -.02):
            with self.assertRaises(ValueError):
                align_teacher(torch.ones(3, 2), hop, 0., frame_times(2))
        with self.assertRaises(ValueError):
            align_teacher(torch.empty(0, 2), .02, 0., frame_times(2))

    def test_pool_occurrences_not_phone_classes(self):
        x = torch.tensor([[0.], [2.], [6.], [8.], [10.]])
        ids = torch.tensor([0, 0, 1, 2, 2])
        pooled = pool_phoneme_spans(x, ids)
        torch.testing.assert_close(pooled[:, 0], torch.tensor([1., 1., 6., 9., 9.]))
        torch.testing.assert_close(x[:, 0], torch.tensor([0., 2., 6., 8., 10.]))

    def test_repeated_or_noncontiguous_ids_rejected(self):
        for ids in ([0, 1, 0], [0, -1, 0], [-2, 0, 1]):
            with self.assertRaises(ValueError):
                pool_phoneme_spans(torch.randn(3, 2), torch.tensor(ids))

    def test_unknown_spans_unchanged(self):
        x = torch.randn(3, 2)
        torch.testing.assert_close(pool_phoneme_spans(x, torch.tensor([-1, -1, -1])), x)


class ObjectiveTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.task = StudentTask(small(), teachers())
        self.rec = record()

    def test_two_teachers_receive_gradients(self):
        terms = self.task.loss(self.rec)
        terms['total'].backward()
        for head in self.task.readouts.values():
            self.assertGreater(float(head.weight.grad.abs().sum()), 0.)
        self.assertGreater(float(self.task.student.convs[0].weight.grad.abs().sum()), 0.)

    def test_teacher_targets_are_detached(self):
        self.rec['teachers']['cvec']['features'].requires_grad_()
        self.task.loss(self.rec)['total'].backward()
        self.assertIsNone(self.rec['teachers']['cvec']['features'].grad)

    def test_fingerprint_mismatch(self):
        self.rec['teachers']['cvec']['checkpoint_sha256'] = '9' * 64
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.task.loss(self.rec)

    def test_no_time_overlap(self):
        specs = [TeacherSpec('cvec', 8, '0'*64, .02, 99.)]
        task = StudentTask(small(), specs)
        self.rec['teachers'].pop('phone')
        self.rec['teachers']['cvec']['offset_seconds'] = 99.
        with self.assertRaisesRegex(ValueError, 'overlapping'):
            task.loss(self.rec)

    def test_logits_not_misread_as_probabilities(self):
        self.rec['teachers']['phone']['features'] *= 2.
        with self.assertRaisesRegex(ValueError, 'probability'):
            self.task.loss(self.rec)

    def test_unverified_pairs_rejected(self):
        self.rec['paired_waveform'] = self.rec['waveform'].clone()
        with self.assertRaisesRegex(ValueError, 'verified'):
            self.task.loss(self.rec)

    def test_identical_pair_loss_zero(self):
        self.rec['paired_waveform'] = self.rec['waveform'].clone()
        self.rec['pair_kind'] = 'verified_timbre_only_same_timing'
        self.assertEqual(float(self.task.loss(self.rec)['paired_content'].detach()), 0.)

    def test_pair_loss_does_not_constrain_performance_head(self):
        self.rec['paired_waveform'] = self.rec['waveform'] * .7
        self.rec['pair_kind'] = 'verified_timbre_only_same_timing'
        self.task.loss(self.rec)['paired_content'].backward()
        self.assertTrue(all(p.grad is None for p in self.task.student.performance.parameters()))

    def test_prosody_branch_receives_gradients(self):
        n = int(frame_lengths(torch.tensor([self.rec['waveform'].numel()]))[0])
        self.rec['prosody'] = torch.tensor([220., 1., .1]).repeat(n, 1)
        self.task.loss(self.rec)['total'].backward()
        self.assertGreater(float(self.task.student.performance[1].weight.grad.abs().sum()), 0.)

    def test_all_unvoiced_finite(self):
        n = int(frame_lengths(torch.tensor([self.rec['waveform'].numel()]))[0])
        self.rec['prosody'] = torch.zeros(n, 3)
        terms = self.task.loss(self.rec)
        terms['total'].backward()
        self.assertTrue(torch.isfinite(terms['total']))

    def test_reverse_gradient_sign(self):
        x = torch.tensor([2.], requires_grad=True)
        reverse_gradient(x, .3).square().sum().backward()
        torch.testing.assert_close(x.grad, torch.tensor([-1.2]))

    def test_speaker_adversary_runs(self):
        task = StudentTask(small(), teachers(), speaker_count=3)
        self.rec['speaker_id'] = 1
        task.loss(self.rec, adversarial_scale=.1)['total'].backward()
        self.assertGreater(float(task.speaker.weight.grad.abs().sum()), 0.)

    def test_loss_weights_rejected(self):
        with self.assertRaises(ValueError):
            self.task.loss(self.rec, pair_weight=-1.)

    def test_synthetic_objective_can_be_optimized(self):
        optimizer = torch.optim.Adam(self.task.parameters(), lr=.002)
        initial = float(self.task.loss(self.rec)['total'].detach())
        for _ in range(30):
            optimizer.zero_grad(set_to_none=True)
            loss = self.task.loss(self.rec)['total']
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.task.parameters(), 1.)
            optimizer.step()
        final = float(self.task.loss(self.rec)['total'].detach())
        self.assertLess(final, initial * .8)
        print('SYNTHETIC_ONLY objective initial=%.6f final=%.6f; not voice-quality evidence' % (initial, final))


class ArtifactAndRunnerTests(unittest.TestCase):
    def test_roundtrip_and_no_overwrite(self):
        model = ContentStudent(small()).eval()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / 'student.pt'
            export_student(model, target, training_steps=1, provenance={'run_label': 'SYNTHETIC_TEST'})
            loaded, meta = load_student(target)
            x = torch.randn(1, 1200)
            torch.testing.assert_close(model(x)['content'], loaded(x)['content'], rtol=0, atol=0)
            self.assertEqual(meta['run_label'], 'SYNTHETIC_TEST')
            with self.assertRaises(FileExistsError):
                export_student(model, target, training_steps=1, provenance={'test': True})

    def test_reject_zero_step_export(self):
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(ValueError):
            export_student(ContentStudent(small()), Path(directory)/'a.pt', training_steps=0, provenance={'test': True})

    def test_legacy_artifact_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'legacy.pt'
            torch.save({'model': {}}, path)
            with self.assertRaises(ValueError):
                load_student(path)

    def test_split_group_leakage(self):
        with self.assertRaisesRegex(ValueError, 'leakage'):
            runner.validate_splits([(Path('a'), 'song')], [(Path('b'), 'song')])

    def test_manifest_missing_or_duplicate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'train.jsonl'
            manifest.write_text(json.dumps({'path': 'missing.pt', 'source_group':'song'})+'\n')
            with self.assertRaises(ValueError):
                runner.load_manifest(manifest)
            (root/'missing.pt').touch()
            manifest.write_text(manifest.read_text()*2)
            with self.assertRaises(ValueError):
                runner.load_manifest(manifest)

    def test_end_to_end_synthetic_training_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for split, count in [('train', 3), ('valid', 1)]:
                lines = []
                for i in range(count):
                    group = split + str(i)
                    torch.save(record(1360, group), root / (group+'.pt'))
                    lines.append(json.dumps({'path':group+'.pt', 'source_group':group}))
                (root/(split+'.jsonl')).write_text('\n'.join(lines)+'\n')
            config = {'student':asdict(small()),'teachers':[asdict(t) for t in teachers()],
                      'epochs':1,'accumulation':2,'run_label':'SYNTHETIC_CLI_TEST'}
            (root/'cfg.json').write_text(json.dumps(config))
            args = SimpleNamespace(config=str(root/'cfg.json'),train=str(root/'train.jsonl'),
                                   valid=str(root/'valid.jsonl'),output=str(root/'output'),device='cpu')
            runner.run(args)
            report = json.loads((root/'output/metrics.jsonl').read_text())
            self.assertEqual(report['optimizer_updates'], 2)
            self.assertFalse(report['audio_quality_measured'])
            loaded, meta = load_student(root/'output/student_epoch_001.pt')
            self.assertEqual(meta['run_label'], 'SYNTHETIC_CLI_TEST')
            self.assertTrue(torch.isfinite(loaded(torch.zeros(1,800))['content']).all())


if __name__ == '__main__':
    unittest.main()
