"""CPU integration tests with actual So-VITS prior, flow, posterior and NSF decoder.

Synthetic tensors/random weights test mechanics, NOT perceptual singing quality.
Run: python -m unittest discover -s tests -p 'test_architecture.py' -v
"""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import SynthesizerTrn, SynthesizerInfer
from modules.voice_adapter import VoiceAdapter, configure_trainable, optimizer_groups
from modules.model_io import (build_model, compatible_state, export_adapter, export_runtime,
                              initialize_generator, load_adapter, load_runtime,
                              model_arguments, normalize_state, save_exclusive)
from modules.training_setup import prepare_generator


def config(rank=0, auto_f0=True):
    return {'train': {'segment_size': 16},
            'data': {'filter_length': 16, 'sampling_rate': 16000, 'hop_length': 4},
            'model': dict(inter_channels=8, hidden_channels=8, filter_channels=16,
                          n_heads=2, n_layers=1, kernel_size=3, p_dropout=.1,
                          resblock='1', resblock_kernel_sizes=[3], resblock_dilation_sizes=[[1,3,5]],
                          upsample_rates=[2,2], upsample_initial_channel=16, upsample_kernel_sizes=[4,4],
                          gin_channels=4, ssl_dim=6, n_speakers=3, n_flow_layer=2,
                          use_automatic_f0_prediction=auto_f0, adapter_rank=rank),
            'spk': {'voice': 0}}


def inputs(batch=1, length=8):
    return (torch.randn(batch,6,length), torch.full((batch,length),220.),
            torch.ones(batch,length), torch.zeros(batch,1,dtype=torch.long))


def waveform(model, data=None):
    c,f0,uv,g = inputs() if data is None else data
    return model.eval().infer(c,f0,uv,g=g,seed=42)[0]


def save_base(path, cfg):
    model=build_model(cfg)
    torch.save({'model':model.state_dict()}, path)
    return model


def synthetic_step(model, optimizer, batch=2):
    c,f0,uv,g=inputs(batch)
    lengths=torch.full((batch,),8)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    y,_,mask,latent,*_=model(c,f0,uv,torch.randn(batch,9,8),g=g,
                            c_lengths=lengths,spec_lengths=lengths)
    z,zp,mp,lp,mq,lq=latent
    # Differentiable waveform loss + prior alignment; the full GAN has separate tests.
    loss=y.square().mean()+(mp-zp.detach()).square().mean()
    loss.backward()
    optimizer.step()
    return float(loss.detach())


class ArchitectureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def setUp(self):
        torch.manual_seed(67)

    def test_legacy_checkpoint_shapes_are_preserved(self):
        cfg=config()
        model=build_model(cfg)
        self.assertEqual(model.enc_q.enc.n_layers,16)
        cfg['model']['n_layers_q']=3
        legacy_template=build_model(cfg)
        self.assertEqual(legacy_template.enc_q.enc.n_layers,16)
        self.assertEqual({k:tuple(v.shape) for k,v in model.state_dict().items()},
                         {k:tuple(v.shape) for k,v in legacy_template.state_dict().items()})

    def test_explicit_posterior_depth_controls_actual_network(self):
        cfg=config();cfg['model']['posterior_layers']=3
        self.assertEqual(build_model(cfg).enc_q.enc.n_layers,3)

    def test_invalid_dimensions_rejected(self):
        for field,value in [('posterior_layers',0),('posterior_layers',True),('adapter_rank',-1),('adapter_rank',1.5)]:
            cfg=config();cfg['model'][field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):build_model(cfg)

    def test_inference_never_allocates_posterior(self):
        m=build_model(config(),inference=True)
        self.assertFalse(hasattr(m,'enc_q'))
        self.assertFalse(any(k.startswith('enc_q.') for k in m.state_dict()))

    def test_runtime_refuses_training_and_forward(self):
        m=build_model(config(),inference=True)
        with self.assertRaises(RuntimeError):m.train()
        c,f0,uv,g=inputs()
        with self.assertRaises(RuntimeError):m(c,f0,uv,None,g=g)
        with self.assertRaises(ValueError):configure_trainable(m)

    def test_compact_and_full_waveforms_match_exactly(self):
        for keep in (False,True):
            with self.subTest(keep_f0=keep):
                cfg=config();full=build_model(cfg);compact=build_model(cfg,inference=True,keep_f0=keep)
                kept,_=compatible_state(compact,full.state_dict())
                compact.load_state_dict(kept,strict=True)
                data=inputs()
                torch.testing.assert_close(waveform(full,data),waveform(compact,data),rtol=0,atol=0)

    def test_optional_f0_prediction_keeps_outputs(self):
        full=build_model(config()).eval();small=build_model(config(),inference=True)
        small.load_state_dict(compatible_state(small,full.state_dict())[0],strict=True)
        c,f0,uv,g=inputs()
        a=full.infer(c,f0,uv,g=g,predict_f0=True)
        b=small.infer(c,f0,uv,g=g,predict_f0=True)
        for left,right in zip(a,b):torch.testing.assert_close(left,right,rtol=0,atol=0)

    def test_pruned_f0_is_not_silently_ignored(self):
        m=build_model(config(),inference=True,keep_f0=False)
        c,f0,uv,g=inputs()
        with self.assertRaises(ValueError):m.infer(c,f0,uv,g=g,predict_f0=True)

    def test_zero_initialized_adapters_preserve_output(self):
        cfg=config();base=build_model(cfg)
        cfg['model']['adapter_rank']=4;adapt=build_model(cfg)
        kept,missing=compatible_state(adapt,base.state_dict(),allow_new_adapters=True)
        adapt.load_state_dict(kept,strict=False)
        self.assertTrue(missing)
        data=inputs()
        torch.testing.assert_close(waveform(base,data),waveform(adapt,data),rtol=0,atol=0)

    def test_adapter_gradients_and_frozen_backbone(self):
        m=build_model(config(4,False));report=configure_trainable(m,'adapters')
        self.assertLess(report['trainable_generator_parameters'],report['generator_parameters'])
        frozen={k:v.detach().clone() for k,v in m.named_parameters() if not v.requires_grad}
        optimizer=torch.optim.AdamW(optimizer_groups(m),lr=.01)
        for _ in range(3):self.assertTrue(torch.isfinite(torch.tensor(synthetic_step(m,optimizer))))
        for name,p in m.named_parameters():
            if name in frozen:
                self.assertIsNone(p.grad);torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)
        self.assertGreater(m.prior_adapter.up.weight.grad.abs().sum().item(),0)
        self.assertGreater(m.decoder_adapter.up.weight.grad.abs().sum().item(),0)
        self.assertEqual(len(optimizer.state),sum(1 for p in m.parameters() if p.requires_grad))

    def test_adapter_mode_disables_frozen_dropout(self):
        m=build_model(config(4));configure_trainable(m,'adapters');m.train()
        self.assertTrue(m.training);self.assertTrue(m.prior_adapter.training)
        self.assertFalse(m.enc_p.training);self.assertFalse(m.enc_q.training)
        configure_trainable(m,'full')
        self.assertTrue(m.enc_p.training)
        self.assertTrue(all(p.requires_grad for p in m.parameters()))

    def test_unused_speaker_rows_do_not_decay(self):
        m=build_model(config(4,False));configure_trainable(m,'adapters+speaker')
        unused=m.emb_g.weight[1:].detach().clone()
        optimizer=torch.optim.AdamW(optimizer_groups(m,weight_decay=.5),lr=.01)
        synthetic_step(m,optimizer)
        torch.testing.assert_close(m.emb_g.weight[1:],unused,rtol=0,atol=0)

    def test_no_adapters_mode_fails_explicitly(self):
        with self.assertRaises(ValueError):configure_trainable(build_model(config()),'adapters')
        with self.assertRaises(ValueError):configure_trainable(build_model(config(4)),'unknown')

    def test_pointwise_adapter_crop_consistency(self):
        a=VoiceAdapter(8,4,4)
        nn.init.normal_(a.up.weight)
        x=torch.randn(2,8,12);f0=torch.rand(2,12)*400;g=torch.randn(2,4,1)
        torch.testing.assert_close(a(x,f0,g)[:,:,3:9],a(x[:,:,3:9],f0[:,3:9],g),rtol=0,atol=0)

    def test_adapter_mask_preserves_padding(self):
        a=VoiceAdapter(8,4,4);nn.init.normal_(a.up.weight)
        x=torch.randn(1,8,6);mask=torch.tensor([[[1,1,1,0,0,0]]])
        y=a(x,torch.ones(1,6)*220,torch.randn(1,4,1),mask=mask)
        torch.testing.assert_close(x[:,:,3:],y[:,:,3:],rtol=0,atol=0)

    def test_dynamic_speaker_condition_and_volume(self):
        a=VoiceAdapter(8,4,4);nn.init.normal_(a.up.weight)
        x=torch.randn(2,8,8);g=torch.randn(2,4,8);f0=torch.ones(2,8)*220
        y=a(x,f0,g,uv=torch.ones_like(f0),volume=torch.ones_like(f0))
        self.assertEqual(y.shape,x.shape);self.assertTrue(torch.isfinite(y).all())

    def test_bad_alignment_rejected(self):
        a=VoiceAdapter(8,4,4);x=torch.ones(1,8,6);f0=torch.ones(1,6);g=torch.ones(1,4,1)
        with self.assertRaises(ValueError):a(x,f0[:,:3],g)
        with self.assertRaises(ValueError):a(x,f0,torch.ones(1,4,3))
        with self.assertRaises(ValueError):a(x,f0,g,volume=torch.ones(1,3))
        with self.assertRaises(ValueError):a(x,f0,g,mask=torch.ones(1,6))

    def test_transformer_flow_runtime(self):
        cfg=config(4,False);cfg['model'].update(use_transformer_flow=True,n_layers_trans_flow=1)
        m=build_model(cfg,inference=True)
        self.assertTrue(torch.isfinite(waveform(m)).all())

    def test_depthwise_flow_runtime(self):
        cfg=config(4,False);cfg['model']['use_depthwise_conv']=True
        self.assertTrue(torch.isfinite(waveform(build_model(cfg,inference=True))).all())

    def test_shared_flow_runtime(self):
        cfg=config(4,False);cfg['model']['flow_share_parameter']=True
        self.assertTrue(torch.isfinite(waveform(build_model(cfg,inference=True))).all())

    def test_volume_embedded_full_model(self):
        cfg=config(4,False);cfg['model']['vol_embedding']=True
        m=build_model(cfg,inference=True);c,f0,uv,g=inputs()
        self.assertTrue(torch.isfinite(m.infer(c,f0,uv,g=g,vol=torch.rand_like(f0))[0]).all())

    def test_core_import_has_no_retrieval_or_plotting_dependency(self):
        script='''import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self, fullname, path=None, target=None):
  if fullname.split('.')[0] in {'faiss','librosa','fairseq','sklearn','matplotlib'}:
   raise RuntimeError('Unwanted dependency: '+fullname)
sys.meta_path.insert(0,Block())
from models import SynthesizerTrn
from vdecoder.hifigan.models import Generator
'''
        subprocess.run([sys.executable,'-c',script],cwd=ROOT,check=True,capture_output=True,timeout=30)

    def test_original_model_state_and_waveform_equivalence(self):
        path=os.environ.get('SVC_BASELINE_MODELS')
        if path:
            source=Path(path).read_bytes()
        else:
            try:source=subprocess.check_output(['git','show','730930d337d171479eadf305f96cbed4bb393e77:models.py'],cwd=ROOT,stderr=subprocess.DEVNULL)
            except (OSError,subprocess.CalledProcessError):self.skipTest('Original Git baseline unavailable')
        self.assertEqual(hashlib.sha1(b'blob '+str(len(source)).encode()+b'\0'+source).hexdigest(),
                         '24338fa2c1f6c15e60f5f341c7e3df2301f74eb8')
        # Redirect only extracted pitch helpers, not any neural component.
        code=source.decode().replace('import utils\n','import modules.model_utils as utils\n').replace('from utils import f0_to_coarse','from modules.model_utils import f0_to_coarse')
        scope={};exec(compile(code,'baseline_models.py','exec'),scope)
        kwargs=model_arguments(config());kwargs.pop('inference_only')
        old=scope['SynthesizerTrn'](**kwargs);new=build_model(config())
        new.load_state_dict(old.state_dict(),strict=True)
        data=inputs()
        torch.testing.assert_close(waveform(old,data),waveform(new,data),rtol=0,atol=0)


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1);torch.manual_seed(91)
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.cfg=config();self.base=self.root/'base.pth';self.original=save_base(self.base,self.cfg)
    def tearDown(self):self.tmp.cleanup()

    def test_runtime_round_trip(self):
        out=self.root/'runtime.pth';report=export_runtime(self.base,self.cfg,out)
        loaded,cfg=load_runtime(out);data=inputs()
        torch.testing.assert_close(waveform(self.original,data),waveform(loaded,data),rtol=0,atol=0)
        self.assertLess(report['runtime_state_bytes'],report['source_generator_state_bytes'])
        self.assertFalse(hasattr(loaded,'enc_q'));self.assertFalse(hasattr(loaded,'f0_decoder'))

    def test_export_refuses_overwrite(self):
        out=self.root/'exists';out.write_text('keep me')
        with self.assertRaises(FileExistsError):export_runtime(self.base,self.cfg,out)
        self.assertEqual(out.read_text(),'keep me')

    def test_initializer_validates_before_mutation(self):
        model=build_model(self.cfg);before={k:v.clone() for k,v in model.state_dict().items()}
        state=self.original.state_dict();state['pre.weight']=torch.ones(3,2,1)
        bad=self.root/'bad.pth';torch.save({'model':state},bad)
        with self.assertRaises(ValueError):initialize_generator(model,bad)
        for k,v in model.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)

    def test_partial_adapter_state_is_rejected(self):
        cfg=config(4);model=build_model(cfg);state=dict(self.original.state_dict())
        state['prior_adapter.up.weight']=model.prior_adapter.up.weight.detach().clone()
        with self.assertRaises(ValueError):compatible_state(model,state,True)

    def test_adapter_round_trip_and_base_binding(self):
        cfg=config(4);model=build_model(cfg)
        prepare_generator(model,SimpleNamespace(finetune_mode='adapters',init_generator=self.base))
        optimizer=torch.optim.AdamW(optimizer_groups(model),lr=.01)
        for _ in range(2):synthetic_step(model,optimizer)
        out=self.root/'voice.pth';report=export_adapter(model,self.base,cfg,out)
        self.assertEqual(report['delta_parameters'],224)
        loaded,_=load_adapter(self.base,out);data=inputs()
        torch.testing.assert_close(waveform(model,data),waveform(loaded,data),rtol=0,atol=0)
        other=self.root/'other.pth';save_base(other,self.cfg)
        with self.assertRaisesRegex(ValueError,'different base'):load_adapter(other,out)

    def test_delta_refuses_backbone_drift(self):
        cfg=config(4);model=build_model(cfg);initialize_generator(model,self.base,True)
        configure_trainable(model,'adapters')
        with torch.no_grad():model.pre.weight.add_(1)
        with self.assertRaisesRegex(ValueError,'Backbone changed'):
            export_adapter(model,self.base,cfg,self.root/'voice.pth')

    def test_full_finetune_cannot_masquerade_as_delta(self):
        with self.assertRaises(ValueError):export_adapter(self.original,self.base,self.cfg,self.root/'voice.pth')

    def test_adapter_training_without_base_is_rejected(self):
        with self.assertRaises(ValueError):prepare_generator(build_model(config(4)),SimpleNamespace(finetune_mode='adapters'))

    def test_runtime_corrupt_key_is_rejected(self):
        out=self.root/'runtime.pth';export_runtime(self.base,self.cfg,out)
        payload=torch.load(out,weights_only=True);payload['model'].pop('pre.weight')
        torch.save(payload,out)
        with self.assertRaises(ValueError):load_runtime(out)

    def test_malformed_config_rejected(self):
        for key,value in [('hop_length',3),('sampling_rate',12345)]:
            cfg=copy.deepcopy(self.cfg);cfg['data'][key]=value
            if key=='sampling_rate':cfg['model']['sampling_rate']=16000
            with self.assertRaises(ValueError):model_arguments(cfg)
        cfg=copy.deepcopy(self.cfg);cfg['spk']['bad']=3
        with self.assertRaises(ValueError):model_arguments(cfg)

    def test_ddp_keys_and_mixed_keys(self):
        state=self.original.state_dict()
        self.assertEqual(set(normalize_state({'module.'+k:v for k,v in state.items()})),set(state))
        with self.assertRaises(ValueError):normalize_state({'module.a':torch.ones(1),'b':torch.ones(1)})
        with self.assertRaises(ValueError):normalize_state({'bad':'not a tensor'})

    def test_runtime_artifact_is_not_training_checkpoint(self):
        out=self.root/'runtime.pth';export_runtime(self.base,self.cfg,out)
        with self.assertRaises(ValueError):initialize_generator(self.original,out)

    def test_artifact_cli(self):
        config_path=self.root/'config.json';config_path.write_text(json.dumps(self.cfg))
        out=self.root/'cli-runtime.pth'
        result=subprocess.run([sys.executable,'scripts/svc_artifacts.py','runtime','--checkpoint',str(self.base),
                               '--config',str(config_path),'--output',str(out)],cwd=ROOT,capture_output=True,text=True,timeout=30)
        self.assertEqual(result.returncode,0,result.stderr);self.assertTrue(out.exists())
        self.assertIn('runtime_state_bytes',json.loads(result.stdout))


if __name__=='__main__':unittest.main()
