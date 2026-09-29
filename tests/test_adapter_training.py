"""Real small-generator CPU DDP and trainer-wiring checks, not CUDA/audio scoring."""
import ast
import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from test_architecture import config, inputs, build_model
from modules.training_setup import prepare_generator
from modules.voice_adapter import configure_trainable, optimizer_groups

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('adapter_config_tool', ROOT / 'scripts/make_adapter_config.py')
config_tool = importlib.util.module_from_spec(spec)
spec.loader.exec_module(config_tool)


def ddp_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(87 + rank)
        model = build_model(config(4,False))
        configure_trainable(model, 'adapters')
        wrapped = DDP(model)
        optimizer = torch.optim.AdamW(optimizer_groups(model),lr=.001)
        frozen = {k:p.detach().clone() for k,p in model.named_parameters() if not p.requires_grad}
        for _ in range(3):
            optimizer.zero_grad(set_to_none=True)
            c,f0,uv,g = inputs(batch=2)
            lengths = torch.full((2,),8)
            y,_,_,latent,*_ = wrapped(c,f0,uv,torch.randn(2,9,8),g=g,
                                      c_lengths=lengths,spec_lengths=lengths)
            _,zp,mean,logs,*_ = latent
            (y.square().mean()+(mean-zp.detach()).square().mean()+logs.square().mean()).backward()
            optimizer.step()
        for name,p in model.named_parameters():
            if name in frozen:
                torch.testing.assert_close(p,frozen[name],rtol=0,atol=0)
                assert p.grad is None
        values = torch.cat([p.detach().flatten() for p in model.parameters() if p.requires_grad])
        replicas = [torch.empty_like(values) for _ in range(2)]
        dist.all_gather(replicas,values)
        torch.testing.assert_close(replicas[0],replicas[1],rtol=0,atol=0)
    finally:
        dist.destroy_process_group()


class AdapterTrainingTests(unittest.TestCase):
    def test_config_is_opt_in_and_preserves_source(self):
        cfg=config();snapshot=copy.deepcopy(cfg)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base.pth';base.write_bytes(b'placeholder-not-loaded-by-config-tool')
            result=config_tool.derive_adapter_config(cfg,base)
        self.assertEqual(cfg,snapshot)
        self.assertEqual(result['spk'],cfg['spk'])
        self.assertEqual(result['data'],cfg['data'])
        self.assertEqual(result['model']['adapter_rank'],16)
        self.assertFalse(result['model']['use_automatic_f0_prediction'])
        self.assertEqual(result['train']['finetune_mode'],'adapters')

    def test_config_rejects_missing_base(self):
        with self.assertRaises(ValueError):config_tool.derive_adapter_config(config(),'/missing/base.pth')

    def test_explicit_init_selects_only_adapter_parameters(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)/'base.pth'
            torch.save({'model':build_model(config()).state_dict()},base)
            model=build_model(config(4,False))
            report=prepare_generator(model,SimpleNamespace(init_generator=str(base),finetune_mode='adapters'))
        self.assertTrue(report['initialized_adapter_keys'])
        selected={id(p) for group in optimizer_groups(model) for p in group['params']}
        self.assertEqual(selected,{id(p) for p in model.parameters() if p.requires_grad})

    def test_trainer_prepares_before_optimizer_and_ddp(self):
        tree=ast.parse((ROOT/'train.py').read_text())
        run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='run')
        calls=[n for n in ast.walk(run) if isinstance(n,ast.Call)]
        prepare=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='prepare_generator')
        groups=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='optimizer_groups')
        ddp=next(n.lineno for n in calls if isinstance(n.func,ast.Name) and n.func.id=='DDP')
        self.assertLess(prepare,groups);self.assertLess(groups,ddp)

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(),'Gloo unavailable')
    def test_actual_generator_adapters_two_rank_ddp(self):
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(ddp_worker,args=(str(Path(tmp)/'rendezvous'),),nprocs=2,join=True)


if __name__=='__main__':unittest.main()
