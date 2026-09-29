"""Two-process CPU/Gloo regression of the GAN freeze/unwrapped-forward protocol.

This checks real DDP reducers on tiny networks, not CUDA/NCCL or the audio model.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP


def worker(rank, rendezvous, runtime_path):
    spec = importlib.util.spec_from_file_location('ddp_test_runtime', runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        torch.manual_seed(15 + rank)
        generator, discriminator = DDP(nn.Linear(3, 3)), DDP(nn.Linear(3, 1))
        opt_g = torch.optim.SGD(generator.parameters(), lr=0.01)
        opt_d = torch.optim.SGD(discriminator.parameters(), lr=0.01)
        for step in range(3):
            opt_g.zero_grad(set_to_none=True)
            opt_d.zero_grad(set_to_none=True)
            fake = generator(torch.randn(4, 3))
            # One discriminator wrapped forward, as in the audio discriminator.
            output = discriminator(torch.cat([torch.randn(4, 3), fake.detach()], 0))
            ((output[:4] - 1).square().mean() + output[4:].square().mean()).backward()
            opt_d.step()
            opt_d.zero_grad(set_to_none=True)
            with runtime.frozen_parameters(runtime.unwrap_model(discriminator)):
                (runtime.unwrap_model(discriminator)(fake) - 1).square().mean().backward()
            opt_g.step()
            assert all(p.grad is None for p in discriminator.parameters())
        parameters = torch.cat([p.detach().flatten() for p in generator.parameters()])
        gathered = [torch.empty_like(parameters) for _ in range(2)]
        dist.all_gather(gathered, parameters)
        torch.testing.assert_close(gathered[0], gathered[1])
    finally:
        dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), 'Gloo is unavailable')
class DDPProtocolTests(unittest.TestCase):
    def test_two_rank_reducers_survive_three_gan_steps(self):
        runtime_path = str(Path(__file__).resolve().parents[1] / 'modules/training_runtime.py')
        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(worker, args=(str(Path(tmp) / 'rendezvous'), runtime_path), nprocs=2, join=True)


if __name__ == '__main__':
    unittest.main()
