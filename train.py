"""Experimental low-memory trainer; network/checkpoint tensor shapes are unchanged."""
import logging
import multiprocessing
import os
import random
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.utils.tensorboard import SummaryWriter

import modules.commons as commons
import utils
from data_utils import TextAudioCollate, TextAudioSpeakerLoader
from models import MultiPeriodDiscriminator, SynthesizerTrn
from modules.losses import discriminator_loss, feature_loss, generator_loss, kl_loss
from modules.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from modules.training_runtime import (
    cuda_autocast, frozen_parameters, make_grad_scaler, resolve_precision,
    seed_worker, unwrap_model,
)

logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('numba').setLevel(logging.WARNING)
torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("Audio training requires CUDA; CPU regression tests do not")
    hps = utils.get_hparams()
    n_gpus = torch.cuda.device_count()
    if n_gpus == 1:
        # No process group, communication buckets, or subprocess for one GPU.
        run(0, 1, hps)
    else:
        os.environ.setdefault('MASTER_ADDR', 'localhost')
        os.environ.setdefault('MASTER_PORT', str(hps.train.port))
        mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps))


def run(rank, n_gpus, hps):
    global global_step
    torch.cuda.set_device(rank)
    distributed = n_gpus > 1
    writer = writer_eval = None
    if distributed:
        dist.init_process_group(backend='gloo' if os.name == 'nt' else 'nccl',
                                init_method='env://', world_size=n_gpus, rank=rank)
    try:
        logger = utils.get_logger(hps.model_dir) if rank == 0 else logging.getLogger(__name__)
        if rank == 0:
            logger.info(hps)
            utils.check_git_hash(hps.model_dir)
            writer = SummaryWriter(log_dir=hps.model_dir)
            writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, 'eval'))
        torch.manual_seed(hps.train.seed + rank)
        random.seed(hps.train.seed + rank)
        np.random.seed(hps.train.seed + rank)
        amp_enabled, half_type = resolve_precision(hps.train, torch.device('cuda', rank))
        scaler = make_grad_scaler(amp_enabled, half_type)
        collate_fn = TextAudioCollate()
        all_in_mem = hps.train.all_in_mem  # CPU RAM cache, not GPU VRAM.
        train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps, all_in_mem=all_in_mem)
        if not len(train_dataset):
            raise ValueError('Training file list is empty')
        workers = getattr(hps.train, 'num_workers', min(5, multiprocessing.cpu_count()))
        if all_in_mem:
            workers = 0
        sampler = (DistributedSampler(train_dataset, num_replicas=n_gpus, rank=rank,
                                      shuffle=True, seed=hps.train.seed)
                   if distributed else None)
        loader_rng = torch.Generator().manual_seed(hps.train.seed + rank)
        train_loader = DataLoader(
            train_dataset, num_workers=workers, shuffle=sampler is None, sampler=sampler,
            pin_memory=True, batch_size=hps.train.batch_size, collate_fn=collate_fn,
            worker_init_fn=seed_worker, generator=loader_rng,
            persistent_workers=workers > 0)
        eval_loader = None
        if rank == 0:
            eval_dataset = TextAudioSpeakerLoader(
                hps.data.validation_files, hps, all_in_mem=all_in_mem,
                vol_aug=False, random_crop=False)
            # Dedicated RNG avoids perturbing training when validation iterates.
            eval_loader = DataLoader(
                eval_dataset, num_workers=0, shuffle=False, batch_size=1,
                pin_memory=True, collate_fn=collate_fn,
                generator=torch.Generator().manual_seed(hps.train.seed))

        net_g = SynthesizerTrn(hps.data.filter_length // 2 + 1,
                              hps.train.segment_size // hps.data.hop_length,
                              **hps.model).cuda(rank)
        net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)
        optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate,
                                    betas=hps.train.betas, eps=hps.train.eps)
        optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate,
                                    betas=hps.train.betas, eps=hps.train.eps)
        if distributed:
            net_g = DDP(net_g, device_ids=[rank])
            net_d = DDP(net_d, device_ids=[rank])

        epoch_str, global_step = 1, 0
        # Preserve compatibility with the existing G/D checkpoint format.
        # Exact optimizer/scaler/RNG resumption is not claimed by this patch.
        try:
            g_path = utils.latest_checkpoint_path(hps.model_dir, 'G_*.pth')
            d_path = utils.latest_checkpoint_path(hps.model_dir, 'D_*.pth')
            _, _, _, epoch_str = utils.load_checkpoint(g_path, net_g, optim_g, False)
            _, _, _, epoch_str = utils.load_checkpoint(d_path, net_d, optim_d, False)
            epoch_str = max(epoch_str, 1)
            global_step = int(os.path.splitext(d_path)[0].rsplit('_', 1)[-1]) + 1
        except (IndexError, FileNotFoundError):
            logger.info('No G/D checkpoint pair found; starting without a resume checkpoint')
        # Corrupt/incompatible checkpoints must fail visibly, not silently restart.
        scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
        scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2)
        for epoch in range(epoch_str, hps.train.epochs + 1):
            if sampler is not None:
                sampler.set_epoch(epoch)
            if epoch <= hps.train.warmup_epochs:
                for optimizer in (optim_g, optim_d):
                    for group in optimizer.param_groups:
                        group['lr'] = hps.train.learning_rate * epoch / hps.train.warmup_epochs
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d],
                               scaler, [train_loader, eval_loader], logger,
                               [writer, writer_eval], amp_enabled, half_type)
            scheduler_g.step()
            scheduler_d.step()
    finally:
        if writer is not None:
            writer.close()
            writer_eval.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


def train_and_evaluate(rank, epoch, hps, nets, optims, scaler, loaders, logger,
                       writers, amp_enabled, half_type):
    global global_step
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders
    writer, writer_eval = writers
    generator, discriminator = unwrap_model(net_g), unwrap_model(net_d)
    net_g.train()
    net_d.train()
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats(rank)
    for batch_idx, items in enumerate(train_loader):
        # Release previous gradients BEFORE the next generator forward.
        optim_g.zero_grad(set_to_none=True)
        optim_d.zero_grad(set_to_none=True)
        c, f0, spec, y, spk, lengths, uv, volume = items
        g = spk.cuda(rank, non_blocking=True)
        c, f0 = c.cuda(rank, non_blocking=True), f0.cuda(rank, non_blocking=True)
        spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
        uv, lengths = uv.cuda(rank, non_blocking=True), lengths.cuda(rank, non_blocking=True)
        if volume is not None:
            volume = volume.cuda(rank, non_blocking=True)
        mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                               hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
        with cuda_autocast(amp_enabled, half_type):
            y_hat, ids_slice, z_mask, latent, pred_lf0, norm_lf0, lf0 = net_g(
                c, f0, uv, spec, g=g, c_lengths=lengths, spec_lengths=lengths, vol=volume)
        z, z_p, m_p, logs_p, m_q, logs_q = latent
        y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
        # STFT and reconstruction losses remain FP32 even when convolutions use AMP.
        with cuda_autocast(False, half_type):
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels,
                hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                hps.data.mel_fmin, hps.data.mel_fmax)
        y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size)
        with cuda_autocast(amp_enabled, half_type):
            d_outputs = net_d(y, y_hat.detach())
            with cuda_autocast(False, half_type):
                loss_disc, _, _ = discriminator_loss(d_outputs[0], d_outputs[1])
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)
        optim_d.zero_grad(set_to_none=True)
        del d_outputs

        # Only input gradients are needed here. Bypass DDP for this frozen pass:
        # the discriminator reducer must not expect parameter gradients from it.
        with frozen_parameters(discriminator):
            with cuda_autocast(amp_enabled, half_type):
                _, d_fake, fmap_real, fmap_fake = discriminator(y, y_hat)
                with cuda_autocast(False, half_type):
                    loss_mel = F.l1_loss(y_mel.float(), y_hat_mel.float()) * hps.train.c_mel
                    loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                    loss_fm = feature_loss(fmap_real, fmap_fake)
                    loss_gen, _ = generator_loss(d_fake)
                    loss_lf0 = (F.mse_loss(pred_lf0.float(), lf0.float())
                                if generator.use_automatic_f0_prediction else y_hat.new_zeros(()))
                    loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl + loss_lf0
            scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()
        optim_g.zero_grad(set_to_none=True)
        del d_fake, fmap_real, fmap_fake

        if rank == 0 and global_step % hps.train.log_interval == 0:
            losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
            logger.info('Epoch %s | batch %s/%s | step %s | losses %s',
                        epoch, batch_idx + 1, len(train_loader), global_step,
                        [float(loss.detach()) for loss in losses])
            scalar_dict = {
                'loss/g/total': loss_gen_all.detach(), 'loss/d/total': loss_disc.detach(),
                'learning_rate': optim_g.param_groups[0]['lr'],
                'grad_norm_d': grad_norm_d, 'grad_norm_g': grad_norm_g,
                'loss/g/fm': loss_fm.detach(), 'loss/g/mel': loss_mel.detach(),
                'loss/g/kl': loss_kl.detach(), 'loss/g/lf0': loss_lf0.detach(),
                'memory/peak_allocated_mib': torch.cuda.max_memory_allocated(rank) / 2**20,
                'memory/peak_reserved_mib': torch.cuda.max_memory_reserved(rank) / 2**20,
            }
            image_dict = {
                'slice/mel_org': utils.plot_spectrogram_to_numpy(y_mel[0].detach().cpu().numpy()),
                'slice/mel_gen': utils.plot_spectrogram_to_numpy(y_hat_mel[0].detach().cpu().numpy()),
                'all/mel': utils.plot_spectrogram_to_numpy(mel[0].detach().cpu().numpy()),
            }
            if generator.use_automatic_f0_prediction:
                image_dict['all/lf0'] = utils.plot_data_to_numpy(
                    lf0[0, 0].detach().cpu().numpy(), pred_lf0[0, 0].detach().cpu().numpy())
            utils.summarize(writer=writer, global_step=global_step,
                            images=image_dict, scalars=scalar_dict)

        # Do not retain a training minibatch's tensors during validation.
        del c, f0, spec, y, g, lengths, uv, volume, mel, y_mel, y_hat_mel, y_hat
        del z, z_p, m_p, logs_p, m_q, logs_q, latent, z_mask, pred_lf0, norm_lf0, lf0
        if global_step % hps.train.eval_interval == 0:
            if rank == 0:
                evaluate(hps, net_g, eval_loader, writer_eval, amp_enabled, half_type)
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, 'G_{}.pth'.format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch,
                                      os.path.join(hps.model_dir, 'D_{}.pth'.format(global_step)))
                keep = getattr(hps.train, 'keep_ckpts', 0)
                if keep > 0:
                    utils.clean_checkpoints(path_to_models=hps.model_dir, n_ckpts_to_keep=keep,
                                            sort_by_time=True)
            if dist.is_initialized():
                dist.barrier()
        global_step += 1
    if rank == 0:
        torch.cuda.synchronize(rank)
        logger.info('Epoch %s elapsed %.2f s (includes validation/checkpoints)',
                    epoch, time.perf_counter() - started)


def evaluate(hps, generator, eval_loader, writer_eval, amp_enabled=False, half_type=torch.float32):
    model = unwrap_model(generator)
    was_training = generator.training
    generator.eval()
    image_dict, audio_dict = {}, {}
    limit = getattr(hps.train, 'max_eval_batches', 0)
    try:
        # infer() seeds torch internally; keep validation from resetting training RNG.
        with torch.no_grad(), torch.random.fork_rng(devices=[0]):
            for index, items in enumerate(eval_loader):
                if limit > 0 and index >= limit:
                    break
                c, f0, spec, y, spk, _, uv, volume = items
                c, f0 = c[:1].cuda(0), f0[:1].cuda(0)
                spec, spk, uv = spec[:1].cuda(0), spk[:1].cuda(0), uv[:1].cuda(0)
                if volume is not None:
                    volume = volume[:1].cuda(0)
                with cuda_autocast(amp_enabled, half_type):
                    y_hat, _ = model.infer(c, f0, uv, g=spk, vol=volume)
                # Transfer each result immediately, not the entire validation set at once.
                audio_dict['gen/audio_{}'.format(index)] = y_hat[0].float().cpu()
                audio_dict['gt/audio_{}'.format(index)] = y[0].float().cpu()
                # Plot one fixed example, rather than retaining all GPU spectrograms.
                if index == 0:
                    mel = spec_to_mel_torch(spec, hps.data.filter_length, hps.data.n_mel_channels,
                                           hps.data.sampling_rate, hps.data.mel_fmin, hps.data.mel_fmax)
                    pred_mel = mel_spectrogram_torch(
                        y_hat.squeeze(1).float(), hps.data.filter_length, hps.data.n_mel_channels,
                        hps.data.sampling_rate, hps.data.hop_length, hps.data.win_length,
                        hps.data.mel_fmin, hps.data.mel_fmax)
                    image_dict['gen/mel'] = utils.plot_spectrogram_to_numpy(pred_mel[0].cpu().numpy())
                    image_dict['gt/mel'] = utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
                    del pred_mel, mel
                del c, f0, spec, spk, uv, volume, y_hat
        if audio_dict:
            utils.summarize(writer=writer_eval, global_step=global_step, images=image_dict,
                            audios=audio_dict, audio_sampling_rate=hps.data.sampling_rate)
    finally:
        generator.train(was_training)


if __name__ == '__main__':
    main()
