import os
import random

import numpy as np
import torch
import torch.utils.data

import utils
from modules.mel_processing import spectrogram_torch
from utils import load_filepaths_and_text, load_wav_to_torch


class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """Load aligned, precomputed conditioning features and source audio on CPU."""

    def __init__(self, audiopaths, hparams, all_in_mem=False, vol_aug=True,
                 random_crop=True):
        self.audiopaths = load_filepaths_and_text(audiopaths)
        self.hparams = hparams
        self.max_wav_value = hparams.data.max_wav_value
        self.sampling_rate = hparams.data.sampling_rate
        self.filter_length = hparams.data.filter_length
        self.hop_length = hparams.data.hop_length
        self.win_length = hparams.data.win_length
        self.unit_interpolate_mode = hparams.data.unit_interpolate_mode
        self.spec_len = hparams.train.max_speclen
        self.random_crop = random_crop
        if isinstance(self.spec_len, bool) or not isinstance(self.spec_len, int) or self.spec_len <= 0:
            raise ValueError("train.max_speclen must be a positive integer (feature frames)")
        segment_size = hparams.train.segment_size
        if segment_size <= 0 or segment_size % self.hop_length:
            raise ValueError("train.segment_size must be a positive multiple of data.hop_length")
        if self.spec_len < segment_size // self.hop_length:
            raise ValueError("train.max_speclen must cover at least one generator segment")
        self.spk_map = hparams.spk
        self.vol_emb = hparams.model.vol_embedding
        self.vol_aug = hparams.train.vol_aug and vol_aug
        # Do not reset the process-wide augmentation RNG when validation is built.
        random.Random(hparams.train.seed).shuffle(self.audiopaths)
        self.all_in_mem = all_in_mem
        if self.all_in_mem:
            self.cache = [self.get_audio(p[0]) for p in self.audiopaths]

    def get_audio(self, filename):
        filename = filename.replace("\\", "/")
        audio, sampling_rate = load_wav_to_torch(filename)
        if sampling_rate != self.sampling_rate:
            raise ValueError("Sample rate: expected {}, got {} in {}".format(
                self.sampling_rate, sampling_rate, filename))
        audio_norm = (audio / self.max_wav_value).unsqueeze(0)
        spec_filename = os.path.splitext(filename)[0] + ".spec.pt"
        if os.path.exists(spec_filename):
            spec = torch.load(spec_filename, map_location="cpu", weights_only=True)
        else:
            spec = spectrogram_torch(
                audio_norm, self.filter_length, self.sampling_rate,
                self.hop_length, self.win_length, center=False).squeeze(0)
            torch.save(spec, spec_filename)

        spk = torch.LongTensor([self.spk_map[filename.split("/")[-2]]])
        # Legacy F0 files may be object arrays; use only locally trusted caches.
        f0, uv = np.load(filename + ".f0.npy", allow_pickle=True)
        f0 = torch.tensor(np.asarray(f0, dtype=np.float32))
        uv = torch.tensor(np.asarray(uv, dtype=np.float32))
        c = torch.load(filename + ".soft.pt", map_location="cpu", weights_only=True)
        c = utils.repeat_expand_2d(c.squeeze(0), f0.shape[0], mode=self.unit_interpolate_mode)
        volume = (torch.from_numpy(np.load(filename + ".vol.npy")).float()
                  if self.vol_emb else None)
        lmin = min(c.size(-1), spec.size(-1))
        if abs(c.size(-1) - spec.size(-1)) >= 3:
            raise ValueError("Content/spectrogram frame mismatch in {}".format(filename))
        if abs(audio_norm.shape[1] - lmin * self.hop_length) >= 3 * self.hop_length:
            raise ValueError("Audio/feature alignment mismatch in {}".format(filename))
        if min(f0.numel(), uv.numel()) < lmin or (volume is not None and volume.numel() < lmin):
            raise ValueError("Conditioning sequence is shorter than the spectrogram in {}".format(filename))
        spec, c, f0, uv = spec[:, :lmin], c[:, :lmin], f0[:lmin], uv[:lmin]
        audio_norm = audio_norm[:, :lmin * self.hop_length]
        if volume is not None:
            volume = volume[:lmin]
        return c, f0, spec, audio_norm, spk, uv, volume

    def random_slice(self, c, f0, spec, audio_norm, spk, uv, volume):
        if self.vol_aug and volume is not None and random.choice([True, False]):
            max_amp = float(torch.max(torch.abs(audio_norm))) + 1e-5
            max_shift = min(1, np.log10(1 / max_amp))
            gain = 10 ** random.uniform(-1, max_shift)
            audio_norm = audio_norm * gain
            volume = volume * gain
            spec = spectrogram_torch(
                audio_norm, self.filter_length, self.sampling_rate,
                self.hop_length, self.win_length, center=False)[0]

        # max_speclen used to be read but ignored in favor of a hard-coded 790.
        if spec.shape[1] > self.spec_len:
            last_start = spec.shape[1] - self.spec_len
            start = random.randint(0, last_start) if self.random_crop else last_start // 2
            end = start + self.spec_len
            spec, c = spec[:, start:end], c[:, start:end]
            f0, uv = f0[start:end], uv[start:end]
            audio_norm = audio_norm[:, start * self.hop_length:end * self.hop_length]
            if volume is not None:
                volume = volume[start:end]
        return c, f0, spec, audio_norm, spk, uv, volume

    def __getitem__(self, index):
        item = self.cache[index] if self.all_in_mem else self.get_audio(self.audiopaths[index][0])
        return self.random_slice(*item)

    def __len__(self):
        return len(self.audiopaths)


class TextAudioCollate:
    def __call__(self, batch):
        batch = [b for b in batch if b is not None]
        if not batch:
            raise ValueError("Cannot collate an empty audio batch")
        has_volume = [row[6] is not None for row in batch]
        if any(has_volume) and not all(has_volume):
            raise ValueError("Volume conditioning must be present for every item or none")
        batch = sorted(batch, key=lambda row: row[0].shape[1], reverse=True)
        size, max_c_len = len(batch), batch[0][0].shape[1]
        max_wav_len = max(row[3].shape[1] for row in batch)
        lengths = torch.LongTensor([row[0].shape[1] for row in batch])
        c_padded = torch.zeros(size, batch[0][0].shape[0], max_c_len, dtype=torch.float32)
        f0_padded = torch.zeros(size, max_c_len, dtype=torch.float32)
        spec_padded = torch.zeros(size, batch[0][2].shape[0], max_c_len, dtype=torch.float32)
        wav_padded = torch.zeros(size, 1, max_wav_len, dtype=torch.float32)
        spkids = torch.empty(size, 1, dtype=torch.long)
        uv_padded = torch.zeros(size, max_c_len, dtype=torch.float32)
        volume_padded = torch.zeros(size, max_c_len, dtype=torch.float32) if all(has_volume) else None
        for i, (c, f0, spec, wav, spk, uv, volume) in enumerate(batch):
            length = c.shape[1]
            if spec.shape[1] != length or f0.numel() != length or uv.numel() != length:
                raise ValueError("Unaligned conditioning features in audio batch")
            if volume is not None and volume.numel() != length:
                raise ValueError("Unaligned volume features in audio batch")
            c_padded[i, :, :length] = c
            spec_padded[i, :, :length] = spec
            f0_padded[i, :length] = f0
            wav_padded[i, :, :wav.shape[1]] = wav
            spkids[i, 0] = spk.reshape(-1)[0]
            uv_padded[i, :length] = uv
            if volume_padded is not None:
                volume_padded[i, :length] = volume
        return c_padded, f0_padded, spec_padded, wav_padded, spkids, lengths, uv_padded, volume_padded
