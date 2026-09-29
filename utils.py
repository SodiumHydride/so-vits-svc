"""Legacy public utilities with optional features imported only when requested.

The model's pitch arithmetic lives in modules.model_utils. Keep the historical
public helper names here for old entry points; do not import retrieval, plotting
or audio-analysis libraries just to read a config or checkpoint.
"""
import argparse
import glob
import importlib
import json
import logging
import os
import re
import subprocess
import sys
import traceback
from multiprocessing import cpu_count

import numpy as np
import torch
from torch.nn import functional as F

from modules.model_utils import f0_to_coarse as f0_to_coarse
from modules.model_utils import normalize_f0 as normalize_f0

MATPLOTLIB_FLAG = False
logging.basicConfig(stream=sys.stdout, level=logging.WARN)
logger = logging

# Historical public constants retained for callers; arithmetic has one owner.
f0_bin = 256
f0_max = 1100.0
f0_min = 50.0
f0_mel_min = 1127 * np.log(1 + f0_min / 700)
f0_mel_max = 1127 * np.log(1 + f0_max / 700)

_F0_PREDICTORS = {
    "pm": "PMF0Predictor",
    "crepe": "CrepeF0Predictor",
    "harvest": "HarvestF0Predictor",
    "dio": "DioF0Predictor",
    "rmvpe": "RMVPEF0Predictor",
    "fcpe": "FCPEF0Predictor",
}
_SPEECH_ENCODERS = {
    "vec768l12": "ContentVec768L12",
    "vec256l9": "ContentVec256L9",
    "vec256l9-onnx": "ContentVec256L9_Onnx",
    "vec256l12-onnx": "ContentVec256L12_Onnx",
    "vec768l9-onnx": "ContentVec768L9_Onnx",
    "vec768l12-onnx": "ContentVec768L12_Onnx",
    "hubertsoft-onnx": "HubertSoft_Onnx",
    "hubertsoft": "HubertSoft",
    "whisper-ppg": "WhisperPPG",
    "cnhubertlarge": "CNHubertLarge",
    "dphubert": "DPHubert",
    "whisper-ppg-large": "WhisperPPGLarge",
    "wavlmbase+": "WavLMBasePlus",
}


def get_f0_predictor(f0_predictor, hop_length, sampling_rate, **kargs):
    if f0_predictor not in _F0_PREDICTORS:
        raise Exception("Unknown f0 predictor")
    name = _F0_PREDICTORS[f0_predictor]
    cls = getattr(importlib.import_module("modules.F0Predictor." + name), name)
    options = {"hop_length": hop_length, "sampling_rate": sampling_rate}
    if f0_predictor in ("crepe", "rmvpe", "fcpe"):
        options.update(device=kargs["device"], threshold=kargs["threshold"])
    if f0_predictor in ("rmvpe", "fcpe"):
        options["dtype"] = torch.float32
    return cls(**options)


def get_speech_encoder(speech_encoder, device=None, **kargs):
    if speech_encoder not in _SPEECH_ENCODERS:
        raise Exception("Unknown speech encoder")
    name = _SPEECH_ENCODERS[speech_encoder]
    cls = getattr(importlib.import_module("vencoder." + name), name)
    return cls(device=device)


def get_content(cmodel, y):
    with torch.no_grad():
        c = cmodel.extract_features(y.squeeze(1))[0]
    return c.transpose(1, 2)


def _get_pyplot():
    global MATPLOTLIB_FLAG
    if not MATPLOTLIB_FLAG:
        import matplotlib
        matplotlib.use("Agg")
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        MATPLOTLIB_FLAG = True
    import matplotlib.pylab as plt
    return plt


def _figure_to_numpy(fig, plt):
    """Copy RGB pixels before closing the figure; supported on modern Matplotlib."""
    try:
        fig.canvas.draw()
        return np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    finally:
        plt.close(fig)


def plot_data_to_numpy(x, y):
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 2))
    ax.plot(x)
    ax.plot(y)
    fig.tight_layout()
    return _figure_to_numpy(fig, plt)


def plot_spectrogram_to_numpy(spectrogram):
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(10, 2))
    im = ax.imshow(spectrogram, aspect="auto", origin="lower", interpolation='none')
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Frames")
    ax.set_ylabel("Channels")
    fig.tight_layout()
    return _figure_to_numpy(fig, plt)


def plot_alignment_to_numpy(alignment, info=None):
    plt = _get_pyplot()
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(alignment.transpose(), aspect='auto', origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    if info is not None:
        xlabel += '\n\n' + info
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Encoder timestep')
    fig.tight_layout()
    return _figure_to_numpy(fig, plt)


def load_checkpoint(checkpoint_path, model, optimizer=None, skip_optimizer=False):
    assert os.path.isfile(checkpoint_path)
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')
    iteration = checkpoint_dict['iteration']
    learning_rate = checkpoint_dict['learning_rate']
    if optimizer is not None and not skip_optimizer and checkpoint_dict['optimizer'] is not None:
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
    saved_state_dict = checkpoint_dict['model']
    model = model.to(list(saved_state_dict.values())[0].dtype)
    if hasattr(model, 'module'):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    new_state_dict = {}
    for k, v in state_dict.items():
        try:
            new_state_dict[k] = saved_state_dict[k]
            assert saved_state_dict[k].shape == v.shape, (saved_state_dict[k].shape, v.shape)
        except Exception:
            if "enc_q" not in k or "emb_g" not in k:
                print("%s is not in the checkpoint,please check your checkpoint.If you're using pretrain model,just ignore this warning." % k)
                logger.info("%s is not in the checkpoint" % k)
                new_state_dict[k] = v
    if hasattr(model, 'module'):
        model.module.load_state_dict(new_state_dict)
    else:
        model.load_state_dict(new_state_dict)
    logger.info("Loaded checkpoint '{}' (iteration {})".format(checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    logger.info("Saving model and optimizer state at iteration {} to {}".format(iteration, checkpoint_path))
    state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    torch.save({'model': state_dict, 'iteration': iteration,
                'optimizer': optimizer.state_dict(), 'learning_rate': learning_rate}, checkpoint_path)


def clean_checkpoints(path_to_models='logs/44k/', n_ckpts_to_keep=2, sort_by_time=True):
    """Retain legacy checkpoint selection; never call this on your only backup."""
    ckpts_files = [f for f in os.listdir(path_to_models)
                   if os.path.isfile(os.path.join(path_to_models, f))]

    def name_key(name):
        return int(re.compile("._(\\d+)\\.pth").match(name).group(1))

    def time_key(name):
        return os.path.getmtime(os.path.join(path_to_models, name))

    sort_key = time_key if sort_by_time else name_key

    def x_sorted(prefix):
        return sorted([f for f in ckpts_files if f.startswith(prefix) and not f.endswith("_0.pth")],
                      key=sort_key)

    to_del = [os.path.join(path_to_models, name) for name in
              (x_sorted('G')[:-n_ckpts_to_keep] + x_sorted('D')[:-n_ckpts_to_keep])]
    for name in to_del:
        os.remove(name)
        logger.info(".. Free up space by deleting ckpt %s", name)


def summarize(writer, global_step, scalars=None, histograms=None, images=None, audios=None,
              audio_sampling_rate=22050):
    for k, v in (scalars or {}).items():
        writer.add_scalar(k, v, global_step)
    for k, v in (histograms or {}).items():
        writer.add_histogram(k, v, global_step)
    for k, v in (images or {}).items():
        writer.add_image(k, v, global_step, dataformats='HWC')
    for k, v in (audios or {}).items():
        writer.add_audio(k, v, global_step, audio_sampling_rate)


def latest_checkpoint_path(dir_path, regex="G_*.pth"):
    f_list = glob.glob(os.path.join(dir_path, regex))
    f_list.sort(key=lambda f: int("".join(filter(str.isdigit, f))))
    x = f_list[-1]
    print(x)
    return x


def load_wav_to_torch(full_path):
    from scipy.io.wavfile import read
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate


def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        return [line.strip().split(split) for line in f]


def get_hparams(init=True):
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str, default="./configs/config.json",
                        help='JSON file for configuration')
    parser.add_argument('-m', '--model', type=str, required=True, help='Model name')
    args = parser.parse_args()
    model_dir = os.path.join("./logs", args.model)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    config_save_path = os.path.join(model_dir, "config.json")
    if init:
        with open(args.config, "r") as f:
            data = f.read()
        with open(config_save_path, "w") as f:
            f.write(data)
    else:
        with open(config_save_path, "r") as f:
            data = f.read()
    hparams = HParams(**json.loads(data))
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_dir(model_dir):
    with open(os.path.join(model_dir, "config.json"), "r") as f:
        config = json.load(f)
    hparams = HParams(**config)
    hparams.model_dir = model_dir
    return hparams


def get_hparams_from_file(config_path, infer_mode=False):
    with open(config_path, "r") as f:
        config = json.load(f)
    return InferHParams(**config) if infer_mode else HParams(**config)


def check_git_hash(model_dir):
    source_dir = os.path.dirname(os.path.realpath(__file__))
    if not os.path.exists(os.path.join(source_dir, ".git")):
        logger.warning("%s is not a git repository, therefore hash value comparison will be ignored.",
                       source_dir)
        return
    cur_hash = subprocess.getoutput("git rev-parse HEAD")
    path = os.path.join(model_dir, "githash")
    if os.path.exists(path):
        with open(path) as f:
            saved_hash = f.read()
        if saved_hash != cur_hash:
            logger.warning("git hash values are different. %s(saved) != %s(current)",
                           saved_hash[:8], cur_hash[:8])
    else:
        with open(path, "w") as f:
            f.write(cur_hash)


def get_logger(model_dir, filename="train.log"):
    global logger
    logger = logging.getLogger(os.path.basename(model_dir))
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s\t%(name)s\t%(levelname)s\t%(message)s")
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    handler = logging.FileHandler(os.path.join(model_dir, filename))
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def repeat_expand_2d(content, target_len, mode='left'):
    return (repeat_expand_2d_left(content, target_len) if mode == 'left'
            else repeat_expand_2d_other(content, target_len, mode))


def repeat_expand_2d_left(content, target_len):
    src_len = content.shape[-1]
    target = torch.zeros([content.shape[0], target_len], dtype=torch.float).to(content.device)
    temp = torch.arange(src_len + 1) * target_len / src_len
    current_pos = 0
    for i in range(target_len):
        if i < temp[current_pos + 1]:
            target[:, i] = content[:, current_pos]
        else:
            current_pos += 1
            target[:, i] = content[:, current_pos]
    return target


def repeat_expand_2d_other(content, target_len, mode='nearest'):
    return F.interpolate(content[None, :, :], size=target_len, mode=mode)[0]


def mix_model(model_paths, mix_rate, mode):
    mix_rate = torch.FloatTensor(mix_rate) / 100
    model_tem = torch.load(model_paths[0])
    models = [torch.load(path)["model"] for path in model_paths]
    if mode == 0:
        mix_rate = F.softmax(mix_rate, dim=0)
    for k in model_tem["model"].keys():
        model_tem["model"][k] = torch.zeros_like(model_tem["model"][k])
        for i, model in enumerate(models):
            model_tem["model"][k] += model[k] * mix_rate[i]
    output_path = os.path.join(os.path.curdir, "output.pth")
    torch.save(model_tem, output_path)
    return output_path


def change_rms(data1, sr1, data2, sr2, rate):
    # From RVC: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    import librosa
    rms1 = librosa.feature.rms(y=data1, frame_length=sr1 // 2 * 2, hop_length=sr1 // 2)
    rms2 = librosa.feature.rms(y=data2.detach().cpu().numpy(), frame_length=sr2 // 2 * 2,
                             hop_length=sr2 // 2)
    rms1 = torch.from_numpy(rms1).to(data2.device)
    rms1 = F.interpolate(rms1.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.from_numpy(rms2).to(data2.device)
    rms2 = F.interpolate(rms2.unsqueeze(0), size=data2.shape[0], mode="linear").squeeze()
    rms2 = torch.max(rms2, torch.zeros_like(rms2) + 1e-6)
    data2 *= torch.pow(rms1, torch.tensor(1 - rate)) * torch.pow(rms2, torch.tensor(rate - 1))
    return data2


def train_index(spk_name, root_dir="dataset/44k/"):
    # From RVC: https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI
    import faiss
    n_cpu = cpu_count()
    print("The feature index is constructing.")
    exp_dir = os.path.join(root_dir, spk_name)
    listdir_res = [os.path.join(exp_dir, file) for file in os.listdir(exp_dir)
                   if ".wav.soft.pt" in file]
    if not listdir_res:
        raise Exception("You need to run preprocess_hubert_f0.py!")
    npys = [torch.load(name)[0].transpose(-1, -2).numpy() for name in sorted(listdir_res)]
    big_npy = np.concatenate(npys, 0)
    big_npy_idx = np.arange(big_npy.shape[0])
    np.random.shuffle(big_npy_idx)
    big_npy = big_npy[big_npy_idx]
    if big_npy.shape[0] > 2e5:
        from sklearn.cluster import MiniBatchKMeans
        print("Trying doing kmeans %s shape to 10k centers." % big_npy.shape[0])
        try:
            big_npy = MiniBatchKMeans(n_clusters=10000, verbose=True, batch_size=256 * n_cpu,
                                     compute_labels=False, init="random").fit(big_npy).cluster_centers_
        except Exception:
            print(traceback.format_exc())
    n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
    index = faiss.index_factory(big_npy.shape[1], "IVF%s,Flat" % n_ivf)
    index_ivf = faiss.extract_index_ivf(index)
    index_ivf.nprobe = 1
    index.train(big_npy)
    for i in range(0, big_npy.shape[0], 8192):
        index.add(big_npy[i:i + 8192])
    print("Successfully build index")
    return index


class HParams:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = HParams(**v) if type(v) == dict else v

    def keys(self):
        return self.__dict__.keys()

    def items(self):
        return self.__dict__.items()

    def values(self):
        return self.__dict__.values()

    def __len__(self):
        return len(self.__dict__)

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        return setattr(self, key, value)

    def __contains__(self, key):
        return key in self.__dict__

    def __repr__(self):
        return self.__dict__.__repr__()

    def get(self, index):
        return self.__dict__.get(index)


class InferHParams(HParams):
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            self[k] = InferHParams(**v) if type(v) == dict else v

    def __getattr__(self, index):
        return self.get(index)


class Volume_Extractor:
    def __init__(self, hop_size=512):
        self.hop_size = hop_size

    def extract(self, audio):
        if not isinstance(audio, torch.Tensor):
            audio = torch.Tensor(audio)
        n_frames = int(audio.size(-1) // self.hop_size)
        audio2 = audio ** 2
        audio2 = F.pad(audio2, (int(self.hop_size // 2), int((self.hop_size + 1) // 2)), mode='reflect')
        volume = F.unfold(audio2[:, None, None, :], (1, self.hop_size),
                          stride=self.hop_size)[:, :, :n_frames].mean(dim=1)[0]
        return torch.sqrt(volume)
