import random
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torchaudio as ta
from lightning import LightningDataModule
from torch.utils.data.dataloader import DataLoader

from matcha.data import feature_cache
from matcha.data.bucket_sampler import LengthBucketBatchSampler
from matcha.text import text_to_sequence
from matcha.utils.audio import mel_spectrogram
from matcha.utils.model import fix_len_compatibility, normalize
from matcha.utils.utils import intersperse


def parse_filelist(filelist_path, split_char="|"):
    with open(filelist_path, encoding="utf-8") as f:
        filepaths_and_text = [line.strip().split(split_char) for line in f]
    return filepaths_and_text


class TextMelDataModule(LightningDataModule):
    def __init__(  # pylint: disable=unused-argument
        self,
        name,
        train_filelist_path,
        valid_filelist_path,
        batch_size,
        num_workers,
        pin_memory,
        cleaners,
        add_blank,
        n_spks,
        n_fft,
        n_feats,
        sample_rate,
        hop_length,
        win_length,
        f_min,
        f_max,
        data_statistics,
        seed,
        load_durations,
        feature_cache_dir=None,
        bucket_batching=False,
        bucket_frame_budget=None,
        pad_quantum=None,
    ):
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

    def setup(self, stage: Optional[str] = None):  # pylint: disable=unused-argument
        """Load data. Set variables: `self.data_train`, `self.data_val`, `self.data_test`.

        This method is called by lightning with both `trainer.fit()` and `trainer.test()`, so be
        careful not to execute things like random split twice!
        """
        # load and split datasets only if not loaded already

        self.trainset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.train_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            self.hparams.feature_cache_dir,
        )
        self.validset = TextMelDataset(  # pylint: disable=attribute-defined-outside-init
            self.hparams.valid_filelist_path,
            self.hparams.n_spks,
            self.hparams.cleaners,
            self.hparams.add_blank,
            self.hparams.n_fft,
            self.hparams.n_feats,
            self.hparams.sample_rate,
            self.hparams.hop_length,
            self.hparams.win_length,
            self.hparams.f_min,
            self.hparams.f_max,
            self.hparams.data_statistics,
            self.hparams.seed,
            self.hparams.load_durations,
            self.hparams.feature_cache_dir,
        )

    def train_dataloader(self):
        collate = TextMelBatchCollate(self.hparams.n_spks, pad_quantum=self.hparams.pad_quantum)
        common = dict(
            dataset=self.trainset,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=collate,
            # Keep workers alive between epochs so we don't pay the Windows
            # spawn cost on every validation cycle. Requires num_workers>0;
            # gated so setting num_workers=0 still works.
            persistent_workers=self.hparams.num_workers > 0,
        )
        if not self.hparams.bucket_batching:
            return DataLoader(batch_size=self.hparams.batch_size, shuffle=True, **common)

        # Length-bucketed batching (opt-in; overnight-run recipe - see
        # Final_Training.md and bucket_sampler.py). Requires the feature
        # cache: batch lengths come from its lengths index.
        if self.trainset.cache is None:
            raise ValueError("bucket_batching=true requires the feature cache (feature_cache_dir must not be false)")
        filepaths = [row[0] for row in self.trainset.filepaths_and_text]
        lengths = self.trainset.cache.lengths_for(filepaths, self.trainset._mel_params)
        sampler = LengthBucketBatchSampler(
            lengths,
            batch_size=self.hparams.batch_size,
            frame_budget=self.hparams.bucket_frame_budget,
            seed=self.hparams.seed,
        )
        return DataLoader(batch_sampler=sampler, **common)

    def val_dataloader(self):
        return DataLoader(
            dataset=self.validset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
            collate_fn=TextMelBatchCollate(self.hparams.n_spks, pad_quantum=self.hparams.pad_quantum),
            persistent_workers=self.hparams.num_workers > 0,
        )

    def teardown(self, stage: Optional[str] = None):
        """Clean up after fit or test."""
        pass  # pylint: disable=unnecessary-pass

    def state_dict(self):
        """Extra things to save to checkpoint."""
        return {}

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Things to do when loading checkpoint."""
        pass  # pylint: disable=unnecessary-pass


class TextMelDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filelist_path,
        n_spks,
        cleaners,
        add_blank=True,
        n_fft=1024,
        n_mels=80,
        sample_rate=22050,
        hop_length=256,
        win_length=1024,
        f_min=0.0,
        f_max=8000,
        data_parameters=None,
        seed=None,
        load_durations=False,
        feature_cache_dir=None,
    ):
        self.filepaths_and_text = parse_filelist(filelist_path)
        self.n_spks = n_spks
        self.cleaners = cleaners
        self.add_blank = add_blank
        self.n_fft = n_fft
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.f_min = f_min
        self.f_max = f_max
        self.load_durations = load_durations

        # On-disk feature cache (see matcha/data/feature_cache.py). None ->
        # default location under Datasets/feature_cache; False/"false" ->
        # disabled (original compute-every-epoch path); string -> custom root.
        if feature_cache_dir in (False, "false", "False", 0):
            self.cache = None
        else:
            root = feature_cache_dir or feature_cache.default_cache_root()
            self.cache = feature_cache.FeatureCache(root)
            self._mel_params = (n_fft, n_mels, sample_rate, hop_length,
                                win_length, f_min, f_max)
            self._phon_sigs = feature_cache.phonemizer_signatures()

        if data_parameters is not None:
            self.data_parameters = data_parameters
        else:
            self.data_parameters = {"mel_mean": 0, "mel_std": 1}
        random.seed(seed)
        random.shuffle(self.filepaths_and_text)

    def get_datapoint(self, filepath_and_text):
        if self.n_spks > 1:
            filepath, spk, text = (
                filepath_and_text[0],
                int(filepath_and_text[1]),
                filepath_and_text[2],
            )
        else:
            filepath, text = filepath_and_text[0], filepath_and_text[1]
            spk = None

        text, cleaned_text = self.get_text(text, add_blank=self.add_blank)
        mel = self.get_mel(filepath)

        durations = self.get_durations(filepath, text) if self.load_durations else None

        return {"x": text, "y": mel, "spk": spk, "filepath": filepath, "x_text": cleaned_text, "durations": durations}

    def get_durations(self, filepath, text):
        filepath = Path(filepath)
        data_dir, name = filepath.parent.parent, filepath.stem

        try:
            dur_loc = data_dir / "durations" / f"{name}.npy"
            durs = torch.from_numpy(np.load(dur_loc).astype(int))

        except FileNotFoundError as e:
            raise FileNotFoundError(
                f"Tried loading the durations but durations didn't exist at {dur_loc}, make sure you've generate the durations first using: python matcha/utils/get_durations_from_trained_model.py \n"
            ) from e

        assert len(durs) == len(text), f"Length of durations {len(durs)} and text {len(text)} do not match"

        return durs

    def get_mel(self, filepath):
        # Cached path: raw (un-normalized) mel from the feature cache;
        # normalization ALWAYS happens here at load time because mel_mean/std
        # depend on the dataset combination, not the utterance.
        key = None
        if self.cache is not None:
            key = self.cache.mel_key(filepath, self._mel_params)
            cached = self.cache.get_mel(key)
            if cached is not None:
                mel = torch.from_numpy(cached)
                return normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])

        audio, sr = ta.load(filepath)
        # VCTK ships at 48 kHz (wav48_silence_trimmed), but the configs in
        # this repo target 22050 Hz. Resample on the fly instead of
        # requiring a pre-resampled corpus on disk — the result is cached, so
        # the cost is paid once per corpus rather than per epoch.
        if sr != self.sample_rate:
            audio = ta.functional.resample(audio, orig_freq=sr,
                                           new_freq=self.sample_rate)
            sr = self.sample_rate
        mel = mel_spectrogram(
            audio,
            self.n_fft,
            self.n_mels,
            self.sample_rate,
            self.hop_length,
            self.win_length,
            self.f_min,
            self.f_max,
            center=False,
        ).squeeze()
        if key is not None:
            self.cache.put_mel(key, mel.numpy())
        mel = normalize(mel, self.data_parameters["mel_mean"], self.data_parameters["mel_std"])
        return mel

    def get_text(self, text, add_blank=True):
        # Cached path: pre-intersperse token IDs (the DP phonemizer is the
        # expensive part); intersperse/add_blank applied at load so they are
        # not part of the cache key.
        key = None
        if self.cache is not None:
            key = self.cache.text_key(text, self.cleaners, self._phon_sigs)
            cached = self.cache.get_text(key)
            if cached is not None:
                text_norm, cleaned_text = cached[0].tolist(), cached[1]
                if self.add_blank:
                    text_norm = intersperse(text_norm, 0)
                return torch.IntTensor(text_norm), cleaned_text

        text_norm, cleaned_text = text_to_sequence(text, self.cleaners)
        if key is not None:
            self.cache.put_text(key, text_norm, cleaned_text)
        if self.add_blank:
            text_norm = intersperse(text_norm, 0)
        text_norm = torch.IntTensor(text_norm)
        return text_norm, cleaned_text

    def __getitem__(self, index):
        datapoint = self.get_datapoint(self.filepaths_and_text[index])
        return datapoint

    def __len__(self):
        return len(self.filepaths_and_text)


class TextMelBatchCollate:
    def __init__(self, n_spks, pad_quantum=None):
        self.n_spks = n_spks
        # Optional shape quantization: round padded widths UP to a quantum so
        # the set of distinct batch shapes is small and bounded - this is what
        # keeps torch.compile inside its recompile budget. Must be a multiple
        # of fix_len_compatibility's factor (4). Token lengths are quantized
        # to a fixed 16 for the same reason. Padding is masked out of every
        # loss, so this is compute-shape-only.
        self.pad_quantum = int(pad_quantum) if pad_quantum else None

    def __call__(self, batch):
        B = len(batch)
        y_max_length = max([item["y"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        if self.pad_quantum:
            q = self.pad_quantum
            y_max_length = ((y_max_length + q - 1) // q) * q
        y_max_length = fix_len_compatibility(y_max_length)
        x_max_length = max([item["x"].shape[-1] for item in batch])  # pylint: disable=consider-using-generator
        if self.pad_quantum:
            x_max_length = ((x_max_length + 15) // 16) * 16
        n_feats = batch[0]["y"].shape[-2]

        y = torch.zeros((B, n_feats, y_max_length), dtype=torch.float32)
        x = torch.zeros((B, x_max_length), dtype=torch.long)
        durations = torch.zeros((B, x_max_length), dtype=torch.long)

        y_lengths, x_lengths = [], []
        spks = []
        filepaths, x_texts = [], []
        for i, item in enumerate(batch):
            y_, x_ = item["y"], item["x"]
            y_lengths.append(y_.shape[-1])
            x_lengths.append(x_.shape[-1])
            y[i, :, : y_.shape[-1]] = y_
            x[i, : x_.shape[-1]] = x_
            spks.append(item["spk"])
            filepaths.append(item["filepath"])
            x_texts.append(item["x_text"])
            if item["durations"] is not None:
                durations[i, : item["durations"].shape[-1]] = item["durations"]

        y_lengths = torch.tensor(y_lengths, dtype=torch.long)
        x_lengths = torch.tensor(x_lengths, dtype=torch.long)
        spks = torch.tensor(spks, dtype=torch.long) if self.n_spks > 1 else None

        return {
            "x": x,
            "x_lengths": x_lengths,
            "y": y,
            "y_lengths": y_lengths,
            "spks": spks,
            "filepaths": filepaths,
            "x_texts": x_texts,
            "durations": durations if not torch.eq(durations, 0).all() else None,
        }
