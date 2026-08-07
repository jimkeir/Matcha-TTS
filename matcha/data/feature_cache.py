"""Content-addressed on-disk cache for per-utterance training features.

The training data pipeline recomputes audio decode -> resample -> mel and the
Deep-Phonemizer text -> token-id pass for every sample, every epoch. Both are
pure functions of (source file / text, config), so they are cached here once
and read back on every subsequent epoch AND every subsequent run.

Two independent caches, invalidated by their actual inputs:

  mel:  key = audio path + file size/mtime + all mel-spectrogram parameters.
        Payload: the RAW (un-normalized) mel as fp16. Normalization happens at
        load time in the dataset - mel_mean/std depend on the dataset
        *combination*, so keeping them out of the cache means a stats change
        or corpus extension never invalidates cached mels. fp16 storage: log-
        mel values span roughly [-12, 3]; fp16's ~3 significant digits are far
        inside training-target tolerance (verified against fp32 at rollout).

  text: key = raw text + cleaners list + DP checkpoint size/mtime +
        pronunciation-overrides JSON content. Payload: token IDs (int16) plus
        the cleaned text string. intersperse/add_blank are applied at load so
        add_blank is not part of the key. Content addressing dedupes repeated
        prompts across speakers (CommonVoice reuses sentences heavily).

New or changed datasets are therefore self-healing: changed inputs change the
key, miss, and recompute. Entries for removed data become dead files;
scripts/atc/matcha_tts/prune_feature_cache.py garbage-collects them.

Writes are atomic (per-PID temp file + os.replace) so parallel DataLoader
workers - and parallel runs - can share a cache directory safely.
"""

import hashlib
import os

import numpy as np

# Bump to invalidate every entry wholesale (format/semantics changes).
_FORMAT_VERSION = "1"


def _file_signature(path):
    """size:mtime_ns for a file that is part of a cache key."""
    st = os.stat(path)
    return f"{st.st_size}:{st.st_mtime_ns}"


def _key(parts):
    h = hashlib.sha1()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


class FeatureCache:
    """One cache root holding 'mel' and 'text' entry kinds."""

    def __init__(self, root):
        self.root = root

    def _path(self, kind, key):
        return os.path.join(self.root, kind, key[:2], key + ".npz")

    def _store(self, path, **arrays):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # The temp name must END in .npz: np.savez appends the extension to
        # names without it, which would strand the write under a different
        # filename and turn every subsequent lookup into a miss.
        tmp = f"{path}.{os.getpid()}.tmp.npz"
        try:
            np.savez(tmp, **arrays)
            os.replace(tmp, path)
        except OSError:
            # Cache writes are best-effort: a full disk or a racing writer
            # must not kill the training run.
            try:
                os.remove(tmp)
            except OSError:
                pass

    def _load(self, path):
        try:
            with np.load(path, allow_pickle=False) as z:
                return dict(z)
        except (OSError, ValueError):
            # Missing, or truncated by an interrupted writer: treat as a miss
            # (the entry is rewritten by the compute path).
            return None

    # ------------------------------------------------------------------ mel

    def mel_key(self, filepath, mel_params):
        """mel_params: iterable of the mel_spectrogram config values (numeric).
        Canonicalized through float() so the key is independent of whether a
        config wrote e.g. f_min as 0 or 0.0 - str(0) != str(0.0) silently
        orphaned every mel entry when the prune tool and the yaml disagreed."""
        canon = [repr(float(p)) for p in mel_params]
        return _key(["mel", _FORMAT_VERSION, os.path.abspath(filepath),
                     _file_signature(filepath), *canon])

    def get_mel(self, key):
        entry = self._load(self._path("mel", key))
        if entry is None:
            return None
        return entry["mel"].astype(np.float32)

    def put_mel(self, key, mel_f32):
        self._store(self._path("mel", key), mel=mel_f32.astype(np.float16))

    # ----------------------------------------------------------------- text

    def text_key(self, text, cleaners, extra_signatures):
        """extra_signatures: phonemizer identity (ckpt signature, overrides
        content hash, ...) - anything whose change must invalidate entries."""
        return _key(["text", _FORMAT_VERSION, text, *cleaners, *extra_signatures])

    def get_text(self, key):
        entry = self._load(self._path("text", key))
        if entry is None:
            return None
        return entry["ids"].astype(np.int64), str(entry["cleaned"])

    def put_text(self, key, ids, cleaned_text):
        self._store(self._path("text", key),
                    ids=np.asarray(ids, dtype=np.int16),
                    cleaned=np.str_(cleaned_text))


    # -------------------------------------------------------------- lengths

    def lengths_for(self, filepaths, mel_params):
        """Mel frame count per audio file, for length-bucketed batching -
        WITHOUT decoding audio. Sources, in order: the persisted lengths map
        (instant), the cached mel entry's npy header (cheap), else a
        soundfile-header duration estimate (~1% accurate; replaced by the
        exact value once the mel is cached and the map is rebuilt).

        Keys are mel_key()s, so entries self-invalidate exactly like mels.
        The map lives at <root>/lengths_index.npz; safe to delete any time.
        """
        map_path = os.path.join(self.root, "lengths_index.npz")
        known = {}
        try:
            with np.load(map_path, allow_pickle=False) as z:
                known = dict(zip(z["keys"].tolist(), z["frames"].tolist()))
        except (OSError, KeyError, ValueError):
            pass

        lengths, dirty = [], False
        for fp in filepaths:
            key = self.mel_key(fp, mel_params)
            frames = known.get(key)
            if frames is None:
                frames = self._frames_from_entry(self._path("mel", key))
                if frames is None:
                    frames = self._frames_estimate(fp, mel_params)
                known[key] = frames
                dirty = True
            lengths.append(frames)

        if dirty:
            tmp = f"{map_path}.{os.getpid()}.tmp.npz"
            try:
                os.makedirs(self.root, exist_ok=True)
                np.savez(tmp,
                         keys=np.array(list(known.keys())),
                         frames=np.array(list(known.values()), dtype=np.int32))
                os.replace(tmp, map_path)
            except OSError:
                try:
                    os.remove(tmp)
                except OSError:
                    pass
        return np.asarray(lengths, dtype=np.int64)

    @staticmethod
    def _frames_from_entry(path):
        """Shape of the cached mel from the npy header inside the (stored
        uncompressed) npz - no array data read."""
        import zipfile

        try:
            with zipfile.ZipFile(path) as z, z.open("mel.npy") as f:
                version = np.lib.format.read_magic(f)
                shape, _, _ = np.lib.format._read_array_header(f, version)
                return int(shape[-1])
        except (OSError, KeyError, ValueError, zipfile.BadZipFile):
            return None

    @staticmethod
    def _frames_estimate(filepath, mel_params):
        """Duration-based estimate from container metadata (no decode).
        soundfile, not torchaudio: torchaudio dropped its `info` API when
        decoding moved to torchcodec (torch 2.13-era stack)."""
        import soundfile as sf

        sample_rate, hop_length = int(float(mel_params[2])), int(float(mel_params[3]))
        info = sf.info(filepath)
        resampled = info.frames * sample_rate / info.samplerate
        return max(1, int(resampled // hop_length))


def default_cache_root():
    """<matcha_tts>/Datasets/feature_cache, resolved relative to this file
    (same 3-parents-up convention as cleaners.py's checkpoint fallback)."""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.normpath(os.path.join(here, "..", "..", ".."))
    return os.path.join(root, "Datasets", "feature_cache")


def phonemizer_signatures():
    """Signatures of everything that determines DP phonemizer output, for the
    text-cache key: the checkpoint file and the pronunciation-overrides
    CONTENT. Uses cleaners.py's own resolution so the key always tracks what
    the phonemizer would actually load."""
    import json

    from matcha.text.cleaners import _load_overrides_dict, _resolve_dp_checkpoint

    ckpt = _resolve_dp_checkpoint()
    ckpt_sig = _file_signature(ckpt) if ckpt and os.path.isfile(ckpt) else "unknown"

    overrides = _load_overrides_dict() or {}
    ovr_sig = hashlib.sha1(
        json.dumps(overrides, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    return [f"ckpt={ckpt_sig}", f"ovr={ovr_sig}"]
