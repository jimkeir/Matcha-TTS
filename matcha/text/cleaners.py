""" from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
import os
import re

from unidecode import unidecode

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# Remove brackets
_brackets_re = re.compile(r"[\[\]\(\)\{\}]")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [
    (re.compile(f"\\b{x[0]}\\.", re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def remove_brackets(text):
    return re.sub(_brackets_re, "", text)


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


# ---------------------------------------------------------------------
# eSpeak backend — lazy-initialised so that importing matcha.text.cleaners
# does NOT load libespeak-ng.dll. Any data config that uses
# english_cleaners_dp (the MIT alternative) never touches this path, so
# espeak can be uninstalled entirely. Only english_cleaners2 calls in.
# ---------------------------------------------------------------------
_global_phonemizer = None


def _get_espeak_phonemizer():
    global _global_phonemizer
    if _global_phonemizer is None:
        import phonemizer  # local — avoid module-level side effects
        critical_logger = logging.getLogger("phonemizer")
        critical_logger.setLevel(logging.CRITICAL)
        _global_phonemizer = phonemizer.backend.EspeakBackend(
            language="en-us",
            preserve_punctuation=True,
            with_stress=True,
            language_switch="remove-flags",
            logger=critical_logger,
        )
    return _global_phonemizer


def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = _get_espeak_phonemizer().phonemize([text], strip=True, njobs=1)[0]
    # Added in some cases espeak is not removing brackets
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


# ---------------------------------------------------------------------
# Deep Phonemizer cleaner — MIT-licensed alternative to eSpeak (GPL).
#
# Lazy-loaded on first call so that
#   (a) import-time cost only happens when this cleaner is actually used,
#       and
#   (b) users who stick with english_cleaners2 aren't forced to install
#       deep-phonemizer.
#
# The checkpoint path is resolvable via either a MATCHA_DP_CHECKPOINT env
# var (preferred — keeps this file machine-agnostic) or the
# fallback at matcha_tts/cmudict-0.7b/en_us_cmudict_ipa_forward.pt (matches our repo
# layout and the `curl` download step in the phonemizer-switch plan).
# ---------------------------------------------------------------------
_dp_phonemizer = None


def _resolve_dp_checkpoint():
    """Resolve the DP checkpoint path (MATCHA_DP_CHECKPOINT env var, else the
    repo-relative default). Shared by the phonemizer loader below and the
    feature cache (matcha/data/feature_cache.py), whose text-cache keys must
    track the same file. May return a non-existent path - callers check."""
    ckpt = os.environ.get("MATCHA_DP_CHECKPOINT")
    if not ckpt:
        # Default: repo-relative path matching our download location.
        # Three parents up from this file's directory:
        #   here     = matcha/text/         (dirname of cleaners.py)
        #   ..       = matcha/
        #   ../..    = Matcha-TTS-source/
        #   ../../.. = matcha_tts/
        # The checkpoint lives under Datasets/cmudict-0.7b/ alongside the
        # other corpora; the bare cmudict-0.7b/ root location is accepted
        # as a legacy fallback.
        here = os.path.dirname(os.path.abspath(__file__))
        root = os.path.normpath(os.path.join(here, "..", "..", ".."))
        for cand in (
            os.path.join(root, "Datasets", "cmudict-0.7b", "en_us_cmudict_ipa_forward.pt"),
            os.path.join(root, "cmudict-0.7b", "en_us_cmudict_ipa_forward.pt"),
        ):
            ckpt = cand
            if os.path.isfile(ckpt):
                break
    return ckpt


def _get_dp_phonemizer():
    global _dp_phonemizer
    if _dp_phonemizer is None:
        from dp.phonemizer import Phonemizer
        ckpt = _resolve_dp_checkpoint()
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"DP checkpoint not found at {ckpt}. Set MATCHA_DP_CHECKPOINT "
                f"env var or place the file at the default location. "
                f"Download URL: https://public-asai-dl-models.s3.eu-central-1"
                f".amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt")
        _dp_phonemizer = Phonemizer.from_checkpoint(ckpt)
    return _dp_phonemizer


# Word-level pronunciation overrides loaded from a sibling JSON file. Mirrors
# the C++ ATC runtime overlay (ATC_tts.cpp dp_word_overrides) so a fix added
# in one path can be ported to the other by copying one entry. Sentinel value
# `False` means "tried to load and got nothing" — distinguishes the empty-file
# case (where we can take the fast no-overrides path) from "haven't checked
# yet". Lazy + cached so the file is read at most once per process.
_dp_overrides = None


def _load_overrides_dict():
    """Lazy-load the overrides JSON. Returns dict (possibly empty).

    Search order:
      1. MATCHA_DP_OVERRIDES env var (full path), if set.
      2. Sibling of MATCHA_DP_CHECKPOINT (same directory), if env set.
      3. matcha_tts/cmudict-0.7b/phonemizer_overrides_en_us.json (default).

    Missing file → empty dict (engine behaves as if no overrides exist —
    same code path as before this overlay was added). Bad JSON → empty
    dict + a warning. Schema is flat {word: ipa_string}; keys starting
    with `_` are reserved for human-facing comments and skipped.
    """
    global _dp_overrides
    if _dp_overrides is not None:
        return _dp_overrides

    import json
    candidates = []
    explicit = os.environ.get("MATCHA_DP_OVERRIDES")
    if explicit:
        candidates.append(explicit)
    ckpt = os.environ.get("MATCHA_DP_CHECKPOINT")
    if ckpt:
        candidates.append(os.path.join(
            os.path.dirname(ckpt), "phonemizer_overrides_en_us.json"))
    here = os.path.dirname(os.path.abspath(__file__))
    candidates.append(os.path.normpath(os.path.join(
        here, "..", "..", "..",
        "cmudict-0.7b", "phonemizer_overrides_en_us.json")))

    for path in candidates:
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            logging.getLogger(__name__).warning(
                "Failed to load pronunciation overrides from %s: %s",
                path, e)
            _dp_overrides = {}
            return _dp_overrides
        if not isinstance(raw, dict):
            _dp_overrides = {}
            return _dp_overrides
        _dp_overrides = {
            k: v for k, v in raw.items()
            if isinstance(k, str) and isinstance(v, str) and not k.startswith("_")
        }
        return _dp_overrides

    _dp_overrides = {}
    return _dp_overrides


def _phonemize_with_overrides(text, overrides):
    """Per-word DP with override interception. Used only when overrides are
    non-empty; the no-overrides path skips this and runs DP on the full
    sentence (preserving the old code's exact output for users without an
    overrides file)."""
    import re
    parts = re.findall(r"\S+|\s+", text)
    out_chunks = []
    pending = []  # contiguous run of non-overridden tokens, fed to DP together

    def flush_pending():
        if not pending:
            return
        sub = " ".join(pending)
        ipa = _get_dp_phonemizer()(sub, lang="en_us")
        ipa = remove_brackets(ipa)
        ipa = collapse_whitespace(ipa).strip()
        if ipa:
            out_chunks.append(ipa)
        pending.clear()

    for p in parts:
        if p.isspace():
            continue  # spacing is reconstructed by collapse_whitespace below
        # Override lookup: exact, lowercase, then punct-stripped lowercase.
        # Matches the C++ dp_lookup_word chain so a single overrides JSON
        # fits both paths.
        lookup_keys = [p, p.lower()]
        stripped = p.strip(".,!?;:").lower()
        if stripped and stripped not in lookup_keys:
            lookup_keys.append(stripped)
        ovr = None
        for k in lookup_keys:
            if k in overrides:
                ovr = overrides[k]
                break
        if ovr is not None:
            flush_pending()
            # Re-emit any trailing prosodic punctuation as space-separated
            # IPA tokens, matching what DP would have produced (split_words
            # in the C++ phonemizer treats `.,!?;:` as their own tokens; the
            # text→symbol-id step in Matcha then assigns them their own IDs
            # and intersperse_blanks adds the surrounding pad).
            trailing = ""
            for c in reversed(p):
                if c in ".,!?;:":
                    trailing = c + trailing
                else:
                    break
            out_chunks.append(ovr)
            for c in trailing:
                out_chunks.append(c)
        else:
            pending.append(p)

    flush_pending()
    return collapse_whitespace(" ".join(out_chunks)).strip()


def english_cleaners_dp(text):
    """English cleaner using Deep Phonemizer (MIT) instead of eSpeak-NG (GPL).

    Output format matches english_cleaners2 shape-wise (lowercase IPA string,
    whitespace collapsed). DP's IPA inventory is a subset of Matcha's symbol
    table — verified across ATC phrases and common English pangrams — so no
    symbol-table extension is required.

    Optional word-level overrides (loaded via _load_overrides_dict) win over
    DP and bypass the neural phonemizer for matching words. With no
    overrides file present the function takes a fast path that is
    bit-identical to the pre-overlay implementation.

    Differences vs eSpeak output worth knowing:
    - No stress markers (ˈ, ˌ). Matcha's prosody is learned end-to-end, so
      this just means the model learns to distribute stress from context
      rather than from explicit markers.
    - DP uses ASCII `r` where eSpeak uses `ɹ`; both are in Matcha's symbols.
    - DP's pronunciations sometimes differ from eSpeak's (e.g. DP correctly
      pronounces 'Colne' as 'koʊln'; eSpeak says 'kˈɑːlni'). Neither is
      globally better — DP is CMUdict-derived, eSpeak is rule-based.
    """
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    # DP passes hyphens AND slashes through verbatim; Matcha's symbol
    # table has neither, which triggers KeyError in text_to_sequence.
    # Compound words / conjunctions ("super-imposition", "and/or",
    # "AC/DC") and fractions ("1/2") pronounce identically when
    # tokenised as separate words, so replace both with spaces before
    # DP sees the text. CommonVoice transcripts in particular have a
    # lot of slashes (VCTK had only hyphens, hence the original
    # one-char fix); see prepare_commonvoice_filelists.py for the
    # broader sanitiser that covers other rare characters too.
    text = text.replace("-", " ")
    text = text.replace("/", " ")

    overrides = _load_overrides_dict()
    if overrides:
        return _phonemize_with_overrides(text, overrides)

    phonemes = _get_dp_phonemizer()(text, lang="en_us")
    phonemes = remove_brackets(phonemes)
    phonemes = collapse_whitespace(phonemes)
    return phonemes


def ipa_simplifier(text):
    replacements = [
        ("ɐ", "ə"),
        ("ˈə", "ə"),
        ("ʤ", "dʒ"),
        ("ʧ", "tʃ"),
        ("ᵻ", "ɪ"),
    ]
    for replacement in replacements:
        text = text.replace(replacement[0], replacement[1])
    phonemes = collapse_whitespace(text)
    return phonemes


# I am removing this due to incompatibility with several version of python
# However, if you want to use it, you can uncomment it
# and install piper-phonemize with the following command:
# pip install piper-phonemize

# import piper_phonemize
# def english_cleaners_piper(text):
#     """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
#     text = convert_to_ascii(text)
#     text = lowercase(text)
#     text = expand_abbreviations(text)
#     phonemes = "".join(piper_phonemize.phonemize_espeak(text=text, voice="en-US")[0])
#     phonemes = collapse_whitespace(phonemes)
#     return phonemes
