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


def _get_dp_phonemizer():
    global _dp_phonemizer
    if _dp_phonemizer is None:
        from dp.phonemizer import Phonemizer
        ckpt = os.environ.get("MATCHA_DP_CHECKPOINT")
        if not ckpt:
            # Default: repo-relative path matching our download location.
            # Three parents up from this file's directory:
            #   here     = matcha/text/         (dirname of cleaners.py)
            #   ..       = matcha/
            #   ../..    = Matcha-TTS-source/
            #   ../../.. = matcha_tts/
            # Then into cmudict-0.7b/ where the checkpoint lives.
            here = os.path.dirname(os.path.abspath(__file__))
            ckpt = os.path.normpath(os.path.join(
                here, "..", "..", "..",
                "cmudict-0.7b", "en_us_cmudict_ipa_forward.pt"))
        if not os.path.isfile(ckpt):
            raise FileNotFoundError(
                f"DP checkpoint not found at {ckpt}. Set MATCHA_DP_CHECKPOINT "
                f"env var or place the file at the default location. "
                f"Download URL: https://public-asai-dl-models.s3.eu-central-1"
                f".amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa_forward.pt")
        _dp_phonemizer = Phonemizer.from_checkpoint(ckpt)
    return _dp_phonemizer


def english_cleaners_dp(text):
    """English cleaner using Deep Phonemizer (MIT) instead of eSpeak-NG (GPL).

    Output format matches english_cleaners2 shape-wise (lowercase IPA string,
    whitespace collapsed). DP's IPA inventory is a subset of Matcha's symbol
    table — verified across ATC phrases and common English pangrams — so no
    symbol-table extension is required.

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
