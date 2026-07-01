import argparse
import random
from pathlib import Path

import numpy as np
import torch
from lightning import LightningModule

from matcha.cli import VOCODER_URLS, load_matcha, load_vocoder

DEFAULT_OPSET = 15

SEED = 1234
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


class MatchaWithVocoder(LightningModule):
    def __init__(self, matcha, vocoder):
        super().__init__()
        self.matcha = matcha
        self.vocoder = vocoder

    def forward(self, x, x_lengths, scales, spks=None):
        mel, mel_lengths = self.matcha(x, x_lengths, scales, spks)
        wavs = self.vocoder(mel).clamp(-1, 1)
        lengths = mel_lengths * 256
        return wavs.squeeze(1), lengths


def get_exportable_module(matcha, vocoder, n_timesteps, fixed_out_length=None):
    """
    Return an appropriate `LighteningModule` and output-node names
    based on whether the vocoder is embedded in  the final graph.

    fixed_out_length (int|None): when set, the mel/time dimension is pinned to
    this constant (multiple of 4) for a fully-static export — see
    MatchaTTS.synthesise. Used for the CoreML/MLProgram static models; leave
    None for the default dynamic export.
    """

    def onnx_forward_func(x, x_lengths, scales, spks=None):
        """
        Custom forward function for accepting
        scaler parameters as tensors
        """
        # Extract scaler parameters from tensors
        temperature = scales[0]
        length_scale = scales[1]
        output = matcha.synthesise(x, x_lengths, n_timesteps, temperature, spks, length_scale,
                                   fixed_out_length=fixed_out_length)
        return output["mel"], output["mel_lengths"]

    # Monkey-patch Matcha's forward function
    matcha.forward = onnx_forward_func

    if vocoder is None:
        model, output_names = matcha, ["mel", "mel_lengths"]
    else:
        model = MatchaWithVocoder(matcha, vocoder)
        output_names = ["wav", "wav_lengths"]
    return model, output_names


def get_inputs(is_multi_speaker, dummy_input_length=50):
    """
    Create dummy inputs for tracing. dummy_input_length sets x's traced
    phoneme-axis size; for a static export (time axis not in dynamic_axes)
    this becomes the model's fixed input length, so pass the tier size.
    """
    x = torch.randint(low=0, high=20, size=(1, dummy_input_length), dtype=torch.long)
    x_lengths = torch.LongTensor([dummy_input_length])

    # Scales
    temperature = 0.667
    length_scale = 1.0
    scales = torch.Tensor([temperature, length_scale])

    model_inputs = [x, x_lengths, scales]
    input_names = [
        "x",
        "x_lengths",
        "scales",
    ]

    if is_multi_speaker:
        spks = torch.LongTensor([1])
        model_inputs.append(spks)
        input_names.append("spks")

    return tuple(model_inputs), input_names


def main():
    parser = argparse.ArgumentParser(description="Export 🍵 Matcha-TTS to ONNX")

    parser.add_argument(
        "checkpoint_path",
        type=str,
        help="Path to the model checkpoint",
    )
    parser.add_argument("output", type=str, help="Path to output `.onnx` file")
    parser.add_argument(
        "--n-timesteps", type=int, default=5, help="Number of steps to use for reverse diffusion in decoder (default 5)"
    )
    parser.add_argument(
        "--vocoder-name",
        type=str,
        # Allow "vocos" alongside the upstream HiFi-GAN names. matcha.cli's
        # load_vocoder dispatches on the same string and raises
        # NotImplementedError for unknown names, so the validation still
        # happens — just one layer down rather than at parse time.
        choices=list(VOCODER_URLS.keys()) + ["vocos"],
        default=None,
        help="Name of the vocoder to embed in the ONNX graph",
    )
    parser.add_argument(
        "--vocoder-checkpoint-path",
        type=str,
        default=None,
        help="Vocoder checkpoint to embed  in the ONNX graph for an `e2e` like experience",
    )
    parser.add_argument("--opset", type=int, default=DEFAULT_OPSET, help="ONNX opset version to use (default 15")
    parser.add_argument(
        "--fixed-length", type=int, default=None,
        help="Export a STATIC-shape model: fix the input phoneme axis to this "
             "many tokens (no dynamic 'time' axis). Required together with "
             "--out-frames. Used for the CoreML/MLProgram tier models; omit "
             "for the default dynamic export consumed by CPU/DirectML.",
    )
    parser.add_argument(
        "--out-frames", type=int, default=None,
        help="Static mel/output length (Y_MAX) baked into the decoder when "
             "--fixed-length is set. Must be a multiple of 4 and >= the largest "
             "mel-frame count any chunk of --fixed-length phonemes produces "
             "(undersize = truncated audio). Size from the runtime mel-frame "
             "histogram (GATHER_PHONEME_STATS).",
    )

    args = parser.parse_args()

    static_export = args.fixed_length is not None or args.out_frames is not None
    if static_export:
        if args.fixed_length is None or args.out_frames is None:
            parser.error("--fixed-length and --out-frames must be given together.")
        if args.out_frames % 4 != 0:
            parser.error("--out-frames must be a multiple of 4 (fix_len_compatibility).")

    print(f"[🍵] Loading Matcha checkpoint from {args.checkpoint_path}")
    print(f"Setting n_timesteps to {args.n_timesteps}")

    checkpoint_path = Path(args.checkpoint_path)
    matcha = load_matcha(checkpoint_path.stem, checkpoint_path, "cpu")

    if args.vocoder_name or args.vocoder_checkpoint_path:
        assert (
            args.vocoder_name and args.vocoder_checkpoint_path
        ), "Both vocoder_name and vocoder-checkpoint are required when embedding the vocoder in the ONNX graph."
        vocoder, _ = load_vocoder(args.vocoder_name, args.vocoder_checkpoint_path, "cpu")
    else:
        vocoder = None

    is_multi_speaker = matcha.n_spks > 1

    dummy_len = args.fixed_length if static_export else 50
    dummy_input, input_names = get_inputs(is_multi_speaker, dummy_input_length=dummy_len)
    model, output_names = get_exportable_module(
        matcha, vocoder, args.n_timesteps,
        fixed_out_length=(args.out_frames if static_export else None),
    )

    # Dynamic-axes map. For a static export we keep ONLY batch_size dynamic and
    # drop every 'time' axis, so the phoneme (input) and mel/wav (output)
    # dimensions are concrete integers throughout — the precondition CoreML
    # MLProgram needs. The dynamic export keeps the original time axes.
    if static_export:
        print(f"Static export: input phonemes fixed to {args.fixed_length}, "
              f"output mel frames fixed to {args.out_frames}.")
        dynamic_axes = {
            "x": {0: "batch_size"},
            "x_lengths": {0: "batch_size"},
        }
        if vocoder is None:
            dynamic_axes.update({"mel": {0: "batch_size"}, "mel_lengths": {0: "batch_size"}})
        else:
            print("Embedding the vocoder in the ONNX graph")
            dynamic_axes.update({"wav": {0: "batch_size"}, "wav_lengths": {0: "batch_size"}})
    else:
        # Set dynamic shape for inputs/outputs
        dynamic_axes = {
            "x": {0: "batch_size", 1: "time"},
            "x_lengths": {0: "batch_size"},
        }

        if vocoder is None:
            dynamic_axes.update(
                {
                    "mel": {0: "batch_size", 2: "time"},
                    "mel_lengths": {0: "batch_size"},
                }
            )
        else:
            print("Embedding the vocoder in the ONNX graph")
            dynamic_axes.update(
                {
                    "wav": {0: "batch_size", 1: "time"},
                    "wav_lengths": {0: "batch_size"},
                }
            )

    if is_multi_speaker:
        dynamic_axes["spks"] = {0: "batch_size"}

    # Create the output directory (if not exists)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    model.to_onnx(
        args.output,
        dummy_input,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=args.opset,
        export_params=True,
        do_constant_folding=True,
    )
    print(f"[🍵] ONNX model exported to  {args.output}")


if __name__ == "__main__":
    main()
