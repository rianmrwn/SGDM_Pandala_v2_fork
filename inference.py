import argparse
import os
import sys
from typing import Optional, Sequence

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.append(ROOT)

try:
    import imageio.v2 as imageio
except ImportError:
    imageio = None

from ldm.xformers_state import disable_xformers
from utils.common import instantiate_from_config, load_state_dict


def parse_band_indices(spec: Optional[str]) -> Optional[Sequence[int]]:
    if spec is None:
        return None
    indices = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        try:
            idx = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid band index '{token}'") from exc
        if idx > 0:
            idx -= 1
        indices.append(idx)
    return indices or None


def read_raster(path: str) -> np.ndarray:
    if imageio is not None:
        array = imageio.imread(path)
    else:
        with Image.open(path) as img:
            array = np.array(img)
    if array.ndim == 3 and array.shape[0] <= 16 and array.shape[0] < array.shape[1] and array.shape[0] < array.shape[2]:
        array = np.transpose(array, (1, 2, 0))
    if array.ndim == 2:
        array = array[..., None]
    if array.ndim != 3:
        raise ValueError(f"Unsupported image shape {array.shape} for '{path}'")
    return array


def select_channels(array: np.ndarray, bands: Optional[Sequence[int]]) -> np.ndarray:
    data = array
    if bands is not None:
        if any(idx < 0 or idx >= data.shape[2] for idx in bands):
            raise ValueError(
                f"Band indices {bands} out of range for {data.shape[2]}-channel image"
            )
        data = data[..., bands]
    elif data.shape[2] > 3:
        data = data[..., :3]
    if data.shape[2] == 1:
        data = np.repeat(data, 3, axis=2)
    if data.shape[2] != 3:
        raise ValueError(f"Expected 3 channels after band selection, got {data.shape[2]}")
    return data


def normalize_image(
    array: np.ndarray,
    min_value: Optional[float],
    max_value: Optional[float],
    scale: Optional[float],
) -> np.ndarray:
    data = array.astype(np.float32)
    if scale is not None and scale > 0:
        data /= scale
    else:
        data_min = float(min_value) if min_value is not None else float(data.min())
        data_max = float(max_value) if max_value is not None else float(data.max())
        data = data - data_min
        denom = data_max - data_min
        if denom <= 0:
            denom = 1.0
        data /= denom
    return np.clip(data, 0.0, 1.0)


def array_to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32).contiguous()


def match_size(tensor: torch.Tensor, target_hw: Sequence[int]) -> torch.Tensor:
    if tensor.shape[-2:] != tuple(target_hw):
        tensor = F.interpolate(tensor, size=target_hw, mode="bilinear", align_corners=False)
    return tensor


def apply_padding(tensor: torch.Tensor, pad_h: int, pad_w: int) -> torch.Tensor:
    if pad_h or pad_w:
        tensor = F.pad(tensor, (0, pad_w, 0, pad_h), mode="reflect")
    return tensor


def save_image(path: str, image: np.ndarray, bit_depth: int) -> None:
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    image = np.clip(image, 0.0, 1.0)
    if bit_depth == 16:
        if imageio is None:
            raise RuntimeError("Writing 16-bit output requires imageio (pip install imageio)")
        data = (image * 65535.0).round().astype(np.uint16)
        imageio.imwrite(path, data)
    else:
        data = (image * 255.0).round().astype(np.uint8)
        if imageio is not None:
            imageio.imwrite(path, data)
        else:
            Image.fromarray(data).save(path)


def resolve_device(device_str: str) -> str:
    device = device_str.lower()
    if device.startswith("cuda"):
        if not torch.cuda.is_available():
            print("CUDA not available, falling back to CPU")
            device = "cpu"
    elif device == "mps":
        if not torch.backends.mps.is_available():
            print("MPS not available, falling back to CPU")
            device = "cpu"
    else:
        device = "cpu"
    if not device.startswith("cuda"):
        disable_xformers()
    return device


def prepare_image(
    path: str,
    bands: Optional[Sequence[int]],
    min_value: Optional[float],
    max_value: Optional[float],
    scale: Optional[float],
) -> np.ndarray:
    array = read_raster(path)
    array = select_channels(array, bands)
    return normalize_image(array, min_value, max_value, scale)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-image inference script for SGDM reference-guided super-resolution",
    )
    parser.add_argument("--config", type=str, required=True, help="Path to model config yaml")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--input", type=str, required=True, help="Path to the low-resolution input image (.tif)")
    parser.add_argument("--reference", type=str, default=None, help="Optional reference image path; defaults to the input image")
    parser.add_argument("--style", type=str, default=None, help="Optional style image path; defaults to the reference image")
    parser.add_argument("--output", type=str, required=True, help="Where to save the super-resolved output image")
    parser.add_argument("--bands", type=str, default=None, help="Comma-separated band indices (1-based) to extract, e.g. '4,3,2'")
    parser.add_argument("--min-value", type=float, default=None, help="Optional absolute minimum used for normalization")
    parser.add_argument("--max-value", type=float, default=None, help="Optional absolute maximum used for normalization")
    parser.add_argument("--value-scale", type=float, default=None, help="Optional scale divisor for normalization (applied before min/max)")
    parser.add_argument("--prompt", type=str, default="", help="Optional text prompt to guide the diffusion model")
    parser.add_argument("--steps", type=int, default=50, help="Number of diffusion sampling steps")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Computation device, e.g. 'cuda', 'cuda:0', 'cpu'")
    parser.add_argument("--sample-style", action="store_true", help="Enable style sampling from the normalizing flow")
    parser.add_argument("--flow-mean", type=str, default=None, help="Checkpoint path for the flow mean estimator")
    parser.add_argument("--flow-std", type=str, default=None, help="Checkpoint path for the flow std estimator")
    parser.add_argument("--style-scale", type=float, default=1.0, help="Scaling factor applied when using sampled styles")
    parser.add_argument("--output-bit-depth", type=int, choices=[8, 16], default=8, help="Bit depth for the saved output image")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pl.seed_everything(args.seed)

    device_str = resolve_device(args.device)
    device = torch.device(device_str)
    print(f"Using device: {device_str}")

    print(f"Loading config from {args.config}")
    config = OmegaConf.load(args.config)
    model = instantiate_from_config(config)

    print(f"Loading checkpoint from {args.ckpt}")
    checkpoint = torch.load(args.ckpt, map_location="cpu")
    load_state_dict(model, checkpoint, strict=False)
    model.freeze()
    model.to(device)
    model.eval()
    if hasattr(model, "cond_stage_model") and model.cond_stage_model is not None:
        model.cond_stage_model.eval()
    if hasattr(model, "first_stage_model") and model.first_stage_model is not None:
        model.first_stage_model.eval()

    bands = parse_band_indices(args.bands)

    print(f"Reading source image {args.input}")
    sr_array = prepare_image(args.input, bands, args.min_value, args.max_value, args.value_scale)
    if args.reference:
        ref_array = prepare_image(args.reference, bands, args.min_value, args.max_value, args.value_scale)
    else:
        ref_array = sr_array
    if args.style:
        style_array = prepare_image(args.style, bands, args.min_value, args.max_value, args.value_scale)
    else:
        style_array = ref_array

    sr_tensor = array_to_tensor(sr_array, device)
    ref_tensor = array_to_tensor(ref_array, device)
    style_tensor = array_to_tensor(style_array, device)

    original_h, original_w = sr_tensor.shape[-2:]
    ref_tensor = match_size(ref_tensor, (original_h, original_w))
    style_tensor = match_size(style_tensor, (original_h, original_w))

    pad_h = (64 - original_h % 64) % 64
    pad_w = (64 - original_w % 64) % 64
    sr_tensor = apply_padding(sr_tensor, pad_h, pad_w)
    ref_tensor = apply_padding(ref_tensor, pad_h, pad_w)

    if args.sample_style:
        style_tensor = None
    else:
        style_tensor = apply_padding(style_tensor, pad_h, pad_w)

    batch_size = sr_tensor.shape[0]

    if args.sample_style:
        if not args.flow_mean or not args.flow_std:
            raise ValueError("--flow-mean and --flow-std are required when --sample-style is enabled")
        from model.Flows.mu_sigama_estimate_normflows import CreateFlow

        flow_kwargs = dict(dim=32, num_layers=16, hidden_layers=[16, 64, 64, 32])
        flow_mean = CreateFlow(**flow_kwargs)
        flow_std = CreateFlow(**flow_kwargs)
        flow_mean_state = torch.load(args.flow_mean, map_location="cpu")
        flow_std_state = torch.load(args.flow_std, map_location="cpu")
        load_state_dict(flow_mean, flow_mean_state, strict=True)
        load_state_dict(flow_std, flow_std_state, strict=True)
        model.flow_mean = flow_mean.to(device).eval()
        model.flow_std = flow_std.to(device).eval()
        model.if_sample_style = True
        model.style_scale = args.style_scale
        with torch.no_grad():
            style_mean, _ = model.flow_mean.sample(batch_size)
            style_std, _ = model.flow_std.sample(batch_size)
        model.style_mean = style_mean.to(device).unsqueeze(-1).unsqueeze(-1)
        model.style_std = style_std.to(device).unsqueeze(-1).unsqueeze(-1)
    else:
        model.if_sample_style = False
        model.style_scale = args.style_scale
        model.style_mean = None
        model.style_std = None

    prompt = args.prompt or ""

    with torch.no_grad():
        cond_latent = model.apply_cond_ref_encoder(sr_tensor, ref_tensor, style_tensor)
        cond_text = model.get_learned_conditioning([prompt]).to(device)
        cond = {
            "c_crossattn": [cond_text],
            "sr_ref_cond_latent": [cond_latent],
            "lq": [sr_tensor],
        }
        samples = model.sample_log(cond=cond, steps=args.steps)

    samples = samples[:, :, :original_h, :original_w]
    output = samples.squeeze(0).permute(1, 2, 0).cpu().numpy()

    print(f"Saving result to {args.output}")
    save_image(args.output, output, args.output_bit_depth)


if __name__ == "__main__":
    main()
