"""
GSAOM: Provably Secure Generative Steganography Based on Adjustable Orthogonal Mapping.

Reference:
    Qinghua Zhang and Fangjun Huang,
    "Provably Secure Generative Steganography Based on Adjustable Orthogonal Mapping",
    IEEE Transactions on Dependable and Secure Computing (TDSC), 2026.

This is a single-file, training-free implementation built on Stable Diffusion v1.5.
"""

import os
import math
import json
import argparse

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from diffusers import (
    StableDiffusionPipeline,
    PNDMScheduler,
    DDIMScheduler,
    UniPCMultistepScheduler,
    DEISMultistepScheduler,
    DPMSolverMultistepScheduler,
    DDIMInverseScheduler,
    DPMSolverMultistepInverseScheduler,
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def img_to_latents(img, device, dtype, pipe_inv):
    """Encode a PIL image into the SD latent space."""
    img = transforms.ToTensor()(img)
    img = img.unsqueeze(0).to(device=device, dtype=dtype) * 2. - 1
    posterior = pipe_inv.vae.encode(img).latent_dist
    latents = posterior.mean * 0.18215
    return latents


def cal_bcr(msg, msgo):
    """Bit correct ratio."""
    count = 0
    for x in range(0, msg.shape[0]):
        if msg[x] == msgo[x]:
            count = count + 1
    return 1. * count / msg.shape[0]


def gray_to_bin(gray_arr, extend):
    """Gray code -> binary.

    Within each block of length `extend`, the first bit serves as the seed, and
    the remaining bits are computed via cumulative XOR.
    """
    bin_arr = gray_arr.clone().long()
    for i in range(len(gray_arr)):
        if i % extend == 0:
            bin_arr[i] = gray_arr[i]
        else:
            bin_arr[i] = int(gray_arr[i]) ^ int(bin_arr[i - 1])
    return bin_arr


def bin_to_gray(bin_arr, extend):
    """Binary -> Gray code.

    Within each block of length `extend`, the first bit serves as the seed, and
    the remaining bits are computed via adjacent XOR.
    """
    gray_arr = bin_arr.clone().long()
    for i in range(len(bin_arr)):
        if i % extend == 0:
            gray_arr[i] = bin_arr[i]
        else:
            gray_arr[i] = int(bin_arr[i]) ^ int(bin_arr[i - 1])
    return gray_arr


# --------------------------------------------------------------------------- #
# Adjustable Orthogonal Mapping (AOM)
# --------------------------------------------------------------------------- #
def _get_orth_matrices(shape, args):
    """Build key-driven orthogonal matrices via QR decomposition."""
    torch.manual_seed(args.seed * 3 + args.mseed)
    noise_matrix = torch.randn(shape[0], shape[1], shape[3], shape[3])

    mats = []
    for i in range(shape[1]):
        a = noise_matrix[0, i, :, :]
        q, _ = torch.linalg.qr(a)
        mats.append(q)
    return mats


def orthogonal_transform(latents, shape, args):
    lat_in = latents.detach().cpu().reshape(
        shape[0], shape[1], shape[2], shape[3]).float()

    mats = _get_orth_matrices(shape, args)

    out = torch.zeros_like(lat_in)
    for i in range(shape[1]):
        out[0, i, :, :] = lat_in[0, i, :, :] @ mats[i]
    return out.to(device=latents.device, dtype=latents.dtype)


def inverse_orthogonal_transform(z, shape, args):
    mats = _get_orth_matrices(shape, args)
    zz = z.reshape(shape[0], shape[1], shape[2], shape[3])

    out = torch.zeros_like(zz)
    for i in range(shape[1]):
        out[0, i, :, :] = zz[0, i, :, :] @ mats[i].t()
    return out


def message_to_latents(msg, device, shape, args):
    total_elements = shape[0] * shape[1] * shape[2] * shape[3]
    b = args.bits_per_element
    s = args.strength

    if b == 1:
        msg_len = total_elements // s
        assert msg.size(0) == msg_len

        msg_float = msg.float()
        signs = msg_float * 2.0 - 1.0
        expanded = torch.zeros(msg_len, s, dtype=torch.float32)
        expanded[:, 0] = signs * math.sqrt(s)
        expanded = expanded.reshape(-1)

        if expanded.size(0) < total_elements:
            pad = torch.zeros(total_elements - expanded.size(0), dtype=torch.float32)
            expanded = torch.cat([expanded, pad], dim=0)

        latents = expanded.reshape(shape)

        np.random.seed(args.seed)
        seed_shuffle = np.random.randint(0, 10000, size=(shape[1], shape[2]))
        latents_shuffled = torch.zeros_like(latents)

        for i in range(shape[1]):
            for j in range(shape[2]):
                np.random.seed(seed_shuffle[i, j])
                ind = np.random.permutation(shape[3])
                latents_shuffled[0, i, j, :] = latents[0, i, j, torch.from_numpy(ind).long()]

        latents = latents_shuffled.to(device=device, dtype=args.dtype)
        latents = orthogonal_transform(latents, shape, args)
        return latents

    else:
        assert s == 1
        assert msg.size(0) == total_elements * b

        bits = msg.reshape(total_elements, b).long()
        weights = (2 ** torch.arange(b - 1, -1, -1)).long()
        k = (bits * weights).sum(dim=1)

        D = math.sqrt((2 ** (2 * b) - 1) / 3.0)
        z = (2.0 * k.float() - (2 ** b - 1)) / D
        latents = z.reshape(shape).to(device=device, dtype=args.dtype)
        latents = orthogonal_transform(latents, shape, args)
        return latents


def latents_to_message(latents, shape, args):
    total_elements = shape[0] * shape[1] * shape[2] * shape[3]
    b = args.bits_per_element
    s = args.strength

    z = latents.detach().cpu().reshape(shape[0], shape[1], shape[2], shape[3]).float()
    z = inverse_orthogonal_transform(z, shape, args)

    if b == 1:
        np.random.seed(args.seed)
        seed_shuffle = np.random.randint(0, 10000, size=(shape[1], shape[2]))
        z_unshuffled = torch.zeros_like(z)

        for i in range(shape[1]):
            for j in range(shape[2]):
                np.random.seed(seed_shuffle[i, j])
                ind = np.random.permutation(shape[3])
                inv_ind = np.empty_like(ind)
                inv_ind[ind] = np.arange(shape[3])
                z_unshuffled[0, i, j, :] = z[0, i, j, torch.from_numpy(inv_ind).long()]

        z_flat = z_unshuffled.reshape(-1)
        msg_len = total_elements // s
        idx = torch.arange(msg_len) * s
        z_picks = z_flat[idx]
        msg_out = (z_picks > 0).float()
        return msg_out

    else:
        z_flat = z.reshape(-1)
        D = math.sqrt((2 ** (2 * b) - 1) / 3.0)
        k_float = (z_flat * D + (2 ** b - 1)) / 2.0
        k = torch.round(k_float).long().clamp(0, (1 << b) - 1)

        bits_list = [(k >> (b - 1 - i)) & 1 for i in range(b)]
        msg_out = torch.stack(bits_list, dim=1).reshape(-1).float()
        return msg_out


# --------------------------------------------------------------------------- #
# Hiding / extraction
# --------------------------------------------------------------------------- #
def message_hiding(smsg, device, shape, args, pipe, width, height):
    latents_msg = message_to_latents(smsg, device, shape, args)

    stego_image = pipe(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_steps,
        width=width,
        height=height,
        latents=latents_msg,
    ).images[0]

    return stego_image, latents_msg


def message_extraction(image, device, shape, args, pipe_inv, width, height):
    latents_image = img_to_latents(image, device, args.dtype, pipe_inv)

    latents_extract, _ = pipe_inv(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        guidance_scale=args.guidance_inv_scale,
        num_inference_steps=args.num_steps,
        width=width,
        height=height,
        output_type='latent',
        return_dict=False,
        latents=latents_image,
    )

    msgout = latents_to_message(latents_extract, shape, args)
    return msgout, latents_extract


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
PROMPT_POOL = [
    "A dog playing fetch in a backyard.",
    "A farmer harvesting crops in a field.",
    "A quiet library with people reading.",
    "A bridge over a calm river at sunset.",
    "A child flying a kite in a park.",
    "A lighthouse standing on a rocky shore.",
]


def build_parser():
    parser = argparse.ArgumentParser(
        description="GSAOM: generative steganography via adjustable orthogonal mapping")

    parser.add_argument("--model_path", default="./models/stable-diffusion-v1-5", type=str,
                        help="Path or Hub id of the Stable Diffusion v1.5 checkpoint")
    parser.add_argument("--output_dir", default="./results", type=str,
                        help="Directory for stego images and logs")
    parser.add_argument("--device", default=None, type=str,
                        help="e.g. cuda:0 / cpu (auto-detected by default)")

    parser.add_argument("--dtype", default=torch.float16, help="dtype")
    parser.add_argument("--width", default=512, type=int,
                        help="Stego image width in pixels (must be a multiple of 8)")
    parser.add_argument("--height", default=512, type=int,
                        help="Stego image height in pixels (must be a multiple of 8)")
    parser.add_argument("--image_size", default=None, type=int,
                        help="Shortcut: sets both --width and --height to this value")
    parser.add_argument("--num_steps", default=50, type=int, help="Number of diffusion steps")
    parser.add_argument("--guidance_scale", default=7.5, type=float, help="CFG scale (generation)")
    parser.add_argument("--guidance_inv_scale", default=7.5, type=float, help="CFG scale (inversion)")

    parser.add_argument("--prompt", default=PROMPT_POOL[0], type=str, help="Prompt")
    parser.add_argument("--negative_prompt", default="", type=str, help="Negative prompt")

    parser.add_argument("--strength", default=1, type=int,
                        help="Spreading factor s: each message bit occupies s latent elements")
    parser.add_argument("--bits_per_element", default=1, type=int,
                        help="Bits carried by each latent element")
    parser.add_argument("--seed", default=10001, type=int,
                        help="Shared key: controls permutation and the orthogonal matrices")
    parser.add_argument("--mseed", default=1, type=int, help="Message seed")

    parser.add_argument("--num_prompt", default=1, type=int,
                        help="Number of prompts to evaluate (max 6)")
    parser.add_argument("--num_mseed", default=1, type=int,
                        help="Number of random messages per prompt")
    return parser


def resolve_resolution(args):
    """Resolve (width, height) and validate against the VAE downsample factor.

    The AOM core is resolution-agnostic: the orthogonal matrices are built on the
    width dimension (W x W) and applied as (H x W) @ (W x W), and the permutation
    is applied row-wise along W. So rectangular latents work as long as sender and
    receiver agree on the exact shape.
    """
    if args.image_size is not None:
        args.width = args.height = args.image_size

    if args.width % 8 != 0 or args.height % 8 != 0:
        raise ValueError(
            f"--width and --height must be multiples of 8 (VAE downsample factor), "
            f"got {args.width}x{args.height}")
    if args.width <= 0 or args.height <= 0:
        raise ValueError(f"Resolution must be positive, got {args.width}x{args.height}")

    return args.width, args.height


def main():
    args = build_parser().parse_args()

    # Constraint: at least one of strength and bits_per_element must be 1.
    if args.bits_per_element != 1 and args.strength != 1:
        raise ValueError("At least one of --strength and --bits_per_element must be 1")
    if args.bits_per_element != 1:
        args.strength = 1
    elif args.strength != 1:
        args.bits_per_element = 1

    device = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')

    width, height = resolve_resolution(args)
    shape = [1, 4, height // 8, width // 8]   # [B, C, H, W] in latent space
    total_elements = shape[0] * shape[1] * shape[2] * shape[3]

    num_prompt = min(args.num_prompt, len(PROMPT_POOL))
    num_mseed = args.num_mseed

    pipe = StableDiffusionPipeline.from_pretrained(
        args.model_path, safety_checker=None, torch_dtype=args.dtype)
    pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
    pipe.to(device)

    pipe_inv = StableDiffusionPipeline.from_pretrained(
        args.model_path, safety_checker=None, torch_dtype=args.dtype)
    pipe_inv.scheduler = DPMSolverMultistepInverseScheduler.from_config(pipe_inv.scheduler.config)
    pipe_inv.to(device)

    sth = args.strength * 100 + args.bits_per_element
    save_dir = os.path.join(
        args.output_dir, "orthogonal", f"{width}x{height}", f"s{args.strength}_b{args.bits_per_element}")
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(vars(args) | {
            "shape": shape,
            "width": width,
            "height": height,
            "payload_bits": math.floor(total_elements * args.bits_per_element / args.strength),
        }, f, indent=2, default=str)

    bcr = np.zeros((num_prompt, num_mseed))

    for ip in range(num_prompt):
        for im in range(num_mseed):
            args.prompt = PROMPT_POOL[ip]
            args.mseed = im

            torch.manual_seed(args.mseed)
            msg_length = math.floor(total_elements * args.bits_per_element / args.strength)
            smsg = torch.randint(0, 2, [msg_length])

            if args.bits_per_element > 1:
                smsg_embed = gray_to_bin(smsg, args.bits_per_element)
            else:
                smsg_embed = smsg

            stego_image, latents_msg = message_hiding(
                smsg_embed, device, shape, args, pipe, width, height)

            stego_name = os.path.join(save_dir, f"prompt_{ip}_mseed_{im}.png")
            stego_image.save(stego_name)

            image_saved = Image.open(stego_name).convert("RGB")
            assert image_saved.size == (width, height), (
                f"Stego size {image_saved.size} != requested {(width, height)}")
            msg_out, latents_extract = message_extraction(
                image_saved, device, shape, args, pipe_inv, width, height)

            if args.bits_per_element > 1:
                msg_out = bin_to_gray(msg_out, args.bits_per_element)

            torch.save({"secret": smsg, "recovered": msg_out},
                       os.path.join(save_dir, f"msg_prompt_{ip}_mseed_{im}.pt"))

            bcr[ip, im] = cal_bcr(smsg, msg_out)
            print(f"prompt {ip}  mseed {im}  BCR = {bcr[ip, im]:.6f}")

    bcr_mean = float(np.mean(bcr))
    print(f"Resolution: {width}x{height}   |   Latent shape: {shape}")
    print(f"Payload: {msg_length} bits   |   Mean BCR: {bcr_mean:.6f}")


if __name__ == "__main__":
    main()
