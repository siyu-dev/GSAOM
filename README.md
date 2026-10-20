# GSAOM

**Official implementation of *Provably Secure Generative Steganography Based on Adjustable Orthogonal Mapping***

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://img.shields.io/badge/DOI-10.1109%2FTDSC.2025.3604222-blue)](https://doi.org/10.1109/TDSC.2025.3604222)
[![Framework](https://img.shields.io/badge/Built%20with-%F0%9F%A4%97%20Diffusers-ffd21e)](https://github.com/huggingface/diffusers)

---

## Description

This repository provides the official implementation of **GSAOM**, a provably secure generative steganography method based on **Adjustable Orthogonal Mapping (AOM)**, proposed in *"Provably Secure Generative Steganography Based on Adjustable Orthogonal Mapping"* (IEEE Transactions on Dependable and Secure Computing, Vol. 23, No. 1, pp. 34–48, 2026).

Unlike cover-editing steganography, GSAOM synthesizes the stego image directly from the secret message, leaving no modification traces on any cover image. The sender employs AOM to convert the secret message into a latent variable that follows the standard normal distribution, and feeds it into a pretrained diffusion model to generate a high-quality stego image. The receiver inverts the diffusion process to recover the latent variable and extracts the secret message through the inverse mapping of AOM.

Since the mapped latent variable and the latent variable randomly sampled during regular image generation are identically distributed, GSAOM achieves **provable security**. Meanwhile, AOM supports **adjustable payload capacity** and maximizes the distinguishability of the extracted message, giving GSAOM advantages in capacity and robustness against common channel distortions.

```
Secret message  ──► [ information adjustment ]  ┐
                ──► [ permutation            ]  ├─► AOM ─► z ~ N(0, I) ─► [ Diffusion ] ─► Stego image
                ──► [ orthogonal transform   ]  ┘                                              │
                                                                                               ▼
Recovered msg   ◄── [ inverse AOM ] ◄── ẑ ◄── [ diffusion inversion ] ◄────────────────────────┘
```

---

## Installation

```bash
git clone https://github.com/siyu-dev/GSAOM.git
cd GSAOM

conda create -n gsaom python=3.10 -y
conda activate gsaom
pip install -r requirements.txt
```

`requirements.txt` pins only lower bounds so that the code installs cleanly on CPU / CUDA / ROCm machines.

## Model weights

This implementation is built on **Stable Diffusion v1.5**. Download the checkpoint from Hugging Face and pass its path via `--model_path`:

```bash
huggingface-cli download runwayml/stable-diffusion-v1-5 --local-dir ./models/stable-diffusion-v1-5
```

---

## Quick start

Run the full hide–extract pipeline and report the bit correct ratio (BCR):

```bash
python GSAOM_SD15.py \
    --model_path ./models/stable-diffusion-v1-5 \
    --width 512 --height 512 \
    --bits_per_element 1 \
    --strength 1 \
    --seed 10001 \
    --output_dir ./results
```

Stego images, the recovered bitstreams and a `config.json` are written to `results/orthogonal/<W>x<H>/s<strength>_b<bits_per_element>/`.

**Rectangular images are supported.** Width and height are independent and only need to be multiples of 8 (the VAE downsample factor):

```bash
# 768 x 512 landscape
python GSAOM_SD15.py --width 768 --height 512

# 512 x 768 portrait
python GSAOM_SD15.py --width 512 --height 768

# shortcut: set both at once
python GSAOM_SD15.py --image_size 768
```

Evaluate over multiple prompts and messages:

```bash
python GSAOM_SD15.py --num_prompt 6 --num_mseed 5
```

---

## Capacity configuration

Let `H = height / 8`, `W = width / 8` be the latent resolution and `N = 4 × H × W` the number of latent elements. The payload is

```
capacity = floor(N × b / s)   bits
```

where `b = --bits_per_element` and `s = --strength`. **At least one of `b` and `s` must be 1.** Larger `s` (or smaller `b`) spreads each bit over more latent elements and improves robustness; larger `b` raises capacity.

Payload at `b = 1, s = 1` for common resolutions:

| Resolution | Latent `H × W` | `N` | Payload |
|---|---|---:|---:|
| 512 × 512 | 64 × 64 | 16,384 | 16,384 bit (2.00 KiB) |
| 768 × 512 | 64 × 96 | 24,576 | 24,576 bit (3.00 KiB) |
| 512 × 768 | 96 × 64 | 24,576 | 24,576 bit (3.00 KiB) |
| 1024 × 512 | 64 × 128 | 32,768 | 32,768 bit (4.00 KiB) |

Capacity–robustness trade-off at 512 × 512 (`N = 16,384`):

| `b` (bits/element) | `s` (strength) | Payload | Use case |
|:---:|:---:|:---:|---|
| 1 | 1 | 16,384 bit | Maximum capacity |
| 1 | 2 | 8,192 bit | Balanced |
| 1 | 4 | 4,096 bit | Robustness-oriented |
| 2 | 1 | 32,768 bit | High capacity (Gray coding) |
| 4 | 1 | 65,536 bit | Maximum capacity (Gray coding) |

When `b > 1`, message bits are Gray-coded before embedding so that a decision error in one quantization level corrupts only a single bit.

---

## Repository structure

```
GSAOM/
├── GSAOM_SD15.py
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

Key functions in `GSAOM_SD15.py`:

| Function | Role |
|---|---|
| `message_to_latents` | AOM forward mapping: message → `N(0, I)` latent |
| `latents_to_message` | AOM inverse mapping: latent → message |
| `orthogonal_transform` / `inverse_orthogonal_transform` | Key-driven orthogonal projection |
| `gray_to_bin` / `bin_to_gray` | Gray coding for `b > 1` |
| `message_hiding` / `message_extraction` | End-to-end pipeline |

---

## Citation

If you find this work useful, please cite:

```bibtex
@ARTICLE{zhang2026gsaom,
  author  = {Zhang, Qinghua and Huang, Fangjun},
  journal = {IEEE Transactions on Dependable and Secure Computing},
  title   = {Provably Secure Generative Steganography Based on Adjustable Orthogonal Mapping},
  year    = {2026},
  volume  = {23},
  number  = {1},
  pages   = {34--48},
  doi     = {10.1109/TDSC.2025.3604222}
}
```

---

## Acknowledgements

The diffusion backbone is based on [🤗 Diffusers](https://github.com/huggingface/diffusers) and [Stable Diffusion v1.5](https://huggingface.co/runwayml/stable-diffusion-v1-5).
