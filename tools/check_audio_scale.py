"""Which scale is the frozen audio latent in?

The H3 audio VAE normalizes its latents with the per-channel mean and std it
ships with, so a latent that is already in the VAE's own space has a spread
close to those constants. The frozen endpoint's audio stream has to be brought
to that space before it can be decoded.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/check_audio_scale.py
"""
import glob
import os
import sys

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, CORE)
sys.path.insert(0, os.path.join(REPO, "tools"))

import torch  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

from test_live import OUTPUT  # noqa: E402

AUDIO_VAE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\vae\minimax_h3_audio_vae_fp32.safetensors"


def main():
    manifest = glob.glob(os.path.join(OUTPUT, "selflift_drafts", "*.json"))[0]
    tensors = comfy.utils.load_torch_file(manifest.replace(".json", ".safetensors"), safe_load=True)
    audio = tensors["x0.1"].float()
    print(f"  the frozen audio latent: {tuple(audio.shape)}")
    per_channel = audio[0].reshape(32, -1)
    print(f"  its per-channel std: mean {per_channel.std(dim=1).mean():.4f} "
          f"min {per_channel.std(dim=1).min():.4f} max {per_channel.std(dim=1).max():.4f}")

    vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(AUDIO_VAE, safe_load=True))
    first = vae.first_stage_model
    print(f"  the VAE's own latents_std: mean {float(first.latents_std.mean()):.4f}")
    print(f"  the VAE's own latents_mean: mean {float(first.latents_mean.mean()):.4f}")

    # What the decoder makes of the latent as it stands and after a scale.
    for scale in (1.0, 0.25):
        out = vae.decode(audio / scale).detach()
        print(f"  decoded at 1/{scale:g}: shape {tuple(out.shape)} "
              f"min {float(out.min()):+.4f} max {float(out.max()):+.4f} "
              f"rms {float(out.pow(2).mean().sqrt()):.4f}")


if __name__ == "__main__":
    main()
