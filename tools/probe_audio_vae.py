"""What shape does the MiniMax H3 audio VAE hand back?

``comfy_extras.nodes_audio.vae_decode_audio`` moves the last axis to the
channel axis, which only makes sense when the decoder returns [B, L, C].
``MiniMaxH3AudioVAE.decode`` documents [B, 2, L]. One of the two is wrong for
H3, so measure it instead of guessing.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/probe_audio_vae.py
"""
import os
import sys

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
sys.path.insert(0, CORE)
os.chdir(CORE)

import torch  # noqa: E402
import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402
import folder_paths  # noqa: E402

PATH = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\vae\minimax_h3_audio_vae_fp32.safetensors"


def main():
    sd = comfy.utils.load_torch_file(PATH, safe_load=True)
    vae = comfy.sd.VAE(sd=sd)
    print("latent_dim        ", vae.latent_dim)
    print("latent_channels   ", vae.latent_channels)
    print("output_channels   ", vae.output_channels)
    print("audio_sample_rate ", getattr(vae, "audio_sample_rate", None))
    print("output_sample_rate", getattr(vae, "audio_sample_rate_output", None))
    print("upscale_ratio     ", vae.upscale_ratio)

    latent = torch.zeros(1, 32, 2, 10)
    audio = vae.decode(latent)
    print("decoded shape     ", tuple(audio.shape))
    print("movedim(-1, 1)    ", tuple(audio.movedim(-1, 1).shape))

    from comfy_extras import nodes_audio
    out = nodes_audio.vae_decode_audio(vae, {"samples": latent})
    print("core helper       ", tuple(out["waveform"].shape), out["sample_rate"])


if __name__ == "__main__":
    main()
