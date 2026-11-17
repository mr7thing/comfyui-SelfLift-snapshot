"""Probe how the H3 preview VAE (taeh3) can be decoded, and what a 5D latent decodes to.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/probe_taeh3.py
"""
import os
import sys
import traceback

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
sys.path.insert(0, CORE)

import torch  # noqa: E402

import comfy.sd  # noqa: E402
import comfy.utils  # noqa: E402

PATH = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Shared\models\vae_approx\taeh3.safetensors"


def probe_core_vae():
    print("\n=== comfy.sd.VAE(sd=taeh3) ===")
    try:
        sd = comfy.utils.load_torch_file(PATH, safe_load=True)
        vae = comfy.sd.VAE(sd=sd, device=torch.device("cpu"))
        print("  first_stage_model:", type(vae.first_stage_model).__name__)
        print("  latent_channels:", vae.latent_channels, "upscale_ratio:", vae.upscale_ratio,
              "latent_dim:", vae.latent_dim)
        latent = torch.zeros(1, 24, 3, 8, 10)
        out = vae.decode(latent)
        print("  decode([1,24,3,8,10]) ->", tuple(out.shape))
    except Exception as error:
        print("  FAILED:", type(error).__name__, error)
        traceback.print_exc()


def probe_kj_decoder():
    print("\n=== KJ TinyVAEDecoder ===")
    sys.path.insert(0, os.path.join(CORE, "custom_nodes", "comfyui-kjnodes"))
    try:
        from nodes.tiny_vae import load_tiny_vae_decoder
    except Exception as error:
        print("  import failed:", error)
        return
    try:
        decoder = load_tiny_vae_decoder("taeh3.safetensors", device=torch.device("cpu"))
        print("  decoder:", type(decoder).__name__)
        print("  latent_channels:", decoder.latent_channels, "upscale_ratio:", decoder.upscale_ratio)
        latent = torch.zeros(1, 24, 3, 8, 10)
        frames = decoder.decode_video(latent)
        print("  decode_video([1,24,3,8,10]) ->", tuple(frames.shape), "range",
              float(frames.min()), float(frames.max()))
    except Exception as error:
        print("  FAILED:", type(error).__name__, error)
        traceback.print_exc()


def probe_h3_latent_format():
    print("\n=== H3 latent format ===")
    import comfy.latent_formats
    fmt = comfy.latent_formats.MiniMaxH3AV()
    print("  channels:", fmt.latent_channels, "spatial:", fmt.spacial_downscale_ratio,
          "temporal:", fmt.temporal_downscale_ratio, "taesd:", fmt.taesd_decoder_name)


probe_core_vae()
probe_kj_decoder()
probe_h3_latent_format()
