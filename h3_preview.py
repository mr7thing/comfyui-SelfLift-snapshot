"""Preview videos drawn with the fast decoder in ``models/vae_approx``.

The preview decoder (taeh3 for MiniMax H3) is the same small stand-in for the
real video VAE that ComfyUI's own previews use: it decodes in a fraction of the
time. It carries the clip's frame count as well -- H3's five latent tokens
cover seventeen pixel frames -- so a preview drawn from it has the same
duration as the clip it stands in for.

The soundtrack is drawn the same way: the audio latent goes through the H3
audio VAE, which is small next to the video one.
"""

from __future__ import annotations

import logging
import os
from fractions import Fraction

import numpy as np
import torch

import comfy.sd
import comfy.utils
import folder_paths

#: The preview decoder's file name starts with this, as core's previewer finds it.
PREVIEW_VAE = "taeh3"

#: AAC wants whole frames of this many samples.
AUDIO_FRAME = 1024

_vaes: dict = {}


def list_preview_vaes():
    """The fast decoders installed in ``models/vae_approx``, by file name."""
    return list(folder_paths.get_filename_list("vae_approx"))


def preview_vae(name: str = PREVIEW_VAE):
    """The preview decoder in ComfyUI's ``vae_approx`` folder, loaded once and kept.

    Args:
        name: A file name from :func:`list_preview_vaes`, or the beginning of
            one (``'taeh3'`` finds ``taeh3.safetensors``).

    Returns:
        The loaded VAE, or ``None`` where no such file is installed.
    """
    found = next((entry for entry in list_preview_vaes()
                  if entry == name or entry.startswith(name)), None)
    if found is None:
        logging.warning("SelfLift: no preview VAE in models/vae_approx matching '%s'", name)
        return None
    path = folder_paths.get_full_path("vae_approx", found)
    key = (path, os.path.getmtime(path))
    held = _vaes.get(key)
    if held is None:
        held = comfy.sd.VAE(sd=comfy.utils.load_torch_file(path, safe_load=True))
        if hasattr(held.first_stage_model, "show_progress_bar"):
            held.first_stage_model.show_progress_bar = False
        _vaes.clear()
        _vaes[key] = held
    return held


def decode_frames(vae, latent):
    """Decode an H3 video latent into frames.

    Args:
        vae: The preview VAE.
        latent: ``[B, 24, T, h, w]`` in the VAE's own latent space.

    Returns:
        ``[T', H, W, 3]`` frames in ``[0, 1]``; ``T'`` is the pixel frame count.
    """
    pixels = vae.decode(latent)
    if pixels.ndim == 5:
        pixels = pixels[0]      # [T, H, W, C]
    elif pixels.ndim == 4:
        pixels = pixels[0]      # [H, W, C]: one frame, no time axis
    return pixels.detach().float().clamp(0.0, 1.0)


def decode_audio(audio_vae, latent):
    """Decode an H3 audio latent into a waveform.

    Args:
        audio_vae: The H3 audio VAE (``models/vae/minimax_h3_audio_vae_*``).
        latent: ``[B, 32, 2, T]`` in the audio VAE's own latent space.

    Returns:
        ``(waveform, sample_rate)``; the waveform is ``[B, C, L]`` in [-1, 1].
    """
    audio = audio_vae.decode(latent).movedim(-1, 1)   # [B, C, L]
    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio = audio / std
    rate = getattr(audio_vae, "audio_sample_rate_output",
                   getattr(audio_vae, "audio_sample_rate", 32000))
    return audio.clamp(-1.0, 1.0), int(rate)


def write_video(frames, path, fps=24.0, crf=18, audio=None, audio_sample_rate=32000):
    """Write frames as an mp4.

    The frames keep the resolution they were decoded at: the preview is the
    low-resolution prefix as it is, not a scaled-up copy of it.

    Args:
        frames: ``[T, H, W, 3]`` in [0, 1].
        path: Where the mp4 goes.
        fps: The frame rate.
        crf: The x264 quality knob.
        audio: Optional ``[C, L]`` waveform in [-1, 1]; it becomes the mp4's track.
        audio_sample_rate: The sample rate of that waveform.
    """
    import av

    height, width = frames.shape[1], frames.shape[2]
    height -= height % 2
    width -= width % 2
    container = av.open(path, mode="w")
    try:
        stream = container.add_stream("libx264", rate=Fraction(round(fps * 1000), 1000))
        stream.width = width
        stream.height = height
        stream.pix_fmt = "yuv420p"
        stream.options = {"crf": str(crf)}
        audio_stream = None
        layout = "stereo"
        if audio is not None:
            layout = "mono" if audio.shape[0] == 1 else "stereo"
            audio_stream = container.add_stream("aac", rate=audio_sample_rate, layout=layout)
            audio_stream.bit_rate = 128000
        for frame in frames:
            array = torch.clamp(frame[:height, :width] * 255.0, 0.0, 255.0)
            container.mux(stream.encode(av.VideoFrame.from_ndarray(
                array.to(torch.uint8).cpu().numpy(), format="rgb24")))
        container.mux(stream.encode())
        if audio_stream is not None:
            _mux_audio(container, audio_stream, audio, audio_sample_rate, layout)
    finally:
        container.close()
    return path


def _mux_audio(container, stream, waveform, sample_rate, layout):
    """Encode a waveform into the container's audio stream."""
    import av

    channels, length = waveform.shape
    padding = (-length) % AUDIO_FRAME
    if padding:
        waveform = torch.nn.functional.pad(waveform, (0, padding))
    resampler = av.AudioResampler(format="fltp", layout=layout, rate=sample_rate)
    frame = av.AudioFrame.from_ndarray(
        np.ascontiguousarray(waveform.float().cpu().numpy()), format="fltp", layout=layout)
    frame.sample_rate = sample_rate
    parts = list(resampler.resample(frame)) + list(resampler.resample(None))
    written = 0
    for part in parts:
        part.pts = written
        part.time_base = Fraction(1, sample_rate)
        container.mux(stream.encode(part))
        written += part.samples
    container.mux(stream.encode())
    logging.info("SelfLift draft: %s audio, %d channels, %d samples at %d Hz",
                 layout, channels, length, sample_rate)
