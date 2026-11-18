"""SelfLift progressive-resolution sampler nodes.

Run the first sampling steps on a spatially downscaled latent, lift the clean
endpoint to the target resolution with the training-free Artifact-Aware
Consistency Lift (arXiv:2609.02036), re-noise it at the transition sigma, and
finish the schedule at full resolution.

Two front-ends share the same engine:
- SelfLiftH3Sampler: MiniMax H3 audio-video (nested AV latents; the audio stream
  has no spatial dimensions and continues through the reused Euler boundary step).
  This is an experimental extension beyond the paper's image-model evaluation.
- SelfLiftImageSampler: compatible rectified-flow image models.
"""

import json
import logging
import os
import time

import torch

import comfy.k_diffusion.sampling
import comfy.model_management
import comfy.model_patcher
import comfy.model_sampling
import comfy.nested_tensor
import comfy.sample
import comfy.samplers
import comfy.utils
import folder_paths
import latent_preview

from . import selflift
from . import h3_upscaler
from . import h3_tiling
from . import h3_tst
from . import h3_preview
from .diagnostics import log_memory


class _StageTimer:
    def __init__(self, stage, device, resolution=None, pixel_scale=1):
        self.stage = stage
        self.resolution = resolution
        self.pixel_scale = pixel_scale
        self.device = torch.device(device)
        self.synchronize = os.environ.get("SELFLIFT_TIMING_SYNC", "0") == "1" and self.device.type == "cuda"
        self.started = self.previous = self._now()
        timing_mode = "CUDA-synchronized wall time" if self.synchronize else "wall time; no forced CUDA sync"
        suffix = ""
        if resolution is not None:
            pixels = tuple(int(value * self.pixel_scale) for value in resolution)
            suffix = f" latent={resolution} pixels={pixels}"
        logging.info("[SelfLift timing] %s start (%s)%s", stage, timing_mode, suffix)

    def _now(self):
        if self.synchronize:
            torch.cuda.synchronize(self.device)
        return time.perf_counter()

    def mark(self, label):
        current = self._now()
        logging.info("[SelfLift timing] %s %s: %.3fs", self.stage, label, current - self.previous)
        self.previous = current

    def finish(self):
        current = self._now()
        logging.info("[SelfLift timing] %s total: %.3fs; after last mark: %.3fs",
                     self.stage, current - self.started, current - self.previous)


def _upscaler_input():
    models = ["none"] + h3_upscaler.list_upscaler_models()
    h3_models = [name for name in models[1:] if "h3" in name.lower()]
    default = h3_models[0] if h3_models else "none"
    return (models, {"default": default,
                     "tooltip": "External H3 latent upscaler (models/latent_upscale_models). The first detected H3 model is selected by default; 'none' uses nearest-neighbor lifting."})


def _default_preview_vae():
    """The fast decoder a run draws its preview with when nothing else is asked for."""
    models = h3_preview.list_preview_vaes()
    return next((name for name in models if name.startswith(h3_preview.PREVIEW_VAE)),
                models[0] if models else "none")


def _preview_vae_input():
    """The fast decoders in models/vae_approx, so a run can draw its own preview."""
    models = ["none"] + h3_preview.list_preview_vaes()
    return (models, {"default": _default_preview_vae(),
                     "tooltip": "Fast decoder from models/vae_approx used to draw the preview mp4 (taeh3 for MiniMax H3). 'none' skips the preview."})


def _streams(samples):
    if samples.is_nested:
        return list(samples.unbind()), True
    return [samples], False


def _pack(streams, nested):
    if nested:
        return comfy.nested_tensor.NestedTensor(streams)
    return streams[0]


def _validate_sampling(model_sampling, sampler):
    if not isinstance(model_sampling, comfy.model_sampling.CONST):
        raise ValueError("SelfLift requires a rectified-flow model")
    if not isinstance(sampler, comfy.samplers.KSAMPLER) or sampler.sampler_function is not comfy.k_diffusion.sampling.sample_euler:
        raise ValueError("SelfLift requires the standard Euler sampler")
    if sampler.extra_options.get("s_churn", 0.0) != 0.0:
        raise ValueError("SelfLift requires Euler with s_churn=0")


def _validate_schedule(sigmas, transition_step):
    if sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("SelfLift: sigmas must be a one-dimensional floating-point tensor")
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("SelfLift: sigmas must be finite and nonnegative")
    if sigmas.numel() < 2:
        return
    if not isinstance(transition_step, int) or not 1 <= transition_step <= sigmas.numel() - 2:
        raise ValueError(f"SelfLift: transition_step {transition_step} out of range for {sigmas.numel() - 1} steps")
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("SelfLift: sigmas must be non-increasing")
    if (sigmas[:-1] <= 0).any():
        raise ValueError("SelfLift: only the final sigma may be zero")
    if sigmas[transition_step] >= 1:
        raise ValueError("SelfLift: the high-resolution starting sigma must be less than 1")


def _validate_draft_schedule(sigmas, transition_step):
    """The draft schedule: the prefix may run the whole step count.

    A draft never resumes at a second resolution, so it needs no step left over
    for the high-resolution stage.
    """
    if sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("SelfLift: sigmas must be a one-dimensional floating-point tensor")
    if not torch.isfinite(sigmas).all() or (sigmas < 0).any():
        raise ValueError("SelfLift: sigmas must be finite and nonnegative")
    if sigmas.numel() < 2:
        return
    if not isinstance(transition_step, int) or not 1 <= transition_step <= sigmas.numel() - 1:
        raise ValueError(f"SelfLift: draft transition_step {transition_step} out of range for {sigmas.numel() - 1} steps")
    if (sigmas[1:] > sigmas[:-1]).any():
        raise ValueError("SelfLift: sigmas must be non-increasing")
    if (sigmas[:-1] <= 0).any():
        raise ValueError("SelfLift: only the final sigma may be zero")


def _validate_latent_input(latent_image):
    """Validate the latent template; returns the normalized noise_mask or None.

    Accepted mask shapes: [B, H, W], [B, 1, H, W], [B, 1, T, H, W] (1 = generate,
    0 = keep the original content). Spatial dims must match the latent; a time
    length of 1 shares the mask over all frames. Normalized to [B, 1, H, W] for
    image latents and [B, 1, T|1, H, W] for video latents.
    """
    streams, _ = _streams(latent_image["samples"])
    if not streams or streams[0].ndim not in (4, 5):
        raise ValueError("SelfLift: expected a 4D image or 5D video latent size template")
    for stream in streams:
        if stream.ndim == 0 or any(size == 0 for size in stream.shape) or stream.shape[0] != streams[0].shape[0]:
            raise ValueError("SelfLift: latent streams must be nonempty and have the same batch size")
    mask = latent_image.get("noise_mask")
    if mask is None:
        return None
    video = streams[0].ndim == 5
    b, H, W = streams[0].shape[0], streams[0].shape[-2], streams[0].shape[-1]
    if mask.ndim == 3:  # [B, H, W]
        mask = mask[:, None]
    if video and mask.ndim == 4:  # [B, 1, H, W] shared over time
        mask = mask[:, :, None]
    if mask.ndim != (5 if video else 4) or mask.shape[1] != 1:
        raise ValueError("SelfLift: noise_mask must have shape [B, H, W], [B, 1, H, W], or [B, 1, T, H, W]")
    if mask.shape[0] != b:
        raise ValueError(f"SelfLift: noise_mask batch {mask.shape[0]} does not match the latent batch {b}")
    if tuple(mask.shape[-2:]) != (H, W):
        # masks are accepted at any resolution and resized to the latent grid,
        # matching ComfyUI's Set Latent Noise Mask convention
        lead = mask.shape[:-2]
        mask = torch.nn.functional.interpolate(
            mask.reshape(-1, 1, *mask.shape[-2:]).float(), size=(H, W), mode="bilinear", align_corners=False
        ).reshape(*lead, H, W)
    if video and mask.shape[2] not in (1, streams[0].shape[2]):
        raise ValueError(f"SelfLift: noise_mask time length {mask.shape[2]} does not match the latent frames {streams[0].shape[2]}")
    if not torch.isfinite(mask).all():
        raise ValueError("SelfLift: noise_mask must be finite")
    return mask.float().clamp(0.0, 1.0)


def _mask_blend_fn(anchor, mask):
    """post-CFG hook pinning x0 to the clean anchor outside the generate region.

    Exact under Euler: with the unmasked x0 pinned to the original latent each
    step, the unmasked state tracks original + the step's noise exactly.
    """
    def fn(args):
        d = args["denoised"]
        m = mask.to(device=d.device, dtype=d.dtype)
        return d * m + anchor.to(device=d.device, dtype=d.dtype) * (1.0 - m)
    return fn


def _euler_step(state, denoised, sigma, sigma_next):
    step = ((sigma_next - sigma) / sigma).to(device=state.device, dtype=state.dtype)
    return state + (state - denoised.to(state)) * step


def _resize_keyframes(cond, h, w):
    """Keyframe cond latents share the generation grid; resize them to the low-res one."""
    out = []
    for tensor, d in cond:
        kfs = d.get("minimax_keyframes")
        if kfs is None:
            out.append((tensor, d))
            continue
        d = d.copy()
        resized = []
        for kf in kfs:
            kf = dict(kf)
            lat = kf.get("latent")
            if lat is not None and (lat.shape[-2] != h or lat.shape[-1] != w):
                if lat.ndim == 5:
                    # spatial-only resize per frame; never interpolates across time
                    batch, channels, frames = lat.shape[:3]
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float().permute(0, 2, 1, 3, 4).reshape(batch * frames, channels, lat.shape[-2], lat.shape[-1]),
                        size=(h, w), mode="bilinear", align_corners=False
                    )
                    resized_latent = resized_latent.reshape(batch, frames, channels, h, w).permute(0, 2, 1, 3, 4)
                else:
                    resized_latent = torch.nn.functional.interpolate(
                        lat.float(), size=(h, w), mode="bilinear", align_corners=False
                    )
                # per-channel, per-frame mean match: fixes bilinear's color drift without
                # restoring variance the low-res grid cannot represent
                source_mean = lat.float().mean(dim=(-2, -1), keepdim=True)
                resized_mean = resized_latent.mean(dim=(-2, -1), keepdim=True)
                kf["latent"] = (resized_latent + (source_mean - resized_mean)).to(lat)
            resized.append(kf)
        d["minimax_keyframes"] = resized
        out.append((tensor, d))
    return out


def _debug_dump(vae, latents):
    """Decode transition intermediates to PNGs when SELFLIFT_DEBUG=1."""
    if os.environ.get("SELFLIFT_DEBUG", "0") != "1":
        return
    out_dir = os.path.join(os.path.dirname(__file__), "debug")
    os.makedirs(out_dir, exist_ok=True)
    logging.warning("SelfLift: debug decoding enabled; intermediate video decodes can substantially increase time and memory")
    from PIL import Image
    for name, lat in latents.items():
        if lat is None:
            continue
        img = vae.decode(lat)
        if img.ndim == 5:
            img = img.reshape(-1, img.shape[-3], img.shape[-2], img.shape[-1])
        frame = (img[0].float().cpu().numpy().clip(0.0, 1.0) * 255).round().astype("uint8")
        Image.fromarray(frame).save(os.path.join(out_dir, name + ".png"))


def _setup_run(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, latent_upsample,
               latent_lifter=None, highres_tiling=False, model_hires=None, draft=False):
    """Validate a run and build what both sampling stages share.

    Nothing here samples; the returned dict is what the stage functions read.
    """
    if draft:
        _validate_draft_schedule(sigmas, transition_step)
    else:
        _validate_schedule(sigmas, transition_step)
    if not 0.25 <= lowres_scale <= 1.0:
        raise ValueError("SelfLift: lowres_scale must be between 0.25 and 1")
    if not 0.0 <= rho <= 1.0:
        raise ValueError("SelfLift: rho must be between 0 and 1")
    if not 0.0 <= w_min <= w_max <= 1.0:
        raise ValueError("SelfLift: weights must satisfy 0 <= w_min <= w_max <= 1")
    noise_mask = _validate_latent_input(latent_image)
    if highres_tiling and noise_mask is not None:
        raise ValueError("SelfLift: noise_mask is not compatible with highres_tiling")

    model_sampling = model.get_model_object("model_sampling")
    _validate_sampling(model_sampling, sampler)

    streams, nested = _streams(comfy.sample.fix_empty_latent_channels(
        model, latent_image["samples"], latent_image.get("downscale_ratio_spacial", None),
        latent_image.get("downscale_ratio_temporal", None)))
    hires_base = model_hires if model_hires is not None else model
    high_model = h3_tiling.tiled_model(hires_base, [tuple(stream.shape) for stream in streams]) if highres_tiling else hires_base
    video = streams[0].ndim == 5
    if video:
        b, c, t, H, W = streams[0].shape
    else:
        b, c, H, W = streams[0].shape
        t = None
    h = max(2, round(H * lowres_scale / 2) * 2)
    w = max(2, round(W * lowres_scale / 2) * 2)
    low_shape = (b, c, t, h, w) if video else (b, c, h, w)
    if draft:
        logging.info("[SelfLift draft] low_latent=%s target_latent=%s spatial_lift=(%.4f, %.4f) "
                     "low_nfe=%d of %d steps cfg=%.4f mask=%s",
                     low_shape, tuple(streams[0].shape), H / h, W / w,
                     transition_step, sigmas.numel() - 1, cfg,
                     "none" if noise_mask is None else f"{tuple(noise_mask.shape)}")
    else:
        logging.info("[SelfLift plan] low_latent=%s target_latent=%s spatial_lift=(%.4f, %.4f) "
                     "low_nfe=%d high_nfe=%d sigma_prediction=%.8g sigma_resume=%.8g "
                     "rho=%.4f weights=(%.4f, %.4f) direct_lift=%s pixel_anchor=%s cfg=%.4f mask=%s hires_model=%s",
                     low_shape, tuple(streams[0].shape), H / h, W / w,
                     transition_step, sigmas.numel() - 1 - transition_step,
                     sigmas[transition_step - 1].item(), sigmas[transition_step].item(),
                     rho, w_min, w_max,
                     "skipped" if rho == 1.0 and w_min == 1.0 else "external" if latent_lifter is not None else latent_upsample,
                     rho > 0.0 and w_max > 0.0, cfg,
                     "none" if noise_mask is None else f"{tuple(noise_mask.shape)}",
                     "custom" if model_hires is not None else "same")

    device = comfy.model_management.intermediate_device()
    source_video = streams[0].to(device)
    audio_streams = [s.to(device) for s in streams[1:]]

    def masked_model(base, video_anchor, m):
        # pin the unmasked region to the clean anchor via a post-CFG x0 blend;
        # ComfyUI's denoise_mask path is unusable here: the high-res resume
        # latent deliberately carries noise, which would contaminate its anchor
        if m is None:
            return base
        patched = base.clone()
        if nested:
            flat_mask = m.reshape(b, 1, -1).repeat(1, 1, c).to(video_anchor.device)
            flat_anchor = video_anchor.reshape(b, 1, -1)
            for s in audio_streams:
                flat_mask = torch.cat([flat_mask, torch.ones_like(s.reshape(b, 1, -1))], dim=-1)
                flat_anchor = torch.cat([flat_anchor, s.reshape(b, 1, -1)], dim=-1)
            hook = _mask_blend_fn(flat_anchor, flat_mask)
        else:
            hook = _mask_blend_fn(video_anchor, m)
        patched.model_options = comfy.model_patcher.set_model_options_post_cfg_function(patched.model_options, hook)
        return patched

    if video:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(t, h, w), mode="trilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    else:
        low_video = torch.nn.functional.interpolate(
            source_video.float(), size=(h, w), mode="bilinear", align_corners=False
        ).to(dtype=source_video.dtype)
    m_full = m_low = None
    if noise_mask is not None:
        if video and noise_mask.shape[2] == 1 and t > 1:
            m_full = noise_mask.expand(b, 1, t, H, W)
        else:
            m_full = noise_mask
        if video:
            m_low = torch.nn.functional.interpolate(
                m_full.reshape(b * t, 1, H, W).float(), size=(h, w), mode="bilinear", align_corners=False
            ).reshape(b, 1, t, h, w)
        else:
            m_low = torch.nn.functional.interpolate(
                m_full.float(), size=(h, w), mode="bilinear", align_corners=False)
        m_low = m_low.clamp(0.0, 1.0)
    low_anchor = low_video
    video_anchor = source_video
    low_latent = _pack([low_video] + audio_streams, nested)
    low_model = masked_model(model, low_anchor, m_low)
    high_model = masked_model(high_model, video_anchor, m_full)
    del source_video, low_video
    del streams
    noise_low = comfy.sample.prepare_noise(low_latent, seed, latent_image.get("batch_index", None))

    total_steps = sigmas.shape[-1] - 1
    callback = latent_preview.prepare_callback(model, total_steps)
    disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
    if video:  # keyframe cond latents share the generation grid on MiniMax H3
        positive_low = _resize_keyframes(positive, h, w)
        negative_low = _resize_keyframes(negative, h, w)
    else:
        positive_low, negative_low = positive, negative

    latent_format = model.get_model_object("latent_format")
    resolution_scale = getattr(latent_format, "spacial_downscale_ratio", 1)

    return {
        "model": model, "vae": vae, "sampler": sampler, "sigmas": sigmas, "seed": seed, "cfg": cfg,
        "latent_image": latent_image, "transition_step": transition_step,
        "lowres_scale": lowres_scale, "rho": rho, "w_min": w_min, "w_max": w_max,
        "latent_upsample": latent_upsample, "latent_lifter": latent_lifter,
        "highres_tiling": highres_tiling, "model_hires": model_hires,
        "model_sampling": model_sampling, "latent_format": latent_format,
        "device": device, "nested": nested, "video": video,
        "shape": (b, c, t, H, W), "low_shape": low_shape, "latent_hw": (h, w),
        "noise_mask": noise_mask, "m_full": m_full, "m_low": m_low,
        "video_anchor": video_anchor, "low_anchor": low_anchor,
        "low_latent": low_latent, "low_model": low_model, "high_model": high_model,
        "positive_low": positive_low, "negative_low": negative_low, "noise_low": noise_low,
        "callback": callback, "disable_pbar": disable_pbar, "total_steps": total_steps,
        "resolution_scale": resolution_scale,
    }


def _run_low_stage(run):
    """Sample the low-resolution prefix and hand back the frozen state."""
    model = run["model"]
    sigmas = run["sigmas"]
    transition_step = run["transition_step"]
    h, w = run["latent_hw"]
    t, H, W = run["shape"][2], run["shape"][3], run["shape"][4]
    transition = {}
    low_evaluations = 0

    def callback_low(step, x0, x, total):
        nonlocal low_evaluations
        step = low_evaluations
        low_evaluations += 1
        if low_evaluations > transition_step:
            raise RuntimeError("SelfLift: too many low-resolution callbacks for the Euler schedule")
        if step == transition_step - 1:
            transition["state"] = x
            transition["x0"] = x0
        # Preview decoders expect the target grid.  Decode a temporary lifted
        # preview while keeping the sampler state at its true low resolution.
        preview_x0 = x0
        if run["video"]:
            preview_streams, preview_nested = _streams(x0)
            preview_video = torch.nn.functional.interpolate(
                preview_streams[0].float(), size=(t, H, W), mode="trilinear", align_corners=False
            ).to(preview_streams[0].dtype)
            preview_x0 = _pack([preview_video] + preview_streams[1:], preview_nested)
            del preview_video, preview_streams
        elif x0.ndim == 4:
            preview_x0 = torch.nn.functional.interpolate(
                x0.float(), size=(H, W), mode="bilinear", align_corners=False
            ).to(x0.dtype)
        result = run["callback"](step, preview_x0, preview_x0, run["total_steps"])
        del preview_x0
        low_timer.mark(f"step {step + 1}/{transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    low_timer = _StageTimer("low_resolution", run["low_model"].load_device, (h, w),
                            run["resolution_scale"])
    log_memory("low_resolution start", run["low_model"].load_device)
    comfy.samplers.sample(run["low_model"], run["noise_low"], run["positive_low"], run["negative_low"],
                          run["cfg"], run["low_model"].load_device,
                          run["sampler"], sigmas[:transition_step + 1], run["low_model"].model_options,
                          latent_image=run["low_latent"], callback=callback_low,
                          disable_pbar=run["disable_pbar"], seed=run["seed"])
    if low_evaluations != transition_step:
        raise RuntimeError(f"SelfLift: expected {transition_step} low-resolution callbacks, received {low_evaluations}; check sampler wrappers")
    low_timer.finish()
    log_memory("low_resolution end", model.load_device)
    for spent in ("low_latent", "noise_low", "positive_low", "negative_low"):
        run.pop(spent, None)
    return {"state": transition["state"], "x0": transition["x0"]}

def _run_transition(run, state, x0):
    """Lift the frozen endpoint to the target resolution and re-noise it.

    The final low-resolution model evaluation is the Eq. 3 prediction. Its Euler
    update is discarded and rebuilt here, preserving NFE.
    """
    model = run["model"]
    vae = run["vae"]
    sigmas = run["sigmas"]
    transition_step = run["transition_step"]
    device = run["device"]
    latent_format = run["latent_format"]
    model_sampling = run["model_sampling"]
    nested = run["nested"]
    H, W = run["shape"][3], run["shape"][4]
    rho, w_min, w_max = run["rho"], run["w_min"], run["w_max"]

    transition_timer = _StageTimer("transition", model.load_device, (H, W), run["resolution_scale"])
    low_streams, nested = _streams(state)
    x0_streams, _ = _streams(x0)
    sigma_k = sigmas[transition_step - 1]
    sigma_next = sigmas[transition_step]
    auxiliary_next = [_euler_step(stream.to(device), denoised.to(device), sigma_k, sigma_next)
                      for stream, denoised in zip(low_streams[1:], x0_streams[1:])]
    z0_low_vae = latent_format.process_out(x0_streams[0].float()).to(device)
    del low_streams, x0_streams
    transition_timer.mark("prepare_endpoint")
    log_memory("transition endpoint_ready", model.load_device)

    # Artifact-Aware Consistency Lift (Eqs. 4-9); skip branches the weights discard
    need_pix = rho > 0.0 and w_max > 0.0
    need_lat = not (rho >= 1.0 and w_min >= 1.0 and w_max >= 1.0)
    z_lat_vae, z_pix_vae = selflift.paired_lifts(
        z0_low_vae, vae, (H, W), run["latent_upsample"], run["latent_lifter"],
        need_lat=need_lat, need_pix=need_pix)
    transition_timer.mark("paired_lifts")
    log_memory("transition lifts_ready", model.load_device)
    z_lat = latent_format.process_in(z_lat_vae) if z_lat_vae is not None else None
    z_pix = latent_format.process_in(z_pix_vae) if z_pix_vae is not None else None
    z0_high = selflift.artifact_aware_consistency_lift(z_lat, z_pix, rho, w_min, w_max, mask=run["m_full"])
    if run["m_full"] is not None:
        # restore the keep-region with the true original instead of any lifted estimate
        m_cast = run["m_full"].to(device=z0_high.device, dtype=z0_high.dtype)
        z0_high = z0_high * m_cast + run["video_anchor"] * (1.0 - m_cast)
    if os.environ.get("SELFLIFT_DEBUG", "0") == "1":
        _debug_dump(vae, {
            "z0_low": z0_low_vae,
            "z_lat": z_lat_vae,
            "z_pix": z_pix_vae,
            "z0_high": latent_format.process_out(z0_high),
        })
    z0_high = z0_high.to(device)
    del z0_low_vae, z_lat_vae, z_pix_vae, z_lat, z_pix
    transition_timer.mark("correction_and_debug")
    log_memory("transition correction_ready", model.load_device)

    # Re-noise the corrected video/image endpoint at the sigma of the reused model
    # evaluation, then complete that Euler interval without another denoiser call.
    video_noise = comfy.sample.prepare_noise(z0_high, (run["seed"] + 1) % (1 << 64),
                                             run["latent_image"].get("batch_index", None)).to(z0_high)
    video_state = model_sampling.noise_scaling(sigma_k, video_noise, z0_high)
    next_streams = [_euler_step(video_state, z0_high, sigma_k, sigma_next)] + auxiliary_next
    del z0_high, video_noise, video_state, auxiliary_next
    resume_streams = [model_sampling.inverse_noise_scaling(sigma_next, s) for s in next_streams]
    resume_latent = model.model.process_latent_out(_pack(resume_streams, nested))
    resume_noise = _pack([torch.zeros_like(s) for s in resume_streams], nested)
    del next_streams, resume_streams
    transition_timer.mark("renoise")
    transition_timer.finish()
    log_memory("transition end / high_resolution start", model.load_device)
    return {"latent": resume_latent, "noise": resume_noise}


def _run_high_stage(run, resume, positive, negative):
    """Finish the schedule at the target resolution."""
    model = run["model"]
    sigmas = run["sigmas"]
    transition_step = run["transition_step"]
    total_steps = run["total_steps"]
    H, W = run["shape"][3], run["shape"][4]
    high_evaluations = 0

    def callback_high(step, x0, x, total):
        nonlocal high_evaluations
        step = high_evaluations
        high_evaluations += 1
        if high_evaluations > total_steps - transition_step:
            raise RuntimeError("SelfLift: too many high-resolution callbacks for the Euler schedule")
        result = run["callback"](step + transition_step, x0, x, total_steps)
        high_timer.mark(f"step {step + 1}/{total_steps - transition_step}" + (" (includes setup)" if step == 0 else ""))
        return result

    high_timer = _StageTimer("high_resolution", model.load_device, (H, W), run["resolution_scale"])
    if run["highres_tiling"]:
        logging.info("[SelfLift plan] automatic high-resolution tiling enabled; preparation selects the tile count")
    out = comfy.samplers.sample(run["high_model"], resume["noise"], positive, negative, run["cfg"],
                                model.load_device, run["sampler"], sigmas[transition_step:],
                                run["high_model"].model_options,
                                latent_image=resume["latent"], callback=callback_high,
                                disable_pbar=run["disable_pbar"], seed=run["seed"])
    if high_evaluations != total_steps - transition_step:
        raise RuntimeError(f"SelfLift: expected {total_steps - transition_step} high-resolution callbacks, received {high_evaluations}; check sampler wrappers")

    result = run["latent_image"].copy()
    result["samples"] = out.to(device=comfy.model_management.intermediate_device(),
                                dtype=comfy.model_management.intermediate_dtype())
    high_timer.finish()
    log_memory("high_resolution end", model.load_device)
    return result


def progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                       transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, latent_lifter=None,
                       highres_tiling=False, model_hires=None):
    _validate_schedule(sigmas, transition_step)
    if sigmas.numel() < 2:
        return latent_image
    run = _setup_run(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                     transition_step, lowres_scale, rho, w_min, w_max, latent_upsample,
                     latent_lifter=latent_lifter, highres_tiling=highres_tiling, model_hires=model_hires)
    frozen = _run_low_stage(run)
    resume = _run_transition(run, frozen["state"], frozen["x0"])
    return _run_high_stage(run, resume, positive, negative)


class SelfLiftH3Sampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "Video VAE used for the pixel re-encode anchor at the resolution transition."}),
            "latent_image": ("LATENT", {"tooltip": "Target-resolution H3 AV latent defining size and duration. Existing latent content and noise masks are accepted."}),
            "sampler": ("SAMPLER", {"tooltip": "Standard Euler only; SelfLift reuses its transition-step prediction to keep the original NFE count."}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "Number of low-resolution denoiser evaluations. The paper uses 6 of 8 NFEs for its 8-step image model; H3 requires independent validation."}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "Spatial scale of the low-resolution prefix (paper: 0.5)."}),
            "rho": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of highest-risk spatiotemporal locations corrected toward the pixel-VAE anchor. The H3 default 0 uses only the external latent upscaler and skips the VAE round trip. For SelfLift-zero with upscaler_model=none, start near 0.6."}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Correction-strength floor. H3's widespread nearest-lift error can require 1.0; 0.5 is the paper's image-model setting."}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Correction-strength ceiling. Keep at 1.0 for the H3 SelfLift-zero diagnostic."}),
            "upscaler_model": _upscaler_input(),
        }, "optional": {
            "model_hires": ("MODEL", {"tooltip": "Optional: model used for the high-resolution stage instead of `model` (e.g. a different checkpoint or LoRA stack). Must share the same architecture and latent format. The low-resolution prefix always runs on `model`."}),
            "highres_tiling": ("BOOLEAN", {"default": False, "label_on": "高分辨率分块：开启", "label_off": "高分辨率分块：关闭", "tooltip": "Experimental: select 1–8 spatial tiles from available memory at high-resolution preparation. Audio input and references remain complete; only the first tile's audio prediction is retained. Quality and speed may change."}),
            "upscaler_unload": ("BOOLEAN", {"default": True, "label_on": "放大后卸载：开启", "label_off": "放大后卸载：关闭", "tooltip": "Unload the external H3 upscaler from VRAM right after the lift, before the high-resolution stage loads. Disable only if the upscaler is reused between runs and VRAM is not tight."}),
            "preview_vae": _preview_vae_input(),
            "preview_video": ("BOOLEAN", {"default": True, "label_on": "预览视频：开启", "label_off": "预览视频：关闭", "tooltip": "Draw the finished clip with the fast preview decoder in models/vae_approx and write it as an mp4. It is a quick look, not the deliverable: the real VAE still decodes the latent you connect."}),
            "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "Frame rate of the preview mp4. Match the frame rate of the finished clip so the two can be compared frame by frame."}),
            "audio_vae": ("VAE", {"tooltip": "Optional: the H3 audio VAE. Connect it and the preview mp4 carries the soundscape too."}),
            "run_name": ("STRING", {"default": "selflift", "tooltip": "Name for the files this run writes: the preview mp4 and the workflow JSON."}),
        }, "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"}}

    RETURN_TYPES = ("LATENT", "STRING", "STRING")
    RETURN_NAMES = ("latent", "preview_video", "workflow_json")
    FUNCTION = "sample"
    CATEGORY = "selflift"
    OUTPUT_NODE = True

    def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, upscaler_model, model_hires=None, highres_tiling=False,
               upscaler_unload=True, preview_vae=None, preview_video=True, fps=25.0, audio_vae=None,
               run_name="selflift", prompt=None, extra_pnginfo=None):
        if preview_vae is None:
            preview_vae = _default_preview_vae()
        if rho == 0.0 and upscaler_model == "none":
            raise ValueError(
                "SelfLift H3: rho=0 with upscaler_model=none disables both SelfLift-zero correction "
                "and external latent upscaling; choose rho>0 or select an external H3 upscaler."
            )
        lifter = None
        if upscaler_model != "none":
            if rho > 0.0 and w_max > 0.0:
                logging.warning("SelfLift H3: rho > 0 with an external upscaler is a hybrid experiment; select upscaler_model=none to test the paper's SelfLift-zero direct route")
            lifter = lambda z, hw: h3_upscaler.learned_latent_lift(z, hw, upscaler_model, force_unload=upscaler_unload)
        result = progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                    transition_step, lowres_scale, rho, w_min, w_max, "nearest",
                                    latent_lifter=lifter, highres_tiling=highres_tiling, model_hires=model_hires)

        params = {
            "seed": seed, "cfg": cfg, "transition_step": transition_step, "lowres_scale": lowres_scale,
            "rho": rho, "w_min": w_min, "w_max": w_max, "upscaler_model": upscaler_model,
            "highres_tiling": highres_tiling, "upscaler_unload": upscaler_unload,
            "preview_vae": preview_vae, "fps": fps, "steps": int(sigmas.numel() - 1),
            "latent_shape": [list(stream.shape) for stream in _streams(result["samples"])[0]],
        }
        video_path = ""
        json_path = ""
        stem = _run_path("run", run_name or "selflift")
        if preview_video:
            latent_format = model.get_model_object("latent_format")
            video_path = _draw_preview(stem + ".mp4", result["samples"], latent_format,
                                       preview_vae, audio_vae, 1.0, fps) or ""
        json_path = _workflow_json(stem + ".json", params, prompt, extra_pnginfo,
                                   {"preview_video": video_path})
        return {"ui": {"preview_video": [video_path], "workflow_json": [json_path]},
                "result": (result, video_path, json_path)}


class SelfLiftH3Draft:
    """Run only the low-resolution part of a SelfLift H3 run.

    The draft samples the low-resolution prefix of the run and stops there, so
    it costs a fraction of the full run. With the same seed, prompt and reference
    setup, the draft is literally the beginning of a full run at the same
    `transition_step`: the full run lifts that prefix to the target resolution
    and finishes the schedule, so the draft shows what the full run starts
    from -- at low resolution.

    Set `transition_step` to the whole step count for a complete low-resolution
    sample instead: readable, but a different trajectory from a full run.

    The preview is drawn with the small decoder in `models/vae_approx` (taeh3),
    at the prefix's own resolution. Nothing is scaled up: the preview is what
    the low-resolution prefix actually looks like, which is the point of it.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "latent_image": ("LATENT", {"tooltip": "Target-resolution H3 AV latent defining size and duration. The draft samples the low-resolution prefix of this grid."}),
            "sampler": ("SAMPLER", {"tooltip": "Standard Euler only."}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 5.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 2, "min": 1, "max": 10000, "tooltip": "Number of low-resolution denoiser evaluations. Use the full run's value to preview its prefix; use the whole step count for a complete low-resolution sample."}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "Spatial scale of the low-resolution prefix. It sets the preview's resolution: the preview is this fraction of the target, and its sampling cost follows its own area, not the target's."}),
            "preview_video": ("BOOLEAN", {"default": True, "label_on": "预览视频：开启", "label_off": "预览视频：关闭", "tooltip": "Draw the prefix with the fast preview decoder in models/vae_approx and write it as an mp4 at the prefix's own resolution. No scaling: the preview stays as small as the prefix is."}),
        }, "optional": {
            "preview_vae": _preview_vae_input(),
            "preview_video": ("BOOLEAN", {"default": True, "label_on": "预览视频：开启", "label_off": "预览视频：关闭", "tooltip": "Draw the prefix with the fast preview decoder in models/vae_approx and write it as an mp4 at the prefix's own resolution. No scaling: the preview stays as small as the prefix is."}),
            "fps": ("FLOAT", {"default": 25.0, "min": 1.0, "max": 120.0, "step": 1.0, "tooltip": "Frame rate of the preview mp4. Match the frame rate of the full run so the two can be compared frame by frame."}),
            "audio_vae": ("VAE", {"tooltip": "Optional: the H3 audio VAE. Connect it and the preview mp4 carries the soundscape the model had at the end of the prefix."}),
        }}

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("draft_latent", "preview_video")
    FUNCTION = "sample"
    CATEGORY = "selflift"
    OUTPUT_NODE = True

    def sample(self, model, positive, negative, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, preview_vae=None, preview_video=True, fps=25.0,
               audio_vae=None):
        if preview_vae is None:
            preview_vae = _default_preview_vae()
        run = _setup_run(model, positive, negative, None, latent_image, sampler, sigmas, seed, cfg,
                         transition_step, lowres_scale, 0.0, 0.0, 0.0, "nearest", draft=True)
        frozen = _run_low_stage(run)
        draft = {"samples": frozen["x0"].to(device=comfy.model_management.intermediate_device(),
                                            dtype=comfy.model_management.intermediate_dtype())}
        if "batch_index" in latent_image:
            draft["batch_index"] = latent_image["batch_index"]
        logging.info("[SelfLift draft] low-resolution latent %s", tuple(draft["samples"].shape))
        path = ""
        if preview_video:
            path = _write_preview_video(run, frozen["x0"], preview_vae, audio_vae, fps) or ""
        return {"ui": {"preview_video": [path]},
                "result": (draft, path)}


def _audio_scale(run):
    """The factor the sampler carries the audio stream at (H3: shift / audio_shift).

    The sampler works on the audio stream scaled onto the video schedule, so a
    frozen endpoint has to be brought back to the audio VAE's own scale before
    it can be decoded.
    """
    scale = getattr(run["model_sampling"], "audio_scale", None)
    if scale is not None:
        return float(scale)
    getter = getattr(run["model"].model, "audio_scale", None)
    if getter is not None:
        return float(getter())
    return 1.0


def _write_preview_video(run, x0, preview_vae, audio_vae=None, fps=25.0):
    """Draw the low-resolution prefix with the fast preview decoder.

    The frames keep the prefix's own resolution: scaling them up to the target
    would only invent detail the prefix never had.
    """
    return _draw_preview(_run_path("draft"), x0, run["latent_format"], preview_vae,
                          audio_vae, _audio_scale(run), fps)


def _draw_preview(path, samples, latent_format, preview_vae, audio_vae=None, audio_scale=1.0, fps=25.0):
    """Draw a latent with the fast preview decoder and write it as an mp4.

    The frames keep the resolution the latent carries: scaling them up would
    only invent detail the run never had.

    Args:
        path: Where the mp4 goes.
        samples: The latent to draw -- the video stream and, for H3, the audio.
        latent_format: The model's latent format, for `process_out`.
        preview_vae: The fast decoder's file name, or 'none'.
        audio_vae: The H3 audio VAE, for the soundtrack.
        audio_scale: The scale the sampler carries the audio stream at.
        fps: The frame rate of the mp4.

    Returns:
        The path, or ``None`` where there is no preview to draw.
    """
    if preview_vae == "none":
        return None
    vae = h3_preview.preview_vae(preview_vae)
    if vae is None:
        return None
    streams, nested = _streams(samples)
    latent = latent_format.process_out(streams[0].float())
    frames = h3_preview.decode_frames(vae, latent)
    audio = None
    sample_rate = 32000
    if audio_vae is not None and nested and len(streams) > 1:
        audio_latent = streams[1].float()
        if audio_scale != 1.0:
            audio_latent = audio_latent / audio_scale
        waveform, sample_rate = h3_preview.decode_audio(audio_vae, audio_latent)
        audio = waveform[0].detach().float().cpu()
    h3_preview.write_video(frames, path, fps=fps, audio=audio, audio_sample_rate=sample_rate)
    logging.info("[SelfLift] preview %s at %dx%d", path, frames.shape[2], frames.shape[1])
    return path


def _run_folder():
    """Where the runs keep the files they write themselves."""
    folder = os.path.join(folder_paths.get_output_directory(), "selflift")
    os.makedirs(folder, exist_ok=True)
    return folder


def _run_path(prefix, name=""):
    """A file name for one run's own output: the name and the time."""
    stem = prefix + ("_" + name if name else "") + "_" + time.strftime("%Y%m%d-%H%M%S")
    return os.path.join(_run_folder(), stem)


def _workflow_json(path, params, prompt=None, extra_pnginfo=None, outputs=None):
    """Save a run as a JSON file: what it was, and the graph that made it.

    The API graph comes from the run itself and the UI graph from the
    extra data ComfyUI passes along, so the file is enough to repeat the run
    or to see afterwards what was asked for.
    """
    payload = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "prompt": prompt or {},
        "workflow": (extra_pnginfo or {}).get("workflow", {}),
        "outputs": outputs or {},
    }
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=1)
    logging.info("[SelfLift] workflow %s", path)
    return path


class SelfLiftImageSampler:
    """SelfLift-zero for compatible rectified-flow image backbones."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "positive": ("CONDITIONING",),
            "negative": ("CONDITIONING",),
            "vae": ("VAE", {"tooltip": "VAE used for the pixel re-encode anchor at the resolution transition."}),
            "latent_image": ("LATENT", {"tooltip": "Target-resolution latent defining the output size. Existing latent content and noise masks are accepted."}),
            "sampler": ("SAMPLER", {"tooltip": "Standard Euler only; SelfLift reuses its transition-step prediction to keep the original NFE count."}),
            "sigmas": ("SIGMAS",),
            "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff, "control_after_generate": True}),
            "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0, "step": 0.1, "round": 0.01}),
            "transition_step": ("INT", {"default": 6, "min": 1, "max": 10000, "tooltip": "Number of low-resolution denoiser evaluations. Paper: 3 of 4 for FLUX.2-Klein, 6 of 8 for Z-Image-Turbo."}),
            "lowres_scale": ("FLOAT", {"default": 0.5, "min": 0.25, "max": 1.0, "step": 0.05, "tooltip": "Spatial scale of the low-resolution prefix (paper: 0.5)."}),
            "rho": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Fraction of locations corrected toward the pixel-VAE anchor. Paper: 0.4 for FLUX.2-Klein, 0.3 for Z-Image-Turbo."}),
            "w_min": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            "w_max": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            "latent_upsample": (["nearest", "bilinear"], {"default": "nearest", "tooltip": "Interpolation for the direct latent lift (paper: nearest)."}),
        }, "optional": {
            "model_hires": ("MODEL", {"tooltip": "Optional: model used for the high-resolution stage instead of `model` (e.g. a different checkpoint or LoRA stack). Must share the same architecture and latent format. The low-resolution prefix always runs on `model`."}),
        }}

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "selflift"

    def sample(self, model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
               transition_step, lowres_scale, rho, w_min, w_max, latent_upsample, model_hires=None):
        return (progressive_sample(model, positive, negative, vae, latent_image, sampler, sigmas, seed, cfg,
                                   transition_step, lowres_scale, rho, w_min, w_max, latent_upsample,
                                   model_hires=model_hires),)


class SelfLiftH3TST:
    """Training-free Temporal State Transport correction (arXiv:2609.08505) for MiniMax H3."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "tau": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.05,
                              "tooltip": "Homeostatic correction strength; 0.2 is the paper setting. 0 disables correction while keeping the diagnostic active."}),
            "log_diagnostics": ("BOOLEAN", {"default": True, "label_on": "诊断日志：开启", "label_off": "诊断日志：关闭",
                                            "tooltip": "Log per-forward Spectral Tension and correction statistics to the console. Note: TST is skipped with a warning when highres_tiling is enabled on the SelfLift sampler."}),
        }}

    RETURN_TYPES = ("MODEL",)
    FUNCTION = "patch"
    CATEGORY = "selflift"

    def patch(self, model, tau, log_diagnostics):
        return (h3_tst.patch_model(model, tau, log_diagnostics),)


NODE_CLASS_MAPPINGS = {
    "SelfLiftH3Sampler": SelfLiftH3Sampler,
    "SelfLiftH3Draft": SelfLiftH3Draft,
    "SelfLiftImageSampler": SelfLiftImageSampler,
    "SelfLiftH3TST": SelfLiftH3TST,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SelfLiftH3Sampler": "SelfLift Progressive Sampler (MiniMax H3)",
    "SelfLiftH3Draft": "SelfLift H3 Draft (low-resolution preview)",
    "SelfLiftImageSampler": "SelfLift Progressive Sampler (Image)",
    "SelfLiftH3TST": "H3 Temporal State Transport (TST)",
}
