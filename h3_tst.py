"""Temporal State Transport (TST) correction for MiniMax H3 joint attention.

Inference-time port of TST (arXiv:2609.08505) to H3's single-stream packed
attention. H3 has no separate temporal attention, so per-head frame-level
transport operators A (F x F) are built from spatial mean-pooled post-RoPE
queries/keys of the video segment, which is always the last packed segment.
Spectral Tension T = H_row - H_vN diagnoses fragmented (T<0) versus over-mixed
(T>0) temporal states; the homeostatic query temperature gamma =
exp(tau_eff * T) rescales only video-row queries. tau_eff follows the paper's
cosine schedules: stronger in deeper layers and earlier denoising steps.

Anchored to ComfyUI sources:
- video segment last: comfy/ldm/minimax/model.py PackedLayout ("target audio
  then target video, always the last two segments")
- injection point: comfy/ldm/modules/attention.py wrap_attn, which checks
  transformer_options["optimized_attention_override"] in every backend (and
  take()s AttentionTensorContainers before calling the override)
- per-forward video shape: WrappersMP.DIFFUSION_MODEL wraps MiniMaxH3Model._forward

Known limitation: not compatible with SelfLift highres_tiling. Wrappers run in
registration order, so the TST wrapper sits outside the tiling wrapper and only
sees the full-resolution shape; tile attention calls fail the length guard and
TST skips with a warning instead of mislocating the tile's video rows.
"""

import logging
import math
import os
import time
from functools import partial

import torch

import comfy.patcher_extension
from comfy.ldm.modules.attention import AttentionTensorContainer


def _unwrap(tensor):
    # H3 wraps q/k/v in AttentionTensorContainer, but wrap_attn take()s them
    # before calling this override (it has no container_function attribute),
    # so plain tensors are the common case here.
    return tensor.peek() if isinstance(tensor, AttentionTensorContainer) else tensor


def _spectral_tension(q, k, v0, frames, rows_per_frame, eps=1e-8):
    """Per-head signed Spectral Tension of the frame-level transport operator.

    q, k: [1, heads, S, head_dim] post-norm post-RoPE; video rows start at v0.
    Returns [heads] fp32; T>0 over-mixing, T<0 fragmented (paper Eqs. 1-3).
    """
    heads = q.shape[1]
    head_dim = q.shape[-1]
    qv = q[0, :, v0:].reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    kv = k[0, :, v0:].reshape(heads, frames, rows_per_frame, head_dim).mean(dim=2).float()
    a = (qv @ kv.transpose(-2, -1) * head_dim**-0.5).softmax(dim=-1).clamp(min=eps)
    log_f = math.log(frames)
    h_row = -(a * a.log()).sum(dim=-1).mean(dim=-1) / log_f
    gram = a @ a.transpose(-2, -1)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=eps)
    eig = torch.linalg.eigvalsh(gram / trace[:, None, None]).clamp(min=eps)
    eig = eig / eig.sum(dim=-1, keepdim=True).clamp(min=eps)
    h_vn = -(eig * eig.log()).sum(dim=-1) / log_f
    return h_row - h_vn


def _exact_tension(q, k, v0, frames, rows_per_frame, eps=1e-8, chunk=64):
    """Exact frame-mass operator tension, for calibrating the pooled prototype.

    Per query token: full-row softmax over the whole packed sequence, video-key
    mass summed per frame, averaged over the frame's query rows, rows
    renormalized over frames (the true joint-attention transport operator).
    q, k: [1, heads, S, head_dim]; returns [heads] fp32 like _spectral_tension.
    """
    heads = q.shape[1]
    head_dim = q.shape[-1]
    qv = q[0, :, v0:]
    k_all = k[0].float()
    log_f = math.log(frames)
    a = q.new_zeros(heads, frames, frames, dtype=torch.float32)
    for start in range(0, frames * rows_per_frame, chunk):
        stop = min(start + chunk, frames * rows_per_frame)
        logits = (qv[:, start:stop].float() @ k_all.transpose(-2, -1)) * head_dim**-0.5
        m = logits.softmax(dim=-1)[..., v0:]
        m = m.reshape(heads, stop - start, frames, rows_per_frame).sum(dim=-1)
        idx = torch.arange(start, stop, device=m.device) // rows_per_frame
        a.scatter_add_(1, idx.view(1, -1, 1).expand(heads, -1, frames), m)
        del logits, m
    a = a / rows_per_frame
    a = (a / a.sum(dim=-1, keepdim=True).clamp(min=eps)).clamp(min=eps)
    h_row = -(a * a.log()).sum(dim=-1).mean(dim=-1) / log_f
    gram = a @ a.transpose(-2, -1)
    trace = gram.diagonal(dim1=-2, dim2=-1).sum(dim=-1).clamp(min=eps)
    eig = torch.linalg.eigvalsh(gram / trace[:, None, None]).clamp(min=eps)
    eig = eig / eig.sum(dim=-1, keepdim=True).clamp(min=eps)
    h_vn = -(eig * eig.log()).sum(dim=-1) / log_f
    return h_row - h_vn


def _attention_override(state):
    def override(func, q, k, v, heads, mask=None, **kwargs):
        frames = state["frames"]
        rows = state["rows_per_frame"]
        qt = _unwrap(q)
        s = qt.shape[2]
        # video rows are the last frames*rows of the packed sequence; the guard
        # also rejects token-refiner calls (text length is far below frames*rows).
        # Only video attention calls advance the layer counter, so a refiner
        # running inside a forward cannot shift the layer schedule.
        if frames is None or frames < 2 or s <= frames * rows:
            if frames is not None and frames >= 2:
                state["guard_misses"] += 1
            return func(q, k, v, heads, mask=mask, **kwargs)
        state["calls_this_forward"] += 1
        # GPU time via events: host-side perf_counter would absorb the sync stall
        # from the float() conversions below (they wait on the whole GPU queue,
        # not just TST's kernels) and report the forward's runtime as TST's
        timed = state["log_diagnostics"]
        if timed and qt.is_cuda:
            tst_start, tst_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            tst_start.record()
        elif timed:
            tst_start = time.perf_counter()
        layers = state["layers"]
        layer = (state["calls_this_forward"] - 1) % layers
        total_steps = state["total_steps"]
        step = min(state["step"], total_steps - 1)
        layer_weight = 0.5 - 0.5 * math.cos(math.pi * layer / (layers - 1)) if layers > 1 else 1.0
        step_weight = 0.5 + 0.5 * math.cos(math.pi * step / (total_steps - 1)) if total_steps > 1 else 1.0
        tension = _spectral_tension(qt, _unwrap(k), s - frames * rows, frames, rows)
        gamma = torch.exp(tension * (state["tau"] * layer_weight * step_weight))
        # calibration probe needs pre-correction q; the scaling below is in place
        exact_probe = state["exact_debug"] and layer == layers // 2
        q_pre = qt.clone() if exact_probe else None
        if state["tau"] != 0.0:
            qt[0, :, s - frames * rows:].mul_(gamma.to(qt.dtype)[:, None, None])
        if state["log_diagnostics"]:
            if qt.is_cuda:
                tst_end.record()
                state["fw_events"].append((tst_start, tst_end))
            else:
                state["fw_tst_time"].append(time.perf_counter() - tst_start)
            state["fw_abs_tension"].append(float(tension.abs().mean()))
            # within-call check (paper appendix Figure A): tension of the corrected
            # operator must drop below the pre-correction value
            post = _spectral_tension(qt, _unwrap(k), s - frames * rows, frames, rows) if state["tau"] != 0.0 else tension
            state["fw_abs_tension_post"].append(float(post.abs().mean()))
            state["fw_gamma"].append(float(gamma.mean()))
            state["fw_active"].append(float(((gamma - 1.0).abs() > 0.03).float().mean()))
            if layer == layers - 1 and state["fw_gamma"]:
                tst_ms = 1000.0 * sum(state["fw_tst_time"])
                for ev_start, ev_end in state["fw_events"]:
                    ev_end.synchronize()
                    tst_ms += ev_start.elapsed_time(ev_end)
                n = len(state["fw_abs_tension"])
                logging.info(
                    "[H3 TST] step %d/%d frames=%d grid_rows=%d pre|T|=%.4f post|T|=%.4f mean_gamma=%.4f active_heads=%.0f%% tst_time=%.2fs",
                    state["step"], total_steps, frames, rows,
                    sum(state["fw_abs_tension"]) / n,
                    sum(state["fw_abs_tension_post"]) / len(state["fw_abs_tension_post"]),
                    sum(state["fw_gamma"]) / len(state["fw_gamma"]),
                    100.0 * sum(state["fw_active"]) / len(state["fw_active"]),
                    tst_ms / 1000.0)
                state["fw_abs_tension"].clear()
                state["fw_abs_tension_post"].clear()
                state["fw_gamma"].clear()
                state["fw_active"].clear()
                state["fw_tst_time"].clear()
                state["fw_events"].clear()
        if exact_probe:
            exact = _exact_tension(q_pre, _unwrap(k), s - frames * rows, frames, rows)
            del q_pre
            logging.info(
                "[H3 TST exact] step %d layer %d/%d: proto_T=%+.4f exact_T=%+.4f proto|T|=%.4f exact|T|=%.4f sign_agree=%.0f%%",
                state["step"], layer, layers,
                float(tension.mean()), float(exact.mean()),
                float(tension.abs().mean()), float(exact.abs().mean()),
                100.0 * float(((exact > 0) == (tension > 0)).float().mean()))
        return func(q, k, v, heads, mask=mask, **kwargs)
    return override



def _check_packed_order(state, transformer_options):
    """Check that the video segment really is the last one in the packed sequence.

    TST locates the video rows by counting backwards from the end of the packed
    sequence: v0 = s - frames * rows_per_frame. That is only right while the
    video segment is the last one, which is what core 0.45 guarantees today
    ("target audio (none for the image model) then target video, always last").

    If a new conditioning kind is ever packed AFTER the video -- a video
    reference, a mask reference, a second reel -- then v0 points into that
    trailing segment instead of into the video, and the correction lands on the
    wrong rows. Nothing raises: the tension numbers still print, they are just
    computed over the wrong rows. That is the failure this check exists for.

    The core hands the layout over in transformer_options["minimax_h3_layout"]
    (the segment table it builds for the attention patches), so the check reads
    the real spans instead of guessing them.
    """
    layout = transformer_options.get("minimax_h3_layout")
    if layout is None:
        return
    segments = getattr(layout, "segments", None)
    if not segments:
        return
    frames = state.get("frames")
    rows = state.get("rows_per_frame")
    if not frames or not rows:
        return
    start, stop, kind = segments[-1]
    video_rows = frames * rows
    if kind == "video" and (stop - start) == video_rows:
        state["layout_checked"] = True
        return
    # something sits after the video segment: TST would locate the wrong rows
    trailing = ", ".join("%s[%d:%d]" % (k, a, b) for a, b, k in segments[-3:])
    message = (
        "[H3 TST] the packed sequence does not end with the video segment: the last "
        "segment is '%s' and the trailing spans are %s. TST locates the video rows by "
        "counting backwards from the end of the sequence, so it would correct the wrong "
        "rows and the result would silently get worse, not better. This happens when a "
        "new conditioning kind is packed after the video (a video reference, a mask "
        "reference, a second reel). Either remove the trailing segment or take the TST "
        "node out of the model chain. Expected the last segment to be 'video' with %d "
        "rows (frames=%d x rows_per_frame=%d), found '%s' with %d rows. The layout comes "
        "from the core, under the transformer_options key 'minimax_h3_layout'."
    ) % (kind, trailing, video_rows, frames, rows, kind, stop - start)
    state["layout_error"] = message
    # Raised on every forward, not only the first: the point of the check is that
    # the run stops, and a dedup guard would let a caught exception let the next
    # forward through with the same wrong layout.
    raise ValueError(message)

def _forward_wrapper(state, executor, x, timestep, context, transformer_options, minimax_payload=None, **kwargs):
    if state["calls_this_forward"] > 0:
        state["layers"] = state["calls_this_forward"]
    elif state["guard_misses"] > 0 and not state["warned_inactive"]:
        # every attention call of the previous forward was shorter than the
        # registered video segment: the wrapper sits outside a spatial-tiling
        # wrapper (wrappers run in registration order) and only saw the
        # full-resolution shape, so TST cannot locate the tile's video rows
        logging.warning("[H3 TST] inactive: packed sequence shorter than the expected video segment; "
                        "TST is not compatible with highres_tiling and was skipped")
        state["warned_inactive"] = True
    state["calls_this_forward"] = 0
    state["guard_misses"] = 0
    # diagnostics off leaves fw_tst_time/fw_events unreported; drop them per forward instead of leaking
    state["fw_tst_time"].clear()
    state["fw_events"].clear()
    video = x[0] if isinstance(x, (list, tuple)) else x
    if isinstance(video, torch.Tensor) and video.ndim == 5:
        state["frames"] = int(video.shape[2])
        state["rows_per_frame"] = ((video.shape[3] + 1) // 2) * ((video.shape[4] + 1) // 2)
    else:
        state["frames"] = None
    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is not None:
        sigma = float(timestep.flatten()[0]) / 1000.0
        state["total_steps"] = max(1, int(sigmas.numel()) - 1)
        state["step"] = int((sigmas.float().cpu() - sigma).abs().argmin())
    # the layout is rebuilt by the core for every forward, so this re-checks it
    # every forward too: the segment order can change between steps if a
    # conditioning block is added or removed mid-run.
    _check_packed_order(state, transformer_options)
    return executor(x, timestep, context, transformer_options, minimax_payload=minimax_payload, **kwargs)



def _check_override_slot(model):
    """Report an attention override that is already in the single slot.

    The core keeps one attention-rewrite entry point, and it is a single slot:
    whoever writes it last wins. TST writes it, so anything already sitting
    there is either another rewrite that is about to be overwritten, or a
    rewrite that already overwrote TST.

    The override is a closure, so its name says nothing about its owner. What
    can be said is that it is not TST's own: TST installs into the slot only
    here, so a slot that is already filled when this runs came from somebody
    else. Refusing to start is the honest answer -- the run would otherwise
    look fine and silently lose one of the two corrections.
    """
    try:
        options = model.model_options or {}
        transformer_options = options.get("transformer_options") or {}
    except Exception:
        return None
    existing = transformer_options.get("optimized_attention_override")
    if existing is None:
        return None
    return existing

def patch_model(model, tau, log_diagnostics=True):
    state = {"tau": float(tau), "log_diagnostics": bool(log_diagnostics),
             "exact_debug": os.environ.get("SELFLIFT_TST_EXACT", "0") == "1",
             "frames": None, "rows_per_frame": None,
             "step": 0, "total_steps": 1, "layers": 50, "calls_this_forward": 0,
             "guard_misses": 0, "warned_inactive": False,
             "layout_checked": False, "layout_error": None,
             "fw_abs_tension": [], "fw_abs_tension_post": [], "fw_gamma": [], "fw_active": [],
             "fw_tst_time": [], "fw_events": []}
    existing = _check_override_slot(model)
    if existing is not None:
        raise ValueError(
            "SelfLift H3 TST: this model already carries an attention override in the core's "
            "single 'optimized_attention_override' slot, and that slot only keeps one writer: "
            "the last one wins. Installing TST now would silently replace it (and the other "
            "way round, a rewrite installed after this node would silently replace TST). Both "
            "corrections are supposed to run together, and the slot cannot hold them. The "
            "override's first argument is the layer below it, so the two can be chained "
            "instead: wrap the existing override with TST's, or wrap TST's with it. Until that "
            "is done, take one of the two nodes out of the chain."
        )
    patched = model.clone()
    patched.add_wrapper_with_key(comfy.patcher_extension.WrappersMP.DIFFUSION_MODEL,
                                 "selflift_h3_tst", partial(_forward_wrapper, state))
    patched.model_options.setdefault("transformer_options", {})["optimized_attention_override"] = _attention_override(state)
    return patched
