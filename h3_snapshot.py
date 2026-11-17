"""Freeze a paused SelfLift H3 draft, and read it back later.

A draft stops at the end of the low-resolution prefix, where the sampler state
(x at the transition sigma) and the Eq. 3 endpoint prediction are both known.
Everything the rest of the run needs -- the latent content, the noise mask, the
prompt conditioning, the sigma schedule and the SelfLift parameters -- is written
beside it, so a later run continues from the frozen state instead of sampling the
prefix again.

The tensors go into a safetensors file; everything else goes into a JSON
manifest next to it. The conditioning is the awkward part: a CONDITIONING is a
list of ``[tensor, dict]`` pairs whose dicts may hold tensors of their own (H3
keeps the encoded reference images there), so the manifest replaces every tensor
with a reference into the safetensors file.
"""

from __future__ import annotations

import json
import logging
import os
import time

import torch

import comfy.nested_tensor
import comfy.utils

#: Bumped when the on-disk layout changes; snapshots of another version are refused.
FORMAT_VERSION = 1

#: Conditioning keys ComfyUI rebuilds for every run. Saving them would freeze the
#: area and mask a condition had under a different noise shape.
RUNTIME_KEYS = frozenset({"area", "mask", "start_t", "end_t", "device"})

#: How a tensor is written in the manifest.
TENSOR_REF = "__tensor__"


class Snapshot:
    """What one paused H3 run left behind."""

    def __init__(self, state, x0, original, mask, positive, negative, sigmas, seed, cfg,
                 params, meta=None, nested=False):
        self.state = state        # x at the transition sigma, packed as H3 packs it
        self.x0 = x0              # the Eq. 3 endpoint prediction, in the same packing
        self.nested = nested      # whether those streams are packed as a NestedTensor
        self.original = original  # the target-resolution latent content (the keep-region anchor)
        self.mask = mask          # the noise mask, or None
        self.positive = positive  # CONDITIONING
        self.negative = negative  # CONDITIONING
        self.sigmas = sigmas      # the whole sigma schedule
        self.seed = seed
        self.cfg = cfg
        self.params = params      # the SelfLift parameters the run was started with
        self.meta = meta or {}    # provenance: the model, the workflow, the draft video


def _pack_meta(value, tensors, key):
    """Rewrite a conditioning meta dict so every tensor in it becomes a reference."""
    if isinstance(value, torch.Tensor):
        return {TENSOR_REF: tensors.put(key, value)}
    if isinstance(value, dict):
        return {name: _pack_meta(item, tensors, f"{key}.{name}")
                for name, item in value.items() if name not in RUNTIME_KEYS}
    if isinstance(value, (list, tuple)):
        return [_pack_meta(item, tensors, f"{key}.{index}") for index, item in enumerate(value)]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _unpack_meta(value, tensors):
    """The inverse of :func:`_pack_meta`."""
    if isinstance(value, dict):
        if len(value) == 1 and TENSOR_REF in value:
            return tensors[value[TENSOR_REF]]
        return {name: _unpack_meta(item, tensors) for name, item in value.items()}
    if isinstance(value, list):
        return [_unpack_meta(item, tensors) for item in value]
    return value


def _pack_conditioning(conditioning, tensors, prefix):
    entries = []
    for index, entry in enumerate(conditioning):
        tensor = entry[0]
        meta = entry[1] if len(entry) > 1 else {}
        key = tensors.put(f"{prefix}{index}.tensor", tensor)
        entries.append({"tensor": key, "meta": _pack_meta(meta, tensors, f"{prefix}{index}.meta")})
    return entries


def _unpack_conditioning(entries, tensors):
    return [[tensors[entry["tensor"]], _unpack_meta(entry["meta"], tensors)] for entry in entries]


class _Tensors:
    """The tensor file, with the bookkeeping safetensors asks for.

    safetensors refuses to write two tensors that share memory, so a value that
    appears twice in one snapshot is copied the second time.
    """

    def __init__(self):
        self.tensors = {}
        self._seen = set()

    def put(self, key, value):
        tensor = value.detach().cpu()
        pointer = tensor.untyped_storage().data_ptr() if tensor.numel() else 0
        if pointer in self._seen:
            tensor = tensor.clone()
        else:
            self._seen.add(pointer)
        self.tensors[key] = tensor.contiguous()
        return key


def _store(tensors, value, key):
    """Write a tensor, or a packed NestedTensor, into the tensor file.

    MiniMax H3's latent holds its video and audio streams in one NestedTensor,
    so what has to be frozen is not always a plain tensor.

    Returns:
        The manifest entry that says where it went.
    """
    if getattr(value, "is_nested", False):
        keys = []
        for index, stream in enumerate(value.unbind()):
            keys.append(tensors.put(f"{key}.{index}", stream))
        return {"nested": keys, "shape": [list(stream.shape) for stream in value.unbind()]}
    tensors.put(key, value)
    return {"key": key, "shape": list(value.shape)}


def _fetch(tensors, entry):
    """The inverse of :func:`_store`."""
    if "nested" in entry:
        return comfy.nested_tensor.NestedTensor([tensors[key] for key in entry["nested"]])
    return tensors[entry["key"]]


def save(snapshot, stem):
    """Write ``<stem>.safetensors`` and ``<stem>.json``; returns the manifest path."""
    tensors = _Tensors()
    manifest = {
        "version": FORMAT_VERSION,
        "created": time.time(),
        "params": snapshot.params,
        "seed": snapshot.seed,
        "cfg": snapshot.cfg,
        "nested": bool(snapshot.nested),
        "sigmas": [float(value) for value in snapshot.sigmas],
        "state": _store(tensors, snapshot.state, "state"),
        "x0": _store(tensors, snapshot.x0, "x0"),
        "positive": _pack_conditioning(snapshot.positive, tensors, "positive."),
        "negative": _pack_conditioning(snapshot.negative, tensors, "negative."),
        "meta": snapshot.meta,
    }
    if snapshot.original is not None:
        manifest["original"] = _store(tensors, snapshot.original, "original")
    if snapshot.mask is not None:
        manifest["mask"] = _store(tensors, snapshot.mask, "mask")

    folder = os.path.dirname(stem)
    if folder:
        os.makedirs(folder, exist_ok=True)
    comfy.utils.save_torch_file(tensors.tensors, stem + ".safetensors",
                                metadata={"format": "selflift-h3-snapshot",
                                          "version": str(FORMAT_VERSION)})
    manifest_path = stem + ".json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    return manifest_path


def load(path):
    """Read a snapshot back; ``path`` may name the manifest or the tensor file."""
    stem = path[:-len(".json")] if path.endswith(".json") else path
    if stem.endswith(".safetensors"):
        stem = stem[:-len(".safetensors")]
    manifest_path = stem + ".json"
    tensors_path = stem + ".safetensors"
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f"SelfLift: no snapshot manifest at {manifest_path}")
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    version = manifest.get("version")
    if version != FORMAT_VERSION:
        raise ValueError(f"SelfLift: snapshot version {version} cannot be read by this build "
                         f"(expected {FORMAT_VERSION})")
    tensors = comfy.utils.load_torch_file(tensors_path, safe_load=True)

    original = _fetch(tensors, manifest["original"]) if "original" in manifest else None
    mask = _fetch(tensors, manifest["mask"]) if "mask" in manifest else None
    return Snapshot(
        state=_fetch(tensors, manifest["state"]),
        x0=_fetch(tensors, manifest["x0"]),
        nested=manifest.get("nested", False),
        original=original,
        mask=mask,
        positive=_unpack_conditioning(manifest["positive"], tensors),
        negative=_unpack_conditioning(manifest["negative"], tensors),
        sigmas=torch.tensor(manifest["sigmas"], dtype=torch.float32),
        seed=manifest["seed"],
        cfg=manifest["cfg"],
        params=manifest.get("params", {}),
        meta=manifest.get("meta", {}),
    )
