"""Bring a saved UI workflow's SelfLift sampler in line with the node definition.

The node grew a preview decoder, a preview toggle, a frame rate, an audio VAE
and a run name, and it now hands back the preview path and the workflow JSON
beside the latent. A saved workflow keeps its old widget list, so the new
widgets have to be added in the right place -- the frontend reads widget values
by position, and a mismatch shifts every value after it.

The patch keeps the sockets first and the widgets after them, the order the
frontend writes them in, and leaves the existing values where they are.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/patch_sampler_node.py <workflow.json>
"""
import json
import os
import sys

SAMPLER = 171
AUDIO_VAE = 184

SOCKETS = [
    ("model", "MODEL"), ("positive", "CONDITIONING"), ("negative", "CONDITIONING"),
    ("vae", "VAE"), ("latent_image", "LATENT"), ("sampler", "SAMPLER"), ("sigmas", "SIGMAS"),
    ("model_hires", "MODEL"), ("audio_vae", "VAE"),
]

WIDGETS = [
    ("seed", "INT"), ("cfg", "FLOAT"), ("transition_step", "INT"), ("lowres_scale", "FLOAT"),
    ("rho", "FLOAT"), ("w_min", "FLOAT"), ("w_max", "FLOAT"), ("upscaler_model", "COMBO"),
    ("highres_tiling", "BOOLEAN"), ("upscaler_unload", "BOOLEAN"),
    ("preview_vae", "COMBO"), ("preview_video", "BOOLEAN"), ("fps", "FLOAT"), ("run_name", "STRING"),
]

WIDGET_VALUES = {
    "preview_vae": "taeh3.safetensors",
    "preview_video": True,
    "fps": 25.0,
    "run_name": "selflift",
}


def node_of(workflow, node_id):
    for node in workflow["nodes"]:
        if node["id"] == node_id:
            return node
    raise KeyError(f"no node {node_id}")


def entry(name, kind, link=None, widget=False, shape=None):
    item = {"label": name, "localized_name": name, "name": name, "type": kind}
    if widget:
        item["widget"] = {"name": name}
    if shape is not None:
        item["shape"] = shape
    item["link"] = link
    return item


def patch(path):
    with open(path, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)

    sampler = node_of(workflow, SAMPLER)
    if "preview_vae" in sampler.get("widgets_values_named", {}):
        print(f"  {path} is already patched")
        return
    old_links = {item["name"]: item.get("link") for item in sampler["inputs"]}

    sampler["inputs"] = []
    for name, kind in SOCKETS:
        item = entry(name, kind, old_links.get(name))
        if name in ("model_hires", "audio_vae"):
            item["shape"] = 7
        sampler["inputs"].append(item)
    for name, kind in WIDGETS:
        sampler["inputs"].append(entry(name, kind, widget=True))

    named = dict(sampler.get("widgets_values_named", {}))
    values = list(sampler.get("widgets_values", []))
    # The seed's own control value sits right after it.
    named.update(WIDGET_VALUES)
    values.extend(WIDGET_VALUES[name] for name, _ in WIDGETS
                  if name in WIDGET_VALUES)
    sampler["widgets_values"] = values
    sampler["widgets_values_named"] = named

    sampler["outputs"].append({
        "label": "preview_video", "localized_name": "preview_video",
        "name": "preview_video", "type": "STRING", "links": None,
    })
    sampler["outputs"].append({
        "label": "workflow_json", "localized_name": "workflow_json",
        "name": "workflow_json", "type": "STRING", "links": None,
    })

    # The audio VAE, so the preview carries the soundscape too.
    audio_slot = [name for name, _ in SOCKETS].index("audio_vae")
    already = any(link[1] == AUDIO_VAE and link[3] == SAMPLER for link in workflow["links"])
    if not already:
        link_id = max(link[0] for link in workflow["links"]) + 1
        workflow["links"].append([link_id, AUDIO_VAE, 0, SAMPLER, audio_slot, "VAE"])
        source = node_of(workflow, AUDIO_VAE)
        source["outputs"][0].setdefault("links", []).append(link_id)
        sampler["inputs"][audio_slot]["link"] = link_id

    with open(path, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, ensure_ascii=False, indent=2)
    print(f"  patched {path}")


if __name__ == "__main__":
    patch(sys.argv[1])
