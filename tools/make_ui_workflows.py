"""Derive the preview graph from the user's own workflow.

The preview graph is that workflow with the sampler swapped for the draft
node, so the two share every other node: the model chain, the prompt, the
reference images, the sigma schedule and the decode tail. Editing the user's
graph in place keeps every other node's widget order correct -- rebuilding the
graph from scratch would not.

Two things are changed on purpose:

  * the seed stops randomizing, so the preview and the finished clip share it
  * the preview drops to one frame per step, because the step preview decodes
    the whole clip otherwise

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/make_ui_workflows.py
"""
import copy
import json
import os
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = (r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
          r"\user\default\workflows\H3 SelfLift Dual Sample｜Strong Action Ref to Video.json")
OUT = os.path.join(REPO, "workflows")

SAMPLER = 171
PREVIEW = 153
SAVE_VIDEO = 274
AUDIO_VAE = 184


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


def drop_link(workflow, link_id):
    """Remove a link, clearing both of its ends where they still exist."""
    for index, link in enumerate(workflow["links"]):
        if link[0] == link_id:
            origin_id, origin_slot, target_id, target_slot = link[1], link[2], link[3], link[4]
            origin = node_of(workflow, origin_id)
            links = origin["outputs"][origin_slot].get("links")
            if links and link_id in links:
                links.remove(link_id)
            target = node_of(workflow, target_id)
            if target_slot < len(target["inputs"]):
                target["inputs"][target_slot]["link"] = None
            del workflow["links"][index]
            return
    raise KeyError(f"no link {link_id}")


def retarget(workflow, link_id, slot):
    for link in workflow["links"]:
        if link[0] == link_id:
            link[4] = slot
            return
    raise KeyError(f"no link {link_id}")


def add_link(workflow, link_id, origin_id, origin_slot, target_id, target_slot, kind):
    """Add a link and register it on both of its ends."""
    workflow["links"].append([link_id, origin_id, origin_slot, target_id, target_slot, kind])
    origin = node_of(workflow, origin_id)
    links = origin["outputs"][origin_slot].setdefault("links", [])
    links.append(link_id)
    target = node_of(workflow, target_id)
    target["inputs"][target_slot]["link"] = link_id


def build_preview(source):
    workflow = copy.deepcopy(source)
    workflow["id"] = str(uuid.uuid4())
    workflow["revision"] = 0

    # The link surgery comes first; the node's input list is rebuilt afterwards.
    drop_link(workflow, 396)              # the draft node has no vae input
    retarget(workflow, 348, 3)
    retarget(workflow, 349, 4)
    retarget(workflow, 410, 5)
    for link in list(workflow["links"]):
        if link[3] == SAMPLER and link[1] == AUDIO_VAE:
            drop_link(workflow, link[0])   # the draft gets its own audio VAE link

    sampler = node_of(workflow, SAMPLER)
    sampler["type"] = "SelfLiftH3Draft"
    sampler["properties"]["Node name for S&R"] = "SelfLiftH3Draft"
    sampler["inputs"] = [
        entry("model", "MODEL", 344),
        entry("positive", "CONDITIONING", 343),
        entry("negative", "CONDITIONING", 346),
        entry("latent_image", "LATENT", 348),
        entry("sampler", "SAMPLER", 349),
        entry("sigmas", "SIGMAS", 410),
        entry("seed", "INT", widget=True),
        entry("cfg", "FLOAT", widget=True),
        entry("transition_step", "INT", widget=True),
        entry("lowres_scale", "FLOAT", widget=True),
        entry("preview_vae", "COMBO", widget=True),
        entry("preview_video", "BOOLEAN", widget=True),
        entry("fps", "FLOAT", widget=True),
        entry("audio_vae", "VAE", shape=7),
        entry("run_name", "STRING", widget=True),
    ]
    # The seed stays put, so the preview and the finished clip are one run.
    sampler["widgets_values"] = [684655983592117, "fixed", 1, 2, 0.4,
                                 "taeh3.safetensors", True, 25.0, "selflift"]
    sampler["widgets_values_named"] = {
        "seed": 684655983592117, "control_after_generate": "fixed",
        "cfg": 1, "transition_step": 2, "lowres_scale": 0.4,
        "preview_vae": "taeh3.safetensors", "preview_video": True, "fps": 25.0,
        "run_name": "selflift",
    }
    sampler["outputs"] = [sampler["outputs"][0]]
    sampler["outputs"].append({
        "label": "preview_video", "localized_name": "preview_video",
        "name": "preview_video", "type": "STRING", "links": None,
    })
    sampler["outputs"].append({
        "label": "workflow_json", "localized_name": "workflow_json",
        "name": "workflow_json", "type": "STRING", "links": None,
    })
    # The draft's own mp4 carries the soundscape: the audio VAE comes along.
    add_link(workflow, max(link[0] for link in workflow["links"]) + 1,
             AUDIO_VAE, 0, SAMPLER, 13, "VAE")

    # One preview frame per step: the step preview decodes the whole clip otherwise.
    preview = node_of(workflow, PREVIEW)
    for index, item in enumerate(preview["inputs"]):
        if item["name"] == "preview_frames" and item.get("link"):
            drop_link(workflow, item["link"])
            item["link"] = None
            preview["widgets_values"][index] = 1
            preview["widgets_values_named"]["preview_frames"] = 1

    save = node_of(workflow, SAVE_VIDEO)
    save["widgets_values"][0] = "video/draft"
    return workflow


def main():
    with open(SOURCE, "r", encoding="utf-8") as handle:
        source = json.load(handle)
    name = "H3 SelfLift Preview｜Low-res Draft.json"
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(build_preview(source), handle, ensure_ascii=False, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
