"""Derive the two redesigned graphs from the user's own workflow.

The draft graph is that workflow with the sampler swapped for the freezing
node; the resume graph is the same model chain feeding the resuming node.
Editing the user's graph in place keeps every other node's widget order
correct -- rebuilding the graph from scratch would not.

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
VIDEO_VAE = 215
AUDIO_VAE = 184
TAIL = (121, 122, 249, 274)      # VAEDecodeAudio, VAEDecode, CreateVideo, SaveVideo
# The prompt branch goes away in the resume graph: the snapshot carries the
# conditioning, the reference images and the sigma schedule.
PROMPT_BRANCH = (136, 173, 224, 128, 216, 115, 132, 131, 332, 333)
# Left-overs from the source workflow that the resume graph has no use for.
LOOSE = (222, 227, 228, 238)


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
    workflow["links"].append([link_id, origin_id, origin_slot, target_id, target_slot, kind])
    origin = node_of(workflow, origin_id)
    origin["outputs"][origin_slot].setdefault("links", []).append(link_id)
    target = node_of(workflow, target_id)
    if target_slot < len(target["inputs"]):
        target["inputs"][target_slot]["link"] = link_id
    workflow["last_link_id"] = max(workflow.get("last_link_id", 0), link_id)


def drop_node(workflow, node_id):
    node = node_of(workflow, node_id)
    for link in list(workflow["links"]):
        if link[1] == node_id or link[3] == node_id:
            drop_link(workflow, link[0])
    workflow["nodes"] = [item for item in workflow["nodes"] if item["id"] != node_id]


def build_draft(source):
    workflow = copy.deepcopy(source)
    workflow["id"] = str(uuid.uuid4())
    workflow["revision"] = 0

    for node_id in TAIL:
        drop_node(workflow, node_id)

    # The link surgery comes first; the node's input list is rebuilt afterwards.
    drop_link(workflow, 396)              # the draft node has no vae input
    retarget(workflow, 348, 3)
    retarget(workflow, 349, 4)
    retarget(workflow, 410, 5)
    add_link(workflow, 554, AUDIO_VAE, 0, SAMPLER, 6, "VAE")

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
        entry("audio_vae", "VAE", 554, shape=7),
        entry("seed", "INT", widget=True),
        entry("cfg", "FLOAT", widget=True),
        entry("transition_step", "INT", widget=True),
        entry("lowres_scale", "FLOAT", widget=True),
        entry("rho", "FLOAT", widget=True),
        entry("w_min", "FLOAT", widget=True),
        entry("w_max", "FLOAT", widget=True),
        entry("upscaler_model", "COMBO", widget=True),
        entry("snapshot_name", "STRING", widget=True),
        entry("draft_video", "BOOLEAN", widget=True),
        entry("highres_tiling", "BOOLEAN", widget=True, shape=7),
        entry("upscaler_unload", "BOOLEAN", widget=True, shape=7),
    ]
    sampler["widgets_values"] = [684655983592117, "randomize", 1, 5, 0.4, 0.6, 0.5, 1,
                                 "minimax_h3_latent_upscaler_3d_fp16_V2.safetensors",
                                 "h3-draft", True, True, True]
    sampler["outputs"][0]["links"] = []
    return workflow


def build_resume(source):
    workflow = copy.deepcopy(source)
    workflow["id"] = str(uuid.uuid4())
    workflow["revision"] = 0

    for node_id in tuple(PROMPT_BRANCH) + tuple(LOOSE):
        if node_id in {item["id"] for item in workflow["nodes"]}:
            drop_node(workflow, node_id)
    for link_id in (343, 346, 348, 410):
        if any(link[0] == link_id for link in workflow["links"]):
            drop_link(workflow, link_id)
    retarget(workflow, 396, 1)
    retarget(workflow, 349, 2)

    sampler = node_of(workflow, SAMPLER)
    sampler["type"] = "SelfLiftH3Resume"
    sampler["properties"]["Node name for S&R"] = "SelfLiftH3Resume"
    sampler["inputs"] = [
        entry("model", "MODEL", 344),
        entry("vae", "VAE", 396),
        entry("sampler", "SAMPLER", 349),
        entry("snapshot_path", "STRING", widget=True),
    ]
    sampler["widgets_values"] = [""]
    sampler["outputs"][0]["links"] = [351, 353]
    return workflow


def main():
    with open(SOURCE, "r", encoding="utf-8") as handle:
        source = json.load(handle)
    for name, workflow in (("H3 SelfLift Draft｜Freeze and Preview.json", build_draft(source)),
                           ("H3 SelfLift Resume｜Continue from Snapshot.json", build_resume(source))):
        path = os.path.join(OUT, name)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(workflow, handle, ensure_ascii=False, indent=2)
        print(f"  wrote {path}")


if __name__ == "__main__":
    main()
