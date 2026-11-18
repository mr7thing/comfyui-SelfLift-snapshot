"""Derive the preview graph from the user's own workflow.

The preview graph is that workflow with the sampler swapped for the preview
sampler and the save node put in between the sampler and the decoders, so the
two graphs share every other node: the model chain, the prompt, the reference
images, the sigma schedule and the decode tail. Editing the user's graph in
place keeps every other node's widget order correct -- rebuilding the graph
from scratch would not.

The preview sampler draws the preview and writes the run down; the save node
files those files away and passes the latent on. The decoders hang off the save
node, so the graph works in both modes: in preview mode the save node keeps the
latent to itself, so nothing downstream runs.

One thing is changed on purpose: the step preview drops to one frame per step,
because the step preview decodes the whole clip otherwise.

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

SAMPLER = 171        # SelfLiftH3Sampler in the user's workflow
SAVE = 400           # the node this script adds
PREVIEW = 153        # the step preview (ModelPreviewOverrideKJ)
MODEL_NODE = 153     # where the sampler's model comes from
AUDIO_VAE = 184      # the H3 audio VAE


def node_of(workflow, node_id):
    for node in workflow["nodes"]:
        if node["id"] == node_id:
            return node
    raise KeyError(f"no node {node_id}")


def link_input(name, kind, link=None, shape=None):
    item = {"label": name, "localized_name": name, "name": name, "type": kind}
    if shape is not None:
        item["shape"] = shape
    item["link"] = link
    return item


def widget(name, kind):
    return {"label": name, "localized_name": name, "name": name, "type": kind,
            "widget": {"name": name}, "link": None}


def output(name, kind, links=None):
    return {"label": name, "localized_name": name, "name": name, "type": kind, "links": links}


def add_link(workflow, link_id, origin_id, origin_slot, target_id, target_slot, kind):
    """Add a link and register it on both of its ends."""
    workflow["links"].append([link_id, origin_id, origin_slot, target_id, target_slot, kind])
    origin = node_of(workflow, origin_id)
    links = origin["outputs"][origin_slot].get("links")
    if links is None:
        links = origin["outputs"][origin_slot]["links"] = []
    if link_id not in links:
        links.append(link_id)
    target = node_of(workflow, target_id)
    target["inputs"][target_slot]["link"] = link_id


def preview_sampler(sampler, clip_fps):
    """Turn the sampler into the preview sampler: it draws the preview itself.

    The widget order follows the node's own: the seed's `control_after_generate`
    button is one entry of its own, right after the seed.
    """
    sampler["type"] = "SelfLiftH3DraftSampler"
    sampler["properties"]["Node name for S&R"] = "SelfLiftH3DraftSampler"

    links = [item for item in sampler["inputs"] if not item.get("widget")]
    widgets = [item for item in sampler["inputs"] if item.get("widget")]
    names = [item["name"] for item in widgets]
    at = names.index("upscaler_model") + 1
    widgets[at:at] = [widget("draft_mode", "BOOLEAN"),
                      widget("preview_vae", "COMBO"),
                      widget("fps", "FLOAT")]
    sampler["inputs"] = links + widgets

    named = sampler["widgets_values_named"]
    values = list(sampler["widgets_values"])
    at = list(named.keys()).index("upscaler_model") + 1
    if values[at - 1] != named["upscaler_model"]:
        raise RuntimeError("the widget values and the named values do not line up")
    values[at:at] = [True, "taeh3.safetensors", clip_fps]
    sampler["widgets_values"] = values

    named = {}
    for key, value in sampler["widgets_values_named"].items():
        named[key] = value
        if key == "upscaler_model":
            named["draft_mode"] = True
            named["preview_vae"] = "taeh3.safetensors"
            named["fps"] = clip_fps
    sampler["widgets_values_named"] = named

    # The audio VAE comes with the preview: the preview mp4 carries the soundscape.
    links.append(link_input("audio_vae", "VAE", shape=7))
    sampler["inputs"] = links + widgets

    sampler["outputs"] = [
        {"label": "LATENT", "localized_name": "Latent", "name": "LATENT", "type": "LATENT",
         "links": []},
        output("draft_video", "STRING"),
        output("workflow_json", "STRING"),
        output("workflow_api", "STRING"),
    ]
    return sampler


def save_node(links, clip_fps):
    """The node that files the run away and passes the latent on."""
    return {
        "id": SAVE,
        "type": "SelfLiftH3Save",
        "pos": [2640.7, 700.0],
        "size": [420, 240],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            link_input("latent", "LATENT", links["latent"]),
            link_input("draft_video", "STRING", links["draft_video"]),
            link_input("workflow_json", "STRING", links["workflow_json"]),
            link_input("workflow_api", "STRING", links["workflow_api"]),
            widget("output_folder", "STRING"),
            widget("file_name", "STRING"),
        ],
        "outputs": [
            output("latent", "LATENT", []),
        ],
        "properties": {"Node name for S&R": "SelfLiftH3Save", "ver": "1.0"},
        "widgets_values": ["", "", "", "selflift", "selflift"],
        "widgets_values_named": {"draft_video": "", "workflow_json": "", "workflow_api": "",
                                 "output_folder": "selflift", "file_name": "selflift"},
        "color": "#323", "bgcolor": "#535",
    }


def build_preview(source):
    workflow = copy.deepcopy(source)
    workflow["id"] = str(uuid.uuid4())
    workflow["revision"] = 0

    clip_fps = float(node_of(workflow, 249)["widgets_values"][0])
    sampler = preview_sampler(node_of(workflow, SAMPLER), clip_fps)

    # The decoders now hang off the save node; the sampler feeds the save node.
    decoder_links = []
    for link in workflow["links"]:
        if link[1] == SAMPLER and link[3] in (121, 122):
            decoder_links.append(link)
    if len(decoder_links) != 2:
        raise RuntimeError(f"expected two decoder links off the sampler, found {len(decoder_links)}")

    link_id = max(link[0] for link in workflow["links"])
    new_links = {
        "latent": link_id + 1,
        "draft_video": link_id + 2,
        "workflow_json": link_id + 3,
        "workflow_api": link_id + 4,
        "audio_vae": link_id + 5,
    }

    for link in decoder_links:
        link[1] = SAVE
    sampler["outputs"][0]["links"] = [new_links["latent"]]

    workflow["nodes"].append(save_node(new_links, clip_fps))
    node_of(workflow, SAVE)["outputs"][0]["links"] = [link[0] for link in decoder_links]
    add_link(workflow, new_links["latent"], SAMPLER, 0, SAVE, 0, "LATENT")
    add_link(workflow, new_links["draft_video"], SAMPLER, 1, SAVE, 1, "STRING")
    add_link(workflow, new_links["workflow_json"], SAMPLER, 2, SAVE, 2, "STRING")
    add_link(workflow, new_links["workflow_api"], SAMPLER, 3, SAVE, 3, "STRING")
    audio_slot = next(index for index, item in enumerate(sampler["inputs"])
                      if item["name"] == "audio_vae")
    add_link(workflow, new_links["audio_vae"], AUDIO_VAE, 0, SAMPLER, audio_slot, "VAE")
    workflow["last_link_id"] = new_links["audio_vae"]
    workflow["last_node_id"] = max(int(workflow.get("last_node_id") or 0), SAVE)

    # The save node runs where the sampler used to hand over to the decoders.
    after = int(sampler.get("order") or 0)
    for node in workflow["nodes"]:
        if node["id"] == SAVE:
            node["order"] = after + 1
        elif int(node.get("order") or 0) > after:
            node["order"] = int(node["order"]) + 1

    # One preview frame per step: the step preview decodes the whole clip otherwise.
    preview = node_of(workflow, PREVIEW)
    for index, item in enumerate(preview["inputs"]):
        if item["name"] == "preview_frames" and item.get("link"):
            preview["widgets_values"][index] = 1
            preview["widgets_values_named"]["preview_frames"] = 1

    return workflow


def main():
    with open(SOURCE, "r", encoding="utf-8") as handle:
        source = json.load(handle)
    name = "H3 SelfLift Draft Mode.json"
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(build_preview(source), handle, ensure_ascii=False, indent=2)
    print(f"  wrote {path}")


if __name__ == "__main__":
    main()
