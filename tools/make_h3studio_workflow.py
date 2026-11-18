"""Derive the H3 Studio x SelfLift graph from H3 Studio's own example.

H3 Studio's "Basic Usage" graph drives the model with SamplerCustomAdvanced.
SelfLift's sampler wants the same three things that graph hands to it -- the
positive conditioning, the target latent, and the sigma schedule -- and the
schedule keeps coming from H3 Studio's H3BasicScheduler, so the Guide node's
v2v_denoise output still drives the restyle amount.

The base graph is read from the user's own workflows folder and edited in
place, so every node keeps the widget order H3 Studio expects. What changes:

  * BasicGuider, SamplerCustomAdvanced and RandomNoise go away: SelfLift runs
    the schedule itself.
  * The TST node goes between the LoRA and the sampler. It is optional --
    delete it and the graph is the same without the correction. It cannot run
    together with the sampler's high-resolution tiling, and the sampler says so
    out loud if both are on.
  * The decoders hang off the save node, so the draft mode can hold the latent
    back and nothing downstream runs.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/make_h3studio_workflow.py

  --small     a cheap variant: a small canvas, a short clip, and a first frame
              at 0.65 strength, so the keyframe path is really walked
  --out PATH  write somewhere else
"""
import json
import os
import sys
import urllib.request
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = (r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
          r"\user\default\workflows\H3 Studio｜Basic Usage.json")
OUT = os.path.join(REPO, "workflows", "H3 Studio x SelfLift Draft Mode.json")

GUIDE = 159        # H3StudioImageToVideo
SCHEDULER = 153    # H3BasicScheduler
UNET = 127         # UNETLoader
VIDEO_VAE = 119
AUDIO_VAE = 120
SAMPLER_SELECT = 162
VAE_DECODE = 122
VAE_DECODE_AUDIO = 121
# BasicGuider, SamplerCustomAdvanced, RandomNoise, and the Turbo LoRA: that one
# comes from the ComfyUI-MiniMax-H3-Turbo pack, which is a separate install, and
# the base graph would not run without it. The model goes straight to the
# sampler instead.
DROP = (126, 125, 129, 141)

TST = 500
DRAFT = 501
SAVE = 502
ZERO = 503

NOTE = (
    "H3 Studio x SelfLift. The Guide node hands its positive conditioning and its "
    "target latent to the SelfLift draft sampler, and its v2v_denoise output still "
    "drives H3 Basic Scheduler's denoise socket -- that is what makes the restyle "
    "amount work here. The decoders hang off the save node, so the draft mode keeps "
    "the latent to itself and nothing downstream runs. The TST node is optional: "
    "delete it for the plain run. TST and the sampler's high-resolution tiling "
    "cannot run together; the sampler refuses that combination rather than "
    "silently skipping the correction. The Turbo LoRA of H3 Studio's own example "
    "is not in this graph: it comes from the separate ComfyUI-MiniMax-H3-Turbo "
    "pack, and the model goes straight to the sampler without it."
)


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


def node_of(workflow, node_id):
    for node in workflow["nodes"]:
        if node["id"] == node_id:
            return node
    raise KeyError(f"no node {node_id}")


def tst_node(model_link):
    return {
        "id": TST,
        "type": "SelfLiftH3TST",
        "pos": [-1660.0, 4620.0],
        "size": [340, 130],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [link_input("model", "MODEL", model_link)],
        "outputs": [output("MODEL", "MODEL", [])],
        "properties": {"Node name for S&R": "SelfLiftH3TST", "ver": "1.0"},
        "widgets_values": [0.2, True],
        "color": "#432",
        "bgcolor": "#653",
    }


def draft_node(model_link, positive_link, vae_link, latent_link, sampler_link, sigmas_link,
               audio_vae_link):
    inputs = [
        link_input("model", "MODEL", model_link),
        link_input("positive", "CONDITIONING", positive_link),
        link_input("negative", "CONDITIONING", None),
        link_input("vae", "VAE", vae_link),
        link_input("latent_image", "LATENT", latent_link),
        link_input("sampler", "SAMPLER", sampler_link),
        link_input("sigmas", "SIGMAS", sigmas_link),
        link_input("model_hires", "MODEL", None, shape=7),
        link_input("audio_vae", "VAE", audio_vae_link, shape=7),
    ]
    inputs += [
        widget("seed", "INT"),
        widget("cfg", "FLOAT"),
        widget("transition_step", "INT"),
        widget("lowres_scale", "FLOAT"),
        widget("rho", "FLOAT"),
        widget("w_min", "FLOAT"),
        widget("w_max", "FLOAT"),
        widget("upscaler_model", "COMBO"),
        widget("draft_mode", "BOOLEAN"),
        widget("preview_vae", "COMBO"),
        widget("fps", "FLOAT"),
        widget("highres_tiling", "BOOLEAN"),
        widget("upscaler_unload", "BOOLEAN"),
    ]
    return {
        "id": DRAFT,
        "type": "SelfLiftH3DraftSampler",
        "pos": [-1180.0, 4400.0],
        "size": [430, 300],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": inputs,
        "outputs": [
            output("latent", "LATENT", []),
            output("draft_video", "STRING", []),
            output("workflow_json", "STRING", []),
            output("workflow_api", "STRING", []),
        ],
        "properties": {"Node name for S&R": "SelfLiftH3DraftSampler", "ver": "1.0"},
        # seed, control_after_generate, cfg, transition_step, lowres_scale, rho,
        # w_min, w_max, upscaler_model, draft_mode, preview_vae, fps, tiling, unload
        "widgets_values": [123456789, "randomize", 1.0, 4, 0.4, 0.0, 0.5, 1.0,
                           "minimax_h3_latent_upscaler_3d_fp16_V2.safetensors",
                           True, "taeh3.safetensors", 24.0, False, True],
        "color": "#243",
        "bgcolor": "#356",
    }


def save_node(latent_link, video_link, json_link, api_link):
    return {
        "id": SAVE,
        "type": "SelfLiftH3Save",
        "pos": [-640.0, 4400.0],
        "size": [420, 220],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [
            link_input("latent", "LATENT", latent_link),
            link_input("draft_video", "STRING", video_link),
            link_input("workflow_json", "STRING", json_link),
            link_input("workflow_api", "STRING", api_link),
            widget("output_folder", "STRING"),
            widget("file_name", "STRING"),
        ],
        "outputs": [output("latent", "LATENT", [])],
        "properties": {"Node name for S&R": "SelfLiftH3Save", "ver": "1.0"},
        # the three string inputs are widgets too, and they come first
        "widgets_values": ["", "", "", "selflift", "selflift"],
        "color": "#323",
        "bgcolor": "#535",
    }


def zero_node(cond_link):
    return {
        "id": ZERO,
        "type": "ConditioningZeroOut",
        "pos": [-1180.0, 4180.0],
        "size": [220, 50],
        "flags": {},
        "order": 0,
        "mode": 0,
        "inputs": [link_input("conditioning", "CONDITIONING", cond_link)],
        "outputs": [output("CONDITIONING", "CONDITIONING", [])],
        "properties": {"Node name for S&R": "ConditioningZeroOut", "ver": "1.0"},
        "widgets_values": [],
    }


SMALL = {"width": 512, "height": 288, "length": 5,
         "first_frame_strength": 0.65, "first_frame_file": "article-01.jpg"}

# H3 Studio's example names checkpoints that are not on every machine. These are
# the same models in other quantisations; the generator only swaps when the
# name the example uses is missing here.
MODEL_SWAP = {
    "minimax_h3_ref2va_pruned_int8_convrot.safetensors":
        ["minimax_h3_ref2va_pruned_w4a8_mixed.safetensors"],
    "minimax_h3_video_vae_fp16.safetensors":
        ["minimax_h3_video_vae_int8_convrot.safetensors"],
}


def available_models(server="http://127.0.0.1:8188"):
    """The checkpoints the running ComfyUI can load, per loader node."""
    try:
        with urllib.request.urlopen(server + "/object_info", timeout=30) as handle:
            info = json.load(handle)
    except Exception as exc:
        print(f"  (cannot ask {server} what it has: {exc})")
        return {}
    found = {}
    for node in ("UNETLoader", "VAELoader"):
        required = ((info.get(node) or {}).get("input") or {}).get("required") or {}
        for key, spec in required.items():
            if key.endswith("_name") and isinstance(spec[0], list):
                found[node] = spec[0]
    return found


def swap_models(workflow, available):
    """Point the loaders at checkpoints this machine actually has."""
    for node in workflow["nodes"]:
        if node["type"] not in available:
            continue
        values = node.get("widgets_values") or []
        if not values or not isinstance(values[0], str):
            continue
        current = values[0]
        if current in available[node["type"]]:
            continue
        for candidate in MODEL_SWAP.get(current, []):
            if candidate in available[node["type"]]:
                values[0] = candidate
                print(f"  {node['type']}: {current} -> {candidate}")
                break


def shrink_guide(workflow, values):
    """Set the Guide node's canvas.

    The node keeps its values in three places: the positional widget list, the
    named one, and its own `properties.h3_widget_values`. The frontend reads
    the last one, so all three have to agree.
    """
    for node in workflow["nodes"]:
        if node["type"] != "H3StudioImageToVideo":
            continue
        named = node.get("widgets_values_named") or {}
        for key, value in values.items():
            named[key] = value
        node["widgets_values_named"] = named
        properties = node.get("properties") or {}
        stored = properties.get("h3_widget_values")
        if isinstance(stored, dict):
            for key, value in values.items():
                stored[key] = value
        widgets = node.get("widgets_values") or []
        for index, widget in enumerate(widgets):
            if isinstance(widget, str) and widget in values:
                widgets[index] = values[widget]
        properties["h3_widget_values"] = stored
        node["properties"] = properties


def main():
    small = "--small" in sys.argv
    out = OUT
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]

    with open(SOURCE, "r", encoding="utf-8") as handle:
        workflow = json.load(handle)

    nodes = {node["id"]: node for node in workflow["nodes"]}

    # the sampler chain of H3 Studio goes away
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] not in DROP]
    kept_links = [link for link in workflow["links"] if link[1] not in DROP and link[3] not in DROP]
    gone = {link[0] for link in workflow["links"]} - {link[0] for link in kept_links}
    workflow["links"] = kept_links
    for node in workflow["nodes"]:
        for item in node.get("outputs") or []:
            if item.get("links"):
                item["links"] = [link for link in item["links"] if link not in gone]
        for item in node.get("inputs") or []:
            if item.get("link") in gone:
                item["link"] = None
    nodes = {node["id"]: node for node in workflow["nodes"]}

    # the guide's conditioning and latent used to feed the guider and the sampler
    guide_positive = nodes[GUIDE]["outputs"][0]
    guide_positive["links"] = []
    guide_latent = nodes[GUIDE]["outputs"][1]
    guide_latent["links"] = []

    next_link = 400
    links = []

    def connect(origin_id, origin_slot, target_id, target_slot, kind):
        nonlocal next_link
        link_id = next_link
        next_link += 1
        links.append([link_id, origin_id, origin_slot, target_id, target_slot, kind])
        origin = nodes[origin_id]["outputs"][origin_slot]
        origin.setdefault("links", []).append(link_id)
        nodes[target_id]["inputs"][target_slot]["link"] = link_id
        return link_id

    # build the three new nodes with their inputs in place, then wire them
    tst = tst_node(None)
    draft = draft_node(None, None, None, None, None, None, None)
    save = save_node(None, None, None, None)
    zero = zero_node(None)
    workflow["nodes"] += [tst, draft, save, zero]
    nodes = {node["id"]: node for node in workflow["nodes"]}

    connect(UNET, 0, TST, 0, "MODEL")
    connect(UNET, 0, SCHEDULER, 0, "MODEL")
    connect(TST, 0, DRAFT, 0, "MODEL")
    connect(GUIDE, 0, DRAFT, 1, "CONDITIONING")
    connect(GUIDE, 0, ZERO, 0, "CONDITIONING")
    connect(ZERO, 0, DRAFT, 2, "CONDITIONING")
    connect(VIDEO_VAE, 0, DRAFT, 3, "VAE")
    connect(GUIDE, 1, DRAFT, 4, "LATENT")
    connect(SAMPLER_SELECT, 0, DRAFT, 5, "SAMPLER")
    connect(SCHEDULER, 0, DRAFT, 6, "SIGMAS")
    connect(AUDIO_VAE, 0, DRAFT, 8, "VAE")
    connect(DRAFT, 0, SAVE, 0, "LATENT")
    connect(DRAFT, 1, SAVE, 1, "STRING")
    connect(DRAFT, 2, SAVE, 2, "STRING")
    connect(DRAFT, 3, SAVE, 3, "STRING")
    connect(SAVE, 0, VAE_DECODE, 0, "LATENT")
    connect(SAVE, 0, VAE_DECODE_AUDIO, 0, "LATENT")

    workflow["links"] += links

    # SelfLift runs Euler; the base graph picked a multistep sampler
    select = nodes[SAMPLER_SELECT]
    select["widgets_values"] = ["euler"]

    # the note says what the graph is
    for node in workflow["nodes"]:
        if node["type"] == "Note":
            node["widgets_values"] = [NOTE]
            node["size"] = [520, 150]

    workflow["id"] = str(uuid.uuid4())
    workflow["revision"] = 0
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])

    if small:
        shrink_guide(workflow, SMALL)
    swap_models(workflow, available_models())

    with open(out, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, ensure_ascii=False, indent=2)
    print("wrote", out)
    print(f"  {len(workflow['nodes'])} nodes, {len(workflow['links'])} links")


if __name__ == "__main__":
    main()
