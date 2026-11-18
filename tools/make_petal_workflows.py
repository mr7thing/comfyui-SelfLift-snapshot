#!/usr/bin/env python
"""Build the two petal demo workflows out of the petal workflow the user runs.

    Petal Room - SelfLift Draft Mode.json
        The user's own petal workflow with the content rewritten: one reference
        picture instead of two, and a prompt that asks for petals to rain down
        in the flower room and gather into the person of <Picture 1>. The
        native MiniMax guide stays where it was.

    Petal Room - H3 Studio x SelfLift.json
        The same graph with H3 Studio brought in: the guide becomes
        H3StudioImageToVideo, the schedule comes from H3BasicScheduler with its
        denoise wired to the guide's v2v_denoise, and the picture is wired as a
        weakened first frame -- so the keyframe path (and the noise restore
        that goes with it) is really walked.

The canvas follows the picture: the source is 832x1248, so the graph asks the
ResolutionSelector for 2:3 and lands on 544x800. On the old 16:9 canvas the
first frame would have been cover-cropped to a letterbox band.

Run with the ComfyUI interpreter:

    ComfyUI/.venv/Scripts/python.exe tools/make_petal_workflows.py

    --install   also copy both into the ComfyUI user workflows folder
"""
import json
import os
import shutil
import sys
import uuid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(REPO, "workflows", "H3 SelfLift Draft Mode.json")
TEMPLATE = os.path.join(REPO, "workflows", "H3 Studio x SelfLift Draft Mode.json")
OUT_A = os.path.join(REPO, "workflows", "Petal Room - SelfLift Draft Mode.json")
OUT_B = os.path.join(REPO, "workflows", "Petal Room - H3 Studio x SelfLift.json")
USER_WORKFLOWS = (r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui"
                  r"\ComfyUI\user\default\workflows")

# node ids in the source workflow (the user's own petal workflow)
GUIDE = 136        # MiniMaxH3ReferenceToVideo
SCHEDULER = 223    # BasicScheduler
EXTEND = 224       # ExtendIntermediateSigmas
CLIP = 128
VAE_VIDEO = 215
VAE_AUDIO = 184
PICTURE = 332      # Image Load -> ref_image_0
PICTURE2 = 333     # Image Load -> ref_image_1  (dropped: one picture only)
PROMPT = 216
RESOLUTION = 115
LENGTH = 131
ZERO = 173         # ConditioningZeroOut
MODEL_OUT = 153    # ModelPreviewOverrideKJ
DRAFT = 171
SAVE = 400
DANGLING = (222, 227, 228)   # nothing connected to them

# node ids in the H3 Studio template (the graph that was already run end to end)
H3_GUIDE = 159     # H3StudioImageToVideo
H3_SCHEDULER = 153  # H3BasicScheduler
H3_NOTE = 161      # Note

# ids for the nodes workflow B adds
NEW_GUIDE = 600
NEW_SCHEDULER = 601
NEW_NOTE = 602

ASPECT = "2:3 (Portrait Photo)"
MEGAPIXELS = 0.4
MULTIPLE = 32

# the picture: a flower room whose floor is carpeted in pink petals, with a
# woman made of petals lying on it
PICTURE_FILE = "Qwen_image_2.1_00038.png"

PROMPT_TEXT = """subject_definitions:
<Subject 1> is the woman in <Picture 1>: she is a human shape built entirely from densely layered pink cherry-blossom petals, long petal hair fanning out across the floor, her face still buried in the layers of petals, her eyes closed.
<Subject 2> is the sunlit room in <Picture 1>: white walls, a tall multi-paned window on the left pouring in soft daylight, the floor completely carpeted in pink petals, a few loose petals drifting down through the light.

summary:
[reference generation] The target video takes place inside <Subject 2>. Loose petals rain down through the window light and drift across the room, while petals lying near the floor lift, stream and coil through the air and gather onto <Subject 1>, tightening her outline until the shape reads unmistakably as a person, and the last of the petals come to rest on her and on the floor around her.

retention_analysis:
<Subject 1> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the petal-built body, the long petal hair fanning across the floor, the closed eyes and the pale pink-and-white palette of <Picture 1> are retained; her surface ripples and shifts as the petals settle onto it.
<Subject 2> (appears in [Shot 1], [Shot 2], [Shot 3]): fully_preserved - the white walls, the tall multi-paned window with soft daylight, and the floor carpeted in pink petals are retained throughout.
<Picture 1>: fully_preserved - the high-angle framing, the petal-formed woman lying among scattered petals near the window, and the petals drifting in the light anchor the look of the shot.

detailed_description:
The target video is in a dreamy, ethereal fantasy style with soft diffused daylight, a pale pink-and-white palette, and gentle slow-motion movement.
[Shot 1] The shot opens on the frame of <Picture 1>: inside <Subject 2>, the camera looks down at <Subject 1> lying among the scattered petals near the window, and a few loose petals drift down slowly through the light. The camera pushes in with small amplitude at slow speed. Around 00:01.200, the petals lying closest to her begin to tremble and lift.
[Shot 2] At 00:02.600, the camera cuts to a close side shot. The lifted petals swirl inward in long slow ribbons, stream onto her torso and limbs and tuck into the layers that cover her, and her outline tightens shoulder by shoulder; the petal hair shifts and fans further across the floor. The camera tilts up with small amplitude from the spreading petal hair to her shoulders.
[Shot 3] At 00:04.000, the camera cuts to a medium shot from a slightly lower angle. <Subject 1> lies complete on the petal floor, her surface rippling once and settling as a last small shower of petals drifts down and comes to rest on her shoulders and along the floor around her. She stays still, eyes closed, while the light shifts gently across the room and the camera holds on the final frame.

overall_soundscape: A soft, airy breeze drifts through the open window, carrying the delicate rustle of slowly falling petals. A faint whisper of petal-on-petal rustling builds as the petals lift and stream onto her body, and softens again when the last of them settle on the floor.

non_diegetic_music: A slow, ethereal arrangement of soft piano notes, airy string pads and delicate harp arpeggios plays at a gentle tempo, swelling gently as the petals gather, and resolving into a quiet, sustained chord when the figure is complete."""

NOTE_A = (
    "Petal Room - SelfLift draft mode. One reference picture "
    "(Qwen_image_2.1_00038.png, from the input folder): a flower room whose "
    "floor is carpeted in pink "
    "petals, with a woman made of petals lying on it. The prompt asks for the "
    "petals to rain down and to gather into the person of <Picture 1>.\n"
    "draft_mode is ON on the sampler: only the low-resolution prefix is sampled, "
    "the preview mp4 and the two workflow files are written, and the save node "
    "keeps the latent so the decoders downstream stay quiet. Turn draft_mode off "
    "for the full run -- the archived workflow always carries draft_mode off, so "
    "the file you drop back on the canvas makes the clip, not a draft of it."
)

NOTE_B = (
    "Petal Room - H3 Studio x SelfLift. The same graph and the same picture, but "
    "the guide is H3 Studio's H3StudioImageToVideo and the schedule comes from "
    "H3BasicScheduler, whose denoise is wired to the guide's v2v_denoise.\n"
    "The picture is wired to first_frame with first_frame_strength 0.65, so H3 "
    "Studio writes it as a weakened keyframe: the low-resolution prefix has to "
    "resize that keyframe onto its own grid, and SelfLift area-resamples it and "
    "puts the keyframe's noise back -- the log line says so.\n"
    "TST is not in this graph: the attention override slot already belongs to the "
    "BlockSparseAttention node in the model chain. The decoders hang off the save "
    "node, so draft_mode keeps the latent and nothing downstream runs."
)


# ---------------------------------------------------------------- graph helpers

def load(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def node_of(workflow, node_id):
    for node in workflow["nodes"]:
        if node["id"] == node_id:
            return node
    raise KeyError(f"no node {node_id}")


def drop_links(workflow, link_ids):
    gone = set(link_ids)
    workflow["links"] = [link for link in workflow["links"] if link[0] not in gone]
    for node in workflow["nodes"]:
        for item in node.get("inputs") or []:
            if item.get("link") in gone:
                item["link"] = None
        for item in node.get("outputs") or []:
            if item.get("links"):
                item["links"] = [link for link in item["links"] if link not in gone]


def drop_node(workflow, node_id):
    """Take a node and every wire that touches it out of the graph."""
    touched = [link[0] for link in workflow["links"]
               if link[1] == node_id or link[3] == node_id]
    drop_links(workflow, touched)
    workflow["nodes"] = [node for node in workflow["nodes"] if node["id"] != node_id]


def connect(workflow, link_id, origin_id, origin_slot, target_id, target_slot, kind):
    workflow["links"].append(
        [link_id, origin_id, origin_slot, target_id, target_slot, kind])
    origin = node_of(workflow, origin_id)["outputs"][origin_slot]
    origin.setdefault("links", []).append(link_id)
    node_of(workflow, target_id)["inputs"][target_slot]["link"] = link_id


def link_input(name, kind, link=None):
    return {"label": name, "localized_name": name, "name": name, "type": kind,
            "link": link}


def output(name, kind, links=None):
    return {"label": name, "localized_name": name, "name": name, "type": kind,
            "links": links}


def set_input(workflow, node_id, name, link_id):
    for item in node_of(workflow, node_id)["inputs"]:
        if item["name"] == name:
            item["link"] = link_id
            return
    raise KeyError(f"node {node_id} has no input {name}")


def set_widget(workflow, node_id, index, value):
    node = node_of(workflow, node_id)
    values = node.get("widgets_values") or []
    while len(values) <= index:
        values.append(None)
    values[index] = value
    node["widgets_values"] = values


def set_named_widget(workflow, node_id, key, value):
    node = node_of(workflow, node_id)
    named = node.get("widgets_values_named")
    if isinstance(named, dict):
        named[key] = value
        node["widgets_values_named"] = named
    properties = node.get("properties") or {}
    stored = properties.get("h3_widget_values")
    if isinstance(stored, dict) and key in stored:
        stored[key] = value
    properties["h3_widget_values"] = stored
    node["properties"] = properties


def resolution_of(aspect, megapixels, multiple):
    """What the ResolutionSelector hands out, so the widgets can be kept honest."""
    ratios = {
        "1:1 (Square)": (1, 1), "2:3 (Portrait Photo)": (2, 3),
        "3:2 (Photo)": (3, 2), "3:4 (Portrait Standard)": (3, 4),
        "4:3 (Standard)": (4, 3), "9:16 (Portrait Widescreen)": (9, 16),
        "16:9 (Widescreen)": (16, 9), "21:9 (Ultrawide)": (21, 9),
    }
    w_ratio, h_ratio = ratios[aspect]
    scale = ((megapixels * 1024 * 1024) / (w_ratio * h_ratio)) ** 0.5
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return width, height


def guide_values(node, values):
    """Set a value on an H3 Studio guide node.

    The guide keeps its values in three places: the positional widget list, the
    named one, and its own properties.h3_widget_values. The frontend reads the
    last one, so all three have to agree.
    """
    named = node.get("widgets_values_named") or {}
    order = list(named.keys())
    for key, value in values.items():
        named[key] = value
    node["widgets_values_named"] = named

    properties = node.get("properties") or {}
    stored = properties.get("h3_widget_values")
    if isinstance(stored, dict):
        for key, value in values.items():
            if key in stored:
                stored[key] = value
    properties["h3_widget_values"] = stored
    node["properties"] = properties

    widgets = node.get("widgets_values") or []
    for index, key in enumerate(order):
        if key in values and index < len(widgets):
            widgets[index] = values[key]
    node["widgets_values"] = widgets


def to_core_load_image(workflow, node_id, filename):
    """Make the picture node core's LoadImage.

    The petal workflow loads its pictures through WAS Suite's "Image Load",
    whose list names output files by their subfolder -- a file that sits in the
    root of the output folder is not in that list at all, and the prompt is
    refused before anything runs. Core's LoadImage takes the input folder by
    name, and the same picture is in there, byte for byte.
    """
    node = node_of(workflow, node_id)
    keep = (node.get("outputs") or [{}])[0].get("links") or []
    node["type"] = "LoadImage"
    node["inputs"] = [link_input("image", "COMBO"),
                      link_input("upload", "IMAGEUPLOAD")]
    node["outputs"] = [output("IMAGE", "IMAGE", keep),
                       output("MASK", "MASK", None)]
    node["widgets_values"] = [filename, "image"]
    node["properties"] = {"Node name for S&R": "LoadImage"}
    node["size"] = [315, 314]
    for item in node["inputs"]:
        item["link"] = None
    return node


def note_node(template, node_id, text, pos):
    node = json.loads(json.dumps(template))
    node["id"] = node_id
    node["pos"] = pos
    node["size"] = [520, 190]
    node["widgets_values"] = [text]
    named = node.get("widgets_values_named")
    if isinstance(named, dict) and "text" in named:
        named["text"] = text
        node["widgets_values_named"] = named
    return node


def finish(workflow):
    workflow["id"] = str(uuid.uuid4())
    workflow["revision"] = 0
    workflow["last_node_id"] = max(node["id"] for node in workflow["nodes"])
    workflow["last_link_id"] = max(link[0] for link in workflow["links"])


# ------------------------------------------------------------------- workflow A

def build_a():
    """The petal workflow with the content rewritten and one picture."""
    workflow = load(SOURCE)

    for node_id in DANGLING:
        drop_node(workflow, node_id)
    # the second picture goes: one reference picture is enough
    drop_node(workflow, PICTURE2)
    # the picture comes from the input folder, through core's loader
    to_core_load_image(workflow, PICTURE, PICTURE_FILE)

    width, height = resolution_of(ASPECT, MEGAPIXELS, MULTIPLE)

    # the canvas follows the picture
    resolution = node_of(workflow, RESOLUTION)
    resolution["widgets_values"] = [ASPECT, MEGAPIXELS, MULTIPLE]
    named = resolution.get("widgets_values_named")
    if isinstance(named, dict):
        named["aspect_ratio"] = ASPECT
        named["megapixels"] = MEGAPIXELS
        named["multiple"] = MULTIPLE
        resolution["widgets_values_named"] = named

    # the guide's own width/height widgets are stale once a link feeds them
    set_widget(workflow, GUIDE, 1, width)
    set_widget(workflow, GUIDE, 2, height)
    set_named_widget(workflow, GUIDE, "width", width)
    set_named_widget(workflow, GUIDE, "height", height)

    # the new prompt
    set_widget(workflow, PROMPT, 0, PROMPT_TEXT)
    set_named_widget(workflow, PROMPT, "value", PROMPT_TEXT)

    template = load(TEMPLATE)
    workflow["nodes"].append(
        note_node(node_of(template, H3_NOTE), NEW_NOTE, NOTE_A,
                  node_of(workflow, PROMPT)["pos"]))

    finish(workflow)
    return workflow


# ------------------------------------------------------------------- workflow B

def build_b():
    """The same graph, with H3 Studio's guide and scheduler in it."""
    workflow = build_a()
    template = load(TEMPLATE)

    guide_pos = node_of(workflow, GUIDE)["pos"]
    scheduler_pos = node_of(workflow, SCHEDULER)["pos"]

    # the native guide and the core scheduler go; their slots are taken over
    drop_node(workflow, GUIDE)
    drop_node(workflow, SCHEDULER)

    width, height = resolution_of(ASPECT, MEGAPIXELS, MULTIPLE)

    # --- the H3 Studio guide, copied from the graph that already ran
    guide = json.loads(json.dumps(node_of(template, H3_GUIDE)))
    guide["id"] = NEW_GUIDE
    guide["pos"] = guide_pos
    guide["inputs"] = [
        link_input("clip", "CLIP"),
        link_input("vae", "VAE"),
        link_input("first_frame", "IMAGE"),
        link_input("last_frame", "IMAGE"),
        link_input("ref_videos.ref_video_0", "IMAGE"),
        link_input("ref_video_audios.ref_video_audio_0", "AUDIO"),
        link_input("audio_vae", "VAE"),
        link_input("ref_images.ref_image_0", "IMAGE"),
        link_input("ref_masks.ref_mask_0", "MASK"),
        link_input("ref_audios.ref_audio_0", "AUDIO"),
        link_input("middle_frames.middle_frame_0", "IMAGE"),
        link_input("v2v_images", "IMAGE"),
        link_input("v2v_audio", "AUDIO"),
        link_input("prompt", "STRING"),
        link_input("width", "INT"),
        link_input("height", "INT"),
        link_input("length", "INT"),
    ]
    guide["outputs"] = [
        output("positive", "CONDITIONING", []),
        output("LATENT", "LATENT", []),
        output("v2v_denoise", "FLOAT", []),
    ]
    guide_values(guide, {
        "prompt": PROMPT_TEXT,
        "width": width,
        "height": height,
        "length": 124,
        "first_frame_strength": 0.65,
        "last_frame_strength": 1.0,
        "first_frame_file": "",
        "last_frame_file": "",
        "ref_image_files": "",
        "ref_image_size": "max",
        "v2v_denoise": 1.0,
    })
    workflow["nodes"].append(guide)

    # --- H3 Studio's scheduler: core's BasicScheduler with denoise on a socket
    scheduler = json.loads(json.dumps(node_of(template, H3_SCHEDULER)))
    scheduler["id"] = NEW_SCHEDULER
    scheduler["pos"] = scheduler_pos
    scheduler["inputs"] = [link_input("model", "MODEL"),
                           link_input("denoise", "FLOAT")]
    scheduler["outputs"] = [output("SIGMAS", "SIGMAS", [])]
    scheduler["widgets_values"] = ["simple", 8, "slice (core)"]
    named = scheduler.get("widgets_values_named")
    if isinstance(named, dict):
        named.update({"scheduler": "simple", "steps": "8",
                      "denoise_mode": "slice (core)"})
        scheduler["widgets_values_named"] = named
    workflow["nodes"].append(scheduler)

    # --- the wires
    connect(workflow, 600, CLIP, 0, NEW_GUIDE, 0, "CLIP")
    connect(workflow, 601, VAE_VIDEO, 0, NEW_GUIDE, 1, "VAE")
    connect(workflow, 602, PICTURE, 0, NEW_GUIDE, 2, "IMAGE")
    connect(workflow, 603, VAE_AUDIO, 0, NEW_GUIDE, 6, "VAE")
    connect(workflow, 604, PROMPT, 0, NEW_GUIDE, 13, "STRING")
    connect(workflow, 605, RESOLUTION, 0, NEW_GUIDE, 14, "INT")
    connect(workflow, 606, RESOLUTION, 1, NEW_GUIDE, 15, "INT")
    connect(workflow, 607, LENGTH, 1, NEW_GUIDE, 16, "INT")

    connect(workflow, 608, NEW_GUIDE, 0, DRAFT, 1, "CONDITIONING")
    connect(workflow, 609, NEW_GUIDE, 0, ZERO, 0, "CONDITIONING")
    connect(workflow, 610, NEW_GUIDE, 1, DRAFT, 4, "LATENT")
    connect(workflow, 611, NEW_GUIDE, 2, NEW_SCHEDULER, 1, "FLOAT")
    connect(workflow, 612, NEW_SCHEDULER, 0, EXTEND, 0, "SIGMAS")
    connect(workflow, 613, MODEL_OUT, 0, NEW_SCHEDULER, 0, "MODEL")

    # the note says what changed
    for node in workflow["nodes"]:
        if node["id"] == NEW_NOTE:
            node["widgets_values"] = [NOTE_B]
            named = node.get("widgets_values_named")
            if isinstance(named, dict) and "text" in named:
                named["text"] = NOTE_B
                node["widgets_values_named"] = named

    finish(workflow)
    return workflow


def write(workflow, path):
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(workflow, handle, ensure_ascii=False, indent=2)
    print(f"wrote {path}")
    print(f"  {len(workflow['nodes'])} nodes, {len(workflow['links'])} links")


def main():
    install = "--install" in sys.argv

    a = build_a()
    write(a, OUT_A)

    b = build_b()
    write(b, OUT_B)

    if install:
        for path in (OUT_A, OUT_B):
            target = os.path.join(USER_WORKFLOWS, os.path.basename(path))
            shutil.copyfile(path, target)
            print("installed", target)


if __name__ == "__main__":
    main()
