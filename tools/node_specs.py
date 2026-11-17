"""Print the current input definitions of the nodes the H3 workflows use.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/node_specs.py [ClassType ...]
"""
import json
import sys
import urllib.request

SERVER = "http://127.0.0.1:8188"

DEFAULT = [
    "UNETLoader", "LoraLoaderBypassModelOnly", "LoraLoaderModelOnly", "BlockSparseAttention",
    "ModelAttentionBackend", "ModelPreviewOverrideKJ", "CLIPLoader", "VAELoader", "Image Load",
    "PrimitiveStringMultiline", "ResolutionSelector", "ComfyMathExpression",
    "ExtendIntermediateSigmas", "BasicScheduler", "KSamplerSelect", "MiniMaxH3ReferenceToVideo",
    "CreateVideo", "SaveVideo", "VAEDecode", "VAEDecodeAudio", "SelfLiftH3Draft", "SelfLiftH3Resume",
]


def main():
    names = sys.argv[1:] or DEFAULT
    with urllib.request.urlopen(f"{SERVER}/object_info", timeout=60) as handle:
        info = json.loads(handle.read().decode("utf-8"))
    for name in names:
        node = info.get(name)
        print(f"=== {name} ===")
        if node is None:
            print("  (not registered)")
            continue
        for section in ("required", "optional"):
            for input_name, spec in (node.get("input", {}).get(section, {}) or {}).items():
                kind = "COMBO*" if isinstance(spec[0], list) else spec[0]
                extras = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
                default = extras.get("default", "")
                options = extras.get("options", "")
                if isinstance(options, list) and len(options) > 6:
                    options = options[:6] + ["..."]
                print(f"  [{section}] {input_name} : {kind} default={default!r} options={options!r}")


if __name__ == "__main__":
    main()
