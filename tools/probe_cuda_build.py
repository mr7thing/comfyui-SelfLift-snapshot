"""Did the CUDA 13 build actually land, and did the quantized kernels wake up?

Prints the torch build, the architectures it carries, whether the GPU still
runs on it, and what comfy-kitchen says about its backends -- the CUDA one is
the one ComfyUI disables below CUDA 13.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/probe_cuda_build.py
"""
import sys
import time

CORE = r"C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI"
sys.path.insert(0, CORE)

import torch  # noqa: E402


def main():
    print(f"  torch {torch.__version__}  cuda {torch.version.cuda}")
    print(f"  architectures: {torch.cuda.get_arch_list()}")
    print(f"  device: {torch.cuda.get_device_name(0)}  capability {torch.cuda.get_device_capability(0)}")
    if torch.cuda.is_available():
        size = 4096
        left = torch.randn(size, size, device="cuda", dtype=torch.float16)
        right = torch.randn(size, size, device="cuda", dtype=torch.float16)
        torch.cuda.synchronize()
        started = time.perf_counter()
        for _ in range(20):
            left @ right
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        print(f"  20 x 4096^3 fp16 matmul: {elapsed:.3f}s -> {20 * 2 * size ** 3 / elapsed / 2**30:.1f} TFLOP/s")
    else:
        print("  CUDA is NOT available")

    print("\n  comfy-kitchen backends:")
    try:
        import comfy.quant_ops  # noqa: F401
        import comfy_kitchen as ck
        for name in ("cuda", "triton", "eager"):
            backend = ck.registry.get_backend(name) if hasattr(ck.registry, "get_backend") else None
            print(f"    {name}: {backend}")
    except Exception as error:  # noqa: BLE001
        print("    could not read the registry:", type(error).__name__, error)


if __name__ == "__main__":
    main()
