"""How fast can this machine feed the GPU?

The H3 workflow runs on a 16 GB card with an 11.2 GB denoiser plus a 15 GB text
encoder and a 2.7 GB video VAE, so ComfyUI's DynamicVRAM keeps the denoiser
staged in host memory. If every forward pass has to pull those weights across
the bus, that cost is paid per sampling step and does not care how many tokens
the step has.

Run with the ComfyUI interpreter:
  ComfyUI/.venv/Scripts/python.exe tools/probe_host_to_device.py
"""
import time

import torch


def timed(label, function, repeats=2):
    function()
    best = None
    for _ in range(repeats):
        started = time.perf_counter()
        function()
        elapsed = time.perf_counter() - started
        best = elapsed if best is None else min(best, elapsed)
    print(f"  {label}: {best:.2f}s")
    return best


def main():
    device = torch.device("cuda")
    size = 11 * 1024 ** 3 // 4          # 11 GB of fp32, the size of the denoiser
    count = size // 4

    pageable = torch.empty(count, dtype=torch.float32)
    pinned = torch.empty(count, dtype=torch.float32).pin_memory()

    print(f"  moving {size / 2**20:.0f} MiB")

    for label, source in (("pageable host memory", pageable), ("pinned host memory", pinned)):
        target = torch.empty(count, dtype=torch.float32, device=device)
        elapsed = timed(label, lambda source=source, target=target: target.copy_(source))
        print(f"    -> {size / 2**30 / elapsed:.2f} GiB/s")
        del target
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
