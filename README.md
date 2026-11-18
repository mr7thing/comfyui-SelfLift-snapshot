# comfyui-SelfLift

[中文说明](README_CN.md)

Progressive-resolution sampling for ComfyUI: run the early denoising steps at low resolution, lift the result to full resolution, and finish there — faster generation with no training. Based on the [SelfLift paper](https://arxiv.org/abs/2609.02036) (SelfLift-zero) for rectified-flow image models, plus an experimental MiniMax H3 audio-video adaptation that the paper does not validate. Also included: an experimental H3 port of [TST](https://arxiv.org/abs/2609.08505) temporal-attention correction.

All nodes are in the `selflift` category. The samplers use `sampler`/`sigmas` inputs like `SamplerCustom`; connect the standard `euler` sampler from `KSamplerSelect` and the model's normal scheduler (other samplers are rejected).

## What this branch changes

Three things, in three separate places. `SelfLiftH3Sampler` and `SelfLiftImageSampler` are **byte-identical to the previous revision** — same inputs, same outputs, same numbers. Everything new lives on new nodes, or behind a switch that is off by default.

### 1. A draft mode, on its own nodes

`SelfLiftH3DraftSampler` gives the H3 sampler a `draft_mode` switch without touching the sampler. With it on, the run is a **draft**: only the low-resolution prefix is sampled — no lift, no high-resolution steps — and the latent does not travel on. The prefix is drawn with the fast decoder (`taeh3`), so the draft carries the soundscape too when `audio_vae` is connected. The node hands back the latent plus the three files the run produced: the preview mp4, the canvas workflow and the API prompt.

`SelfLiftH3Save` files those three files away and decides what happens to the latent. A draft run: the files move into the folder and under the name it was given, and the latent is kept, so the decoders downstream stay quiet. A full run: the latent passes straight through and nothing is written.

The switch lives on the sampler and the save node reads it from there, so the two cannot disagree about what the run was — one switch, not two. The operational details are in *Draft mode* below.

### 2. What had to be adapted inside the sampler

Three shared helpers grew one switch, `h3_studio_coop`, **defaulting to off**:

| Helper | What the switch guards |
| --- | --- |
| `_setup_run` | the tiling/TST conflict check and the extra log lines, both inside `if h3_studio_coop:` |
| `progressive_sample` | forwards the switch, nothing else |
| `_resize_keyframes` | the keyframe resizing was refactored so the 5D and 4D branches share one interpolation call; with the switch off the result is bit-identical to before |

The rest of the sampling chain — `_run_low_stage`, `_run_transition`, `_run_high_stage`, `_streams`, `_euler_step`, `_mask_blend_fn` and the validators — is untouched.

Two smaller adaptations, both on the draft path only:

- The preview is written into a staging folder and **moved** into place by the save node, so a half-written file is never filed.
- The save node stops the downstream decoders with ComfyUI's `ExecutionBlocker` when the run is a draft. Without it the decoders would run on the preview latent and quietly produce a second, wrong video.

### 3. What it does for H3 Studio

[ComfyUI-H3Studio](https://github.com/mr7thing/ComfyUI-H3Studio) decides *what the model gets*; SelfLift decides *how it is sampled*. The two share exactly one thing: the conditioning dictionary. The draft sampler reads two keys out of it, both written by H3 Studio's guide node:

| Key | What it is | What SelfLift does with it |
| --- | --- | --- |
| `minimax_keyframes` | the keyframe list | the low-resolution prefix has to resize the keyframes onto its own grid |
| `minimax_keyframes[i]["noise_aug"]` | `= strength × 0.999` | how much noise the keyframe carries, and therefore how much has to be put back |

**Keyframe noise is put back.** H3 Studio weakens a keyframe by blending noise into it (`z' = a·z + (1-a)·noise`) and keeps `a` in the condition as `noise_aug`. Averaging that keyframe down to the low-resolution grid averages its noise away with it — and bilinear, which is what the plain sampler uses, aliases the noise instead of averaging it, so the noise survives as a pattern rather than as noise. The draft sampler resamples by **area** (the correct filter for downscaling) and adds the missing noise back in, scaled by `1-noise_aug`, from a seed derived from the run's own seed, so the same run lands on the same noise. Measured offline on a keyframe at strength 0.65 (`noise_aug = 0.6493`, 35.1% noise): the per-axis standard deviation of the noise went from 0.2289 — *above* the 0.1200 a true average would give, which is what aliasing looks like — to 0.3479 against a target of 0.3507, 0.8% off. A keyframe with no `noise_aug` gets no noise added: its strength is not knowable, and guessing would damage it.

**`highres_tiling` and TST are refused together.** The tiling wrapper sits inside TST's, so tiled attention never reaches the correction and TST would be skipped without a trace. The draft sampler raises instead of leaving it to a console warning.

**A start below pure noise is called out.** When the schedule starts under sigma 1.0 the run begins from the source latent (v2v / img2img semantics), and the log says so.

**Multi-frame keyframe blocks are called out.** Motion-context pins get downscaled together with everything else in the low-resolution prefix, so the join can be softer here than in a native H3 Studio run. The keyframe grid rule leaves no other option.

The wiring table is in *Using it with H3 Studio* below; the contract on both sides is in [`COLLABORATION.md`](COLLABORATION.md).

## SelfLift Progressive Sampler (Image)

The paper's SelfLift-zero for rectified-flow image backbones. Connect 4D image latents (e.g. *Empty Latent Image*) and the model's own VAE.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `transition_step` | 6 | Steps run at low resolution. Paper: 6 of 8 for Z-Image-Turbo, 3 of 4 for FLUX.2-Klein |
| `lowres_scale` | 0.5 | Spatial scale of the low-res prefix |
| `rho` | 0.3 | Fraction of high-risk locations corrected toward the pixel-VAE anchor. FLUX.2-Klein: 0.4. 0 disables the anchor |
| `w_min` / `w_max` | 0.5 / 1.0 | Correction strength range inside the selected locations |
| `latent_upsample` | nearest | Direct-lift interpolation; `bilinear` is an option |
| `model_hires` (optional) | — | Separate model for the high-resolution stage (e.g. a different checkpoint or LoRA stack). Must share the same architecture and latent format; the low-res prefix always runs on `model` |

The transition adds no denoiser evaluations: an N-step schedule stays exactly N NFEs, plus one VAE decode → upscale → re-encode round trip unless `rho=0`.

## SelfLift Progressive Sampler (MiniMax H3)

Experimental H3 audio-video adaptation (not paper-validated). Two modes:

- **External upscaler (default)**: an installed H3 checkpoint with `rho=0` — a learned latent-only lift. Practical default, but not SelfLift-zero.
- **SelfLift-zero**: `upscaler_model=none` with `rho>0`. Suggested H3 starting point: `rho=0.6`, `w_min=w_max=1` (see *Diagnosing H3*).

Extra inputs: `upscaler_model`, `upscaler_unload` (default on: unloads the upscaler from VRAM right after the lift, before the high-resolution stage) and `highres_tiling` (experimental: splits the high-res phase into 1–8 spatial tiles to save VRAM; only the first tile's audio is kept, there is no cross-tile attention, ControlNet is unsupported, and quality/speed may change).

### Optional H3 upscaler

Download from [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) into `ComfyUI/models/latent_upscale_models/` and restart. The node selects the first filename containing `h3`, otherwise `none` (nearest-neighbor lift).

## Draft mode (MiniMax H3)

Two nodes, and the plain sampler stays out of it: `SelfLiftH3Sampler` is untouched, the draft mode lives in its own nodes.

| Node | What it does |
| --- | --- |
| `SelfLiftH3DraftSampler` | The sampler with a draft mode. `draft_mode` on: a draft -- only the low-resolution prefix is sampled, no lift and no high-resolution steps, and the latent does not travel on. The prefix is drawn with the fast decoder (`taeh3`) and the run is written down as two standard workflows with the draft mode switched off. `draft_mode` off: the full run, and the node behaves exactly like `SelfLiftH3Sampler`. It outputs the latent and the three files the run produced. |
| `SelfLiftH3Save` | Files those files away and decides what happens to the latent. A draft run: the files move into the folder and under the name it was given, and the latent is kept, so the decoders downstream stay quiet. A full run: the latent passes straight through and nothing is written. |

The switch lives on the sampler, and the save node reads it from there, so the two cannot disagree about what the run was — one switch, not two.

**The files**, and where they come from:

| Output of `SelfLiftH3DraftSampler` | What it is | Filed by `SelfLiftH3Save` as |
| --- | --- | --- |
| `draft_video` | The prefix drawn with the fast decoder from `models/vae_approx` (`preview_vae`, `taeh3` for H3), at the prefix's own resolution — nothing is scaled up, because a scaled-up preview only invents detail the prefix never had. Connect `audio_vae` and it carries the soundscape too. | `<folder>/<name>_<time>.mp4` |
| `workflow_json` | A standard ComfyUI workflow: the canvas graph when the run came from the canvas, the API prompt when it did not. Drop the file on the canvas and it opens. | `<folder>/<name>_<time>.workflow.json` |
| `workflow_api` | The API prompt of the same run, ready for `POST /prompt`. | `<folder>/<name>_<time>.api.json` |

The save node takes `output_folder` (a folder under the ComfyUI output directory, `selflift` by default) and `file_name` (the name the files are filed under, `selflift` by default; the time of the run is added to it). It moves the files rather than copying them: the sampler leaves them in a staging folder under `output/selflift/.staging/`, and this node is where they end up.

There is no other run record: the parameters are the nodes' own widget values in the workflow and the inputs in the API prompt, so they are already in both files.

**The workflow that comes out is the run that produces.** The sampler's `draft_mode` is off in it, so the file you drop back on the canvas makes the clip, not a draft of it. The graph is the one you ran: the decoders hang off the save node and the save node passes the latent on.

**The draft is as big as the prefix, not as big as the target.** Its pixel count is `target megapixels × lowres_scale`, decoded as it is. That is the whole sizing question: raising `lowres_scale` makes the draft bigger and leaves the finished clip alone, while raising the target makes both bigger — every high-resolution step and the lift's VAE round trip scale with the target's area.

**The same trajectory, not a second sample.** With the same seed, prompt, reference setup and `transition_step` as the full run, the draft is literally the first stretch of that run: what the full run does next is lift the prefix to the target resolution, re-noise it and finish the schedule. The composition, the subject and the movement in the draft are the ones the finished clip grows out of.

Two ways to use it, depending on `transition_step`:

- **The same value the full run uses** (full run 5, draft 5): the draft is the opening of the clip, still carrying a lot of noise — cheap, but hard to read.
- **The whole step count** (4 in a four-step schedule): the low-resolution grid runs the entire schedule and the result is readable, with the framing of the finished clip but not its pixels — two resolutions on one trajectory, not two samples.

For both at once: raise the full run's `transition_step` so more of the content is settled at low resolution, and give the draft the same value.

**The references line up.** The keep-region anchor and the noise mask are interpolated onto the prefix grid automatically; the reference images themselves follow `ref_image_size='match'`, which scales them to the current generation resolution. Measured against each other, the `'match'` and the `'max'` encodings give nearly the same picture, so this is not the sensitive part; what matters is that **the draft and the full run use the same reference setup**, and that the draft lands on the same grid as the prefix (same grid plus same seed means the same noise means the same trajectory).

`workflows/h3-draft.api.json` is the draft graph in the API format, ready for `POST /prompt`. `workflows/H3 SelfLift Draft Mode.json` is the same graph in the UI format, derived from the "H3 SelfLift Dual Sample" workflow with the save node put between the sampler and the decoders.

## H3 Temporal State Transport (TST)

A `MODEL` → `MODEL` patch node that improves the temporal stability of H3 videos at inference time — no training, no extra model, ~1–2% runtime overhead. Typical problems it counteracts: details that flicker or morph between frames (logos, on-screen text, textures), identity drift of people and outfits, and physically implausible motion. It works with any standard sampler node, not only the SelfLift sampler. It cannot create detail the model does not have.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `tau` | 0.2 | Correction strength. 0.2 is the recommended value; 0.5 is visibly too strong. 0 disables correction but keeps diagnostics |
| `log_diagnostics` | on | Logs one `[H3 TST]` line per model forward |

How it works, in one paragraph: instead of blindly strengthening cross-frame attention, TST measures a signed "Spectral Tension" of the frame-level attention operator and corrects only imbalanced heads — sharpening over-mixed ones, softening fragmented ones — with stronger correction in deeper layers and earlier steps. Theory and derivations are in the [paper](https://arxiv.org/abs/2609.08505).

The diagnostics line reports latent frames, grid rows per frame, tension before/after the within-call correction, mean gamma, the share of corrected heads, and TST's own runtime. On real H3 runs the correction direction matched the exact attention operator for 89–100% of heads. Set `SELFLIFT_TST_EXACT=1` before starting ComfyUI to log the exact-operator comparison on one layer per forward (debug only, slow).

Limitations: MiniMax H3 models only (other models are skipped silently); **not compatible with `highres_tiling`** (skipped with a console warning).

## Using it with H3 Studio

[ComfyUI-H3Studio](https://github.com/mr7thing/ComfyUI-H3Studio) decides *what the model gets* — keyframes, references, the timeline — and SelfLift decides *how it is sampled*. The two share exactly one thing: the conditioning dictionary. The full contract is in [`COLLABORATION.md`](COLLABORATION.md); this is the short version.

| From | To | Note |
| --- | --- | --- |
| `UNETLoader` | `SelfLiftH3TST` | TST only acts on a patched model |
| `H3StudioImageToVideo` → `positive` | `SelfLiftH3DraftSampler` → `positive` | |
| `H3StudioImageToVideo` → `positive` | `ConditioningZeroOut` → `negative` | |
| `H3StudioImageToVideo` → `LATENT` | `SelfLiftH3DraftSampler` → `latent_image` | sets the target size and duration |
| `H3StudioImageToVideo` → `v2v_denoise` | `H3BasicScheduler` → `denoise` | **required** — unwired, v2v silently becomes a full denoise |
| `H3BasicScheduler` → `SIGMAS` | `SelfLiftH3DraftSampler` → `sigmas` | not a KSampler denoise |
| `KSamplerSelect` (`euler`) | `SelfLiftH3DraftSampler` → `sampler` | SelfLift supports Euler only |
| video / audio `VAELoader` | `vae` / `audio_vae` | the preview mp4 carries the soundscape |

A ready-made graph ships in `workflows/H3 Studio x SelfLift Draft Mode.json`; `tools/make_h3studio_workflow.py` regenerates it and swaps in the model names actually present on this machine.

Two more graphs ship as worked examples, both cut from one real run: a flower room whose floor is carpeted in pink petals, with a woman made of petals lying on it.

- `workflows/Petal Room - SelfLift Draft Mode.json` — the same content on the stock MiniMax guide. One reference picture, and a prompt that asks for the petals to rain down and to gather into the person of that picture. The canvas is 544x800, the shape of the picture.
- `workflows/Petal Room - H3 Studio x SelfLift.json` — the same picture and the same prompt, with H3 Studio's guide in it. The picture is wired to `first_frame` at strength 0.65, so H3 Studio hands the sampler a *weakened keyframe*: the low-resolution prefix has to resize that keyframe onto its own grid, and the log line saying the noise was put back is the cooperation path doing its work.

`tools/make_petal_workflows.py` regenerates both.

What the draft sampler adds for H3 Studio (the plain `SelfLiftH3Sampler` is untouched, byte for byte):

- **Keyframe noise is put back.** H3 Studio weakens a keyframe by blending noise into it (`z' = a·z + (1-a)·noise`) and keeps `a` in the condition as `noise_aug`. Averaging that keyframe down to the low-resolution grid averages its noise away with it, so the draft sampler resamples by area rather than bilinear and adds the missing noise back in, scaled by `1-noise_aug`. Every keyframe is area-resampled — area is the correct filter for downscaling and bilinear aliases — but a keyframe with no `noise_aug` gets no noise added, because its strength is not knowable and guessing would damage it.
- **`highres_tiling` and TST are refused together.** The tiling wrapper sits inside TST's, so tiled attention never reaches the correction and TST would be skipped without a trace. The draft sampler raises instead.
- **A start below pure noise is called out.** When the schedule starts under sigma 1.0 the run begins from the source latent (v2v / img2img semantics), and the log says so.
- **Multi-frame keyframe blocks are called out.** Motion-context pins get downscaled together with everything else in the low-resolution prefix, so the join can be softer here than in a native H3 Studio run.

## Diagnosing H3

Same prompt and seed, `transition_step=6`, `lowres_scale=0.5`:

| Test | `upscaler_model` | `rho` | Weights | Meaning |
| --- | --- | ---: | --- | --- |
| Direct route | `none` | 0 | any | Nearest-neighbor latent lift only |
| Pixel route | `none` | 1 | `1 / 1` | Pure H3 VAE pixel anchor |
| Paper-like SelfLift-zero | `none` | 0.3 | `0.5 / 1` | Paper's image parameters |
| Strong H3 SelfLift-zero | `none` | 0.6 | `1 / 1` | Tested H3 starting point |
| External lifter | H3 checkpoint | 0 | any | Learned H3 lift |

A controlled single-seed H3 run found: clean native baseline and clean pure pixel anchor; widespread artifacts on the nearest route that the paper's image setting (`rho=0.3`) mostly left visible; `rho=0.6` with `w_min=w_max=1` removed the main artifacts. Single-seed evidence only — H3's direct-lift error is broader than on the paper's image backbones, so start strong and reduce only if the result is overly smooth.

## Logging and environment variables

- `[SelfLift plan]` — resolved low/target latent shapes, NFE counts, transition sigmas, enabled lift routes
- `[SelfLift timing]` — wall-clock time per step and per stage (low-res, transition, high-res)
- `[SelfLift upscaler]` / `[SelfLift tiling memory]` — upscaler inference and tiling memory estimates (heuristics, not measured peaks)
- `[H3 TST]` — TST per-forward diagnostics
- `SELFLIFT_TIMING_SYNC=1` — CUDA-synchronized timing (slower; for diagnostics)
- `SELFLIFT_MEMORY_LOG=1` — host/device memory snapshots at stage boundaries
- `SELFLIFT_DEBUG=1` — dump transition intermediates as PNGs under `debug/` (slow, memory-hungry)
- `SELFLIFT_TST_EXACT=1` — TST exact-operator calibration probe (debug only)

## Notes and limitations

- Only standard Euler with `s_churn=0`.
- `noise_mask` is supported on both samplers with Set Latent Noise Mask semantics (1 = generate, 0 = keep the original content; use an initialized latent to have content worth keeping). Masks at any resolution are resized to the latent grid, and a time length of 1 is shared over all frames. The keep-region is pinned to the original latent at every step, and the artifact-aware correction is restricted to the generate region. `noise_mask` is not compatible with `highres_tiling`.
- Use the VAE belonging to the sampled model so the pixel anchor stays in the same latent space.
- H3's 768-pixel short edge becomes 384 px at `lowres_scale=0.5`, which may sit outside the backbone's training distribution — validate per model.
- SelfLift-rich (the distilled lifter + On-Policy Self Recovery) requires training and is not included.

## References and acknowledgements

- SelfLift paper: [SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036)
- TST paper and code: [Temporal State Transport in Video Generation](https://arxiv.org/abs/2609.08505), [lytang63/temporal-state-transport](https://github.com/lytang63/temporal-state-transport)
- Optional MiniMax H3 latent upscaler checkpoint and download: [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
- Original ComfyUI integration and inference implementation: [LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

Thanks to LBH-123-AI for publishing the MiniMax H3 latent upscaler weights and ComfyUI implementation. They made the optional learned H3 lifting path in this plugin possible. This external lifter remains separate from the SelfLift paper's SelfLift-rich model.

```
@article{wen2026selflift,
  title={SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition},
  author={Wen, Tingyan et al.},
  journal={arXiv:2609.02036},
  year={2026}
}
```
