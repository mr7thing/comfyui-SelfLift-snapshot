# comfyui-SelfLift（中文说明）

[English README](README.md)

ComfyUI 渐进分辨率采样：前期去噪步骤跑低分辨率，把结果提升到全分辨率并收尾——免训练加速生成。基于 [SelfLift 论文](https://arxiv.org/abs/2609.02036)（SelfLift-zero，面向 rectified-flow 图像模型），外加论文未验证的 MiniMax H3 音视频实验性适配，以及 [TST](https://arxiv.org/abs/2609.08505) 时间注意力校正的 H3 实验性移植。

全部节点位于 `selflift` 分类。采样器节点使用与 `SamplerCustom` 相同的 `sampler`/`sigmas` 接口；`KSamplerSelect` 必须选标准 `euler`，调度器沿用模型默认（其它采样器会被拒绝）。

## SelfLift Progressive Sampler (Image)

论文 SelfLift-zero，用于 rectified-flow 图像骨干。接入 4D 图像 latent（如 *Empty Latent Image*）和模型所属 VAE。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `transition_step` | 6 | 低分辨率阶段步数。论文：Z-Image-Turbo 8 步取 6，FLUX.2-Klein 4 步取 3 |
| `lowres_scale` | 0.5 | 前缀的空间缩放 |
| `rho` | 0.3 | 向像素 VAE 锚点修正的高风险位置比例。FLUX.2-Klein 取 0.4；0 关闭锚点 |
| `w_min` / `w_max` | 0.5 / 1.0 | 选中位置的修正强度范围 |
| `latent_upsample` | nearest | 直接提升插值；可选 `bilinear` |
| `model_hires`（可选） | — | 高分辨率阶段使用的另一个模型（如不同 checkpoint 或 LoRA 组合）。必须与 `model` 同架构、同 latent 格式；低分辨率前缀始终用 `model` |

过渡不增加去噪评估次数：N 步调度仍严格等于 N 次 NFE；`rho=0` 以外的情况多一次 VAE 解码 → 上采样 → 重编码往返。

## SelfLift Progressive Sampler (MiniMax H3)

H3 音视频实验性适配（论文未验证）。两种模式：

- **外部 upscaler（默认）**：安装 H3 checkpoint 且 `rho=0`——学习式纯 latent 提升。实用默认，但不是 SelfLift-zero。
- **SelfLift-zero**:`upscaler_model=none` 且 `rho>0`。H3 建议起点：`rho=0.6`、`w_min=w_max=1`（见"诊断 H3")。

额外输入：`upscaler_model`、`upscaler_unload`（默认开启：放大完立即把 upscaler 从显存卸载，再进入高分辨率阶段）与 `highres_tiling`（实验性：把高分辨率阶段切成 1–8 个空间块以省显存；只保留首块音频，块间无全局注意力，不支持 ControlNet，画质和速度可能变化）。

### 可选 H3 upscaler

从 [LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler) 下载，放入 `ComfyUI/models/latent_upscale_models/` 后重启。节点默认选第一个文件名含 `h3` 的模型，找不到则为 `none`(nearest 提升）。

## 草稿模式（MiniMax H3）

两个节点，原来的采样器一点没动：`SelfLiftH3Sampler` 保持原样，草稿模式全部在它自己的新节点里。

| 节点 | 作用 |
| --- | --- |
| `SelfLiftH3DraftSampler` | 带草稿模式的采样器。`draft_mode` 开：草稿——只采样低分辨率前缀，不做放大、不做高分辨率阶段，latent 也不往下游传。前缀用快速解码器（`taeh3`）画出来，这次运行写成两份标准工作流（草稿模式翻成关）。`draft_mode` 关：完整流程，这个节点和普通采样器没有任何区别。它输出 latent，以及这次运行产出的三个文件。 |
| `SelfLiftH3Save` | 把这些文件归档，并决定 latent 怎么办。草稿模式：文件挪到它指定的目录和名字下，latent 自己留着，后面的解码链不跑。生产模式：latent 直接直通，什么都不写。 |

开关放在采样节点上，保存节点从那里读它——这样两个节点不可能对"这次是草稿还是生产"产生分歧：一个开关，不是两个。

**三个文件从哪来到哪去**：

| `SelfLiftH3DraftSampler` 的输出 | 是什么 | `SelfLiftH3Save` 存成 |
| --- | --- | --- |
| `draft_video` | 用 `models/vae_approx` 里的快速解码器（H3 是 `taeh3`，由 `preview_vae` 指定）画出的前缀，保持前缀自己的分辨率——**不做任何放大**，放大的预览只会凭空造出前缀没有的细节。接上 `audio_vae` 还会带上声音。 | `<目录>/<名字>_<时间>.mp4` |
| `workflow_json` | **标准格式的 ComfyUI 工作流**：从界面触发时是画布上的图，从 API 触发时是 API 提示图——两种前端都认，把文件拖进画布就能打开。 | `<目录>/<名字>_<时间>.workflow.json` |
| `workflow_api` | 同一次运行的 API 提示图，可直接 `POST /prompt`。 | `<目录>/<名字>_<时间>.api.json` |

保存节点上填 `output_folder`（ComfyUI 输出目录下的子文件夹，默认 `selflift`）和 `file_name`（文件名前缀，默认 `selflift`，后面自动加运行时间）。文件是**移动**而不是复制：采样节点先把它们留在 `output/selflift/.staging/` 暂存区，保存节点负责把它们归位。

没有别的运行记录文件：参数本来就是工作流里各节点自己的 widget 值、API 图里各节点的 inputs，两份文件里都已经有了。

**存下来的工作流就是能出成片的那次运行。** 里面采样节点的 `draft_mode` 是关的，所以把文件拖回画布，跑出来的是成片，不是草稿。图还是你跑的那张：解码链挂在保存节点后面，保存节点把 latent 直通下去。

**草稿有多大，取决于前缀有多大。** 草稿的像素数是 `目标像素 × lowres_scale`，按原样解码。所以想要更清楚的草稿，应该调 `lowres_scale`（前缀变大，成片不变），而不是调目标分辨率（草稿和成片一起变大——成片的每个高分辨率步骤、以及提升时的 VAE 往返都按目标面积涨）。

**同一条轨迹，不是另一个样本。** 用与完整运行相同的 seed、提示词、参考图和 `transition_step` 跑草稿，得到的正是完整运行**一开始的那一段**：完整运行接下来做的，就是把它提升到目标分辨率、重新加噪、继续跑完。所以草稿里的构图、主体、动态走向就是成片的走向。

两种用法，取决于 `transition_step`：

- **与完整运行相同**（完整运行用 2，草稿也用 2）：草稿就是成片的开头，画面还带着不少噪声——省钱，但看不清。
- **等于总步数**（4 步日程就填 4）：低分辨率把整条调度跑完，画面清晰可读，构图与成片一致，像素不同——它和成片是同一条轨迹上的两个分辨率，而不是两个样本。

想两者兼得：把完整运行的 `transition_step` 调大，让内容在低分辨率上定得更完整，草稿沿用同一个值。

**参考复原会对齐。** keep-region 锚点和 noise mask 会自动插值到前缀网格；参考图本身用 `ref_image_size='match'` 缩到当前生成分辨率。实测 `'match'` 与 `'max'` 两种编码出来的参考，画面几乎一致，所以这一步不是关键；关键是**草稿和完整运行用同一套参考设置**，以及草稿落在与前缀相同的网格上（同一网格 + 同一 seed = 同一份噪声 = 同一条轨迹）。

`workflows/h3-draft.api.json` 是 API 格式的草稿图，可直接 `POST /prompt`。`workflows/H3 SelfLift Draft Mode.json` 是 UI 格式版本，由"H3 SelfLift Dual Sample"工作流派生，把保存节点插在采样节点和解码链之间。

## H3 Temporal State Transport (TST)

`MODEL` → `MODEL` 补丁节点，推理时改善 H3 视频的时间稳定性——免训练、免额外模型，运行开销约 1–2%。主要对付：帧间闪烁或变形的小细节（logo、画面文字、纹理）、人物与服装的身份漂移、不符合物理的运动。可配合任意标准采样节点，不限于 SelfLift 采样器。它不能凭空生成模型没有的细节。

| 参数 | 默认 | 含义 |
| --- | --- | --- |
| `tau` | 0.2 | 校正强度。0.2 为推荐值；0.5 已明显过强。0 关闭校正但保留诊断 |
| `log_diagnostics` | 开 | 每次模型前向输出一行 `[H3 TST]` 诊断 |

原理一句话：不盲目加强跨帧注意力，而是先测量帧级注意力算子的带符号"谱张力"，只校正失衡的头——过混合的锐化、碎片化的软化——并在更深层和更早去噪步施加更强校正。理论和推导见[论文](https://arxiv.org/abs/2609.08505)。

诊断行包含：latent 帧数、每帧网格行数、调用内校正前/后的张力、平均 γ、被校正头比例、TST 自身耗时。真实 H3 运行上，校正方向与精确注意力算子的逐头一致率为 89–100%。调试时可在启动 ComfyUI 前设 `SELFLIFT_TST_EXACT=1`，在每次前向的一层上输出精确算子对比（很慢，仅调试用）。

限制：仅支持 MiniMax H3（其它模型静默跳过）;**与 `highres_tiling` 不兼容**（跳过并给出控制台警告）。

## 和 H3 Studio 一起用

[ComfyUI-H3Studio](https://github.com/mr7thing/ComfyUI-H3Studio) 决定**给模型什么**（关键帧、参考、时间线），
SelfLift 决定**怎么采**。两边真正共享的只有一样东西：条件字典。
完整契约见 [`COLLABORATION.md`](COLLABORATION.md)，这里是速查版。

| 从 | 到 | 说明 |
| --- | --- | --- |
| `UNETLoader` | `SelfLiftH3TST` | TST 只在打过补丁的模型上生效 |
| `H3StudioImageToVideo` → `positive` | `SelfLiftH3DraftSampler` → `positive` | |
| `H3StudioImageToVideo` → `positive` | `ConditioningZeroOut` → `negative` | |
| `H3StudioImageToVideo` → `LATENT` | `SelfLiftH3DraftSampler` → `latent_image` | 决定目标尺寸和时长 |
| `H3StudioImageToVideo` → `v2v_denoise` | `H3BasicScheduler` → `denoise` | **必须接** —— 不接 v2v 会静默退化成满 denoise |
| `H3BasicScheduler` → `SIGMAS` | `SelfLiftH3DraftSampler` → `sigmas` | 不是 KSampler 的 denoise |
| `KSamplerSelect`（`euler`） | `SelfLiftH3DraftSampler` → `sampler` | SelfLift 只支持 Euler |
| 视频 / 音频 `VAELoader` | `vae` / `audio_vae` | 预览 mp4 会带上声音 |

现成的工作流在 `workflows/H3 Studio x SelfLift Draft Mode.json`；
`tools/make_h3studio_workflow.py` 可以重新生成它，并把模型名换成本机实际有的那些。

草稿采样器为 H3 Studio 额外做的几件事（原 `SelfLiftH3Sampler` 一个字节没动）：

- **关键帧的噪声被补回来。** H3 Studio 用 `z' = a·z + (1-a)·noise` 弱化关键帧，
  并把 `a` 以 `noise_aug` 的形式留在条件里。把这个关键帧平均到低分辨率网格上时，
  噪声也跟着被平均掉了 —— 所以草稿采样器改用面积采样（而不是双线性），
  再按 `1-noise_aug` 把缺的噪声补回去。**所有关键帧都走面积采样**（面积采样才是缩小时的
  正确滤波器，双线性会混叠）；但没有 `noise_aug` 的关键帧**一分噪声都不补** ——
  它的强度不可知，瞎补会改坏它。
- **TST 和 `highres_tiling` 一起用会直接报错。** tiling 的 wrapper 坐在 TST 里面，
  分块 attention 永远到不了校正，TST 会被静默跳过。所以草稿采样器选择报错而不是装作没事。
- **不是从纯噪声起步时会说明。** 调度表起点低于 sigma 1.0 时，这次运行是从源 latent
  部分去噪起步（v2v / 图生图语义），日志里会写清楚。
- **多帧关键帧块会说明。** motion context 钉住的帧和别的关键帧一起被降分辨率，
  所以这里的接缝可能不如原生 H3 Studio 流程干净。

## 诊断 H3

固定同一 prompt 和 seed,`transition_step=6`、`lowres_scale=0.5`：

| 测试 | `upscaler_model` | `rho` | 权重 | 含义 |
| --- | --- | ---: | --- | --- |
| 直接路径 | `none` | 0 | 任意 | 只用 nearest latent 提升 |
| 像素路径 | `none` | 1 | `1 / 1` | 纯 H3 VAE 像素锚点 |
| 论文式 SelfLift-zero | `none` | 0.3 | `0.5 / 1` | 论文图像参数 |
| H3 强修正 SelfLift-zero | `none` | 0.6 | `1 / 1` | H3 实测起点 |
| 外部提升器 | H3 checkpoint | 0 | 任意 | 学习式 H3 提升 |

单 seed 对照结果：原生基线和纯像素锚点干净；nearest 路径出现大范围伪影，论文图像参数（`rho=0.3`）仍保留大部分；`rho=0.6` 且 `w_min=w_max=1` 消除了主要伪影。仅为单 seed 证据——H3 的直接提升误差比论文图像骨干更广，建议从强修正开始，过度平滑再降低。

## 日志与环境变量

- `[SelfLift plan]` — 实际的低/目标 latent 形状、NFE 数、过渡 sigma、启用的提升路径
- `[SelfLift timing]` — 逐步与分阶段（低分辨率/过渡/高分辨率）墙钟耗时
- `[SelfLift upscaler]` / `[SelfLift tiling memory]` — upscaler 推理与分块的内存估算（启发式，非实测峰值）
- `[H3 TST]` — TST 逐前向诊断
- `SELFLIFT_TIMING_SYNC=1` — CUDA 同步计时（更慢，仅诊断用）
- `SELFLIFT_MEMORY_LOG=1` — 阶段边界的主机/显存快照
- `SELFLIFT_DEBUG=1` — 把过渡中间结果导出 PNG 到 `debug/`（慢且耗内存）
- `SELFLIFT_TST_EXACT=1` — TST 精确算子校准探针（仅调试用）

## 注意事项与限制

- 只接受 `s_churn=0` 的标准 Euler。
- 两个采样器都支持 `noise_mask`（Set Latent Noise Mask 语义：1 = 生成，0 = 保留原内容；需要有初始化 latent 才有可保留的内容）。任意分辨率的 mask 都会缩放到 latent 网格，时间长度为 1 时全帧共享。保留区域在每一步都被钉到原始 latent，artifact-aware 修正也只作用于生成区域。`noise_mask` 与 `highres_tiling` 不兼容。
- 必须使用采样模型所属的 VAE，保证像素锚点处于同一 latent 空间。
- H3 的 768 像素短边在 `lowres_scale=0.5` 时为 384 像素，可能超出骨干训练分布——请按模型自行验证。
- SelfLift-rich（蒸馏提升器 + On-Policy Self Recovery）需要训练，不在本插件内。

## 引用与致谢

- SelfLift 论文：[SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition](https://arxiv.org/abs/2609.02036)
- TST 论文与代码：[Temporal State Transport in Video Generation](https://arxiv.org/abs/2609.08505)，[lytang63/temporal-state-transport](https://github.com/lytang63/temporal-state-transport)
- 可选 MiniMax H3 latent upscaler 权重与下载：[LBH-123-AI/Minimax_h3_latent_Upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler)
- 原始 ComfyUI 集成与推理实现：[LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler](https://github.com/LBH-123-AI/Comfyui_Minimax_h3_latent_Upscaler)

感谢 LBH-123-AI 公开 MiniMax H3 latent upscaler 权重和 ComfyUI 实现，使本插件能够提供可选的 H3 学习式提升路径。该外部 lifter 与 SelfLift 论文中的 SelfLift-rich 模型仍是彼此独立的实现。

```
@article{wen2026selflift,
  title={SelfLift: Accelerating Few-Step Diffusion via Self-Recovering Resolution Transition},
  author={Wen, Tingyan et al.},
  journal={arXiv:2609.02036},
  year={2026}
}
```
