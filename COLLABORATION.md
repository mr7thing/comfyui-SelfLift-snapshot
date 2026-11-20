# SelfLift × H3 Studio 协作契约（SelfLift 侧）

> 这是 H3 Studio 那份 [`COLLABORATION.md`](https://github.com/mr7thing/ComfyUI-H3Studio/blob/master/COLLABORATION.md)
> 的**镜像视角**。同一套约束，从采样侧写一遍。两份文档冲突时，**以两边的实测为准**，
> 不要靠猜。
>
> 适用范围：本机 ComfyUI **0.45.0**，MiniMax H3（`comfy/ldm/minimax/model.py`）。

---

## 1. 分工

| | H3 Studio | SelfLift |
|---|---|---|
| 仓库 | `mr7thing/ComfyUI-H3Studio`（fork of `shootthesound/ComfyUI-H3Studio`） | `mr7thing/comfyui-SelfLift-snapshot`（fork of `facok/comfyui-SelfLift`） |
| 角色 | **条件侧**：关键帧 / 参考 / 时间线 / reel —— 决定「给模型什么」 | **采样侧**：两段分辨率渐进采样 + TST 注意力校正 —— 决定「怎么采」 |
| 改不动的部分 | 关键帧 latent 必须与生成分辨率同网格（§3.3） | `optimized_attention_override` 单槽归 TST（§5.1） |

**唯一的交汇面是条件字典**，唯一的硬约束是 §3.3。除此之外各自随便改，
改完跑对方的检查脚本。

---

## 2. SelfLift 从条件字典里读什么

SelfLift 不认识 H3 Studio 的节点，只认识条件字典里的键：

| 键 | 谁写 | SelfLift 怎么用 |
|---|---|---|
| `minimax_keyframes` | H3 Studio（`h3_studio.py:2020`）、核心 guide | `_resize_keyframes()`：低分辨率段要把关键帧缩到低分辨率网格上 |
| `minimax_keyframes[i]["latent"]` | 同上 | 5D `[B,C,T,h,w]`，按帧做空间缩放，**不跨时间插值** |
| `minimax_keyframes[i]["noise_aug"]` | H3 Studio（`strength < 1.0` 且 row-aug 生效时） | `= strength × 0.999`。噪声占比 `1 - noise_aug`，低分辨率段据此补噪 |
| `minimax_keyframes[i]["strength"]` | H3 Studio（fork `feat/keep-keyframe-strength` 起保留） | 「这个关键帧到底被削弱了没有」。**没有它就分不清「满强度」和「标签缺失」**（§6.4） |

条件字典的顶层键 (`minimax_refs` / `minimax_frame_count` / `minimax_prompt_text`)
SelfLift 一个都不读，也不改。参考块用自己的网格，不受 §3.3 约束，SelfLift 原样传下去。

---

## 3. SelfLift 侧守的边界

### 3.1 `optimized_attention_override` 是 TST 独占的

核心的 `wrap_attn`（`comfy/ldm/modules/attention.py:232`）认的唯一注意力改写入口，
**单槽**：谁后写谁覆盖。

TST 写这个槽（`h3_tst.py:223`）。SelfLift 内部**只有 TST 一个**注意力改写，
所以现在没有自冲突。但规矩留着：

> **SelfLift 以后要加第二个注意力层的改动，必须和 TST 串起来**（`override` 的第一个
> 参数就是下层的 `func`，可以链式套），不能各写各的。

TST 用 `model_options["wrappers"]["diffusion_model"]["selflift_h3_tst"]` 注册自己，
**调用时**读，所以和 H3 Studio 的类级 `_forward` 替换可以共存，先后装都行。

### 3.2 packed 段序：video 必须在最后

TST 假设「video 行是 packed 序列的最后一段」，按
`rows_per_frame = ((H+1)//2) × ((W+1)//2)` 定位视频行。

核心 0.45 的段序是 `[text, cond, ref_img, audio, video]`，video 确实最后
（已实测 `frames × rows_per_frame` 与布局的 video 段行数严格相等）。

**SelfLift 侧现在有运行期自保**（§6.2）：每次 forward 都读核心放在
`transformer_options["minimax_h3_layout"]` 里的段表，核对最后一段确实是
`video`、且行数等于 `frames × rows_per_frame`；不是就直接抛错，不再默默打偏。

### 3.3 关键帧 latent 必须与生成分辨率同网格

这条两边都得守。SelfLift 的做法：

- **高分辨率段**：直接用原始条件，关键帧已经是全分辨率，一个字节不动。
- **低分辨率段**：关键帧必须缩到低分辨率网格 —— §3.3 硬约束下唯一合法的做法。

---

## 4. 已经做的配合（逐条对应对方的软肋）

> 全部落在 `SelfLiftH3DraftSampler` 这条路径上。
> **`SelfLiftH3Sampler` 一个字节都没改**，行为与 upstream 完全一致。

### 4.1 §6.1 关键帧噪声被抹平 → 已解决

**问题**：H3 Studio 先 flow-blend 把噪声混进关键帧（`z' = a·z + (1-a)·noise`），
SelfLift 的 `_resize_keyframes` 再做双线性缩放 + 均值匹配 —— 噪声被空间平滑掉了。

**做法**（`nodes.py` 的 `_resize_keyframes(..., restore_noise=True)`）：

1. **面积采样**（`F.interpolate(mode="area")`）代替双线性。
   双线性是**混叠**（alias），它把噪声的高频挪到别处，还会让关键帧钉住的几何漂移；
   面积采样是**平均**，这才是「噪声占比」能算出来的前提。
2. **按 `1-noise_aug` 精确补噪**。
   设噪声占比 `f = 1 - noise_aug`，源网格 `H₀×W₀` 降到目标 `H×W`，
   一个输出像素平均了 `cells = (H₀/H)·(W₀/W)` 个源格 → 平均后噪声 std 掉到 `1/√cells`，
   所以补 `std = f·√(1 - 1/cells)`。
3. **确定性 seed**：`(seed + (index+1)·0x9E3779B1) mod 2⁶³`，CPU generator。
   同一个 seed 复跑落在同一个噪声上，换 seed 就换。

**实测**（512×288 目标、低分辨率段 8×12 latent、关键帧 strength 0.65）：

```
[SelfLift H3 Studio] positive keyframe 1: 18x32 -> 8x12, area-resampled and the noise put back
                     (noise_aug 0.6493, so 35.1% of the keyframe is noise).
```

`0.6493 = 0.65 × 0.999` ✓。补噪后每轴 std 0.34786，目标 0.35065，偏差 0.8%。

**没有 `noise_aug` 的关键帧不会被补噪** —— 那是全强度的帧，或者是被
`weaken_cond_latent`（另一条不标 `noise_aug` 的弱化路径）改过的帧，强度信息不可知，
瞎补会改坏它。fork 从 `feat/keep-keyframe-strength` 起把 `strength` 留在字典里，
所以这个「不可知」现在是**可以判定的**：见 §6.4。

注意区分两件事：**缩放方式**统一是「所有关键帧都用面积采样」（面积采样才是缩小时的
正确滤波器，双线性会混叠）；**补多少噪声**才按 `noise_aug` 分情况 —— 有标签就按
`1-noise_aug` 补，没有标签就一分不补。

### 4.2 §6.2 motion_context 接缝打折 → 检测 + 日志 + 文档降级

**SelfLift 侧无解**。motion context 钉住的尾帧也在 `minimax_keyframes` 里，
§3.3 的网格硬约束下它必须跟着低分辨率段一起缩。对方的 §7.2
（加 `minimax_h3_studio_meta` 标出 `pinned`）是真正的解法 —— 有了那个键，
SelfLift 可以对钉住的帧选「不降分辨率」。

现在能做的是**说清楚**：低分辨率段里出现 T>1 的多帧关键帧块时，
SelfLift 会打印一条降级说明，告诉你接缝在这里可能不如原生 H3 Studio 流程干净。

### 4.3 §6.3 TST 在 `highres_tiling` 下静默失效 → 直接报错

**问题**：wrapper 按注册顺序跑，TST 坐在 tiling wrapper 外面，
分块 attention 触发长度保护 → TST 跳过，只留一条 warning。

**做法**：草稿节点在 setup 期就检查，**直接 `raise ValueError`**，不给静默退化的机会。

识别方式不看「有没有 attention override」，而看 **TST 自己注册的 wrapper key**：

```python
model.wrappers["diffusion_model"]["selflift_h3_tst"]
```

所以别人的 attention override 不会被误伤。

### 4.4 §6.4 v2v 静默退化成满 denoise → 接线 + 显式日志

v2v 是**纯接线问题**，SelfLift 吃的是 `SIGMAS` 不是 KSampler 的 `denoise`：

```
H3StudioImageToVideo.v2v_denoise  ->  H3BasicScheduler.denoise  ->  SIGMAS  ->  SelfLift.sigmas
```

SelfLift 侧补一条日志：当 `sigmas[0] < 1.0` 时说明「这次不是从纯噪声起步，
是从源 latent 部分去噪起步（v2v / img2img 语义）」。

没有 v2v 源时 H3 Studio 会如实发 `denoise 1.0`，日志里会看到
`no v2v source -- emitting denoise 1.0` —— 那是**正确**的，不是接线漏了。

---

## 5. 接线配方（H3 Studio × SelfLift 草稿模式）

```
UNETLoader ──> SelfLiftH3TST ──> model ─────────────────────────┐
                                                               │
H3StudioImageToVideo ──> positive ─────────────────────────────┤
                       ├──> LATENT ──> latent_image            │
                       └──> v2v_denoise ──> H3BasicScheduler.denoise
                                                     │
                                          H3BasicScheduler ──> SIGMAS ──┐
KSamplerSelect (euler) ──> sampler ─────────────────────────────────────┤
                                                                       v
                                          SelfLiftH3DraftSampler ──> latent
                                                                    ├──> draft_video
                                                                    ├──> workflow_json
                                                                    ├──> workflow_api
VAELoader(video) ──> vae ─────────────────────┐                      │
VAELoader(audio) ──> audio_vae ───────────────┼──────────────────────┤
                                               │                      v
                                               │              SelfLiftH3Save
                                               │                      │
                                               │                      └──> latent ──> VAEDecode ──> CreateVideo ──> SaveVideo
                                               └──> (采样器自己用它画预览 mp4)
```

**要点**

| 项 | 要求 | 为什么 |
|---|---|---|
| 模型链 | UNET → **TST** → 采样器 | TST 只在模型 wrapper 上生效 |
| `sampler` | **必须 `KSamplerSelect: euler`** | SelfLift 复用 transition-step 的预测，只支持 Euler |
| `sigmas` | 必须来自 **`H3BasicScheduler`**，不能用 KSampler 的 denoise | 见 §4.4 |
| `negative` | `Guide.positive` → `ConditioningZeroOut` → `negative` | H3 Studio 的 positive 带条件元信息，负向走同一份字典 |
| `vae` / `audio_vae` | 视频 VAE 接 `vae`，音频 VAE 接 `audio_vae` | 预览 mp4 带声音 |
| `latent_image` | 接 `Guide.LATENT` | 定义目标尺寸和时长 |
| 草稿模式 | `draft_mode=true` | 只跑低分辨率前缀；latent 被 Save 节点拦下，下游解码器不跑 |

**直接可用的工作流**：`workflows/H3 Studio x SelfLift Draft Mode.json`
（生成器：`tools/make_h3studio_workflow.py`，`--small` 出小画布测试变体）。

生成器会**按本机 `/object_info` 替换模型名**，所以换机器不用手改。

---

## 6. 运行时自保（这一节原来全是待办）

四项里有三项已经落地成**会响的东西**，一项是跨两边的改动。共同原则：
**该报错就报错，该说清楚就说清楚，说了要能当行动指令用。**

| # | 事项 | 状态 | 触发条件 | 现在的行为 |
|---|---|---|---|---|
| A | **override 单槽冲突** | ✅ 已做 | 装 TST 时槽里已经有别人 | **抛错拒绝安装** |
| B | **packed 段序自保** | ✅ 已做 | video 不再是最后一段 | **每次 forward 抛错** |
| C | **`minimax_h3_studio_meta`** | 🟡 降级提示 | 对方还没标 `pinned` | **warning**，说清接缝会打折 |
| D | **`strength` 保留** | ✅ 已做（跨两边） | —— | 只在**真损失**时 warning |

### 6.1 A：单槽冲突 → 拒绝安装

`optimized_attention_override` 是单槽，后写覆盖前写。`h3_tst.py` 的
`_check_override_slot()` 在 `patch_model()` 里看这个槽：

- **空** → 正常装。
- **不空** → **直接抛错**，不装。报错写清四件事：单槽的覆盖语义、
  两个改写本来应该一起跑而一个槽装不下、链式包裹的具体做法
  （override 的第一个参数就是下层的 `func`）、以及在处理完之前
  把哪一条从链上拿掉。

> **A 的固有局限**：只能发现「装 TST **之前**槽就满了」。
> **装完之后被别人覆盖**，运行期发现不了 —— 事后没人能说清谁覆盖了谁。

### 6.2 B：packed 段序 → 每次 forward 都抛

TST 用 `v0 = s - frames × rows_per_frame` 从**序列末尾往前数**定位视频行，
所以「video 是最后一段」是硬前提。

`_check_packed_order()` 每次 forward 读核心放在
`transformer_options["minimax_h3_layout"]` 里的段表（`layout.segments`，
`(start, stop, kind)` 的连续段表），核对两件事：最后一段的 kind 必须是
`video`；它的行数必须等于 `frames × rows_per_frame`。

不满足就 **抛 `ValueError`**，报错里写明：最后一段是什么、末尾三段
各自的 kind 和行号、期望多少行、为什么这会让 TST 打偏、怎么退出，
以及段序是从哪个 `transformer_options` 键读来的。

> **为什么不警告而是报错**：这是唯一一类「完全静默且结果错误」的事故
> —— TST 的张力数字照打，只是算在了错的行上。警告会被忽略。
>
> **为什么不按长度推算**：只有 video 后面还压了别的段时，长度推算和
> 正常布局长得一模一样 —— 只有段表能分开这两种情况。
>
> **为什么每次 forward 都抛，不去重**：去重会让被上层捕获的异常在下一次
> forward 失效，等于给检查开后门。
>
> **依赖点**：核心内部约定 `transformer_options["minimax_h3_layout"]`。
> 核心升级若改了名，这个检查会**静默失效**（拿不到段表就直接放行）。

### 6.3 C：motion_context 多帧块 → warning

多帧关键帧块（motion context / 视频锚）会被低分辨率段一起缩到小网格，
钉住的帧因此被「减半再抬回来」，接缝处可能有一帧的软。
`nodes.py` 的 `_note_h3_studio_conditions()` 会在每次这样的运行里
warning 一次，写清四件事：**原因**（关键帧必须坐在生成分辨率网格上，
低分辨率段有自己的小网格）、**可见效果**（接缝处一帧的软或小跳，
在头发/树叶/织物这种密纹理上最明显，镜头一动就明显）、
**本地能不能修**（不能）、**真正的解法**（对方在
`minimax_h3_studio_meta` 里标 `pinned`，SelfLift 就可以让这种帧保持原分辨率）。

### 6.4 D：分不清「满强度」和「标签缺失」→ 只在真损失时报

`noise_aug` **只在 flow-blend 那条路上写**。所以一个没有标签的关键帧，
要么本来就是满强度，要么是被 `weaken_cond_latent` 改过（弱化路径不写标签）。
前者没事，后者**噪声比例未知，低分辨率缩放之后补不回来**。

只靠标签分不出来，但**两侧一起就能分清**：

- **H3 Studio 侧**（fork `feat/keep-keyframe-strength`）：把 `strength`
  留在关键帧字典里。它对核心是惰性的（核心只读 `latent` 和 `noise_aug`），
  只告诉对面这个帧到底被削弱了没有。
- **SelfLift 侧**：按下面的表判定，**只在真损失时 warning**。

| 关键帧的状态 | 判定 | 行为 |
|---|---|---|
| 有 `noise_aug` | 被削弱过，噪声比例精确 | 安静，补噪照做 |
| 有 `strength` 且 `>= 1.0` | 从没被削弱，没有噪声要补 | 安静 |
| 有 `strength` 且 `< 1.0`、无标签 | **真损失**：被削弱了，噪声比例未知 | **warning** |
| 两者都没有 | 分不清是满强度还是标签缺失 | **info**：说清自己看不到什么、为什么、怎么才能看到 |

最后一行还有一条**运行期自己做的推断**：标签只在「被削弱」时写，所以运行里
只要**有一个**带标签的关键帧，就证明 per-row 那套机制当时是通的，同一个运行里
没有标签的关键帧就只能是满强度 —— 这种组合一个字都不说。

多帧块（motion context）不进这张表：它的 latent 是成片素材，不是被削弱的静图，
那是 §6.3 的事。

> **还没对称的一件事**：声音锚（`audio_noise_aug`）走的是同一套逻辑，
> 同样在弱化路径上不写标签，但 SelfLift 不消费声音关键帧，所以没有对等检查。

---

## 7. 验证协议

SelfLift 改完，跑**对方**的检查脚本：

| 脚本 | 位置 | 查什么 | 什么时候跑 |
|---|---|---|---|
| `h3studio_api_drift_check.py` | `works/comfy-dev/` | H3 Studio 引用的 30 个核心符号还在不在 | ComfyUI 升级后 |
| `h3studio_patch_compat_check.py` | `works/comfy-dev/` | row-aug 锚点命中没命中；补丁版与原版的时间步表逐项差分 | ComfyUI 升级后 |
| `h3studio_selflift_coexist_check.py` | `works/comfy-dev/` | H3 Studio 的类级 `_forward` 替换 × TST wrapper 叠加，**17 项** | **SelfLift 每次改完** |

```powershell
$env:CUDA_VISIBLE_DEVICES=''   # 必须屏蔽 GPU，否则抢用户正在跑的任务的显存
$py='C:\Users\dream\AppData\Local\Comfy-Desktop\ComfyUI-Installs\Comfyui\ComfyUI\.venv\Scripts\python.exe'
& $py 'C:\Users\dream\Desktop\works\comfy-dev\h3studio_selflift_coexist_check.py'
& $py 'C:\Users\dream\Desktop\works\comfy-dev\h3studio_patch_compat_check.py'
```

SelfLift 自己还有三个离线检查（**不导入 ComfyUI**，可以随便跑）：

| 脚本 | 查什么 |
|---|---|
| `tools/test_h3_studio_coop.py` | 关键帧缩放路径：原采样器逐比特不变、补噪精度、`_tst_installed` 不误认；以及 C/D 警告的**触发条件**（普通运行不报警、满强度不报警、被削弱却丢标签才报） |
| `tools/test_runtime_guards.py` | 运行时守卫：packed 段序（B）、override 单槽（A）、C/D 警告是不是 warning 而不是 error |
| `tools/check_ui_workflow.py` | 工作流 JSON 的节点 / 连线完整性（要带工作流路径参数） |

**规矩**：任何一边动了「会影响对方的槽位」，先在对方仓库提一条 issue 或者直接跑一遍
对方的检查脚本，把结果贴过去。不要等对方踩到。

---

## 8. 一句话总结

> **H3 Studio 管「给模型什么」，SelfLift 管「怎么采」。**
> 两边唯一的交汇面是条件字典 `minimax_keyframes`，唯一的硬约束是
> 「关键帧 latent 必须与生成分辨率同网格」。
> 除此以外各自随便改，改完跑对方的检查脚本。
