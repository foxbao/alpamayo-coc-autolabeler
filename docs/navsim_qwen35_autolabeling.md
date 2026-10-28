# NAVSIM CoC reasoning 标注运行手册

本文记录如何在当前机器上用 NVIDIA Alpamayo CoC Auto-Labeler 和本地 Qwen3.5 模型，为 Qwen-Drive 的 NAVSIM 场景生成 reasoning 文本候选。它面向后续人工使用和 AI 助手接手维护。

> **实验状态**：这是 NAVSIM 适配实验，不是 NVIDIA 官方发布的 NAVSIM 工作流。生成文本是待审核候选，不是已经确认正确的训练标签。

## 项目关系与处理流程

Alpamayo CoC 仓库原生工作流面向 PhysicalAI AV 数据：生成 meta-action、挑选关键帧、提取视频，再用 VLM 生成 Chain-of-Causation（CoC）文本。NAVSIM 的数据结构不同，因此目前没有直接运行原生 PAI 数据入口。

当前 NAVSIM 适配脚本调用本仓库的 `QwenWrapper`，把每条样本的三路相机历史图像整理成 4 个时间对齐的视频帧，再生成一条 CoC 文本。适配脚本现在位于另一个仓库：[`Qwen-Drive-1.0/scripts/annotate_navsim_coc.py`](../../Qwen-Drive-1.0/scripts/annotate_navsim_coc.py)。

处理时会从 NAVSIM 未来轨迹计算粗略的纵向/横向动作，作为生成阶段的 **teacher-only action hint**，帮助模型理解需要解释的动作。该提示不会写入最终训练样本的 user 消息；脚本会保留源样本 user 消息，只替换 assistant 文本并附加 `reasoning_annotation` 元数据。不过模型在生成时看到了这个提示，因此候选文本可能受到未来动作信息影响，也可能提到历史状态或路线指令。人工审核时要检查文本是否能由当前图像和合法输入支持。

数据流可以概括为：

```text
NAVSIM user 输入（历史状态、导航命令、12 张相机图像）
                 │
                 ├── 3 路相机 × 4 个时间点 → 4 帧三栏视频
                 ├── 未来轨迹 → teacher-only 粗动作提示，仅供生成使用
                 └── Alpamayo QwenWrapper + Qwen3.5-35B-A3B
                                      │
                                      └── 候选 assistant reasoning + 审核页面
```

## 当前机器的环境与路径

| 项目 | 当前值 |
| --- | --- |
| CoC 仓库 | `/home/baojiali/Downloads/public_code/alpamayo-coc-autolabeler` |
| Qwen-Drive 仓库 | `/home/baojiali/Downloads/public_code/Qwen-Drive-1.0` |
| Conda 环境 | 仓库内 `.conda-envs/alpamayo-coc` |
| NAVSIM 场景 JSONL | `/cloud/cloud-ssd1/qwen-drive/datasets/navsim_trainval_v1.1/prepared/navsim_qwen_train_split.jsonl` |
| NAVSIM 图像根目录 | `/cloud/cloud-ssd1/qwen-drive/datasets/navsim_trainval_v1.1` |
| 本地模型缓存 | `/home/baojiali/.cache/alpamayo-coc-models` |
| 当前模型 | `qwen3.5_35b`（Qwen3.5-35B-A3B） |
| 当前 GPU | 8 × NVIDIA GeForce RTX 4090 24 GB |

这些绝对路径只适用于当前机器。AI 助手在其他机器上操作前，应先检查挂载和文件是否存在，不要假设路径相同。

### Conda 与 CUDA 隔离

用下面的命令进入专用环境：

```bash
conda activate /home/baojiali/Downloads/public_code/alpamayo-coc-autolabeler/.conda-envs/alpamayo-coc
```

当前已验证版本：Python 3.12、PyTorch `2.10.0+cu128`、vLLM `0.17.1`、Transformers `4.57.6`。`torch.cuda.is_available()` 返回 `True`，并且 `from vllm import LLM` 可以导入。

当前 CoC checkout 的 HEAD 是 `4dda809`。[`src/coc_labeling/model_clients/vlm_wrappers/qwen.py`](../src/coc_labeling/model_clients/vlm_wrappers/qwen.py) 有一项本地未提交修改：wrapper 从 `VLLM_MAX_MODEL_LEN` 读取上下文上限，默认仍为 128000。NAVSIM 脚本依赖这项修改将上下文限制为 32768；后续 AI 助手操作前应检查 `git status`，不要 reset 或覆盖这项修改。

主机 NVIDIA 驱动和系统 CUDA 无需更改。当前 `nvidia-smi` 显示驱动 580.126.09、CUDA 13.0；Conda 环境中的 PyTorch wheel 自带 CUDA 12.8 运行库，两者可以共存。不要为了此项目升级/替换系统驱动或 `/usr/local/cuda`，也不要把 CoC 依赖装进 `qwen-drive` 环境。

`qwen-drive` 环境单独维护，目前使用 PyTorch `2.8.0+cu128`。在 `alpamayo-coc` 中不应能导入 `qwen_drive`；两个环境的 PyTorch/vLLM 版本不同是预期的隔离结果。

### 从零重建环境

使用仓库内的 Conda 环境。首次创建后，在仓库根目录运行 `conda activate "$PWD/.conda-envs/alpamayo-coc"`：

```bash
conda create -y -p "$PWD/.conda-envs/alpamayo-coc" python=3.12 pip
conda activate /home/baojiali/Downloads/public_code/alpamayo-coc-autolabeler/.conda-envs/alpamayo-coc
cd /home/baojiali/Downloads/public_code/alpamayo-coc-autolabeler
python -m pip install --no-cache-dir \
  --index-url https://download.pytorch.org/whl/cu128 \
  --extra-index-url https://pypi.org/simple \
  -r requirements-lock.txt
python -m pip install --no-cache-dir --no-deps \
  'trajdata @ git+https://github.com/NVlabs/trajdata.git@coc_autolabeling-1.0.0'
python -m pip install --no-cache-dir --no-deps -e .
```

服务器上如果导入 `cv2` 报 `ImportError: libGL.so.1`，将 OpenCV 模块覆盖为 headless wheel：

```bash
python -m pip install --no-cache-dir --force-reinstall --no-deps \
  opencv-python-headless==4.13.0.92
```

CoC 的 Dockerfile 已说明 `trajdata` 声明的 `protobuf==3.19.4` 与锁文件中的新版依赖冲突，所以 `trajdata` 要用 `--no-deps` 安装并保留锁文件里的 protobuf 6.x。当前 `pip check` 对此仍会报告版本约束冲突；CoC、trajdata、vLLM 导入和实际标注运行均已通过。不要为了让 `pip check` 清零而把 protobuf 降级。

### 模型权重

本机已有约 67 GB 的 Qwen3.5-35B-A3B 权重，缓存路径为：

```text
/home/baojiali/.cache/alpamayo-coc-models/models--Qwen--Qwen3.5-35B-A3B
```

脚本默认复用该缓存，不需要重复下载。新机器首次使用时需要按 Hugging Face 仓库许可登录并下载模型；不要把 Hugging Face token 写进文档、脚本、shell 历史或输出文件。

## 运行 NAVSIM 标注

先做 1 条 smoke test，再扩大样本数。脚本会拒绝覆盖已经存在的 `candidates.jsonl`，每次重复运行都应指定新的输出目录。

### 单条 smoke test

```bash
conda activate /home/baojiali/Downloads/public_code/alpamayo-coc-autolabeler/.conda-envs/alpamayo-coc
cd /home/baojiali/Downloads/public_code/Qwen-Drive-1.0
python scripts/annotate_navsim_coc.py \
  --scenes /cloud/cloud-ssd1/qwen-drive/datasets/navsim_trainval_v1.1/prepared/navsim_qwen_train_split.jsonl \
  --image-root /cloud/cloud-ssd1/qwen-drive/datasets/navsim_trainval_v1.1 \
  --output outputs/navsim-coc-qwen35-smoke-new-run/candidates.jsonl \
  --limit 1 \
  --seed 3407 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

### 20 条平衡抽样

下面的命令使用与当前 20 条运行相同的设置；输出目录采用新名字，避免覆盖已有结果：

```bash
conda activate /home/baojiali/Downloads/public_code/alpamayo-coc-autolabeler/.conda-envs/alpamayo-coc
cd /home/baojiali/Downloads/public_code/Qwen-Drive-1.0
python scripts/annotate_navsim_coc.py \
  --scenes /cloud/cloud-ssd1/qwen-drive/datasets/navsim_trainval_v1.1/prepared/navsim_qwen_train_split.jsonl \
  --image-root /cloud/cloud-ssd1/qwen-drive/datasets/navsim_trainval_v1.1 \
  --output outputs/navsim-coc-qwen35-20-new-run/candidates.jsonl \
  --limit 20 \
  --seed 3407 \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.90
```

脚本当前从约 9,018 条场景记录中按粗略未来纵向/横向动作类别轮流抽样，并按 `scene_token` 去重。`--max-model-len 32768` 和显存利用率 `0.90` 已在单张 A800 80 GB 上验证；Qwen3.5 权重加载约占 65.5 GiB。若 GPU 上有其他任务，先检查 `nvidia-smi`，不要同时启动多份模型进程。

常用参数：

- `--limit`：候选条数。
- `--seed`：抽样和生成种子；同样数据与代码下可复现抽样。
- `--model-name`：CoC 注册的模型别名；当前使用 `qwen3.5_35b`。
- `--max-model-len`：上下文上限。Qwen wrapper 的本地修改通过 `VLLM_MAX_MODEL_LEN` 环境变量支持此参数，默认仍为 128000。不要移除该修改后还假设 32768 参数会生效。
- `--gpu-memory-utilization`：传给 vLLM 的显存比例；当前 A800 设置为 `0.90`。
- `--max-new-tokens`：生成上限，默认 512。

## 输出与审核

运行目录包括：

- `candidates.jsonl`：逐场景的 ChatML 训练样本候选。每条的 user 输入保留源 NAVSIM user 消息，assistant 文本替换为生成的 reasoning，并在 `meta_info.reasoning_annotation` 中记录模型、动作提示、规则标记和审核状态。
- `review.html`：图像与候选文本的人工审核页面。
- `manifest.json`：数据源、模型、抽样方式、泄漏控制说明和总数。
- `mosaics/`：每个样本的 4 帧、每帧 3 个相机视图拼图，供视频输入和审核使用。

当前 20 条结果在 Qwen-Drive 仓库：

- [`candidates.jsonl`](../../Qwen-Drive-1.0/outputs/navsim-coc-qwen35-20/candidates.jsonl)
- [`review.html`](../../Qwen-Drive-1.0/outputs/navsim-coc-qwen35-20/review.html)
- [`manifest.json`](../../Qwen-Drive-1.0/outputs/navsim-coc-qwen35-20/manifest.json)

目前 20 条对应 20 个不同 `scene_token`，全部状态为 `candidate_unreviewed`；其中 5 条触发至少一项自动规则。自动规则只检查长度、动作词和少量泄漏词，不能判断图像事实是否正确，也不能证明因果解释成立。审核时逐条对照图像，检查对象/信号/原因是否真实可见、动作方向和时序是否正确、文本是否把路线指令或历史状态误说成视觉证据。保留审核记录；未经审核的候选不要直接用于 SFT。

在本次样本中，模型有时会把历史速度或导航命令写入解释，也会对场景原因做无依据推测。自动标记为零不代表文本正确。未来动作提示还会让输出受到 teacher 信号影响，因此若目标是完全不使用未来信息的在线 reasoning，需要调整标注提示和生成方法，再重新评估。

## 快速验证命令

激活环境后可运行：

```bash
python - <<'PY'
import torch, vllm, transformers, cv2
from vllm import LLM
from coc_labeling.model_clients.vlm_wrappers.qwen import QwenWrapper
print('torch:', torch.__version__)
print('CUDA runtime:', torch.version.cuda)
print('CUDA available:', torch.cuda.is_available())
print('vLLM:', vllm.__version__)
print('Transformers:', transformers.__version__)
print('OpenCV:', cv2.__version__)
PY
```

`nvidia-smi` 的 CUDA 版本指驱动所支持的 CUDA 版本；`torch.version.cuda` 显示当前环境 PyTorch wheel 使用的运行时版本。两者数字不同本身不代表冲突。

## 给后续 AI 助手的操作约定

1. 先读本手册、本仓库的 [`README.md`](../README.md) 和适配脚本，再决定下一步；不要把 NAVSIM 适配误当成上游原生 PAI 数据工作流。
2. 使用 `conda activate /home/baojiali/Downloads/public_code/alpamayo-coc-autolabeler/.conda-envs/alpamayo-coc`。不要在 `qwen-drive` 安装/升级 CoC、vLLM、PyTorch 或 Transformers，也不要更改系统 CUDA。
3. 开始前确认数据、模型缓存、GPU 和输出目录。先跑 `--limit 1`，成功后再增加样本数；输出目录已存在时使用新目录名，不覆盖已有候选。
4. 生成结果保持 `candidate_unreviewed`，先完成人工审核和质量评估，再讨论训练。不要把自动规则通过当作审核通过。
5. 修改 NAVSIM 适配脚本或 CoC wrapper 时，说明改变了哪些输入、泄漏边界和输出格式，并重新做单条 smoke test。
6. 不要删除已有模型缓存或审核输出来释放空间，除非用户明确要求。

## 关联资料

- 本仓库上游流程、模型别名和输出格式：[`README.md`](../README.md)
- 上游锁定依赖与 CUDA 12.8 安装方式：[`requirements-lock.txt`](../requirements-lock.txt)、[`Dockerfile`](../Dockerfile)
- NAVSIM 适配脚本：[`Qwen-Drive-1.0/scripts/annotate_navsim_coc.py`](../../Qwen-Drive-1.0/scripts/annotate_navsim_coc.py)
