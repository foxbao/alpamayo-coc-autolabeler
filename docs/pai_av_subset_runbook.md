# PAI-AV 小样本下载与 CoC 自动标注

这份手册面向新服务器：下载一小部分 PhysicalAI AV 数据，生成 meta-action 和
keyframe，提取前视视频，再用本仓库正式入口运行少量 CoC 推理。

## 先理解下载粒度

PAI-AV 按 chunk 打包。下面的下载器可以从 reasoning 表中抽取 N 个 clip ID，
再下载这些 clip 所在的 chunk 和指定数据组件；**它不会只下载 N 个 clip 的字节**。
同一个 chunk 里其它 clip 也会随 zip 一起下载，因此下载量会随所选 chunk 和组件而变。
先检查目标盘空间，下载后再看 `du -sh "$PAI_ROOT"`。

`--only-reasoning-chunks` 会先取 `clip_index.parquet` 和
`reasoning/ood_reasoning.parquet`，用 reasoning 表挑选 clip ID 并找到 chunk。后续
CoC 标注流程只读取轨迹、视频和本仓库生成的 meta-action；不会把 OOD reference
reasoning 文本传给模型。

## 准备 Hugging Face 访问和下载器

先在 Hugging Face 网页上接受 [PhysicalAI-Autonomous-Vehicles 数据集](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles)
的访问条款。然后在目标服务器上准备 Python 环境。下载脚本来自 NVIDIA 的
[`alpamayo-recipes`](https://github.com/NVlabs/alpamayo-recipes) 仓库，不在本仓库中：

```bash
git clone https://github.com/NVlabs/alpamayo-recipes.git /path/to/alpamayo-recipes
```

在将要运行下载器的 Python 环境里安装依赖并登录。`huggingface_hub` 提供下载和
`hf auth` 命令；`pandas` 与 `pyarrow` 用于读取 reasoning/clip index Parquet 并写小索引。

```bash
python -m pip install huggingface_hub pandas pyarrow
hf auth login
hf auth whoami
```

登录时按 Hugging Face 的交互提示输入 token，不要把 token 放进命令行、脚本或日志。
成功后 token 会保存在该用户的 Hugging Face 本地凭证目录中。

## 准备 CoC 仓库和 Python 环境

在目标服务器上获取本仓库，并按要使用的推理后端安装依赖。建议让下载和标注步骤
使用各自明确的 Python 环境；下面的 `PYTHON_BIN` 要指向已安装本仓库的环境。

```bash
COC_REPO=/path/to/alpamayo-coc-autolabeler
git clone https://github.com/NVlabs/alpamayo-coc-autolabeler.git "$COC_REPO"
cd "$COC_REPO"

# GPT API 或其它非本地 Qwen 后端
python -m pip install -e ".[dev]"

# 本地 Qwen/VLLM 后端：按服务器 CUDA/VLLM 环境安装
# python -m pip install -e ".[dev,qwen]"
```

如果仓库已经存在，不要重复 clone；进入该目录并安装当前 checkout。Qwen 的 VLLM、CUDA
和显卡显存要求请按主 [README](../README.md) 检查。后续命令从仓库根目录运行。

## 下载一个小样本

下面的最小组件组合足以跑本仓库的 PAI CoC 主线：`labels/egomotion` 给 trajdata 和
meta-action 使用；一条前视 camera stream 给 VideoLoader 使用。当前 CoC 配置不需要
下载四路相机、LiDAR、radar 或 calibration。

按新服务器实际情况设置绝对路径：

```bash
PAI_ROOT=/path/to/data/pai_av_subset
RECIPES_ROOT=/path/to/alpamayo-recipes
mkdir -p "$PAI_ROOT"
```

先用 16 个 clip 做一轮比较可靠的小型功能验证。可以把 `16` 调小来减少候选 clip 数，
但**下载空间不会按 clip 数线性缩小**，因为仍按含这些 clip 的 chunk 下载。

```bash
python "$RECIPES_ROOT/scripts/download_pai.py" \
  --only-reasoning-chunks \
  --num-reasoning-clips 16 \
  --random-seed 3407 \
  --camera camera_front_wide_120fov \
  --labels egomotion \
  --output-dir "$PAI_ROOT"
```

下载器会报告抽样数量和对应 chunk ID。完成后检查关键文件：

```bash
test -f "$PAI_ROOT/clip_index.parquet"
test -f "$PAI_ROOT/clip_index_reasoning_mini.parquet"
test -d "$PAI_ROOT/labels/egomotion"
test -d "$PAI_ROOT/camera/camera_front_wide_120fov"
du -sh "$PAI_ROOT"
```

## 为选中的 clip 生成 meta-action 和 keyframe

`meta-action-autolabel --scene_list` 接收每行一个 clip ID 的文本文件。用下载器生成的
mini index 创建它，并建立本次运行目录：

```bash
RUN_ROOT=/path/to/output/pai_av_subset_smoke
mkdir -p "$RUN_ROOT/inputs" "$RUN_ROOT/cache/trajdata" "$RUN_ROOT/meta_action" \
  "$RUN_ROOT/keyframes" "$RUN_ROOT/video"

python - "$PAI_ROOT" "$RUN_ROOT/inputs/scene_list.txt" <<'PY'
from pathlib import Path
import sys

import pandas as pd

dataset_root = Path(sys.argv[1])
output_path = Path(sys.argv[2])
index = pd.read_parquet(dataset_root / "clip_index_reasoning_mini.parquet")
clip_ids = index.index.astype(str).drop_duplicates().tolist()
output_path.write_text("".join(f"{clip_id}\n" for clip_id in clip_ids), encoding="utf-8")
print(f"Wrote {len(clip_ids)} clip IDs to {output_path}")
PY
```

生成所有纵向和横向 ego meta-action。`--scene_list` 限制处理范围，因此不会遍历整个
PAI 数据集。`--num_workers` 可按服务器 CPU 和内存调整：

```bash
meta-action-autolabel \
  --dataset_name pai \
  --meta_action_names all_ego \
  --data_dir "$PAI_ROOT" \
  --cache_dir "$RUN_ROOT/cache/trajdata" \
  --save_dir "$RUN_ROOT/meta_action" \
  --scene_list "$RUN_ROOT/inputs/scene_list.txt" \
  --num_workers 4
```

再从 meta-action 结果中生成 CoC 可消费的相对 keyframe JSON：

```bash
python -m coc_labeling.keyframe_auto_select \
  --meta_action_dir "$RUN_ROOT/meta_action/final_outputs" \
  --min_duration 10 \
  --min_start_ts 20 \
  --max_start_ts 135 \
  --output_dir "$RUN_ROOT/keyframes"
```

检查 `segments_relative_timestamp_sampled.json` 是否存在并含有事件。clip 数不等于
event 数：有些 clip 经过动作时长和起始时间过滤后不会留下 keyframe。

可以先打印事件数量；如果是 0，就先检查 meta-action 输出和 keyframe 的过滤参数，不要
继续提取视频或启动模型：

```bash
python - "$RUN_ROOT/keyframes/segments_relative_timestamp_sampled.json" <<'PY'
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
count = sum(len(rows) for rows in payload.values())
print(f"{count} keyframe events across {len(payload)} action buckets")
if count == 0:
    raise SystemExit("No events found; inspect meta-actions and keyframe filters first.")
PY
```

## 只提取 keyframe 涉及的前视视频

从下载的 camera chunk zip 中抽取索引引用的 clips，而不是手动解压所有内容。先用
`--dry-run` 查看将要提取的文件，再真正执行：

```bash
python scripts/extract_pai_videos_from_index.py \
  --index-file "$RUN_ROOT/keyframes/segments_relative_timestamp_sampled.json" \
  --video-zip-dir "$PAI_ROOT/camera/camera_front_wide_120fov" \
  --output-video-root "$RUN_ROOT/video" \
  --strict \
  --dry-run

python scripts/extract_pai_videos_from_index.py \
  --index-file "$RUN_ROOT/keyframes/segments_relative_timestamp_sampled.json" \
  --video-zip-dir "$PAI_ROOT/camera/camera_front_wide_120fov" \
  --output-video-root "$RUN_ROOT/video" \
  --strict
```

CoC 的 `video_dir` 应指向 `"$RUN_ROOT/video"`；抽取器会在其下建立 `camera/`。

## 用正式 CoC 入口做小规模推理

本仓库的 [批量启动器](../scripts/run_coc_labeling.sh) 会读取上面生成的路径。把
`PYTHON_BIN` 指向当前已安装本仓库的 Python。先用 `--dry-run` 检查命令，再运行推理。

本地 Qwen：

```bash
PYTHON_BIN="$(command -v python)" \
DATA_DIR="$PAI_ROOT" \
RUN_ROOT="$RUN_ROOT" \
MODEL_CACHE_DIR=/path/to/qwen-model-cache \
./scripts/run_coc_labeling.sh \
  --model qwen3.5_35b \
  --exp-name pai_subset_qwen_smoke \
  --limit 3 \
  --dry-run

PYTHON_BIN="$(command -v python)" \
DATA_DIR="$PAI_ROOT" \
RUN_ROOT="$RUN_ROOT" \
MODEL_CACHE_DIR=/path/to/qwen-model-cache \
./scripts/run_coc_labeling.sh \
  --model qwen3.5_35b \
  --exp-name pai_subset_qwen_smoke \
  --limit 3
```

GPT API：

```bash
PYTHON_BIN="$(command -v python)" \
DATA_DIR="$PAI_ROOT" \
RUN_ROOT="$RUN_ROOT" \
./scripts/run_coc_labeling.sh \
  --model gpt-6.1-sol \
  --api-config ~/.config/alpamayo-coc/openai_api.yaml \
  --exp-name pai_subset_gpt_smoke \
  --limit 2 \
  --dry-run
```

配置文件格式可参考 [`tutorials/openai_api.example.yaml`](../tutorials/openai_api.example.yaml)，
把示例复制到本机配置目录后再填写凭证；不要把真实 key 写进仓库。

确认无误后，去掉最后的 `--dry-run` 才会调用 API。GPT 图像输入会产生费用；先用
`--limit 1` 或 `--limit 2`，检查 YAML 和实际 usage，再决定是否扩大。启动器默认按
`(clip_id, event_start_frame)` 去重，避免不同动作 bucket 的同一事件覆盖同一 YAML。

`--limit N` 限制的是**最多 N 个唯一事件**，不是 N 个 clip。可以用
`--segment-index /path/to/another_segments.json` 指定另一份 JSON 索引。

结果写入仓库下的 `experiments/<timestamp>_<exp-name>/`。核对日志中的
`Processed N / N segments successfully`，并检查输出 YAML 数量：

```bash
find experiments -path '*pai_subset_*_smoke*' -name 'cot_*.yaml' -type f | wc -l
```

## 新服务器上还需要准备什么

- CoC Python 环境和本仓库依赖：按前面的安装步骤及主 [README](../README.md) 准备；Qwen
  要安装 Qwen/vLLM 依赖、准备有足够显存的 GPU，并让模型权重可从 Hugging Face 缓存读取。
- GPT 推理：配置本地 OpenAI-compatible YAML；不需要 GPU，但每次请求会收费。
- 数据访问：Hugging Face 账号必须有 PAI-AV 权限；下载 token 使用 `hf auth login` 保存，
  不要写进仓库。
- 存储：按 chunk 下载，不能把 `--num-reasoning-clips 16` 理解为仅下载 16 个视频文件。
  所需磁盘空间取决于抽中的 chunk 和 `--camera`/`--labels` 组件。

之前 16-clip 运行的本机数据目录约 86 GB，因为当时还下载了四路相机、多个 calibration
和 label 组件。那次 run manifest 没有记录下载器命令或随机种子；本手册给出的是可复现的
最小组件路径，不能保证抽出与旧运行相同的 clip IDs。
