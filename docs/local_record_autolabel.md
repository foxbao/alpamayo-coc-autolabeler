# 本地记录 CoC MVP 使用说明

当前入口处理一条记录和一个人工指定的相对时刻，用 front-camera 文件名时间取出 CoC 所需的 2 秒历史、6 秒未来窗口。它读取 localization 文件名时间、JSON 中的 `position.x/y` 和 `orientation`，生成 10 Hz 位置轨迹；位置差分得到地图速度，再用同一时刻的姿态转成车体 `v_long/v_lat`，按 0.5 秒间隔提供给 agent。它不读取 chassis 或 planning，也不自动发现动作。

## 准备预览

在仓库根目录使用现有环境：

```bash
conda activate /mnt/disk1/conda_envs/alpamayo-coc
python -m coc_labeling.local_record_autolabel \
  --input-root /home/baojiali/Downloads/disk3/20261019_lin_007_out \
  --record 20261019153941_record \
  --event-time-sec 21.6 \
  --output-dir outputs/local_record_autolabel_mvp
```

`--event-time-sec` 从该记录第一张 front-camera 图片的文件名时间开始计时。程序按仓库默认的 0.5 秒间隔选 16 张图；每张图必须与目标时刻相差不超过 50 ms，且不能重复使用同一源图。

输出目录会生成：

```text
outputs/local_record_autolabel_mvp/<record>/event_<seconds>s/
  contact_sheet.jpg
  preview.mp4
  frame_manifest.json
  position_trajectory_10hz.json
  position_trajectory.png
  position_motion_audit.json
  agent_input_preview.json
```

`contact_sheet.jpg` 标出相对事件时间、采样误差和源文件名；`preview.mp4` 是带时间标记的 2 fps、8 秒预览；`frame_manifest.json` 记录选帧、绝对 Unix 时间戳、事件索引和当前状态。

`position_trajectory_10hz.json` 保存事件相对 10 Hz 网格上的 map-XY 位置。每个网格点只会在前后原始定位样本相隔不超过 32 ms 时插值；原始间隔超过 32 ms 的区段会单独列在 `raw_sampling_gaps_over_limit` 中。`position_trajectory.png` 显示 10 Hz 轨迹，红点标出跨越原始大间隔的位置段；图上的 map-XY 轨迹方向不代表车头方向。

`position_motion_audit.json` 记录 `-2.0` 至 `+6.0` 秒、每 0.5 秒一个的 17 个位置差分速度点、姿态、车体速度和有效性。对当前四条记录，数据核对支持将 orientation 解释为车体 FLU 到 map-XY 的旋转；地图速度逆旋转后，`v_long = body_x`、`v_lat = -body_y`（右正）。这项判断来自轨迹方向、规划 `theta` 和前视画面运动，JSON 本身没有坐标系标签；详细依据见 [local_record_coordinate_conversion.md](local_record_coordinate_conversion.md)。`linear_velocity.x/y` 只作为候选 FLU 分量与推导结果做一致性审计。`agent_input_preview.json` 展示实际送入 agent 的输入，`ego_text` 包含 `v_long`、`v_lat` 和地面速度大小；姿态缺失时，前两个量显示 `Not Available`，位置速度仍可提供地面速度。默认 prepare-only 不读取 API 凭据、不初始化模型。

## 单次 GPT 推理

人工确认预览中的画面顺序和事件位置后，显式加 `--run-inference`：

```bash
python -m coc_labeling.local_record_autolabel \
  --record 20261019153941_record \
  --event-time-sec 21.6 \
  --run-inference
```

默认模型为 `gpt-6.1-sol`。GPT 配置读取 `~/.config/alpamayo-coc/openai_api.yaml`，字段为 `api_key` 和可选的 `base_url`；也可先设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 覆盖配置文件。凭据不会写入输出或日志。

## VS Code 调试

仓库的 `.vscode/launch.json` 提供两个启动项：`Local record autolabel (prepare preview)` 只生成预览，适合逐步检查读图和选帧；`Local record autolabel (GPT inference; sends API request)` 会调用模型并产生 API 费用。输出默认放在仓库的 `outputs/local_record_autolabel_mvp/` 下。启动前按需修改 `--record`、`--event-time-sec` 和输入目录，然后在 VS Code 的 Run and Debug 面板选择配置并按 F5。配置使用本机的 `alpamayo-coc` Python 环境。

单次推理输出：

```text
<event>/labels/<record>/cot_<event_unix_timestamp_micros>.yaml
<event>/preview_with_coc.mp4
<event>/contact_sheet_with_coc.jpg
```

YAML 由现有 `VLMCoCAgent` 写出，`preview_with_coc.mp4` 使用 `scripts/overlay_coc_result.py` 叠加描述；`contact_sheet_with_coc.jpg` 在原 4×4 联系表上方显示同一条 CoC 描述。输出状态是 `completed_pending_review`，仍需要人工审核。相同参数再次运行时会覆盖同路径输出；若带 `--run-inference`，每次都会重新请求模型并覆盖该事件的 YAML 和可视化结果。

## 代码结构

命令仍从 `python -m coc_labeling.local_record_autolabel` 启动；这个文件负责参数解析和串起流程。具体实现拆在 `src/coc_labeling/local_recording/`：`camera.py` 负责相机时间戳和选帧，`localization.py` 负责原始位置、10 Hz 重采样和速度，`agent.py` 负责组装并调用 CoC agent，`visualization.py` 负责图片预览，`position_artifacts.py` 负责轨迹与运动审计，`artifacts.py` 负责清单和 agent 输入预览，`types.py` 放共享数据结构和参数。

## 当前限制

目前 `all_meta_action_str` 仍使用 `Not Available`；纵向和横向速度由位置差分与 localization 姿态合成，并不等同于车道相对运动或换道方向。CoC 描述仍需结合图片并由人工审核，当前入口也不自动发现事件或批量处理记录。
