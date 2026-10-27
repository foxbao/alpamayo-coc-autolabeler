# 本地记录 CoC MVP 使用说明

当前入口处理一条记录和一个人工指定的相对时刻，用 front-camera 文件名时间取出 CoC 所需的 2 秒历史、6 秒未来窗口。它读取 localization 文件名时间、JSON 中的 `position.x/y` 和 `orientation`，生成 10 Hz 位置轨迹；位置差分得到地图速度，再用同一时刻的姿态转成车体 `v_long/v_lat`，按 0.5 秒间隔提供给 agent。它还在同一条 10 Hz 轨迹上生成 PAI 风格的纵向 meta-action。它不读取 chassis 或 planning，也不自动发现动作。

## 自动筛选整条记录中的 CoC 候选片段

整段记录的动作生成和相机候选筛选分成两个命令。这样调整候选时间间距或覆盖频率时，可以重复第二步，不必重新读取 localization 和计算动作。两步默认共用 `outputs/local_record_candidates/`。

第一步只读取 localization，按 10 Hz 轨迹生成纵向动作点和区间：

```bash
python -m coc_labeling.local_data_labeling \
  --input-root /home/baojiali/Downloads/disk3/20261019_lin_007_out \
  --record \
    20261019153841_record \
    20261019153911_record \
    20261019153941_record \
    20261019154011_record \
  --output-dir outputs/local_record_candidates
```

不传 `--record` 时，会处理 `--input-root` 下所有含 `localization/` 子目录的记录。每条记录会写出 `full_record_meta_action.json`，包含动作点、动作区间和定位时间原点；这些仍是供检查的规则提示，不是人工真值。

第二步只读取 front-camera 图片和第一步生成的动作 JSON，筛选候选事件、检查每个事件的图像窗口，并生成候选联系表：

```bash
python -m coc_labeling.local_keyframe_auto_select \
  --input-root /home/baojiali/Downloads/disk3/20261019_lin_007_out \
  --meta-action-root outputs/local_record_candidates \
  --record \
    20261019153841_record \
    20261019153911_record \
    20261019153941_record \
    20261019154011_record \
  --output-dir outputs/local_record_candidates
```

候选阶段不会重新读取 localization，也不会重新计算 meta-action。缺少 `full_record_meta_action.json` 时，命令会提示先运行 `python -m coc_labeling.local_data_labeling`。`--meta-action-root` 指定第一阶段动作 JSON 的目录，`--output-dir` 指定本阶段候选清单和联系表的输出目录；省略前者时仍使用 `--output-dir`，兼容原有命令。默认每 5 秒生成一个无动作类别的覆盖点，并严格要求最终保留的候选起点至少相隔 2 秒；动作区间起点优先于附近的覆盖点。这个间距规则不保证一个长动作的整个过程都会落在候选窗口中，靠近动作候选的覆盖点也可能被抑制。动作候选默认至少持续 1 秒（10 Hz 下 10 个 tick），也可以只保留指定类别，例如 `--actions "Gentle Deceleration" Stop`；这些筛选不会影响定时覆盖候选。可用 `--minimum-action-duration-sec` 调整动作时长门槛，也可用 `--periodic-interval-sec` 和 `--minimum-separation-sec` 调整覆盖频率及近邻间距。

候选筛选随后会像 PAI 的 `keyframe_auto_select` 一样按动作类别限制样本数：对本次命令处理的所有记录，先验证相机窗口，再为每个纵向动作类别最多保留 `--target-count` 个动作候选，并优先保留持续时间最长的区间，最后在每条记录内应用严格间距规则。如果区间较长的候选因间距规则被抑制，程序会从该类别超额候选中按时长回填与当前已选动作保持足够间距的候选。回填保留已选较长候选，不会回溯替换它们；因此这是类别上限，不保证填满。即使换成几条较短区间可以选满目标数，程序仍优先保留较长区间。默认值为 `500000`，与 PAI 一致，通常不会裁掉当前数据；可设为较小值做平衡抽样。仍被限额排除的项目会保留在清单中，状态为 `filtered`、原因是 `action_type_target_count_exceeded`。定时 `coverage` 候选不参加类别限额，但会遵守相同的起点间距规则。若显式传入 `--record`，类别限额只在这次指定的记录集合内统计。

`--minimum-action-duration-sec` 是候选阶段额外施加的最短动作区间门槛，默认 1 秒。它独立于本地纵向动作分类器生成动作区间时使用的最短长度规则；某个区间即使已经被分类器识别，若短于这个候选门槛，仍不会成为 CoC 候选。

每条记录的动作输入和候选输出都默认位于 `outputs/local_record_candidates/<record>/`；使用 `--meta-action-root` 和 `--output-dir` 可将它们分开放置：

```text
full_record_meta_action.json
segment_candidates.json
contact_sheets/candidate_*.jpg
```

`segment_candidates.json` 以 `action_candidates` 和 `coverage_candidates` 两个数组分别保存动作样本与覆盖样本。动作项会保留规则动作类别、区间起止时间和持续时长；覆盖项的动作类别与动作区间为空。每项也会记录筛选状态和原因，包括被选中、时长或类别过滤、相机窗口不完整、以及与其他候选过近而合并。重跑后，`contact_sheets/` 中不属于当前选中候选的旧 `candidate_*.jpg` 会被清理；其它文件会保留。候选 ID 按本次候选时间顺序生成，改变筛选参数后 ID 可能变化，应以当前 manifest 为准。这个流程用于发现值得人工检查的片段，不调用 GPT，也不直接生成 CoC 标签。

## 从候选生成 CoC 标注

候选生成后，可将所有 `selected` 项交给现有单事件准备流程。默认只生成图片预览、轨迹与定位审计、Agent 输入快照；检查后可对一个候选显式请求 GPT：

```bash
# 对指定记录的全部 selected 候选只准备输入，不调用模型
python -m coc_labeling.local_record_labeling \
  --input-root /home/baojiali/Downloads/disk3/20261019_lin_007_out \
  --candidate-root outputs/local_record_candidates \
  --record 20261019153941_record \
  --output-dir outputs/local_record_autolabel_candidates

# 只准备动作区间候选；改成 coverage 可只准备定时覆盖候选
python -m coc_labeling.local_record_labeling \
  --input-root /home/baojiali/Downloads/disk3/20261019_lin_007_out \
  --candidate-root outputs/local_record_candidates \
  --record 20261019153941_record \
  --candidate-type action_span \
  --output-dir outputs/local_action_span_labels

# 先从 segment_candidates.json 选定 candidate ID，再只推理这一项
python -m coc_labeling.local_record_labeling \
  --input-root /home/baojiali/Downloads/disk3/20261019_lin_007_out \
  --candidate-root outputs/local_record_candidates \
  --record 20261019153941_record \
  --candidate-id candidate_003 \
  --output-dir outputs/local_record_autolabel_candidates \
  --run-inference
```

`--candidate-type` 可设为 `action_span`、`coverage` 或 `all`（默认）。它只控制从候选清单中选哪些事件进入标注，不改变 CoC 模型输入内容；选 `action_span` 时 GPT 仍会看到该事件周围图片、位置推导速度和逐图纵向动作提示。每个候选的相机相对时间直接取自候选清单；单事件入口仍按该时间匹配 16 张图片，并用对应 localization 准备运动数据。同一记录的图片时间表和 localization 文件只读取一次，再供该记录的所有候选复用。标注阶段会校验 manifest 记录的历史、未来和抽帧间隔与当前 CoC 配置一致；不一致时会要求重新筛选候选或恢复对应配置。输出结构与手动单事件标注一致，按记录和事件时间写在 `outputs/local_record_autolabel_candidates/<record>/event_<seconds>s/`。可用 `--record` 限定记录，也可用 `--candidate-id` 限定单个或多个候选；所选 ID 必须属于 `--candidate-type` 指定的候选池。添加 `--run-inference` 会对筛选出的每个候选各发送一次 API 请求并产生费用；未加该参数时不会读取 API 凭据。

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
  longitudinal_meta_action.json
  agent_input_preview.json
```

`contact_sheet.jpg` 标出相对事件时间、采样误差和源文件名；`preview.mp4` 是带时间标记的 2 fps、8 秒预览；`frame_manifest.json` 记录选帧、绝对 Unix 时间戳、事件索引和当前状态。

`position_trajectory_10hz.json` 保存事件相对 10 Hz 网格上的 map-XY 位置。每个网格点只会在前后原始定位样本相隔不超过 32 ms 时插值；原始间隔超过 32 ms 的区段会单独列在 `raw_sampling_gaps_over_limit` 中。`position_trajectory.png` 显示 10 Hz 轨迹，红点标出跨越原始大间隔的位置段；图上的 map-XY 轨迹方向不代表车头方向。

`position_motion_audit.json` 记录 `-2.0` 至 `+6.0` 秒、每 0.5 秒一个的 17 个位置差分速度点、姿态、车体速度和有效性。对当前四条记录，数据核对支持将 orientation 解释为车体 FLU 到 map-XY 的旋转；地图速度逆旋转后，`v_long = body_x`、`v_lat = -body_y`（右正）。这项判断来自轨迹方向、规划 `theta` 和前视画面运动，JSON 本身没有坐标系标签；详细依据见 [local_record_coordinate_conversion.md](local_record_coordinate_conversion.md)。此外，四条记录的 `linear_velocity.x/y` 按车体 FLU 的水平分量解释并用 orientation 旋转到 map 后，与位置差分速度明显更一致，因此代码把“`linear_velocity` 水平分量是车体 FLU”记录为强支持的工作假设；z 轴正方向未单独核实。它仍只用于审计，不直接替代位置推导速度。`agent_input_preview.json` 展示实际送入 agent 的输入，`ego_text` 只包含 `v_long` 和 `v_lat`；`v_ground` 仍保留在 `position_motion_audit.json`，不发送给 Agent。姿态缺失时，前两个量显示 `Not Available`。默认 prepare-only 不读取 API 凭据、不初始化模型。

`longitudinal_meta_action.json` 保存 10 Hz 动作点和清理后的动作区间。分类器使用位置推导的车体前向速度 `v_long`，并使用二维地速作为 Stop 的保护条件；它不读取 chassis、`linear_velocity`、`v_lat` 或 lane map。动作词表与 PAI-AV 一致，包括 `Strong/Gentle Acceleration`、`Strong/Gentle Deceleration`、`Maintain Speed`、`Stop` 和 `Reverse`。这些标签是给模型的参考先验，不是控制指令，仍需人工检查。每张 2 Hz 图片从最近的 10 Hz 动作点取纵向标签。港口没有明确 lane graph，因此 `lateral` 和 `lane` 在 Agent 输入中继续写为 `Not Available`。

本地分类器复用 PAI 的动作词表和主要阈值，并按 10 Hz 轨迹做居中平滑、短段清理和最小速度变化检查；它是针对当前 localization 输入写的轻量规则实现，并未直接调用 PAI 的 trajdata/lane 生成流程。初始阈值还需要在更多港口片段上核对。

## 单次 GPT 推理

人工确认预览中的画面顺序和事件位置后，显式加 `--run-inference`：

```bash
python -m coc_labeling.local_record_autolabel \
  --input-root /home/baojiali/Downloads/disk3/20261019_lin_007_out \
  --record 20261019153941_record \
  --event-time-sec 21.6 \
  --run-inference
```

默认模型为 `gpt-6.1-sol`。GPT 配置读取 `~/.config/alpamayo-coc/openai_api.yaml`，字段为 `api_key` 和可选的 `base_url`；也可先设置 `OPENAI_API_KEY`、`OPENAI_BASE_URL` 覆盖配置文件。凭据不会写入输出或日志。

## VS Code 调试

仓库的 `.vscode/launch.json` 也提供 `Local data labeling (single record; debug)`、`Local keyframe auto-select (single record; debug)` 和 `Local record labeling (selected candidates; prepare only)` 三个启动项，对应动作生成、候选筛选和候选预览。需要模型推理时可在命令行选择候选并显式添加 `--run-inference`。启动前按需修改 `--record`、`--candidate-id` 和输入目录，然后在 VS Code 的 Run and Debug 面板选择配置并按 F5。配置使用本机的 `alpamayo-coc` Python 环境。

单次推理输出：

```text
<event>/labels/<record>/cot_<event_unix_timestamp_micros>.yaml
<event>/preview_with_coc.mp4
<event>/contact_sheet_with_coc.jpg
```

YAML 由现有 `VLMCoCAgent` 写出，`preview_with_coc.mp4` 使用 `scripts/overlay_coc_result.py` 叠加描述；`contact_sheet_with_coc.jpg` 在原 4×4 联系表上方显示同一条 CoC 描述。输出状态是 `completed_pending_review`，仍需要人工审核。相同参数再次运行时会覆盖同路径输出；若带 `--run-inference`，每次都会重新请求模型并覆盖该事件的 YAML 和可视化结果。

## 代码结构

整条记录的三个阶段分别由 `local_data_labeling.py`、`local_keyframe_auto_select.py` 和 `local_record_labeling.py` 提供入口：第一步从 localization 生成纵向 meta-action，第二步结合相机图片筛选候选，第三步准备候选的 CoC 输入，并可通过 `--run-inference` 调用 GPT。手动指定单个事件的入口仍是 `local_record_autolabel.py`。

这些入口复用 `src/coc_labeling/recording_autolabel/` 中的实现：`camera.py` 负责相机时间戳和选帧，`localization.py` 负责原始位置、10 Hz 重采样和速度，`actions.py` 负责本地纵向 meta-action，`segment_candidates.py` 负责候选筛选规则，`agent.py` 负责组装并调用 CoC agent，`visualization.py` 负责图片预览，`position_artifacts.py` 负责轨迹与运动审计，`artifacts.py` 负责清单和 agent 输入预览，`types.py` 放共享数据结构和参数。

## 当前限制

当前只实现纵向 meta-action。`v_lat` 是车体瞬时侧向速度，不等同于港口内的横移、斜行或变道方向，因此没有据此生成横向动作；lane 也没有使用地图规则。纵向标签来自规则阈值，可能受定位噪声和低速抖动影响。手动单事件入口要求用户提供事件时间；整条记录的候选筛选和多候选处理由上述独立阶段完成。CoC 描述仍需结合图片并由人工审核。
