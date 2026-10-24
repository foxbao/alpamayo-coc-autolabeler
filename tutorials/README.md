# CoC autolabeler 教程

这套教程分成两层。`toy_01`–`toy_08` 只用很小的 Python 程序建立直觉；第 09 课开始，逐步靠近仓库里的真实 meta-action、keyframe、loader、VLM 和生产入口。最后可以把一个真实 PAI 事件交给本地 Qwen3.5-35B。

第 09–21 课已经按数据流重新编号。主线是：先得到动作区间，再选真实事件，接着读取视频、轨迹和 meta-action，最后构造消息并调用模型。第 11、13、14、16、17 课是主线旁边的源码加深课；它们解释主线中容易被一笔带过的实现细节，但不会额外接收上一课的 JSON。

`deepseek_review.md` 到 `deepseek_review_6.md` 写于这次重编号之前，其中的课号指旧编号；`deepseek_review_7.md` 起使用当前编号。

## 学习顺序

| 课程 | 你会亲手做什么 | 对照生产代码 |
| --- | --- | --- |
| [toy_01](toy_01_numbers_and_differences.py) | 从速度序列计算逐帧 Δv | `prepare_longitudinal_data` |
| [toy_02](toy_02_rules_to_actions.py) | 按优先级把速度变化变成 action | `classify_chunk` |
| [toy_03](toy_03_actions_to_spans.py) | 合并连续动作并筛选候选事件 | `segment_longitudinal`、`parse_txt_file` |
| [toy_04](toy_04_event_window.py) | 以事件为 t=0，取前后历史 | `MetaActionLoader`、`VideoLoader` |
| [toy_05](toy_05_multimodal_messages.py) | 把文本和视频组织成多模态消息 | `VLMCoCAgent._build_input_message` |
| [toy_06](toy_06_fake_vlm.py) | 用假模型模拟一次结构化输出 | `VLMCoCAgent.run` |
| [toy_07](toy_07_validate_and_save.py) | 验证嵌套 schema 并保存结果 | `OutputSchema`、`process_results` |
| [toy_08](toy_08_full_pipeline.py) | 把 toy 的全部步骤串起来 | 整条生产流程 |
| [09](09_target_and_source_map.py) | 从最终 YAML 反向认识源码地图 | `LabelingAgent` 到 Qwen wrapper |
| [10](10_compute_actions_from_speed.py) | 计算 Δv、加速度和逐帧动作 | `prepare_longitudinal_data`、`classify_chunk` |
| [11](11_smoothing_and_hysteresis.py) | 观察平滑、滞回和切段保护 | `segment_longitudinal` |
| [12](12_make_spans_and_events.py) | 合并 span、汇总 chunk 并选事件 | `segment_longitudinal`、`classify_chunk` |
| [13](13_clean_short_segments.py) | 合并同类段并清理短段 | `clean_longitudinal_segments` |
| [14](14_three_action_channels.py) | 区分纵向、横向和车道动作 | longitudinal/lateral/lane processors |
| [15](15_select_real_keyframe.py) | 从 PAI 清单选一条真实事件 | `write_action_json`、`ParsedSegmentGenerator` |
| [16](16_filter_and_balance_candidates.py) | 按动作、时长、起点筛选并平衡候选 | `keyframe_auto_select.py` |
| [17](17_span_candidate_segment_info.py) | 看懂 span、candidate、`SegmentInfo` 的接缝 | `ActionRecord`、`SegmentInfo` |
| [18](18_mini_loaders.py) | 读取真实动作、视频和轨迹，拼证据 | `VectorLoader`、`VideoLoader`、`MetaActionLoader` |
| [19](19_handwrite_messages.py) | 把证据组织成 8 条多模态消息 | `VLMCoCAgent._build_input_message` |
| [20](20_direct_real_vlm.py) | 调用真实 loaders 和本地 Qwen | `VLMCoCAgent.run` |
| [21](21_production_entry_dummy.py) | 用 dummy 验证生产协调器的单事件流程 | `LabelingAgent`、`VLMFactory` |
| [22](22_direct_gpt_api.py) | 从本地配置文件读取凭证并调用 GPT API | `OpenAIWrapper`、`OutputSchema` |

## 从头运行

先在仓库根目录激活环境：

```bash
conda activate alpamayo-coc
```

先运行 `toy_01`–`toy_08`。然后按下面的顺序运行不加载大模型的课程：

```bash
python tutorials/09_target_and_source_map.py
python tutorials/10_compute_actions_from_speed.py
python tutorials/11_smoothing_and_hysteresis.py
python tutorials/12_make_spans_and_events.py
python tutorials/13_clean_short_segments.py
python tutorials/14_three_action_channels.py
python tutorials/15_select_real_keyframe.py
python tutorials/16_filter_and_balance_candidates.py
python tutorials/17_span_candidate_segment_info.py
python tutorials/18_mini_loaders.py
python tutorials/19_handwrite_messages.py
python tutorials/21_production_entry_dummy.py
```

如果当前 shell 中没有 `python`，可以使用环境里的绝对路径 `/home/baojiali/anaconda3/envs/alpamayo-coc/bin/python`。

第 20 课会加载约 35B 参数的本地模型并占用 GPU，单独运行：

```bash
python tutorials/20_direct_real_vlm.py
```

第 21 课直接创建生产 `LabelingAgent` 处理单个事件，没有运行 `data_labeling.py` 的完整 Hydra 命令行入口。模型名设成 `dummy` 后不会加载 GPU 模型；`DummyWrapper` 会忽略消息和 `OutputSchema`，固定返回占位文本。因此它验证数据加载、消息构造和 YAML 落盘，不验证真实推理或输出 schema。真实模型输出看第 20 课。

第 22 课沿用第 20 课的单事件流程，只把模型后端换成 GPT API。它从
`~/.config/alpamayo-coc/openai_api.yaml` 读取 `api_key`、`base_url` 和 `model`，
在当前 Python 进程内设置凭证，不需要手动 `export`。先复制
`tutorials/openai_api.example.yaml` 并填写本地 key，再运行：

```bash
/home/baojiali/anaconda3/envs/alpamayo-coc/bin/python \
  tutorials/22_direct_gpt_api.py
```

也可以指定其它配置文件：

```bash
/home/baojiali/anaconda3/envs/alpamayo-coc/bin/python \
  tutorials/22_direct_gpt_api.py --config /path/to/openai_api.yaml
```

本课会清除当前进程中的 NVIDIA cloud credential 环境变量，确保使用配置文件里的
标准 OpenAI-compatible endpoint；它不会打印 API key。第一次运行只处理第 15 课选中的一个
事件，调用会产生 API 费用，结果保存到 `tutorials/_outputs/rebuild/real_result/`。

## 主线中的中间产物

主线会在 `tutorials/_outputs/rebuild/` 下生成这些文件：

```text
action_rows.json
    -> spans.json / chunks.json / events.json
    -> one_event.json
    -> evidence.json
    -> messages.json
```

这里的箭头表示数据流，不表示每一课都必须读取上一课的文件：第 10 课写出动作行，第 12 课写出 spans/chunks/events，第 15 课选真实事件，第 18 课写出 evidence，第 19 课写出 messages，第 20/21 课使用生产对象。第 11、13、14、16、17 课是独立的源码加深实验。

## 每个阶段和源码的对应关系

第 09 课先打开一份真实 YAML，列出事件时间、prompt、`final_content` 和源码位置。这样后面的 toy 代码都有明确的目标。

第 10 课先用逐帧规则建立直觉。生产代码在 10 Hz 轨迹上按 `STEP=2` 形成 5 Hz 区间，再用区间的平均加速度、Δv 和平均速度分类。第 11 课展示生产为什么先对加速度做居中 rolling mean，再用进入/退出两个阈值做滞回，并在切段后跳过 3 个 tick、保护尾部短段。

第 12 课把连续相同 action 合成右开区间 `[start, end)`，同时计算生产 `classify_chunk` 所需的 `delta_velocity`、`average_acceleration` 和 `average_speed`。教学代码为了直观，先逐帧分类再合并；生产顺序是先分段，再对每个 chunk 分类。第 13 课演示 `EgoMetaAction` 中“先合并同类段、吸收短段、再合并”的过程。平滑、滞回和短段清理都发生在动作通道内部；`meta_action.post_processing.process_batch` 目前只负责把结果映射到 clip 时间范围、排序并格式化输出，不再额外做平滑、消抖或补洞。

第 14 课说明纵向、横向、车道是三条并行证据。同一时刻可以同时得到纵向 `GentleAcceleration`、横向 `SteerLeft` 和车道 `LeftLaneChange`。toy 覆盖纵向生产的 7 类：`Reverse`、`Stop`、`Strong/GentleAcceleration`、`Strong/GentleDeceleration`、`MaintainSpeed`；横向只演示 `SteerLeft/Right/GoStraight`，车道只演示 `LaneKeep/LaneChange`。生产还包括 `SharpSteer*`、`Reverse*`、`SlightlyShift*`、`Turn*` 和 `FollowCurve*`。显示标签是 `LaneKeep`，loader 和 keyframe 筛选使用的 canonical key 是 `keep_lane`。

第 15 课只把 toy 的动作类别用来选择真实数据 bucket，toy 帧号不会冒充真实事件帧。第 16 课进一步拆开生产 keyframe 选择器的三个过滤条件：动作白名单、最短时长和起始帧范围；随后按动作类别限量，超量时优先保留较长区间。toy 的 `BUILT_IN_ACTIONS` 只放 3 个动作以便演示；生产的 `RESERVE_ALL_ACTIONS = set(ALL_META_ACTIONS)` 覆盖全部约 23 个 canonical longitudinal、lateral、lane 动作。生产同时保存过滤后的 `all` 清单和限量后的 `sampled` 清单。

第 17 课把三个容易混淆的对象并排打印：meta-action 的 `ActionSpan`、keyframe 选择器保存的 `ActionRecord`，以及 `ParsedSegmentGenerator` 交给 loader 的 `SegmentInfo`。10 Hz clipgt index、微秒 timestamp 和源视频帧号属于不同时间轴；真实转换还会通过 `normalize_meta_action`/`canonicalize_action` 把 `LaneKeep` 归一为 `keep_lane`。

第 18 课读取本机 PAI 产物。keyframe 的 10 Hz clipgt 帧不能直接当作 30 FPS 视频帧：生产 `SegmentInfo` 保存事件时间，`VideoLoader` 再通过 timestamp sidecar 找最近的源视频帧。速度投影给出车体瞬时横向速度，约定 `v_lat` 右正；它不能代替车道相对位移或变道方向。第 19 课用这些摘要构造 8 条消息，图像位置用文字占位，正式 prompt 仍由生产 `VLMCoCAgent` 从配置加载。

第 20 课调用真实 `VectorLoader`、`VideoLoader`、`MetaActionLoader` 和 `VLMCoCAgent`，把一个 PAI 事件交给本地 Qwen3.5-35B。模型输出必须符合 `OutputSchema` 的嵌套结构，但结构正确不等于场景判断正确，仍需要人工审核视频、轨迹和动作证据。第 21 课把同一事件交给生产协调器 `LabelingAgent`，验证单事件的数据编排和保存流程：

```text
ActionRecord -> SegmentInfo -> LabelingAgent -> loaders -> VLMCoCAgent/DummyWrapper -> YAML
```

第 21 课不运行真实模型；dummy 固定返回 `Dummy response.`，也不会验证 `OutputSchema`。教程直接调用 `LabelingAgent`，没有覆盖 `data_labeling.py` 的 Hydra 作业配置、恢复和分组批处理入口。

第 22 课的变化只有模型后端：`model_name=gpt5.5` 让 `VLMFactory` 使用
`OpenAIWrapper`。前面的事件、时间戳、视频帧、meta-action、messages 和
`OutputSchema` 都保持不变。

正式批量标注可以使用仓库提供的 `scripts/run_coc_labeling.sh`。它集中填写了本机
PAI 数据路径，支持 `qwen3.5_35b` 和 GPT 后端；GPT 模式会在脚本内部读取
`~/.config/alpamayo-coc/openai_api.yaml`，不需要手动 `export`：

```bash
scripts/run_coc_labeling.sh --model qwen3.5_35b --exp-name pai_qwen35_full
scripts/run_coc_labeling.sh --model gpt-6.1-sol --exp-name pai_gpt61_full
```

第一次正式运行建议先用 `--filter '[gentle_acceleration]'` 做小范围检查。完整运行会
处理 keyframe 清单中的全部事件；GPT API 会按图片和文本输入量收费。脚本默认不保存
裁剪视频，并默认去重相同 clip/event frame，结果仍写入 `experiments/`。

最后可以回到第 09 课重新查看这份 YAML。

## 本机数据和模型

课程默认使用此前准备好的本机文件：

- PAI 数据集：`/home/baojiali/Downloads/qwen-drive/datasets/pai_reasoning_16`
- trajdata cache：`/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013/cache/trajdata`
- 提取的视频：`/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013/video`
- meta-action 文本：`/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013/meta_action/final_outputs`
- keyframe 清单：`/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013/keyframes/segments_relative_timestamp_sampled.json`
- 模型别名：`qwen3.5_35b`
- 模型缓存：`/home/baojiali/.cache/alpamayo-coc-models`

如果路径在另一台机器上不同，请修改对应教程顶部的常量和 Hydra override；不要把数据集、模型权重或凭据提交到仓库。`_outputs/` 已加入 `.gitignore`，运行产物不会混进源码变更。
