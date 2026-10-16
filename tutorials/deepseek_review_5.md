# 对 09–16 教程修订版的评审（第五轮）

**评审对象**：`tutorials/09`、`13`、`15`、`16`、`README.md`（`10`/`11`/`12`/`14` 本轮未改）
**验证方式**：对照 `src/` 源码；**实跑** `09` 与 `16`；把 `16` 的产物与 `09` 打开的真实 YAML、`15` 的真实 Qwen 产物、`13`/`14` 的中间产物逐项对账
**结论**：第四轮点名的 **A–E、H、I 全部修好，且逐条验证通过**；新增的 `16` 能真正跑通。链的**头（上游）和尾（生产入口）都接上了**。仅剩两条上轮已标"可选/次要"的未动，另引入两个很轻的小瑕疵。

---

## 一、已核对改对（附实测证据）

| 项 | 现在的做法 | 我的验证 |
| --- | --- | --- |
| **A 上游断链** | `09` 对照表新增 `meta_action/data_labeling.py:main/process_clip`、`post_processing.py:process_batch`；`13` docstring 明说该 txt 是 `post_processing` 平滑/消抖/补洞后的产物；`README:80` 补全 | `process_clip` 确在 `data_labeling.py:58`；`09` 实跑通过 |
| **B 生产入口** | 新增 `16`，走 `LabelingAgent → VLMFactory` | **实跑 `16`，EXIT=0**，产出 `production_dummy/run_<stamp>/05ee29be-.../cot_11400000.yaml` |
| **C `09` 打开 YAML** | 现在真的解析并打印 | 实跑输出：`event_start_frame: 114`、`coc_prompt_version: coc_autolabeling_release`、`final_content: {ego_behavior_schema: {...}}`、8 条消息字符数 `[304, 181, 1504, 506, 1464, 151, 1125, 49970]` |
| **D 闭环对账** | `15:120-131` 用 `result["prompt"]` 与 `messages.json` 对骨架 | `VLMCoCAgent.run` 确实返回 `(messages, output_dict)`（`coc_agent.py:480-497`），解包正确 |
| **E `SegmentInfo`** | `09` 对照表 + `13` 两条新 print + `README:82` | `segment_generators.py:47` 存在 |
| **H 对照表完整度** | 扩到 17 条 | **17 个目标逐个验证全部存在**（含 `processor/lateral`、`processor/lane`、`prompts/common.py:OutputSchema`、`agents/vlm/factory.py:VLMFactory`、`labeling_agent.py:__init__/parse_dataset/run`） |
| **I 批量 / 双分支** | `16:83-84` print + `README:88` | 与 `labeling_agent.py:162,196-198` 一致 |

### 本轮两个高含金量的旁证

修订顺带证明了两件事，说明教程不只是"看起来对"：

**1. `16` 的输入 JSON 格式与生产完全一致。**
把 `16` 生成的 `production_single_segment.json` 与 `keyframe_auto_select.py:151-157` 的 `write_action_json`（`asdict(ActionRecord)`）对比，并与真实清单
`.../keyframes/segments_relative_timestamp_sampled.json` 里的记录逐字节比较：

- 顶层键：bucket 名，值是该 bucket 的记录列表 ✅
- 记录字段：`meta_action / clip_id / event_start_frame / event_end_frame / duration` ✅（与真实记录完全相同）
- 选中的记录与真实文件里那条一模一样 ✅

**2. 手搓链与生产入口产出的 prompt 逐字节等价。**
`16` 的 dummy YAML 消息字符数 `[304, 181, 1504, 506, 1464, 151, 1125, 49970]`，与 `09` 打开的预存 YAML、以及 `15` 的真实 Qwen 产物**完全相同**；三者的 `event_start_timestamp` 都是 `11400000`。
即"手搓协调器 == 真的那套"，这是最强的等价性证明，而且是免费得到的。

### `13` / `14` 产物与生产 prompt 的数值对账

- **msg 4（meta-action）**：16 行，t = -2.0 … +5.5，逐行**完全一致**
  （Strong Acceleration ×4 → Gentle Acceleration ×6 → Maintain Speed ×6）。
- **msg 6（ego）**：17 点，t = -2.0 … +6.0，**`max abs diff = 0.0000`**（heading 投影正确）。
  仅格式不同：教程是每采样点一行 `(t, v_long, v_lat)`，生产是 `Longitudinal` / `Lateral` 两条方括号列表。属教学简化，数值无误。

---

## 二、本轮新发现的小问题（都很轻）

### 1. `15` 与 `16` 的对账不对等 —— 建议把对账挪到 `16`，或两边都做

现在是"手搓课（`15`）做了对账，生产课（`16`）反而没做"。但上面那个"逐字节等价"的结论恰恰只能由 `16` 给出。建议在 `16` 末尾加一个最便宜、也最有说服力的断言：**读回自己刚写的 YAML，打印 8 条消息的字符数**，让它与 `09` 打印的那串数字对齐。读者一眼就能确认"换入口 ≠ 换结果"。

### 2. `README:82` 的指代错了

> 生产 `ParsedSegmentGenerator` 会把记录转换成 `SegmentInfo(...)`；**它读取 timestamp sidecar 找到对应源视频帧**。

读 sidecar 的**不是** `ParsedSegmentGenerator`，而是 `VideoLoader.get_index_from_timestamp()`（由 `labeling_agent.py:186-192` 调用）。`SegmentInfo` 只携带 `clipgt_index` 和微秒 `timestamp`，两者都不是源视频帧号。建议把这句拆开写，否则正好在最容易搞混的"两套时间轴"上讲错了主体。

### 3. `16:41` 的 override 是冗余的

`data_loader.keyframe.segment_generator_type=json` 本就是默认值（`base_config_vlm_rel_ts.yaml:48`）。留着无害，也有"显式"的好处；如需精简可去掉（print 里那句 parquet 说明可保留）。

---

## 三、仍未处理的（上轮已标"可选/次要"）

### F. `12` 的 `ACTION_TO_BUCKET` 人为桥接仍在

可顺手补一个很扎实的事实：真实清单其实有 **23 个 bucket**，其中
`steer_left: 13`、`steer_right: 12`、`sharp_steer_right: 2` 是**非空**的——即"横向动作有数据，只是生产故意不喂 VLM"；而 `left_lane_change / right_lane_change / keep_lane / turn_left / turn_right` 等 lane bucket 才是真的 0 条。这正好与 `README:86` 的说法互为印证，写进 `12` 一句即可。

### G. `toy_04` 的"事件窗口"概念仍悬空到 `13` 才回来

影响很小，可不改。

---

## 四、待办清单

1. `16` 末尾读回自己的 YAML，打印 8 条消息字符数（可选：与 `09`/`15` 的数字对照）。
2. `README:82` 拆开"谁读 timestamp sidecar"的主体（`VideoLoader`，不是 `ParsedSegmentGenerator`）。
3. （可选）`12` 补一句 bucket 全貌与 lateral/lane 的区别。
4. （可选）`16:41` 去掉冗余 override。
5. （可选）把 `toy_04` 的窗口概念在 `13` 开头回指一句。

第 1、2 条都可直接用 `16` 的 dummy 输出复现验证，不需要 GPU，也不需要重新加载 Qwen。
