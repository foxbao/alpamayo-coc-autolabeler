# 对 `09`–`15` 重构版的评审（第二轮）

**评审对象**：`tutorials/09_target_and_source_map.py` ~ `15_direct_real_vlm.py`、`README.md`
**验证方式**：实际运行 `09`–`14`，并用 `15` 产出的真实 YAML（`_outputs/rebuild/real_result/run_20261014_033133/05ee29be-.../cot_11400000.yaml`）逐条对齐 `14` 手写的 `messages.json`
**结论**：结构问题已经修好，**这次评审的重点是"信息保真度"**——输入给 LLM 的证据被过度简化，其中一处不是简化，而是偏离了生产设计。

---

## 一、先确认哪些改对了

- 课与课真正串起来了：`11` 读 `action_rows.json`、`12` 读 `events.json`、`13` 读 `one_event.json`、`14` 读 `evidence.json`。中途可以停下来打开文件。
- `meta_action` 半场补上了：`10` 自己算 Δv/加速度/逐帧动作，`11` 自己合并区间、选事件，并在末尾指明 `prepare_longitudinal_data` / `segment_longitudinal` / `classify_chunk`。
- `14` 手写 8 条消息；`15` 不再用 subprocess 包起来，改成直接调用生产 loader 和 `VLMCoCAgent.run()`。
- 消息骨架是对的：`15` 产出的 YAML 里 `prompt` 正好 8 条，role / `m_type` / 顺序与 `14` 手写的完全一致（对照 `coc_agent.py:148-193`）。**这个骨架可以保留，不需要改。**

下面的意见都是在这个正确骨架之内，关于"每一条消息里装了多少信息"。

---

## 二、真实 prompt vs 教程 prompt：逐条体量对比

教程 `14_handwrite_messages.py` 产出 `messages.json`；真实产出是 `15` 的 YAML。对齐后：

| # | 段 | 真实 | 教程 | 关键差异 |
| --- | --- | --- | --- | --- |
| 0 | system | 304 | 31 | 真实含"必须严格按 schema 输出 JSON" |
| 1 | images_prompt | 181 | 40 | 真实含 **2 Hz / 8 s / 前 2 s / 后 6 s** |
| 2 | video | 16 张真实图像数组 | 16 个占位字符串 | 见第五节（可接受的简化） |
| 3 | meta_action_prompt | 506 | 30 | 真实解释 `lane` 字段的用法与边界 |
| 4 | meta_action 内容 | 1464 | 215 | 形态不同，见第三节 |
| 5 | ego_state_prompt | 151 | 31 | 真实含横向符号约定 |
| 6 | ego 内容 | 1125 | 543 | 维度少一半，见第四节 |
| 7 | output_prompts | **49970** | **58** | 决策体系整段缺失，见第二节 |

**合计约 53.7 KB → 1.0 KB（约 2%）**。第 7 条单独就占了真实 prompt 的 93%。

---

## 三、`output_prompts` 是最严重的一处：50 KB 压成一句话

真实第 8 条 = `question_prompt + general_prompt + general_example`（`prompts/vlm/coc.py:592`、`:164`、`:547`）。内容包含：

- **封闭决策表**：
  - 纵向 7 类：`Stop / Yield / Pass-Overtake / Gap-search / Keep Distance / Adapt Speed / Resume Speed`（各带定义）
  - 横向 6 类：`Lane Change / Nudge / Turn / Merge / Split / Keep Lane`
- **优先级策略**（`coc.py:220` 起）：interesting 决策优先；lane change / nudge 最高优先，出现就必须写进输出；不许降级成 `Keep Lane` / `Resume Speed`；Gap-search 的多条强制触发条件……
- **一个完整示例**（`general_example`，约占全文件的 93%）

教程 `14_handwrite_messages.py:49` 把它换成：

```python
"Return one causal sentence using the required JSON schema."
```

**后果**：读者会以为"模型只是自由生成一句因果句"，而真实任务是**在一张带优先级的封闭决策表里做选择，且必须遵守十几条枚举检查规则**。这基本改变了这个任务的性质理解。

而且 `09` 课已经让读者看过最终 YAML 的 `final_content`，却没让他们瞄一眼真正占 93% 的那一段——这个"先看终点"的机会现在也还没用上。

**建议**：不必抄全文。加一张体量表（就是上面这张），再摘 3 段真货各几行：决策分类表、优先级策略开头、`general_example` 开头。让读者知道那句"一句话"背后是什么。

---

## 四、meta_action 证据：不是变短，是变成了另一种数据（且有一处与生产相反）

真实第 4 条（`meta_action_loader.py:383-404`）是**逐采样帧一行、2 Hz、16 行、每行三通道**：

```text
Meta-actions by sampled timestamp (relative to event, unit: seconds):
(t=-2.0s) longitudinal: Strong Acceleration, lateral: Not Available, lane: Not Available
(t=-1.5s) longitudinal: Strong Acceleration, lateral: Not Available, lane: Not Available
...
```

教程产出的是**6 条区间、单通道**（`messages.json` 第 5 条）：

```text
t=-3.4s..+0.0s action=StrongAcceleration
t=-3.1s..-1.0s action=SteerRight
...
```

三个问题：

### 1. `lane` 通道整条消失

`lane`（`Left Lane Change` / `Slightly Shift Left`）正是真实 `meta_action_prompt`（`coc.py:146-155`）专门用一整段强调的 "high-value hints for lateral behavior"，也是第 8 条里一大片 lane-change / nudge 规则的输入依据。教程里完全没有它，读者看不到"横向先验从哪来"。

### 2. 教程喂进了生产刻意排除的信号（**这一条不是简化，是偏离**）

核对 `meta_action_loader.py:273-276`：

```python
if meta_action in meta_action_longitu:
    frame_meta_action[frame_index]["Longitudinal"] = meta_action_str
# if commented out, currently we do not feed this signal into VLM
# elif meta_action in meta_action_lateral:
#     frame_meta_action[frame_index]["Lateral"] = meta_action_str
elif meta_action in meta_action_lane:
    frame_meta_action[frame_index]["Lane"] = meta_action_str
```

**lateral 通道被显式注释掉，原文写着 "currently we do not feed this signal into VLM"。** 而教程的 `evidence.json` / `messages.json` 里赫然有 `SteerRight` / `SteerLeft` / `GoStraight` —— 这些全是 lateral 动作。

也就是说：生产只喂 `longitudinal + lane`，教程却只喂了 `longitudinal + lateral`，**恰好各喂错了另一半**。这一条请优先修。

顺带一提，"生产只喂两个通道、而且注释里点明了原因"本身是个很好的教学点，值得在课里讲出来。

### 3. raw token 没有规范化

真实 loader 会把 raw token 规范化成展示文本（`GentleAcceleration` → `Gentle Acceleration`，见 `meta_action_loader.py:220-233`）。教程直接输出原始 token。建议对齐。

---

## 五、ego 证据：横向维度被开根号消掉了

真实第 6 条（`vector_loader.py:139-203`）是**纵向 / 横向两条序列 + 符号约定句**：

```text
Ego speed time series (relative to event time t=0.0s):
- Longitudinal (m/s): [(t=-2.0s, v_long=4.292 m/s), (t=-1.5s, v_long=5.128 m/s), ...]
- Lateral (m/s): [(t=-2.0s, v_lat=-0.004 m/s), (t=-1.5s, v_lat=0.007 m/s), ...]
Positive lateral speed means rightward motion, negative means leftward motion.
```

教程 `13_mini_loaders.py:116` 用的是：

```python
"speed_magnitude_mps": math.hypot(vx, vy),
```

**横向速度序列正是判断变道 / nudge 的证据，而 CoC 任务的核心就是区分纵向与横向行为**——`hypot` 把最需要的那一维直接消掉了。代码注释里它自己承认这是 "toy 特征"，但从没意识到砍掉的是任务主维。

另外真实序列含 `t=+6.0s`（共 17 点）；教程用 `scene_ts < event_frame + 6*FPS` 开区间，少了最后一点（16 点）。

---

## 六、两段"解码键"被换成了纯标签

- 真实 msg 1 告诉模型 **2 Hz / 8 s / 前 2 s / 后 6 s**（`coc.py:85-92`）；教程 `14:40` 是 `"Analyze the video around event time t=0."`。这不是省修饰，是把读者第 13 课刚算出来的窗口语义从 prompt 里删掉了。
- 真实 msg 3 解释 `lane` 字段怎么用（含 "do not treat them as always correct" 的边界），msg 5 说明符号约定；教程换成了 `"Meta-actions around the event:"` / `"Ego motion evidence around t=0:"` 两个纯标签，等于删掉了数据的读法。

---

## 七、可以接受的简化（但建议补 3 行）

`video` 用文字占位——**这个我认为合理**，因为 `15` 课会做真的。但读者看完 `14` 不知道真实的 video content 是 16 张 JPEG 拼成一段序列、`fps=2`、`limit_mm_per_prompt={"video": 16}`（`model_clients/vlm_wrappers/qwen.py:245-252`、`:492`）。在 `14` 末尾补三行即可。

---

## 八、顺带发现的小问题

1. `13_mini_loaders.py:74` 注释写"未来包含 t=0，共 12 个点"，但代码 `range(-60, 180, 15)` 实际产生 **16** 个点。注释错了，代码是对的（16 与真实帧数一致）。
2. `13_mini_loaders.py:57` 的 `event_frame * 100_000` 只在 clipgt 起点为 0 时成立。本例恰好为 0（`15` 课算出同为 `11400000`，对上了），但这不是通用公式；`15` 用的 `get_clip_timing_info` 才是通用路径。可在 `13` 输出里标注"此处恰好成立"。
3. `15_direct_real_vlm.py:111` 手工拼 `save_root / clip_id / f"cot_{timestamp}.yaml"` 来猜路径，而真实规则在 `utils/general_helpers.py:57-61` 的 `get_vlm_yaml_path()`。建议直接调用该 helper，或使用 `process_results` 的返回值，否则规则一变就会找不到文件。

---

## 九、修改优先级

1. **修 `13`/`14` 的 meta_action 通道**：lateral → lane，按真实形态改成逐帧 2 Hz × 三通道，并说明 lateral 在生产里被注释掉。这是唯一一条"偏离设计而非简化"的问题。
2. **`14` 补 `output_prompts` 的体量与结构**：加体量表，摘决策分类表 / 优先级策略 / 示例开头各几行。
3. **`13`/`14` 的 ego 改成 `v_long` / `v_lat` 双序列 + 符号约定句**，并补 `t=+6.0s` 采样点。
4. 补回 `2 Hz / 2s / 6s / 8s` 窗口说明到 msg 1。
5. `14` 末尾补 3 行说明真实 video content 的形态。
6. 修第八节的三处小问题。

改完之后，读者手里的"简化版 autolabeler"和真实 pipeline 之间就只差"体量"和"边界处理"，而不再差"信息维度"和"任务定义"。
