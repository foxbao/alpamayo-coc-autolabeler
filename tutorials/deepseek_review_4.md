# 对 09–15 教程的**结构 / 编排**评审（第四轮）

**评审对象**：`tutorials/09`–`15`、`tutorials/README.md`
**评审角度**：这一轮**不看单个函数的对错**（前三轮已逐行核过数值），只看教学链的**骨架**——结构上缺了什么、编排上哪里不顺
**验证方式**：逐课对照 `src/` 源码，并用 `grep` 统计关键生产组件在教程中的出现次数
**结论**：**横向已打通**（单事件：keyframe → 真实 loader → 8 条消息 → 真实模型 → YAML，每环都能和源码对上）；**纵向还断着**——上游那一环（10/11 教的算法怎么变成 13 读的文件）和收尾那一环（手搓版怎么变成生产入口）都缺。

---

## 一、结构性缺失

### A. 上游断链：10/11 教的算法，和 13 读的文件之间没有连接 ⚠️ 最重要

`10`/`11` 实现的是 `src/meta_action/processor/longitudinal/` 的算法；`13:68` 直接读

```python
path = RUN / "meta_action/final_outputs" / f"{clip_id}.txt"
```

**但这个 txt 的生产者，正是"用 10/11 那套规则算出来的东西"**：

| 步骤 | 位置 |
| --- | --- |
| CLI 入口 | `src/meta_action/data_labeling.py:245`（`main()`） |
| 逐 clip 推理 | `process_clip` → `TemporalScenario.get_tag_motions` |
| 写原始结果 | `save_result_clip` → `tmp/raw_results/<clip_id>.json`（`data_labeling.py:40,100`） |
| 后处理（平滑/消抖/补洞） | `post_processing.process_batch`（`data_labeling.py:312`、`post_processing.py:207`） |
| 最终产物 | `final_outputs/<clip_id>.txt`（`data_labeling.py:278`） |

**实测**：教程目录内 `data_labeling`、`post_processing`、`SegmentInfo`、`VLMFactory` 全部 **grep 零命中**；`09` 的对照表也没有这两行。

**后果**：读者会以为那个 txt 是数据集自带的，而它其实是"上一阶段用 10/11 的规则跑出来的"。这恰好是 `toy_03` → 真实 `post_processing.py` 的落点——第一轮评审时 `post_processing.py` 还在对照表上，现在整条链上消失了。

### B. 结尾没有"换成真货"的那一步

- `15:47-85` 的 `load_real_event()` 是 `LabelingAgent.run()` 循环体的**手抄**；`15:101` 直接 `VLMCoCAgent(...)`。
- 生产路径是 `data_labeling.main()` → `LabelingAgent`（`__init__` 里建 `VLMFactory`）→ `parse_dataset()` → `run()`；`VLMFactory.run()` 再按 `use_metaaction_check_pipeline` 分流到 `run` / `run_metaaction_check`（`vlm/factory.py:71-75`）。
- 教程里 `LabelingAgent` **只出现两次，都不是"使用"**：
  - `15:48` 一句 docstring（`Mirror LabelingAgent.run(): ...`）
  - `13:219` 一句 print
- `VLMFactory` 零命中；仓库根 README 给的标准命令 `python -m coc_labeling.data_labeling ...` 从未被跑过。

**后果**：读者的目标是"比较接近源码地构建一个 autolabeler"，但他最后停在了一个**手写复制品**上，没跨过最后一步。最有价值的一课应该是："把 15 手搓的协调器换成 `LabelingAgent`，跑同一个事件，得到同样的 YAML"——那才是"我理解了整个 codebase"的瞬间。顺带，不展示 `VLMFactory`，读者永远不知道还有 `run_metaaction_check` 这条二次横向修复路径（默认关闭）。

**关键事实（让这一步变得很便宜）**：`MODEL_WRAPPER_REGISTRY` 里有 `"dummy": DummyWrapper`（`model_clients/vlm_wrappers/factory.py:26`），且 `create_model_wrapper` 按 `model_name` 查表（同文件 `:35-48`）。因此可以

```bash
model_name=dummy   # 走通整条生产协调链（生成器 → loaders → agent → YAML），不需要 GPU
model_name=qwen3.5_35b   # 换回真实推理
```

**这是最干净的教学收尾**：先证明"生产入口能跑通"，再证明"它能产出真标签"。

### C. `09` 的"先看终点"是空转

`09:19-21` 只 **print 了路径**和一句概括，**从没打开它**：

```python
sample = REAL / "experiments/coc/05ee29be-.../cot_11400000.yaml"
print("最终产物:", sample)
print("核心字段：事件时间、提示消息、结构化 final_content，以及可追溯的 prompt 版本。")
```

真正的"先看终点"应该**解析**这个 YAML：打印 8 条消息各自的长度、`final_content`、`coc_prompt_version`。这样既兑现钩子，又给 `14` 的"消息骨架 / 体量"提前埋下伏笔。

而且这是**别人预存的** YAML（`experiments/coc/...`），读者后面自己跑出来的是另一份，**结尾也没有回到它做对照**。

### D. 缺闭环对账：自己的产物从未被读回

`toy_07` 的收尾是 validate + save；`09`–`15` 重造了一条链，却**没有任何一课读回自己的产物**。`15` 跑完写出 YAML，但没有一步"打开它，和你的 `messages.json` 逐条对比"。这恰恰是 toy→real 最有价值的一次对账，现在只能靠读者自觉。

### E. `SegmentInfo` 这个"接缝对象"缺失

生产里 `ParsedSegmentGenerator` 产出 `SegmentInfo`（定义在 `data_loader/segment_generators.py:47`），`LabelingAgent.run()` 再把它拆成 base dict，并在 `labeling_agent.py:169-171` 专门写了注释：

```python
# NOTE this is not actually the frame number from the video.
# It is an interval count at 10Hz from the start of the clipgt data.
"event_start_frame": segment_info.event_start_clipgt_index,
```

教程 `SegmentInfo` 零命中：`12` 直接读原始 keyframe record，`13` 自己算 timestamp。**两套时间轴是全项目最容易搞混的一点**，教程用注释讲了，却没让读者看到承载它的那个类型。

---

## 二、编排上可以更顺的地方

### F. `10`/`11` 在 toy 域，`12` 突然跳到真实域，而且是人为桥接

`10`/`11` 跑在合成速度序列上；`12` 用 `ACTION_TO_BUCKET` 把 toy 动作名映射到真实 bucket（`README:68` 也承认这是"桥接"）。

更顺的接法：`12` 直接拿 `13` 要读的那个**真实 meta-action txt**，按 `keyframe_auto_select.py:89-115` 的 `parse_txt_file` 逻辑（action 集合 + `min_duration` + `start_ts` 范围）筛出区间，再决定选哪条 keyframe。这样"`10`/`11` 的算法 → 真实文件 → keyframe"是连贯的，也顺带补上 `keyframe_auto_select` 的平衡采样（`balance_action_records`，现在只用一句 print 带过）。

### G. `toy_04` 的"事件窗口"概念悬空了三课

`toy_04` 讲"以事件为零点、前 2s 后 6s"；但真实窗口直到 `13`（写成 clipgt 帧号）和 `14`（写进消息）才再出现，中间 `10`/`11`/`12` 都在讲"**怎么找到**这个事件"。不算错，但知识点搁置太久。

### H. `09` 的对照表不完整

现在缺：

| 缺失项 | 应指向 |
| --- | --- |
| 上游后处理 | `src/meta_action/data_labeling.py`、`src/meta_action/post_processing.py`（见 A） |
| lateral / lane 的 processor | `src/meta_action/processor/lateral/`、`processor/lane/`（数据集里明明有 `SteerLeft`/`SteerRight`/`SharpSteerRight`，表里只列了 longitudinal） |
| schema | `src/coc_labeling/prompts/common.py:OutputSchema`（`toy_07` 专门讲过 schema） |
| agent 工厂 | `src/coc_labeling/agents/vlm/factory.py:VLMFactory` |
| 协调器入口 | `labeling_agent.py` 只列了 `run`，没列 `__init__` / `parse_dataset`（生成器和三个 loader 正是在那里建的） |

### I. 两处"只发生一次"的机制没有交代

- **生产是批量的**：`LabelingAgent.run()` 遍历 `segment_info_list`，单条失败就 `continue`（`labeling_agent.py:162,196-198,239-241,253-255`）。教程全程单事件，值得一句话点明。
- **`segment_generator` 有 parquet / json 两条分支**（`labeling_agent.py:71-91`，配置项 `data_loader.keyframe.segment_generator_type`）。教程只用 json，读者读到这里会困惑。

---

## 三、建议的最小改动（按性价比排序）

1. **`09` 对照表补两行**：`meta_action/data_labeling.py`、`post_processing.py`；并在 `13` 开头加一句"**你正在读的 txt，是上一阶段用 10/11 那套规则算出来的**"。（几乎零成本，补上最关键的断链 A）
2. **给 `15` 加收尾小节，或新增 `16` 课**：用 `model_name=dummy` 跑一遍真正的 `LabelingAgent`（走 `VLMFactory`）得到 YAML，再换回 `qwen3.5_35b` 跑真实推理。（把"手搓复制品"换成"生产入口"，是整条链的终点 B）
3. **`09` 真正打开那个 YAML**：解析并打印 8 条消息长度 + `final_content`。（C）
4. **`15` 末尾读回自己产出的 YAML**，和 `messages.json` 逐条对比。（闭环对账 D）
5. （可选）把 `12` 改成用真实 txt 走一遍 `parse_txt_file` 式筛选，去掉人为桥接。（F）
6. （可选）`09`/`13` 补 `SegmentInfo` 与两套时间轴；README 补"批量 + 单条失败 continue"一句。（E、I）

---

## 四、给作者的两点说明

- 这轮**没有**推翻前几轮的结论；消息骨架、证据维度、数值对齐都保持正确。要改的是**链的头和尾**。
- 第 1 条（对照表补行 + `13` 开头一句话）成本极低、收益最大，建议优先做。
- 第 2 条（`dummy` 跑通生产入口）是本次评审的核心建议：它让教程从"我复刻了一份"变成"我跑通了真的那一份"，而且不需要额外 GPU 时间就能验证结构正确。
