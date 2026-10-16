# 对 `09_real_pipeline_map.py` ~ `14_real_pai_one_event.py` 的评审

**评审对象**：`tutorials/09` ~ `14`、`tutorials/_real_workshop_common.py`、`tutorials/README.md` 中关于后半段课程的表述
**参照基准**：`toy_01` ~ `toy_08` 已建立的节奏，以及 rasbt/LLMs-from-scratch 的教学风格
**结论**：内容本身是对的、能跑的、也忠实于源码，但它交付的**不是需求方要的东西**——目标被替换了。

---

## 一、根本问题：目标被悄悄换掉了

需求方对后半段的原话是：

> 希望他能够带着我复现一个简单一点的版本，最后做到能够比较接近源码地构建一个 autolabeler。

关键词是 **复现**、**构建**、**从简单版本逐步收敛到源码**。

而 `README.md` 开头写的是：

> 后半段不再另造一套“相似实现”，而是围绕仓库真实模块解释对象如何流动。

这两句是直接矛盾的。`09`–`14` 实际做的是**逐个文件参观真实 pipeline、import 生产模块、打印中间对象**，即"带我读懂真货"；需求方要的是**"先自己写简化版 → 再和真货对照 → 发现真货多处理了什么"**，即"带我造出来"。

`toy_01`–`08` 之所以让需求方"理解得很舒服"，正是因为它严格遵循后一种节奏：先自己写循环 → 再封装函数 → 每章末尾对照真实实现。`09`–`14` 把"先自己写"这一步整段删掉了，只剩"调用真货 + print"，所以读者会有"我明明理解了 toy，但读真代码还是对不上"的落差感。

**修改方向不是修补细节，而是把后半段的节奏换回复现式。**

---

## 二、具体证据（可逐条核对）

### 1. `10`–`13` 不是复现，是同一个 3 行调用跑了三遍

`11`/`12`/`13` 三课的 `main()` 结构完全同构：

```python
cfg = make_cfg(WORK_ROOT / "one_event.json")
segment_info, vector_loader = create_production_segment(cfg)   # 11:17 / 12:18 / 13:19
# 然后换个字段 print
```

真正属于"你自己写"的只有：

- 第 10 课：在 keyframe 清单里按 `clip_id` + `event_start_frame` 筛一条记录（`10:24-36`）
- 第 14 课：拼一个 subprocess 命令行（`14:77-99`）

其余全是 `import` 生产模块后打印。这与"动手实现"的课程描述（`README.md` 中"课程文件保留了每一步的'动手实现'"）不符。

### 2. 课与课的产物不累积，"顺序推进"的设计没有落地

- `11` 写出 `segment_info.json`（`11:25`）——**没有任何后续课程读它**
- `13` 写出 `message_summary.json`（`13:54`）——**没有任何后续课程读它**

而 `11/12/13` 每次都重新 `create_production_segment()`（`11:17`、`12:18`、`13:19`），每次都重建 trajdata loader（慢）。所以真实情况是"三段互不相干的参观"，不是"分段复现"。README 声称"这一步的产物打开检查，再运行下一步"，但下一课并不消费它，只是重跑一遍同样的东西。

### 3. 最大的洞：`toy_01`–`03` 对应的那一半源码，在 `09`–`14` 里完全消失

`toy_01/02/03` 教的是 **数值 → meta-action → 区间**。它对应的真实实现是 `src/meta_action/`：

- `src/meta_action/processor/longitudinal/{segmentation,classification}.py`
- `src/meta_action/processor/lateral/`、`processor/lane/`
- `src/meta_action/utils/constant.py`（动作空间定义）
- `src/meta_action/post_processing.py`（逐帧动作合并为区间）

对 `09`–`14` 全部文件 grep `processor` / `longitudinal` / `segmentation` / `classification` / `constant.py`：**零命中**。第 09 课的地图 `STAGES`（`09:11-21`）里也没有任何 `src/meta_action/` 条目。

后果：读者从"我理解了动作是怎么算出来的"直接跳到"拿现成的 `all_meta_action_str` 去问 VLM"。**需求方最想复现的那一块（也正是仓库的另一半）被整段跳过了。** 这是本次评审最严重的问题，优先级高于其余所有条目。

### 4. 第 09 课是"文件级"地图，缺"函数级"对照

`09_real_pipeline_map.py` 只做了 `path.exists()` 检查（`09:28`）并列出文件路径。它没有回答读者真正的问题：toy_01 的 `delta speed` 对应真仓库哪个函数、toy_03 的 `make_spans` 对应哪个函数。

此前 `review_deepseek.md` 已指出"缺一张 toy 概念 → 真实文件/函数对照表"，第 09 课本意是补这个坑，但只补到"文件在哪"，没补到"哪一步对应哪个函数"。

### 5. 第 13 课直接调用私有方法

`13_qwen_wrapper.py:31`：

```python
messages = coc_agent._build_input_message(evidence)
```

`_build_input_message` 是下划线开头的内部实现。作为教学这勉强可以接受，但它构成的是"看真货输出 8 条消息"，**没有**"你自己手写这 8 条消息、再对照真货多加了什么"的环节。而消息如何被设计出来，恰恰是 CoC pipeline 的核心。

### 6. 第 14 课把唯一一次真实模型调用塞进 subprocess

前面 13 课都是透明风格（import 真模块、打印中间对象），到最后最关键的一课变成：

```python
subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)   # 14:123
```

整个推理变成黑盒，读者只能等待。恰恰看不到：

- 结构化输出是怎么被约束的（`OutputSchema` → vLLM 结构化解码）
- 结果怎么落盘、落到哪
- `toy_07` 里那个 `{'ego_behavior_schema': {'effect_on_ego_behavior': ...}}` 真实 schema 到底长什么样

而"从 toy 收敛到真货"的最后一步，正是对照这两者。`toy_07_validate_and_save.py:43` 已经写对了真实 schema，但 `09`–`14` 没有任何一课把它和真实产物并排放在一起看。

### 7. 整段硬编码本机绝对路径，不可移植

`_real_workshop_common.py:23-28`：

```python
DATASET_ROOT = Path("/home/baojiali/Downloads/qwen-drive/datasets/pai_reasoning_16")
PREVIOUS_RUN = Path("/home/baojiali/Downloads/qwen-drive/outputs/pai_reasoning_16_official_20261013")
```

并且依赖那次实验遗留的 cache / video / meta_action。这使它只能是"在这台机器上重跑真实 pipeline"，不可能是可移植的教学复现。对当前这位读者（就在这台机器上）也许还能跑，但它与"构建一个 autolabeler"的教学目标无关。

### 8. 缺少收尾对照

需求方学完 14 之后，手里没有一个"toy 版 vs 真实版"的差别清单。`toy_01`–`08` 有意省略了时间戳对齐、文件格式、长 prompt、结构化解码等；这些差异应当在最后一课集中列出并逐条指认到源码位置，否则读者不知道自己的"简化版"和真货差在哪、为什么差。

---

## 三、保留什么

以下素材是有价值的，改版时应当保留复用：

- `10` 课对 keyframe JSON 分桶结构的展示（`{bucket: [records]}`），以及"按 `clip_id` + frame 精确匹配而不是取第一条"的做法
- `12`/`13` 展示的三个 loader 各自贡献哪些字段（`all_fpv_frames` / `all_meta_action_str` / `ego_text`），以及两套时间轴（clipgt 10 Hz vs source video 帧号）的区别
- `13` 展示的真实 8 条消息结构
- `14` 展示的真实 CLI 入口参数与 Hydra override 列表
- `_real_workshop_common.load_production_evidence` 里对 `LabelingAgent.run()` 顺序的复刻（顺序与 `src/coc_labeling/agents/labeling_agent.py:166-247` 基本一致；注意真实代码里 vector 在前、video 居中、meta_action 在后，而 common 里是 vector → meta_action → video，改版时最好对齐）

---

## 四、建议的改版结构

把节奏换成 **"每课手写简化版 → 立刻与真货对照 → 记录真货多处理了什么"**，并补上缺失的 meta_action 半场。每课"自己的实现"的产物要被下一课**真正消费**，而不是各自重跑。

| 课 | 手写什么（不 import 生产模块） | 之后对照什么 |
| --- | --- | --- |
| 09 | toy_08 的 `make_spans` 在真仓库的落点图 | `src/meta_action/processor/*` 的**函数级**对照表（现在只有文件级） |
| 10 | 自己读 meta_action 帧序列、自己按规则判动作 | `meta_action/utils/constant.py` 的动作空间（纵向 7 / 横向 7 / 车道级 9）、5 Hz 判定 |
| 11 | 自己实现 `make_spans` + 事件挑选 | `meta_action/post_processing.py` |
| 12 | 自己写 mini loader 去读同一批文件 | `VectorLoader` / `VideoLoader` / `MetaActionLoader` |
| 13 | 自己手拼 8 条消息 | `coc_agent._build_input_message`（并注明这是私有 API） |
| 14 | 自己直接调 `VLMCoCAgent.run()`（不要用 subprocess 包起来） | 真实 CLI 入口 + 与 `toy_07` schema 并排对照，最后给出"toy vs 真实"差异清单 |

关键差别：每一课读者都亲手写了一段会进入最终 pipeline 的代码，而不是 import 真货来打印。

---

## 五、最低限度改动（如果只做一部分）

按性价比排序：

1. **补 `src/meta_action/` 这一半**（第 3 条）。这是最严重的缺口，价值最高。
2. **把第 14 课的 subprocess 改成直接调用**，让结构化输出与落盘对读者可见；并与 `toy_07` 的 schema 并排对照。
3. **让课与课真正串起来**：`11` 的产物被 `12` 消费，`12` 的产物被 `13` 消费，避免重复跑 loader。
4. **第 09 课补函数级对照表**，把每个 toy 步骤指到具体函数而不是文件。
5. **第 13 课补"自己手写消息"的环节**，再与 `_build_input_message` 对照。
6. **最后一课补"toy vs 真实"差异清单**，逐条指认到源码位置。

改版时如果一次改不完，建议先只重写 09 和 10 两课交需求方确认风格，再继续往后推。
