# 《从零理解 CoC autolabeler》教程评审

**评审对象**：`tutorials/01~08_*.py` + `tutorials/README.md`
**参照风格**：rasbt/LLMs-from-scratch（渐进式、纯标准库可运行、每章末尾小实验）
**验证方式**：8 个脚本全部实际运行通过；逐条对照 `src/`、`docs/`、`README.md` 中的生产实现

## 一、总体评价

**结构和风格是成功的**，可以保留不动：

- 纯标准库、不依赖仓库内部模块/视频/GPU，8 章全部可直接运行，输出正确。
- 渐进式展开到位：先打印原始变量 → 解释动机 → 再封装成函数（第 01→08 章的主线清晰）。
- 每章末尾"小实验"符合 rasbt 的风格；第 02 章的"规则顺序即优先级"演示、第 03 章的"先肉眼写答案再写循环"都是很好的教学手法。

**核心短板**：它教了一个**和真实系统不同的玩具**，但**没有告诉读者哪里不同**。README 结尾说"先理解算法，再阅读生产实现"，却没有给出任何"本章对应真实仓库哪个文件"的锚点。结果读者读真实代码时会大面积对不上。补一张对照表 + 修正第 07 章的 schema，价值会立刻上一个台阶。

---

## 二、必改项（与生产实现的事实偏差）

### 1. 第 07 章的输出 schema 是虚构的（最严重）

教程用 `{ego_behavior, explanation}` 两个字段。真实输出是**一个自由文本句子**：

```yaml
ego_behavior_schema:
  effect_on_ego_behavior: "Nudge left around the parked vehicle while yielding to the pedestrian crossing ahead."
```

出处：`src/coc_labeling/prompts/vlm/coc.py:539-545`、`README.md:436-451`。
这句话本身就是"原因 + 行为"的因果链，这正是 **Chain-of-Causation** 的设计核心。教程把它拆成"行为标签 + 解释"两项，等于把因果链拆散了，与项目前提相反。读者拿着教程去读 `cot_*.yaml` 会完全对不上。

同时第 07 章落盘 `from_scratch_label.json` 也是虚构格式；真实产物是 `experiments/<run_id>/<clip_id>/cot_<timestamp>.yaml`（`utils/general_helpers.py:57-61`）。

### 2. 消息字段名 `kind` 应为 `m_type`

第 05/06/08 章用 `{"role": ..., "kind": "video", ...}`。真实代码用 **`m_type`**，取值 `{text, image, video}`；role 取值 `{system, user, assistant}`。出处：`src/coc_labeling/agents/vlm/coc_agent.py:162`、`vlm_wrappers/cloud.py:226-262`。
教程还特意说"生产代码会把它传给 `infer(json_schema=...)`"来暗示忠实度，字段名就不该是编的。

另外真实消息是 **8 条**（`coc_agent.py:148-180`）：system + images_prompt + video + meta_action_prompt + meta_action 字符串 + ego_state_prompt + ego 文本 + output_prompts。教程只构造了 5 条，且把"指令类"文本简化成一句 `"Explain what happened around t=0."`——而真实 prompt 是 `coc.py:139-594` 约 600 行、含 13 条枚举检查规则和几十条示例。至少要让读者知道真实 prompt 的体量。

### 3. 动作空间砍掉了"横向"这一半

教程只有 4 个纵向类。真实系统（`src/meta_action/utils/constant.py:75-100`）有：

- 纵向 7 类：`GentleAcceleration / StrongAcceleration / GentleDeceleration / StrongDeceleration / MaintainSpeed / Stop / Reverse`
- 横向 7 类：`GoStraight / SteerLeft / SteerRight / SharpSteerLeft / SharpSteerRight / ReverseLeft / ReverseRight`
- 车道级 8+ 类：`LaneKeep / LeftLaneChange / RightLaneChange / SlightlyShiftLeft / SlightlyShiftRight / TurnLeft / TurnRight / FollowCurveLeft / FollowCurveRight`

需注意：

- **没有裸的 `Acceleration`**，只有 `Gentle/StrongAcceleration`；教程把两级减速压成一级，也没有 `Reverse`。
- 带空格的 `Maintain Speed` 确实是真实存在的人类可读展示文本（`data_loader/meta_action_loader.py:59`），但规范名是驼峰 `MaintainSpeed`。
- 真实判定是 **5 Hz**（`STEP = 2`，`constant.py:31-34`），教程是逐帧 10 Hz。
- 真实 CoC 示例里变道、绕行（nudge）、左转占大半（`coc.py:549-590`）——**教程恰好省掉了驾驶中最相关的横向行为**。

### 4. 第 01 章把速度差当"加速度"，但没提除以 Δt

真实代码计算加速度是 `np.diff(speed)/dt`（`processor/longitudinal/segmentation.py:26-54`），10 Hz 下 dt=0.1。
教程用 `speed[i] - speed[i-1]`，得到的是 Δv（单位 m/s），不是加速度（m/s²）。后果是第 02 章的阈值 `-1.0` 看着像有物理含义，其实没有。加一句"真实代码会再除以 Δt，这里省略"即可。

### 5. 第 04 章的事件窗口数字是随机的，且漏掉了"锚点从何而来"

真实窗口（`config/data_loader/default.yaml:17-21`、`video_loader.py:396-478`）：
**历史 2 s / 未来 6 s / 步长 0.5 s**，@10 Hz 抽到 **2 Hz**，共 **16 帧**（4 历史 + 12 未来）——正好对应 Qwen 的 `limit_mm_per_prompt={"video": 16}`（`vlm_wrappers/qwen.py:252`）。
教程的"前 2 帧 + 后 4 帧"（-0.2 s/+0.4 s）应换成这组真实数字。

更重要的是：教程第 04 章直接 `event_start = 3` 任选一帧，**从没说明事件锚点是"meta-action 发生变化的那一帧"**——而这正是整条流水线"决定标注哪里"的原因。教程里"为什么要把逐帧动作合并成区间"（第 03 章）的动机因此是悬空的。

### 6. 第 07 章教的校验模式与真实做法相反

真实系统用**解码期约束**：vLLM `structured_outputs` / guided decoding，以及 OpenAI 的 `.parse(response_format=OutputSchema)`（`qwen.py:76-92`、`cloud.py:121-224`）。agent 里**没有**事后 `model_validate`，直接信任解码结果落盘。
教程花整章教"生成后用 `validate()` 检查再保存"——这是常见教科书模式，但**恰好和这个仓库相反**。建议加一句对比："真实系统在解码时约束输出，而不是生成后再验；两种思路各有取舍。"

### 7. 需要澄清"单次 vs 多阶段"（避免过度声称）

- 真实**生成**默认是**一次** VLM 调用（`coc_agent.py:480-497`）。教程第 06 章的单次 `fake_vlm` 是对的。
- 生成后按横向先验做修复的二次调用（`run_metaaction_check`，`coc_agent.py:499-579`）存在，但由 `use_metaaction_check_pipeline` 控制，**默认关闭**。
- "evidence / behavior / causal 三阶段"**不是生成流程**，而是 `scripts/coc_validation_stage/` 对已产出 `cot_*.yaml` **打分校准**的独立评估流程（2s/6s/8s 三窗口各一次调用）。

如果要在教程里提，请准确表述为"生成是单次调用；另有一套独立的离线评估流程"，不要写成"生成也是多阶段"。

---

## 三、建议改项（教学清晰度）

1. **缺一张"toy 概念 → 真实文件/函数"对照表**。这是性价比最高的一项：每章末尾加一行 `→ src/coc_labeling/xxx.py:yyy`，读者才知道该去读哪里。
2. **术语不统一**：教程叫 "action"/"事件"，真实叫 **meta-action** / **keyframe** / **segment**。建议早期就引入真实术语，避免读者二次映射。
3. **第 06 章的 `fake_vlm` 完全不看 video 消息**（只看 text）。它自己承认了，但这削弱了第 05 章"video 是证据"的论点。可让 `fake_vlm` 断言 video 消息存在，体现视频是必需输入。
4. **章与章玩具格式不一致**：第 05 章是 `t=+0.0s action=Strong Deceleration` / `speed=8.0 m/s`；第 08 章变成 `t=+0.0s Strong Deceleration` / `speed=8.0`（丢了 `action=` 和 `m/s`）。第 08 章自称"重新组合前面步骤"，格式却对不齐。
5. **第 03 与第 08 章口径不一**：第 03 章 `interesting` 含 Stop/Acceleration 且要求 `duration >= 2`；第 08 章只挑 `Strong Deceleration`、无时长过滤。读者会不知道哪个是标准。
6. **建议加一个"先看终点"的第 00 章**：先贴一份真实的 `cot_*.yaml`，再说"我们要从零把它重建出来"。rasbt 会先把最终产物摆出来再回溯。
7. **第 04 章注释与代码差一帧**：注释说"取前两帧和后四帧"，`range(event_start-2, event_start+4)` 实际是前 2 + 锚点 + 后 3。

---

## 四、工程小问题

- `tutorials/_outputs/from_scratch_label.json` **未被 gitignore**（`.gitignore` 只忽略了 `__pycache__`）。整个 `tutorials/` 目前未跟踪，`git add` 会把生成物一起提交。建议加 `tutorials/_outputs/`。
- 第 07 章 `Path("tutorials/_outputs/...")` 是相对路径，只在仓库根目录运行才对；从 `tutorials/` 内运行会生成 `tutorials/tutorials/_outputs/`。建议改成基于 `Path(__file__).parent`。
- `README.md` 写 `python tutorials/01_...`，但本机 `python` 不在 PATH（要走 `conda activate alpamayo-coc` 或用 `python3`）。文档其他处都强调 conda 环境，这里最好统一。

---

## 五、可直接照抄的真实值对照表

| 概念 | 真实值 | 出处 |
|---|---|---|
| 加速度 | `np.diff(speed)/dt`，10 Hz → dt=0.1 | `processor/longitudinal/segmentation.py:26-54` |
| 动作判定频率 | 5 Hz（`STEP = 2`） | `meta_action/utils/constant.py:31-34` |
| 纵向动作 | `GentleAcceleration/StrongAcceleration/GentleDeceleration/StrongDeceleration/MaintainSpeed/Stop/Reverse` | `constant.py:75-83` |
| 横向动作 | `GoStraight/SteerLeft/SteerRight/SharpSteerLeft/SharpSteerRight/ReverseLeft/ReverseRight` | `constant.py:84-92` |
| 事件窗口 | 历史 2 s / 未来 6 s / 步长 0.5 s，@10 Hz → 2 Hz，共 16 帧 | `config/data_loader/default.yaml:17-21`、`video_loader.py:396-478` |
| 帧数佐证 | Qwen `limit_mm_per_prompt={"video": 16}` | `vlm_wrappers/qwen.py:252` |
| 消息条数 | 8 条 | `agents/vlm/coc_agent.py:148-180` |
| 消息字段 | `m_type` ∈ {text, image, video}；role ∈ {system, user, assistant} | `coc_agent.py:162`、`cloud.py:226-262` |
| 输出 schema | `{ego_behavior_schema: {effect_on_ego_behavior: <自由文本因果句>}}` | `prompts/common.py:25-32`、`prompts/vlm/coc.py:539-545` |
| 生成调用次数 | 默认 1 次（可选 +1 次横向修复，默认关闭） | `coc_agent.py:480-497`、`:499-579` |
| 保存路径 | `<save_root>/<clip_id>/cot_<event_start_timestamp>.yaml` | `utils/general_helpers.py:57-61` |
| 校验方式 | 解码期结构化约束（structured decoding / `.parse`），非事后 validate | `qwen.py:76-92`、`cloud.py:121-224` |

---

## 六、建议的最小改动清单（按性价比排序）

1. README 加一张 **toy 概念 → 真实文件/函数** 对照表，每章一句 `→ src/...:line`。
2. 第 07 章改用真实 schema `{ego_behavior_schema: {effect_on_ego_behavior: "<一句话>"}}`，并说明它是**自由文本因果句**；顺带点出真实校验是**解码期约束**。
3. 第 05/06/08 章把 `kind` 改成 `m_type`，消息条数补齐到 8 条的说明。
4. 第 01 章补"真实代码会再除以 Δt"。
5. 第 02 章补横向/车道级动作、Gentle/Strong 分级、Reverse、5 Hz 的说明。
6. 第 04 章改用真实窗口 2 s/6 s/0.5 s → 16 帧，并补"锚点 = meta-action 变化帧"。
7. 修正 `tutorials/_outputs/` 的 gitignore、第 07 章相对路径、以及第 03↔05↔08 的格式不一致。
