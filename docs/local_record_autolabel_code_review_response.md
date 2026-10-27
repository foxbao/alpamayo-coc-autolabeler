# 本地记录自动标注代码评审回复

感谢对 `local_record_autolabel.py` 和 `recording_autolabel/` 的逐项检查。本轮按建议优先处理
入口 re-export、重复计算和重复状态处理，目标是减少重复逻辑，同时保持现有标注输入和审计
结果不变。

本轮只改了本地记录适配器、相机/定位辅助模块、manifest 写入模块和对应测试。评审中提到的
`src/coc_labeling/recording_autolabel/agent.py` 是本项目为本地记录新增的输入适配器；上游 CoC
Agent 实现在 `src/coc_labeling/agents/vlm/coc_agent.py`，本轮没有修改该文件。

## 已采纳

### 1. 收窄 CLI 的 re-export

测试现在从各自的实现模块导入定位、相机、Agent、可视化和产物函数；`local_record_autolabel.py`
的 `__all__` 只保留 CLI 入口 `main`。CLI 仍保留自己的参数解析与流程函数，领域实现不再经由
CLI 模块转发。

评审还指出 `agent.py` 和 `artifacts.py` 跨模块导入了 `camera.py` 的私有时间函数。本轮将
`_event_timestamp` 和 `_timestamp_micros` 改为 `event_timestamp` 和 `timestamp_micros`，让它们
的模块间用途与命名一致。

### 2. 复用定位括区和连续性判断

`sample_position_at()` 现在复用 `_sample_bracket()` 来处理边界、最大采样间隔和插值比例，避免
维护两份相同的二分查找逻辑。

原 `_ensure_continuous_position_interval()` 只被调用方放进 `try/except`，并把“是否连续”编码成
异常与否。本轮改为 `position_interval_is_continuous()`，直接返回布尔值，两个调用处据此设置
`connected_to_previous`。超出覆盖范围、请求区间倒置或遇到过大采样缺口都返回 `False`。

### 3. 只构造一次 10 Hz 运动序列

本地 CoC 的 2 Hz 运动时刻与默认 10 Hz 网格重合。本轮先构造 10 Hz `action_motion`，再按目标
时间抽取 CoC 所需样本；没有使用固定的 `[::5]` 索引。如果将来 CoC 时刻不落在 10 Hz 网格，
代码会回退到原先针对 CoC 时刻单独构造运动样本的路径。

抽取时会汇总两个 CoC 时刻之间所有 10 Hz 小区间的 `connected_to_previous`。因此 CoC 样本上的
该字段仍表示整个 CoC 间隔是否连续，例如 0.5 秒，而不是误用 10 Hz 的 0.1 秒语义。

用 `20261019153941_record` 的 `21.6s` 事件逐字段比较旧的 2 Hz 构造结果和新抽取结果，17 个
样本的所有字段（包含连续性标记）完全一致。

### 4. 合并动作标签扫描和 dataclass 更新

`actions.py` 新增 `_label_runs()`，供短区间清理、速度变化量检查和动作 span 汇总共同使用，
消除三处重复的连续标签扫描。两个标签调整步骤现在用 `dataclasses.replace()` 更新 `label`，
不再逐字段重建 `LongitudinalActionPoint`。

### 5. 一次构造 manifest 和运动统计

准备阶段不再先调用 `_write_frame_manifest()` 写文件、随后读回、补字段再写一次。现在由调用方
收集附加路径和统计，再通过 `extra_fields` 一次传入并写入 manifest。有效样本数各计算一次，
无效数用总数减有效数得到；终端摘要复用同一份 body-velocity 计数。

## 暂缓处理

- `write_position_motion_audit()` 同时生成审计 JSON 和轨迹图，确实较长。本轮没有拆它，以便将
  纯可读性拆分与行为调整分开；`trajectory_samples=None` 分支目前仍由现有测试覆盖，暂时保留。
- `build_position_motion_samples()` 中的可选采样异常处理、临时 `TimedPosition` 和未使用的中心
  位置返回值仍可继续整理，但比本轮的重复运动序列构造涉及更多内部接口，留待单独评估。
- `build_agent_data()` 的 17 个样本约束、预览视频 2 fps 和个人机器的 `DEFAULT_INPUT_ROOT` 仍与
  默认配置耦合。本轮没有改变这些默认行为；建议下一轮单独决定可配置接口和兼容方式。

本轮保留了 Decimal 时间戳、定位缺口限制、坐标系来源说明、审计字段和模块拆分，没有删减
可追溯性信息。

## 验证结果

- `pytest -q`：38 passed。
- 修改涉及的 Ruff 检查：通过。
- `git diff --check`：通过。
- 对上述实际事件逐字段比较旧版 CoC 运动样本与 10 Hz 抽取结果：17/17 完全一致。
- prepare-only 流程成功生成预览、轨迹审计、动作审计和 Agent 输入清单；manifest 的有效/无效
  样本数正确，未调用 GPT。

测试环境只出现 PyTorch 对 `pynvml` 的弃用提示，不影响测试或准备流程。
