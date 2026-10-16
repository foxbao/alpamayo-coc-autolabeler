# 对 17–21 新教程的评审（第六轮）

**评审对象**：新增的 `tutorials/17`–`21`，以及本轮修订的 `10`、`11`、`12`、`13`、`README.md`
**验证方式**：对照 `src/` 源码逐条核对；**实跑 `17`–`21`**（均 EXIT=0）并手算复核 demo 输出
**结论**：17–21 的**技术准确度很高**，我核对的每一条阈值、公式、调用路径都对得上。**主要问题不在"对不对"，而在"放在哪里"**——17–21 全是**上游**（meta-action 生成阶段）的加深课，却排在 15/16（整条链的**下游终点**）之后，读起来像"回锅"而非难度递增。

---

## 〇、先更正我自己的错误

**`README:90` 与 `13` docstring 关于 `post_processing` 的改动是对的，我在第四、五轮说错了。**

我之前写"`post_processing.process_batch` 做平滑/消抖/补洞"。源码不是这样：

```python
# src/meta_action/post_processing.py:199
# Directly write tags without additional cleaning/merging/filling.
```

`git log` 显示该文件长期未改（最后一次改动是 `4dda809`），所以这是我的记述错误，不是版本差异。真正的清理（平滑、滞回、短段合并）在动作通道内部：`processor/longitudinal/segmentation.py` 与 `data_structures/ego_meta_action.py`。

作者这处更正很准，而且已经同步改到 `13` 的 docstring（"第 10、11、17、18 课简化了前一部分"）。**两处一致，没有留下矛盾。** 上一轮评审的 A 项（把上游接上）方向仍然正确，只是我对其**行为的描述**是错的。

---

## 一、17–21 逐课核对（全部通过）

| 课 | 关键断言 | 源码依据 | 结论 |
| --- | --- | --- | --- |
| **17** | 居中滑动平均 | `segmentation.py:85-96`（`rolling(window, center=True, min_periods=1)`） | ✅ |
| 17 | 正阈值 `T_on=T*(1+f)`、`T_off=T*(1-f)`；初始状态用基准阈值 | `segmentation.py:98-133` | ✅ 逐行一致 |
| 17 | `hysteresis_fraction=0.15`、`accel_smoothing_window=5` | `segmentation.py:61-62` | ✅ |
| 17 | 尾 3 tick 不切新段；切段后跳过 3 tick | `segmentation.py:165,173` | ✅ |
| 17 | 还比较**速度**阈值；非加速度列用普通 `>=`（无滞回） | `config.py:19-24`（speed ±0.3、accel ±0.1/±1.25）；`segmentation.py:148-149` | ✅ |
| 17 | demo：直接阈值切换 4 次 → 平滑+滞回 2 次；切段 `[(0,2),(2,10)]` | 手算复核 | ✅ |
| **18** | `clean_longitudinal_segments`；先并同类 → 清短段 → 再并同类 | `ego_meta_action.py:123-215` | ✅ |
| 18 | Δv 相加、平均速度按时长加权、平均加速度按新时长重算 | 与源码同款公式 | ✅ |
| 18 | 两侧同类 → 三段合一（Case A） | `prev_cap == nxt_cap` 分支 | ✅ |
| 18 | 默认 8 tick、MaintainSpeed 12 tick | `processor/longitudinal/config.py:30-31` | ✅ |
| 18 | demo：`dv = 0.5-0.1+0.5 = 0.9`、`avg_spd=5.2`、`avg_accel=0.409` | 手算复核 | ✅ |
| **19** | 横向 `0.0015 rad/m` + `0.015 rad` | `lateral/classification.py:54-55,66-67` | ✅ 数值完全一致 |
| 19 | lane 用 `lane_decision` + `lane_centerline_id_changed` | `lane/classification.py:367-390,586` | ✅ |
| 19 | `LaneKeep` 命名 | `lane/classification.py:586` `TAG = "LaneKeep"` | ✅（`13` 里的 `KeepLane` 才是别名） |
| 19 | 纵向 `accel>=0.15`、`dv>=0.8`、`Stop: avg_speed<=0.12` | `longitudinal/classification.py:28-31,87` | ✅（上界见第二节） |
| **20** | 过滤条件：动作白名单 + `duration>=min_duration` + `min_start_ts<=start<=max_start_ts` | `keyframe_auto_select.py:106-110` | ✅ 三个条件逐字对应 |
| 20 | 超量按 duration 从长到短保留 | `balance_action_records` `:220-250` | ✅ |
| 20 | 内置动作不足保留全部、非内置动作不足报错 | `:228-250`、`RESERVE_ALL_ACTIONS = set(ALL_META_ACTIONS)` `:58` | ✅ |
| 20 | `DEFAULT_MIN_DURATION = 10` | `:62` | ✅ |
| 20 | 输出 `all` / `sampled` 两份 | `:334-335`（`segments_relative_timestamp_all.json` / `..._sampled.json`） | ✅ |
| **21** | `timestamp = clip_start + frame*dt`；`clipgt_index` 反算 | `segment_generators.py:692,135` | ✅ |
| 21 | "最近邻"找视频帧 | `video_loader.py:634-643`（`use_nearest=True`，closest） | ✅ |
| 21 | demo：`ts=1700000011400000`、`index=114`、`video frame=342` | 手算复核 | ✅ |

五个 demo 的输出全部手算复核通过。**`17` 尤其好**：它把 `config.py` 里"2 个速度阈值 + 4 个加速度阈值，且只有加速度走滞回"这个容易漏的结构讲清楚了。

---

## 二、几处小偏差（都很轻，但既然"讲源码"就该说清）

### 1. `19` 的 `classify_longitudinal` 少了 GentleAcceleration 的**上界**

生产要求 `average_acceleration < 1.4`（`longitudinal/classification.py:29` 的 `MaxThreshold`）。toy 只判 `>= 0.15`，因此 `accel=2.0, dv=1.0` 时 toy 返回 `GentleAcceleration`，生产返回 **`StrongAcceleration`**。

同时 toy 是 **4 类**，生产是 **7 类且带优先级**
（`ORDERED_ACTIONS = [Reverse, Stop, StrongAcceleration, GentleAcceleration, StrongDeceleration, GentleDeceleration, MaintainSpeed]`，`classification.py:104-112`）。

横向 toy 漏了 `SharpSteerLeft/Right`、`ReverseLeft/Right`；lane toy 漏了 `SlightlyShiftLeft/Right`、`TurnLeft/Right`、`FollowCurveLeft/Right`。

`10` 已经写了"生产动作还包括 Reverse、横向动作和车道级动作；这里先只做纵向分类"，但 `19` 没有同等披露。**补一行即可**（`19` 结尾已有的"注意：本例只演示三个通道并行"可以扩成一句）。

### 2. `20` 的 `BUILT_IN_ACTIONS` 是 3 元素替身

生产的 `RESERVE_ALL_ACTIONS = set(ALL_META_ACTIONS)`，是**全部 ~23 个 canonical 动作**。toy 用了 `{"gentle_acceleration","strong_deceleration","stop"}`。语义方向对（"内置保留全部、自定义报错"），但读者会以为内置只有这 3 种。建议在注释里写明"生产是全部 canonical 动作，此处只取 3 个做演示"。

### 3. `21` 的 `make_candidate` 用正则转 snake_case

`re.sub(r"(?<!^)(?=[A-Z])", "_", action).lower()` 会把 `LaneKeep` 转成 `lane_keep`，
而生产的 `canonicalize_action` 给的是 `keep_lane`。示例只用 `GentleAcceleration`，所以没暴露；
但这一课的主题恰恰是"对象转换"。建议点一句"真实转换走 `canonicalize_action`，本 toy 用正则近似"。

### 4. `17` 的 `cut_with_production_guard` 把多阈值元组压成单个 bool

生产是 `current_states = tuple(...)` 多阈值联合判据（`segmentation.py:158-161`）。toy 用单个 bool 演示切段冷却，行为能对上，且隐含说明了是演示。**无需改。**

---

## 三、结构与编排（本轮核心）

### 3.1 难度不是变陡，是"掉头"

17–21 比 15/16 **更简单**：纯 stdlib、不加载模型、不读数据。真正突兀的是**位置**——读者刚在 `15` 用 35B 拿到真实 YAML、在 `16` 跑通生产入口，然后被送回 `centered_rolling_mean` 和玩具列表。而且这与 README 自述的弧线冲突：

> README:3 —— "…**最后**把一个真实 PAI 事件交给仓库的生产 loader、`VLMCoCAgent` 和本地 Qwen3.5-35B"

这条弧线在 **15/16 就结束了**，17–21 不在弧线上。

### 3.2 17–21 其实是"上游加深课"，各自都有天然的家

| 新课 | 它加深的是 | 该贴的位置 |
| --- | --- | --- |
| 17 平滑/滞回 | `10`/`11` 的 `segment_longitudinal` | `10` 之后 |
| 18 短段清理 | `11` 的"分类之后" | `11` 之后 |
| 19 三通道 | `13` 的"三字段"来源 | `11` 之后 / `13` 附近 |
| 20 keyframe 过滤+平衡 | `12` | `12` 之后 |
| 21 span→candidate→SegmentInfo | `12`→`13` 的接缝 | `12` 与 `13` 之间 |

换句话说：**`10`/`11` 教的是纵向算法的简化版，`17`/`18` 正是它的"生产保真版"补丁**；**`20` 正是 `12` 用 `ACTION_TO_BUCKET` 一笔带过的那个真实筛选**；**`21` 正是 `12`/`13` 之间缺的那块接缝**。它们不是新话题，是**补洞**。

### 3.3 两个改法（推荐 A）

**A（推荐，改动小）**：位置不动，但在 README 里**明确写成"回顾加深"**，并给出配对关系与可选读序：

> 17–21 是回头加深课，不参与 `messages.json` 这条链。它们分别补 `10`/`11`/`12`/`13` 的生产细节；**可以按主题就近阅读**：读完 `10` 就读 `17`，读完 `11` 读 `18`/`19`，读完 `12` 读 `20`/`21`，再进入 `13`。

**B（曲线单调，改动大）**：把运行顺序排成
`09, 10, 17, 11, 18, 19, 12, 20, 21, 13, 14, 15, 16`，
让每课紧挨它加深的父课，难度单调上升。代价是要重编号（且前几轮 review 是按旧号写的）。

### 3.4 另外两点

- **README:3 / README:52-58 的"后半段每课会读取上一课保存的 JSON"对 17–21 不成立**——它们不写任何产物。加半句"10–14 传递 JSON；17–21 是独立小实验"即可。
- **`19` 的位置最尴尬**：它夹在 `18`（纵向清理）和 `20`（keyframe）之间，与两边都没有共享线索，是纯粹的概念章。若采用 A，建议在 README 里点明它是"三通道总览"。
- **没有收尾课**：系列停在 `21` 的最后一句 print。`toy_07` 当年是 validate+save 收尾；可考虑让 `21` 结尾回指 `09` 打开的那份 YAML，把整条弧线合上。

---

## 四、待办清单

1. `19` 给 `classify_longitudinal` 补上 `1.4` 上界，并说明 toy 是 4 类 / 生产是 7 类带优先级；横向 / lane 的漏项也提一句。
2. `20` 注释里说明 `BUILT_IN_ACTIONS` 是演示替身，生产是全部 canonical 动作。
3. `21` 点明真实转换走 `canonicalize_action`。
4. **README 给 17–21 加一句定位**："回头加深课、不参与产物链、可按下表就近阅读"（配 3.2 的对应表）。
5. （可选）README:3 或 :52 修掉"每课读取上一课 JSON"的泛指。
6. （可选）`21` 结尾回指 `09` 的 YAML，收束整条弧线。

第 1–3 条都只需改注释/print；第 4 条是本轮最关键的一处，成本极低。

---

## 五、给作者的话

- 这轮**没有发现事实性错误**，17–21 的源码对齐做得很扎实；`17` 和 `18` 尤其好。
- 唯一需要我"更正自己"的是 `post_processing` 那一条（见第〇节），作者改得对。
- 主要建议是**编排**：17–21 是好课，只是站错了队。按第 3.3 节的 A 方案加一段 README 定位即可，不必重编号。
