# 对 `13` / `14` / `15` / `README` 修订版的评审（第三轮）

**评审对象**：`tutorials/13_mini_loaders.py`、`14_handwrite_messages.py`、`15_direct_real_vlm.py`、`README.md`
**验证方式**：逐文件对照 `src/` 源码，并用 `15` 真实产出的 YAML
（`_outputs/rebuild/real_result/run_20261014_033133/05ee29be-af7b-4fd6-853f-96a49f0d6bc7/cot_11400000.yaml`）
与实际生成的 `_outputs/rebuild/evidence.json` 做数值比对
**结论**：上一轮点名的三项主要问题全部改对，消息骨架未动，方向正确。**本次修订新引入一处偏差（meta-action 采样多一个点）**，另有一处上一轮未追到底的老问题（`v_lat` 未做航向投影）。

---

## 一、核对确认改对的部分

| 项 | 教程现在的写法 | 源码依据 | 结论 |
| --- | --- | --- | --- |
| meta-action 形态 | 逐采样点 `longitudinal` / `lateral` / `lane` 三字段 | `meta_action_loader.py:383-404`（`process_meta_action`） | 一致 |
| lateral 固定 `Not Available` | 并有两行注释解释原因 | `meta_action_loader.py:273-276`（`# currently we do not feed this signal into VLM`） | 一致，且把教学点讲出来了 |
| lane 通道保留 | 用 `LANE` 集合识别 | `utils/constant.py:93-103`、`meta_action_loader.py:276` | 结构对（命名见第四节 D） |
| msg 1 窗口语义 | `2 Hz / 8 s / 前 2 s / 后 6 s` | `prompts/vlm/coc.py:85-92` + `config/data_loader/default.yaml`（hist 2 / fut 6） | 完全一致 |
| msg 5 符号约定 | “Positive lateral means rightward …” | `prompts/vlm/coc.py:157-162`、`vector_loader.py:201` | 一致 |
| msg 7 决策词表 | 13 项 | `prompts/vlm/coc.py:176-219`（纵向 7 + 横向 6） | 名称逐一对得上 |
| msg 7 优先级 | “interesting 优先、lane change/nudge 不得降级” | `prompts/vlm/coc.py:220-233` | 一致 |
| `15` 落盘路径 | 改用 `get_vlm_yaml_path()` | `utils/general_helpers.py:57-61` | 修掉了猜路径 |
| `13` 视频注释 | 改成 “-2 到 +5.5 秒的 16 帧” | 与生产一致 | 修掉了 |

`README` 新增说明也准确：`cut-in`(20 处) / `construction`(29) / `intersection`(28) 均存在于 `coc.py`；
`output_prompts` 实测 49970 字符 ≈ 49 KB，“约 50 KB” 成立。

消息条数与帧数已核对：真实 YAML 的 `prompt` 为 8 条，视频帧 16 张，与 `14` 手写的骨架完全一致。

`09`–`12` 本轮未改动，符合预期。

---

## 二、本轮新引入的问题：meta-action 采样多了一个点

`13_mini_loaders.py:93`：

```python
for offset in range(-2 * FPS_CLIPGT, 6 * FPS_CLIPGT + 1, 5):
```

`range(-20, 61, 5)` → **17 个点**，t = -2.0 … **+6.0**。

生产的采样是（`meta_action_loader.py:345-370`）：

- 历史：`range(hist_start_frame, event_start_frame, 5)` → 4 点，t = -2.0 ~ -0.5
- 未来：`range(event_start_frame, fut_end_frame, 5)`，其中 `fut_end_frame = event + 6*10 = event+60`
  → 12 点，t = 0.0 ~ **+5.5**
- 合计 **16 点**

**实测确认**：真实 msg 4 为 **16 行**；教程生成的 `evidence.json` 中 `meta_actions` 为 **17 行**。

**更直接的问题**：教程内部自相矛盾了。它自己的视频采样（`13:138`）
`range(source_frame - 2*30, source_frame + 6*30, 15)` 终点为 +5.5 秒、16 帧，与生产一致；
但 meta-action 现在跑到 +6.0。同一段 8 秒窗口，视频给到 +5.5、动作给到 +6.0，
读者会困惑哪个才是标准。

**修法**：去掉 `+ 1`，即 `range(-2 * FPS_CLIPGT, 6 * FPS_CLIPGT, 5)`，回到 16 点（-2.0 … +5.5）。

---

## 三、上一轮未追到底的老问题：ego 的 `v_lat` 不是生产用的那个量

上一轮指出“ego 横向维度被 `hypot` 消掉”。本轮确实把维度加回来了，但加回来的是 **feather 的原始 `vy` 列**，而生产用的是**按航向投影后的横向速度**，两者不是同一个量。

教程 `13:177-179`：

```python
# PAI Feather 的 vx/vy 在本例可直接作为纵向/横向速度教学字段。
"v_long_mps": vx,
"v_lat_mps": vy,
```

生产并非直接取列，而是做了航向投影（`data_loader/trajdata.py:157-170`）：

```python
ego_hist_longitudinal_speed = ego_hist_vel[:, 0] * cos_heading_hist + ego_hist_vel[:, 1] * sin_heading_hist
ego_hist_lateral_speed      = -ego_hist_vel[:, 0] * sin_heading_hist + ego_hist_vel[:, 1] * cos_heading_hist
```

即把 `(vx, vy)` 按**每帧 heading** 旋转到车体坐标系。

**实测比对**（`05ee29be-af7b-4fd6-853f-96a49f0d6bc7 @ frame 114`）：

| t | 生产 `v_long` | 教程 `v_long` | 生产 `v_lat` | 教程 `v_lat` |
| --- | --- | --- | --- | --- |
| -2.0s | 4.292 | 4.2923 ✅ | **-0.004** | **-0.0308** ❌ |
| +6.0s | 8.732 | 8.7320 ✅ | **+0.001** | **+0.0650** ❌ |

纵向对得上，是因为这条 clip 近乎直线行驶（feather `heading ≈ -0.006 rad`），投影近似恒等。
横向对不上——差 8~65 倍，因为生产那一项里含 `vx * cos(h)` 的贡献，而教程报的是未投影的全局 `vy`。

数值上两者在这条直线 clip 里都 ≈ 0，定性结论（直行）不受影响；但只要换成一条转弯 clip，
教程的 `v_lat` 会明显偏大甚至反号。既然这一课的教学价值正是“把横向证据喂给 VLM”，这一维值得做准。

顺带一提，代码注释里的“在本例可直接作为”其实也站不住：**本例的 `v_lat` 就已经不是同一个数了。**

**修法**：feather 本来就有 `heading` 列
（列名：`agent_id, scene_ts, key.clip_id, key.label_class_id, rel_time_seconds, x, y, z, heading, length, width, height, type, source, vx, vy, ax, ay`）。
照 `trajdata.py:157-170` 那两行做一次旋转即可，并在注释里点明“生产在 `trajdata.py` 做的就是这个投影，
教程不再直接使用原始列”。上面那张表可以直接当自测基准。

---

## 四、两处小遗留

### C. 10 课未提 5 Hz

`toy_02` 里有一句“真实 meta-action 判定以 5 Hz 处理”，但 `10` 课没有。
生产的 `STEP = 2`、`DELTA_TIMESTAMP = 0.1`（`meta_action/utils/constant.py:26-31`）
意味着 clipgt 是 10 Hz、动作评定是 5 Hz。`10` 课开头写了“本课使用 10 Hz toy 数据”，
补半句“生产在 10 Hz 数据上按 STEP=2、即 5 Hz 评定”会更连贯，也与 `toy_02` 呼应。

### D. `LANE` 集合命名，以及“本数据集根本没有 lane 动作”

`13:52-62` 用的是 `KeepLane`，而 `constant.py:93` 的 temporal 类名是 `LaneKeep`
（`META_ACTION2TEXT` 把 `LaneKeepTemporal` 映射成 `LaneKeep`）。
coc 侧 loader 两个别名都认（`_ACTION_ALIASES` 中 `KeepLane` 与 `LaneKeep` 均映射到 `keep_lane`），
所以不影响正确性。

但这里有一件更值得写进课里的事实：**把该数据集全部 16 个 meta-action 文件扫一遍，
出现的 token 只有** `SteerRight / GoStraight / SteerLeft / MaintainSpeed / GentleDeceleration /
StrongDeceleration / GentleAcceleration / Stop / StrongAcceleration / SharpSteerRight`
**——一个 lane 动作都没有。**

也就是说在这份 PAI 数据上，lane 通道**永远是 `Not Available`**（生产 msg 4 也是如此）。
建议在 `13` 或 README 里明说这一点，否则读者可能误以为 lane 字段只是个没生效的摆设，
而它其实是生产 prompt 里被重点强调的输入。

---

## 五、待办清单

1. `13:93` 去掉 `+ 1`，让 meta-action 采样回到 16 点（-2.0 … +5.5），与视频窗口一致。
2. `13:177-179` 把 `v_lat` 换成按 heading 投影后的值（`v_long` 保持现状即可），并修正注释。
3. （可选）`10` 课补一句生产按 5 Hz 评定。
4. （可选）`13` 或 README 说明本数据集没有 lane 动作、lane 恒为 `Not Available`。

第 1、2 条都可以直接用 `15` 的真实输出复现验证，不依赖口头断言。
