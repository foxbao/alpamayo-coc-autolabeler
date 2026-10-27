# 本地记录 CoC 适配方案与实现现状

> 本文以当前代码和已检查的本地数据为准。早期“只用位置计算地面速度”的方案已被后续实现取代；本文件说明当前实现做了什么、依据是什么，以及哪些判断仍未得到数据生产端确认。
>
> 数据根目录：`/home/baojiali/Downloads/disk3/20261019_lin_007_out`
>
> 当前验证记录：`20261019153941_record`，人工选定事件时间 `21.6 s`
> 历史评审记录：[`local_record_autolabel_plan_response_v4.md`](local_record_autolabel_plan_response_v4.md) 针对的是更早的图片输入阶段，不是当前实现规格。

## 目标和范围

当前工具用于一条本地记录的一个人工选定时刻：从 front camera 取图，从 localization 估计车辆运动状态，准备 CoC Agent 输入；用户显式传入 `--run-inference` 后，才调用一次模型并生成待人工审核的结果。它是本地数据适配与单事件验证工具，不会自动发现事件，也不做批量数据集标注。

当前不把 `chassis`、`planning` 或旧版 Century 源码作为运行时输入。规划结果和相机画面曾用于离线核对坐标解释；它们不会作为本工具的 CoC Agent 输入。

## 对数据的判断：事实、推断和未知项

定位 JSON 没有坐标系标签或生产端版本信息。因此，“字段名叫 `x/y`”本身不足以确定物理方向；代码当前采用的数据解释是有证据支持的工作假设，不是生产端格式保证。

| 项目 | 当前掌握情况 | 状态 |
| --- | --- | --- |
| localization 文件名中的时间戳 | 用户确认它就是该 localization 样本的采样时间，可用于位置差分 | 已确认 |
| front-camera 图片时间 | 图片文件名包含 Unix 秒时间戳；`--event-time-sec` 从该目录最早图片时间起算 | 代码约定 |
| 多传感器时钟 | 当前将 camera 事件的 Unix 时间直接用于定位序列，没有估计传感器 offset | 尚未单独验证 |
| localization 数值字段 | 当前样本含 `position`、`orientation` 和 `linear_velocity`；JSON 未声明坐标系、四元数方向约定或 producer | 已检查字段；语义不完整 |
| `position.x/y` | 作为平面 map 坐标使用，并从时间差分得到地图平面速度 | 当前处理假设 |
| `orientation` | 按车体局部坐标到 map-XY 的四元数解释；这批数据的局部 x 轴按前向、局部 y 轴按左向解释，即 FLU | 多条数据线索支持的推断 |
| `linear_velocity.x/y/z` | 水平分量按车体 FLU 的 x 前、y 左解释后，与位置差分速度明显更一致；z 轴正方向未单独验证 | 强支持的水平帧工作假设，待生产端确认 |

这里不采用旧 Century 源码作为当前格式依据，因为它可能不对应这批数据的实际生产链路。

### FLU 工作假设的依据与边界

对这批目录中的四条记录，已有统计显示，在速度大于 `1 m/s` 的抽样点上，地图轨迹方向和四元数 yaw 的绝对角差如下。统计采用 0.5 秒中心差分，每 0.5 秒取一个点：

| 记录 | 有效运动样本 | 中位地速 | 轨迹方向与四元数 yaw 绝对角差：中位数 / P90 |
| --- | ---: | ---: | ---: |
| `153841` | 18 | `1.96 m/s` | `1.10° / 3.35°` |
| `153911` | 29 | `2.51 m/s` | `4.15° / 23.40°` |
| `153941` | 39 | `5.77 m/s` | `0.37° / 0.75°` |
| `154011` | 25 | `2.99 m/s` | `0.18° / 0.62°` |

其中 `153941` 事件点附近，轨迹方向约 `−7.28°`、四元数 yaw 约 `−7.20°`；同刻规划轨迹 `theta` 约 `−7.52°`，规划速度为正 `5.66 m/s`。此外，事件前后 15 对、相隔 0.5 秒的前视图像呈现静态背景特征向左右外扩的运动，符合相机主要向前运动的表现。这些线索共同支持局部 x 作为前向的 FLU 解释。

它们仍不能代替生产端的坐标契约：轨迹方向与 yaw 的一致性依赖 yaw 所对应的局部轴；规划 `theta` 的物理约定和相机安装朝向也没有在 localization JSON 中声明。故此结论适用于当前检查过的记录，换数据源、车辆或定位生产链路时必须重新核对。

## 当前代码实际执行的流程

入口为 `python -m coc_labeling.local_record_autolabel`，主要实现位于 `src/coc_labeling/recording_autolabel/`。

1. **确定事件和图片。** 事件相对时间以 front-camera 目录中最早图片的时间戳为起点。按 CoC 时间配置取事件前 2 秒和从事件时刻起的 6 秒窗口，图片目标间隔为 0.5 秒，共 16 张（4 张历史、12 张未来）。单张图片与目标时间相差不得超过 50 ms，源图不能重复使用。
2. **把定位样本放到事件时间轴。** 按 localization 文件名时间戳排序，并把该时间戳当作采样时刻。事件的绝对 Unix 时间直接用于查询定位样本；代码没有拟合跨传感器 offset，也不外推。位置插值两端原始样本间隔上限为 32 ms。
3. **从位置估计地图速度。** 在事件相对 10 Hz 网格上插值 `position.x/y`。速度时刻为 `−2.0` 到 `+6.0` 秒、每 0.5 秒一个，共 17 点；每点以 `t−0.25` 和 `t+0.25` 的位置作 0.5 秒中心差分。水平地速 `v_ground = sqrt(v_map_x² + v_map_y²)` 不依赖姿态。
4. **按工作假设转到车体速度。** 在每个速度时刻对四元数作 SLERP，按“车体 FLU 到 map-XY”解释，并作逆旋转：

   ```text
   v_body_flu = R(q)^T · [v_map_x, v_map_y, 0]
   v_long = v_body_flu.x       # 前向为正
   v_lat  = -v_body_flu.y      # 右向为正；FLU 的 y 轴向左
   ```

   `v_lat` 是车体坐标下的瞬时侧向速度，不等同于变道方向。位置或姿态不可可靠取得时，对应值标为 `Not Available`。
5. **构造 Agent 输入和审核材料。** 17 个时刻的 `v_long`、`v_lat` 写入 `ego_text`；`v_ground` 只保留在审计文件，不发给 Agent，因为它是同一水平速度的无方向大小，等于 body-frame 分量的范数。另用同一条 10 Hz 位置轨迹生成 PAI 风格的纵向 meta-action，并按 0.5 秒图片时刻写入 `all_meta_action_str`；`lateral` 和 `lane` 继续为 `Not Available`。`linear_velocity` 的原值及其按车体 FLU 水平分量工作假设与位置推导速度的差异写入审计文件，但不进入 Agent 输入。四条记录的核对显示，先把原始 `linear_velocity.x/y` 按 x 前、y 左用 orientation 旋转到 map，比直接按 map-XY 解释更一致；这仍需生产端确认，且 z 轴正方向未单独验证。prepare-only 生成帧清单、输入预览、位置轨迹、速度审计和 `longitudinal_meta_action.json`；默认不加载 API 凭据或请求模型。显式传入 `--run-inference` 才生成 CoC YAML、叠字视频和带 CoC 的联系表。

当前实现假设的集中说明见 [`local_record_coordinate_conversion.md`](local_record_coordinate_conversion.md)，使用方式见 [`local_record_autolabel.md`](local_record_autolabel.md)。生成的 `position_motion_audit.json` 会记录时间来源、差分方法、姿态解释、速度映射和坐标不确定性；`longitudinal_meta_action.json` 记录本地纵向分类的阈值、10 Hz 动作点和动作区间。

## 当前验证结果

- `20261019153941_record` 的 localization 序列有 3,698 个样本，覆盖约 30 秒；平均约 123.3 Hz，平均间隔约 8.11 ms。开头有约 424.6 ms 的样本间隔。
- 事件 `21.6 s` 的 10 Hz 轨迹网格覆盖 `−2.3` 到 `+6.3 s`。已有审计记录了 4 段超过 32 ms 的原始间隔；当前事件所需的 17 个位置差分和姿态样本均有效。
- 该事件点的输出约为 `v_long=+5.59 m/s`、`v_lat=+0.01 m/s`、`v_ground=5.59 m/s`。这说明此前把约 `5.6 m/s` 解释为向右运动与 FLU 工作假设不符；它本身不能证明所有记录都采用 FLU。
- `21.6 s` 事件的纵向分类使用 81/81 个有效 10 Hz 速度点；16 个图片时刻在 `agent_input_preview.json` 中得到纵向动作。该次预览在 `t=0.0–1.0 s` 和 `t=2.5–4.0 s` 标为 `Gentle Deceleration`，其余采样时刻为 `Maintain Speed`；这是一轮规则结果，尚需人工结合视频核验。
- 当前回归测试覆盖定位插值、重采样、中心差分、姿态插值、FLU 逆旋转、右正 `v_lat`、纵向动作分类、输入提示和输出审计。最近一次完整测试为 `36 passed`，Ruff 检查通过；prepare-only smoke test 不调用模型。

## 早期方案哪些已被取代

早期方案提出“先只用位置算 `v_ground`，暂不读取姿态；之后再决定是否需要 orientation”。这适用于位置-only 阶段，但不是当前代码状态。只有位置可以得到地面速度大小和 map-XY 运动，不能把速度分解为车体前向/侧向分量；当前 JSON 中实际存在姿态数据，因此后续实现把姿态用于车体系投影，并通过当前记录作了坐标核对。该决定应按上文的 FLU 工作假设理解，不能回头把早期方案当成当前算法说明。

早期方案中“一条记录、一个人工事件、先预览、显式推理、结果待审核”的范围仍然适用；已经实现的 10 Hz 速度、审计图表和 CoC 联系表则是相对早期方案增加的内容。

## 仍需确认的事项

在把这套转换推广到更多数据前，优先确认以下事项：

1. 向当前 localization 数据的生产或维护人员核实 `position` 的参考坐标系、四元数分量顺序与旋转方向、局部轴定义，以及 `linear_velocity` 的坐标系。现有四条记录强支持 `linear_velocity` 为车体 FLU；若无法取得生产端说明，仍应把它限定为这批已抽查记录的适配假设。
2. 核实 camera 与 localization 时间戳是否共享同一时钟和 epoch。用户已确认 localization 文件名时间是采样时刻；这还没有单独证明各传感器时钟之间不存在 offset。
3. 在其余记录和不同运动状态（停车、倒车、转弯、横移）上复查：位置差分、姿态投影、候选 `linear_velocity` 及视频是否相互吻合。确认 `linear_velocity` 的帧语义后，才决定能否把它作为主速度来源或校验真值。
4. 验证 32 ms 原始插值上限和 0.5 秒差分窗口对当前定位采样抖动及速度变化是否合适；若调整，应同时更新审计产物和回归测试。

这些确认完成前，工具继续限定为人工指定事件的单记录验证；不增加自动事件检测或批量标注，也不把 `v_lat` 当作变道标签。CoC 输出仍须人工审核。

## 运行方式

prepare-only：

```bash
conda activate /mnt/disk1/conda_envs/alpamayo-coc
python -m coc_labeling.local_record_autolabel \
  --input-root /home/baojiali/Downloads/disk3/20261019_lin_007_out \
  --record 20261019153941_record \
  --event-time-sec 21.6 \
  --output-dir outputs/local_record_autolabel_mvp
```

确认图片和输入预览后，显式添加 `--run-inference` 才会请求模型。重复运行会覆盖同一事件目录中的结果；运行前应确认输出路径。
