# KL-8 最小读取适配方案

## 目标

让 KL-8 数据能进入当前的本地数据筛选流程。第一版只解决记录发现和路径映射，直接复用现有相机图片、localization 读取函数；不复制或改写原始数据，也不先建设通用数据集框架。

数据根目录为：

```text
/home/baojiali/Downloads/disk1/data/kl_8/v1.0-trainval
```

## 和当前样例的结构差别

当前四条样例记录的必要输入路径是：

```text
<input-root>/<record>/
  camera/front_image/<timestamp>.jpg
  localization/<timestamp>.json
```

KL-8 的对应路径是：

```text
<dataset-root>/sample/<batch>/<record>/
  camera/front_image/<timestamp>.jpg
  localization/<timestamp>.json
```

所以，对筛选流程实际要读的前视图片和 localization 来说，KL-8 主要多了 `sample/<batch>/` 这一层。两边 record 内的必要目录名一致，抽查的 localization JSON 字段也一致。

KL-8 还有平行的 `label/`、`extra_info/` 和数据索引文件；record 内也有去畸变相机、额外传感器等目录。它们不属于第一版筛选所需输入，暂不读取。后续确实需要检测框或 chassis 信息时，再按实际用途添加路径。

## 最小实现

两套数据共用 `coc_labeling.data_loader.sample_dataset.SampleDataset`。它接收数据集名和可选根路径，扫描 `sample/<batch>/<record>/`，并为每条记录返回包含唯一 `<batch>/<record>` ID、record 路径、前视图片路径和 localization 路径的 `SampleRecord`。记录对象调用现有图片与定位读取器，不重复实现解码。

当前默认根目录分别为 `data/vla_kl` 与 `data/kl_8`。可以用下列命令查看发现结果，也可传 `--dataset-root` 覆盖根路径：

```bash
python -m coc_labeling.data_loader.sample_dataset --dataset vla_kl
python -m coc_labeling.data_loader.sample_dataset --dataset kl_8
```

Python 调用示例：

```python
dataset = SampleDataset("kl_8")
record = dataset.get_record("20251027_007/202510270958_record")
images = record.load_front_images()
positions = record.load_localization()
```

meta-action 生成和候选筛选 CLI 也可以直接选择数据集：

```bash
python -m coc_labeling.local_data_labeling --dataset vla_kl
python -m coc_labeling.local_keyframe_auto_select \
  --dataset vla_kl \
  --meta-action-root outputs/local_record_candidates
```

KL-8 使用 `--dataset kl_8`，`--record` 可指定一个或多个 `<batch>/<record>`。两阶段产物都按该 qualified ID 分目录保存；原有 `--input-root` 模式保持兼容。当前接入范围到候选清单、联系表和预览视频，尚未接入 `local_record_labeling` 的 CoC 输入与模型推理阶段。

`discover_records()` 稳定按 batch 和 record 排序，缺少前视图片目录或 localization 目录的记录不会进入结果。`qualified_id` 用于输出目录和清单，避免不同 batch 的同名 record 冲突。

KL-8 的 split pickle、`label/`、`extra_info/`、其他相机、LiDAR 和筛选规则都不放进第一版 adapter。筛选指标与 `keep` / `exclude` / `review` 的产物由独立的 [trainval 筛选方案](kl8_trainval_data_screening_plan.md) 负责。

## 图片时间间隔

当前 CoC 默认配置将图片目标时刻设为每 `0.5` 秒一个，窗口为历史 `2` 秒、未来 `6` 秒，共选择 16 张图片。样例源图片约为 10 Hz，读取时会从中按目标时间挑出这 16 张；送入 GPT 的目标间隔仍是 `0.5` 秒。

KL-8 抽查记录的前视图片间隔中位数约 `0.5` 秒，和目标间隔相近，但事件时刻不一定刚好落在图片时间戳上。当前选帧函数要求每个目标时刻与源图片相差不超过 `50 ms`。在一条 KL-8 记录上，用 `0.1` 秒步长检查 241 个可能事件时刻时，只有 33 个能选齐全部 16 帧。这个结果只是单条记录的探索性检查，但说明“源图片也是约 2 Hz”不能保证现有严格选帧条件直接适用。

因此，第一版 adapter 只验证路径与读取，不调整选帧容差。接入自动标注前，单独确定 KL-8 应如何匹配图片：保留 50 ms 并接受部分事件不可用，还是采用适合低帧率图片的时间容差/采样规则。规则确定后再用多条记录验证。

## 验收

- 合成目录测试能发现多个 batch，生成稳定且唯一的 `qualified_id`，并跳过缺少必要输入目录的记录。
- 对发现的记录，前视图片路径指向 `camera/front_image/`，定位路径指向 `localization/`。
- 现有图片和定位读取函数可以直接读取这些路径；源数据保持只读。
- `vla_kl` 与 `kl_8` 均通过同一 discovery 和共享图片/localization loader；源数据保持只读。
- 真实数据 smoke check 不启动 GPT，也不复制传感器数据。
