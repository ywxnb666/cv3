# HW3 Screw Instance Segmentation

这套代码按“`hw2` 检测 teacher + 实例分割主模型 + 传统弱约束兜底”的思路组织，当前即使还没接入 12 张实例标注，也可以先用 `hw2` 权重在 `hw3/data` 上跑出一版可视化结果。

## 目录

- `run.py`: 推理入口
- `tools/prepare_dataset.py`: 将 Labelme 实例标注转成 YOLO-seg，并抽取 RGBA 实例素材
- `tools/generate_synth.py`: 用实例素材生成 copy-paste 合成数据
- `tools/build_trainval.py`: 自动把真实数据切成 `10 train / 2 val`，并把合成数据并入训练集
- `tools/train_seg.py`: 训练 Ultralytics 实例分割模型
- `screw_seg/`: 数据准备、合成增强、推理融合和可视化逻辑

## 1. 直接推理

先用 `cv` 环境运行：

```bash
/home/ywx/anaconda3/envs/cv/bin/python run.py \
  --data_dir ./data \
  --output_dir ./outputs/inference \
  --config ./configs/default.yaml
```

输出包括：

- `*_overlay.png`: 报告可直接使用的半透明可视化
- `*_instances.json`: 每个实例的类别、来源、box 和像素坐标
- `summary.json`: 汇总

当前若 `seg_weights` 为空，系统会自动退化为 `teacher+classical` 模式。

## 2. 准备 12 张实例标注

如果你的 12 张训练图是 Labelme 标注：

```bash
/home/ywx/anaconda3/envs/cv/bin/python tools/prepare_dataset.py \
  --labelme_dir /path/to/labelme_dir \
  --output_dir ./prepared/train_real
```

这一步会生成：

- `prepared/train_real/images`
- `prepared/train_real/labels`
- `prepared/train_real/instance_assets`
- `prepared/train_real/dataset.yaml`

## 3. 生成合成数据

```bash
/home/ywx/anaconda3/envs/cv/bin/python tools/generate_synth.py \
  --asset_dir ./prepared/train_real/instance_assets \
  --output_dir ./prepared/synth_360 \
  --num_images 360
```

## 4. 训练实例分割模型

先把真实数据和合成数据整理成最终训练集：

```bash
/home/ywx/anaconda3/envs/cv/bin/python tools/build_trainval.py \
  --real_dataset_dir ./prepared/train_real \
  --synth_dataset_dir ./prepared/synth_360 \
  --output_dir ./prepared/final_dataset \
  --val_count 2
```

脚本会优先把实例数更多、重叠更强的真实图放入验证集。

然后训练：

```bash
/home/ywx/anaconda3/envs/cv/bin/python tools/train_seg.py \
  --data_yaml ./prepared/final_dataset/dataset.yaml \
  --output_dir ./outputs/train \
  --model yolo11m-seg.pt \
  --imgsz 1280 \
  --epochs 250 \
  --patience 40 \
  --batch 2 \
  --device 0
```

训练结束后，把最优权重路径填到 `configs/default.yaml` 的 `seg_weights` 字段，再重新运行 `run.py`。

## 5. 还没接入 SAM 2 的部分

当前仓库已经把 SAM 留成可插拔位，但没有强绑安装。你后续在 H200 侧装好 `sam2` 后，只需要把 `configs/default.yaml` 里的 `use_sam` 打开，并补上 checkpoint/config 路径，就可以在现有主流程里继续接“低质量实例送 SAM 精修”。
