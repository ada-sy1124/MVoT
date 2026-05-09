# MVoT-Anole

基于 `GAIR/Anole-7b` 的 MVoT 复刻版（独立于原 Chameleon 硬编码逻辑）。

## 目录
- `anole_utils.py`: 模型/词表/VQ 编解码公共工具
- `probe_image_token_spec.py`: 探测 image token 数量/起止ID/单图生成token数
- `prepare_sft_dataset.py`: 重新编码训练数据（按 Anole 图像 token）
- `compute_distance_table.py`: 重算图像 token 物理距离矩阵
- `custom_trainer.py`: `Lc + alpha * Ld` 自定义损失
- `train.py`: LoRA 训练
- `evaluate.py`: 推理、图像解码与评估

## 0) 先探测 image token 规格（强烈建议先跑）
```bash
python ./MVoT-Anole/probe_image_token_spec.py \
  --model GAIR/Anole-7b \
  --save-json data/anole_image_token_probe.json
```

重点看输出字段：
- `num_image_tokens`
- `min_image_token_id` / `max_image_token_id`
- `is_contiguous_range`
- `generated_image_tokens_count`（模型一次生成中实际输出了多少图像 token）

默认是轻量模式（不加载整模型，速度快）。  
如果你要实测生成长度，再加：
```bash
python ./MVoT-Anole/probe_image_token_spec.py \
  --model GAIR/Anole-7b \
  --with-generate \
  --save-json data/anole_image_token_probe.json
```

## 1) 生成 Anole 版 SFT 数据集
```bash
python ./MVoT-Anole/prepare_sft_dataset.py \
  --model GAIR/Anole-7b \
  --input-jsonl data/训练样本/maze_dataset_merged_600.jsonl \
  --image-root data/600个样本单步image \
  --output-jsonl data/训练样本/sft_dataset_anole.jsonl \
  --meta-json data/训练样本/sft_dataset_anole.meta.json \
  --image-token-start 8712 \
  --image-token-count 7672 \
  --image-token-length 1024 \
  --start-index 0 \
  --end-index 100
```

## 2) 计算 Anole 图像 token 距离矩阵
```bash
python ./MVoT-Anole/compute_distance_table.py \
  --model GAIR/Anole-7b \
  --output data/MSE查询表_anole.pt \
  --image-token-start 8712 \
  --image-token-count 7672 \
  --normalize
```

## 3) LoRA 训练
```bash
python ./MVoT-Anole/train.py \
  --model GAIR/Anole-7b \
  --dataset-jsonl data/训练样本/sft_dataset_anole.jsonl \
  --distance-matrix data/MSE查询表_anole.pt \
  --image-token-start 8712 \
  --image-token-count 7672 \
  --output-dir ./outputs/anole_mvot_lora \
  --final-dir ./outputs/anole_mvot_lora_final \
  --alpha 1.0
```

## 4) 测试并解码图像
`test-jsonl` 需为 `input_ids + labels` 结构（和训练一致）。
```bash
python ./MVoT-Anole/evaluate.py \
  --model GAIR/Anole-7b \
  --adapter-path ./outputs/anole_mvot_lora_final \
  --test-jsonl data/测试样本.jsonl \
  --distance-matrix data/MSE查询表_anole.pt \
  --image-token-start 8712 \
  --image-token-count 7672 \
  --image-token-length 1024 \
  --out-dir data/评估结果_anole \
  --max-samples 10
```

## 说明
- 本套代码不再假设固定图像 token 区间（如 `8704~16895`），而是动态读取 Anole 词表映射。
- 若你使用本地缓存模型，加 `--local-files-only` 即可离线运行。
