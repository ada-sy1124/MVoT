import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
import torch
from PIL import Image
from tqdm import tqdm
from transformers import ChameleonProcessor, ChameleonForConditionalGeneration

# =========================
# ⚙️ 核心参数配置区域
# =========================
INPUT_JSONL = "data/训练样本/maze_dataset_merged_600.jsonl"
IMAGE_ROOT = "data/600个样本单步image"
OUTPUT_JSONL = "data/训练样本/sft_dataset.jsonl"
REQUIRE_IMAGES = True

START_INDEX = 0
END_INDEX = 100 # 测试时设为 102 只跑 1 个样本，跑全量请设为 None

SYSTEM_PROMPT = "system: You are a deterministic visual world model. Map the given observation and action to the next visual state.\nuser: "

def load_jsonl(input_path: str | Path) -> List[Dict[str, Any]]:
    input_path = Path(input_path)
    records: List[Dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw: continue
            records.append(json.loads(raw))
    return records


def save_jsonl(records: List[Dict[str, Any]], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def find_frame_path(sample_dir: Path, frame_index: int) -> Path:
    matches = sorted(sample_dir.glob(f"*_move_{frame_index:04d}.png"))
    if not matches:
        return sample_dir / f"missing_move_{frame_index:04d}.png"
    return matches[0]


def collect_image_paths(sample_index: int, frame_count: int, image_root: str | Path) -> List[str]:
    sample_dir = Path(image_root) / f"sample_{sample_index:06d}"
    return [str(find_frame_path(sample_dir, frame_index)) for frame_index in range(frame_count)]


def validate_image_paths(image_paths: List[str]) -> None:
    missing = [path for path in image_paths if not Path(path).exists()]
    if missing:
        raise FileNotFoundError(f"缺失渲染图像！首先找不到的路径: {missing[0]}")


def build_pure_sft_records(
    samples: List[Dict[str, Any]],
    image_root: str | Path,
    require_images: bool,
    start_index: int = START_INDEX,
    end_index: int | None = END_INDEX,
) -> List[Dict[str, Any]]:
    """
    核心引擎：将图片与文本直接熔铸为底层 Token ID 整数数组！
    """
    print("📦 正在加载分词器与底层 VQ 引擎 (需要十几秒)...")
    processor = ChameleonProcessor.from_pretrained("facebook/chameleon-7b")
    tokenizer = processor.tokenizer
    
    model = ChameleonForConditionalGeneration.from_pretrained(
        "facebook/chameleon-7b", 
        device_map="cuda", 
        torch_dtype=torch.bfloat16
    )
    model.eval()

    # 自适应提取 VQ 引擎
    chameleon_core = model.model
    if hasattr(chameleon_core, "vqmodel"):
        vq_model = chameleon_core.vqmodel
    elif hasattr(chameleon_core, "vq_model"):
        vq_model = chameleon_core.vq_model
    else:
        raise AttributeError("❌ 找不到 VQ 编码器！")

    def encode_image(img_path: str) -> List[int]:
        """将图片直接转换为 1024 个底层物理 Token ID 的列表"""
        img = Image.open(img_path).convert("RGB")
        pixel_values = processor.image_processor(img, return_tensors="pt")["pixel_values"].to(model.device, dtype=torch.bfloat16)
        with torch.no_grad():
            z = vq_model.encode(pixel_values)
            z_latents = z.latents if hasattr(z, "latents") else z[0]
            _, _, vq_indices = vq_model.quantize(z_latents)
        # 变色龙的图像 ID 偏移量为 8704
        return [8704 + idx for idx in vq_indices.view(-1).tolist()]

    sft_records: List[Dict[str, Any]] = []
    selected_samples = samples[start_index:end_index] if end_index else samples[start_index:]

    # 预先 Tokenize 固定的系统提示词
    sys_ids = tokenizer.encode(SYSTEM_PROMPT, add_special_tokens=False)

    for local_index, sample in enumerate(tqdm(selected_samples, desc="🚀 正在铸造纯净物理数据集")):
        sample_index = start_index + local_index
        moves = sample["moves"]
        
        image_paths = collect_image_paths(sample_index, len(moves) + 1, image_root)
        if require_images:
            validate_image_paths(image_paths)

        # 优化：为了避免重复计算，先把该样本的所有图片一次性提成 Token
        encoded_frames = [encode_image(p) for p in image_paths]

        for step_idx, action in enumerate(moves):
            # 获取前后两帧的图像数组
            img1_ids = encoded_frames[step_idx]
            img2_ids = encoded_frames[step_idx + 1]

            # 动态 Tokenize 用户的 Action
            user_text_after_img = f"\nAction: {action}\nassistant: "
            user_ids = tokenizer.encode(user_text_after_img, add_special_tokens=False)

            # ==========================================
            # 🧩 终极拼接：抛弃字符串，直接在底层数组级别拼接
            # ==========================================
            # 1. 组合输入部分 (系统提示 + 帧1 + 动作)
            prompt_ids = sys_ids + img1_ids + user_ids
            
            # 2. 组合答案部分 (帧2 + 结束符)
            target_ids = img2_ids + [tokenizer.eos_token_id]
            
            # 3. 得到完整的模型输入与交叉熵掩码 (-100 表示不计算 Loss)
            input_ids = prompt_ids + target_ids
            labels = [-100] * len(prompt_ids) + target_ids

            sft_records.append(
                {
                    "id": f"maze_env_transition_{sample_index:06d}_{step_idx:04d}",
                    "input_ids": input_ids,
                    "labels": labels,
                    "metadata": {
                        "sample_index": sample_index,
                        "step_index": step_idx,
                        "maze_id": sample.get("id"),
                        "action": action,
                    },
                }
            )

    return sft_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成纯净版 ID 数组格式的多模态 SFT 样本")
    parser.add_argument("--input", default=INPUT_JSONL)
    parser.add_argument("--image-root", default=IMAGE_ROOT)
    parser.add_argument("--output", default=OUTPUT_JSONL)
    parser.add_argument("--start-index", type=int, default=START_INDEX)
    # 不传 --end-index 默认处理到底
    parser.add_argument("--end-index", type=int, default=END_INDEX)
    parser.add_argument("--allow-missing-images", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = load_jsonl(args.input)
    
    records = build_pure_sft_records(
        samples=samples,
        image_root=args.image_root,
        require_images=not args.allow_missing_images,
        start_index=args.start_index,
        end_index=args.end_index,
    )
    
    save_jsonl(records, args.output)
    print("=" * 60)
    print(f"🎉 大功告成！生成的纯净数据集已保存至: {Path(args.output).resolve()}")
    print("✨ 这个 JSONL 里只包含大模型最喜欢的纯数字数组，没有任何杂乱的文本！")


if __name__ == "__main__":
    main()