import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import torch
from PIL import Image
from peft import PeftModel
from transformers import AutoTokenizer, ChameleonForConditionalGeneration, LogitsProcessor, LogitsProcessorList

# ==========================================
# 🚨 物理去势拦截器 (屏蔽文字输出)
# ==========================================
class ImageOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, start_id=8704, end_id=16895, eos_id=None):
        self.start_id = start_id
        self.end_id = end_id
        self.eos_id = eos_id

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float('inf'))
        mask[:, self.start_id : self.end_id + 1] = 0
        if self.eos_id is not None:
            mask[:, self.eos_id] = 0
        return scores + mask


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def build_prompt(record: Dict[str, Any]) -> str:
    lines: List[str] = []
    for turn in record.get("conversations", []):
        role = turn.get("from", "")
        value = turn.get("value", "")
        if role == "assistant":
            break
        lines.append(f"{role}: {value}")
    return "\n".join(lines) + "\nassistant: "


def get_target(record: Dict[str, Any]) -> str:
    for turn in record.get("conversations", []):
        if turn.get("from") == "assistant":
            return str(turn.get("value", ""))
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="把测试样本喂给底模+LoRA，并直接渲染输出图像！")
    p.add_argument("--base-model", default=os.environ.get("MODEL_NAME_OR_PATH", "facebook/chameleon-7b"))
    p.add_argument("--adapter-path", default="./outputs/chameleon_mvot_lora_final1")
    p.add_argument("--checkpoint-path", default="", help="可选：直接指定某个 checkpoint 目录，优先级高于 --adapter-path。")
    p.add_argument("--test-jsonl", default="data/测试样本.jsonl")
    p.add_argument("--max-samples", type=int, default=10)
    p.add_argument("--max-new-tokens", type=int, default=1050) # 一张图需要 1024 个 token
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--save-img-dir", default="data/训练样本/测试生成的图片") 
    p.add_argument("--save-output-json", default="data/训练样本/lora_test_outputs.jsonl")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    adapter_to_use = args.checkpoint_path.strip() or args.adapter_path

    # 1. 准备图片保存目录
    img_out_dir = Path(args.save_img_dir)
    img_out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("🧠 正在加载 Base 模型...")
    base = ChameleonForConditionalGeneration.from_pretrained(
        args.base_model,
        dtype=torch.bfloat16,
        device_map="auto",
        local_files_only=args.local_files_only,
    )
    
    print(f"🔥 正在融合 LoRA 权重: {adapter_to_use}")
    model = PeftModel.from_pretrained(base, adapter_to_use)
    model.eval()

    # ==========================================
    # 🚨 核心修复：自适应提取底层的 VQ 模型
    # ==========================================
    if hasattr(model, "base_model"):
        chameleon_core = model.base_model.model.model
    else:
        chameleon_core = model.model

    if hasattr(chameleon_core, "vqmodel"):
        vq_model = chameleon_core.vqmodel
    elif hasattr(chameleon_core, "vq_model"):
        vq_model = chameleon_core.vq_model
    else:
        raise AttributeError("❌ 找不到 VQ 解码器！请检查 transformers 库版本。")

    # 挂载拦截器，强制只能输出图片
    logits_processor = LogitsProcessorList([
        ImageOnlyLogitsProcessor(start_id=8704, end_id=16895, eos_id=tokenizer.eos_token_id)
    ])

    samples = load_jsonl(args.test_jsonl)[: args.max_samples]
    results: List[Dict[str, Any]] = []

    for i, sample in enumerate(samples, start=1):
        prompt = build_prompt(sample)
        target = get_target(sample)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        print("=" * 80)
        print(f"[{i}/{len(samples)}] 正在渲染样本 id={sample.get('id', 'unknown')}")

        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False, # 贪心解码，确定性世界模型不需要掷骰子
                logits_processor=logits_processor, # 强制拦截文本
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        # 掐头去尾，只保留模型生成的新 Token
        new_ids = output_ids[0][inputs["input_ids"].shape[1] :]
        
        # 提取图像 Token
        image_tokens = new_ids[(new_ids >= 8704) & (new_ids <= 16895)]
        
        # ==========================================
        # 🎨 VQ 解码，将数字渲染为图片
        # ==========================================
        if len(image_tokens) >= 1024:
            try:
                # 1. 转换为 VQ 索引 (从 0 开始)
                vq_indices = image_tokens[:1024] - 8704
                vq_indices = vq_indices.reshape(1, 32, 32)
                
                # 2. 查表获取特征并解码为像素
                z = vq_model.quantize.get_codebook_entry(vq_indices, shape=(1, 32, 32, vq_model.config.embed_dim))
                z = z.permute(0, 3, 1, 2).contiguous()
                decoded_pixels = vq_model.decode(z)
                
                # 3. 像素值规整并转为 PIL 图像
                decoded_pixels = torch.clamp((decoded_pixels + 1.0) / 2.0, min=0.0, max=1.0)
                decoded_pixels = decoded_pixels[0].permute(1, 2, 0).cpu().numpy()
                decoded_pixels = (decoded_pixels * 255).astype("uint8")
                
                final_image = Image.fromarray(decoded_pixels)
                
                # 4. 保存图像
                img_name = f"pred_{i:03d}_{sample.get('id', 'unknown')}.png"
                img_path = img_out_dir / img_name
                final_image.save(img_path)
                print(f"✅ 图像解码成功！已保存为: {img_path}")
            except Exception as e:
                print(f"❌ 图像解码发生错误: {e}")
        else:
            print(f"⚠️ 警告：模型只生成了 {len(image_tokens)} 个图像 Token，凑不够 1024 个，无法生成完整图片！")

        pred_text = tokenizer.decode(new_ids, skip_special_tokens=True).strip()
        item = {
            "index": i - 1,
            "id": sample.get("id"),
            "target": target,
            "prediction_text": pred_text,
            "metadata": sample.get("metadata", {}),
        }
        results.append(item)

    # 记录 JSONL (可选，留作对比分析用)
    out_json_path = Path(args.save_output_json)
    out_json_path.parent.mkdir(parents=True, exist_ok=True)
    with out_json_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print("=" * 80)
    print(f"🎉 全部测试完毕！")
    print(f"🖼️ 生成的图片已统一保存在: {img_out_dir.resolve()}")


if __name__ == "__main__":
    main()