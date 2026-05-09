import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import torch
from transformers import AutoConfig, AutoTokenizer, ChameleonForConditionalGeneration, LogitsProcessor, LogitsProcessorList

from anole_utils import (
    DEFAULT_IMAGE_TOKEN_COUNT,
    DEFAULT_IMAGE_TOKEN_LENGTH,
    DEFAULT_IMAGE_TOKEN_START,
    image_token_ids_from_range,
    load_jsonl,
)


class ImageOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, image_token_ids: List[int], eos_id: int):
        self.allowed = sorted(set(int(x) for x in image_token_ids) | {int(eos_id)})

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float("inf"))
        mask[:, self.allowed] = 0
        return scores + mask


def build_probe_prompt(tokenizer: Any) -> torch.Tensor:
    text = (
        "system: You are a deterministic visual world model. "
        "Map the given observation and action to the next visual state.\n"
        "user: <image>\nAction: up\nassistant: "
    )
    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor([ids], dtype=torch.long)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Probe image-token specification for Anole/Chameleon-family model.")
    p.add_argument("--model", default="GAIR/Anole-7b")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--test-jsonl", default="data/测试样本.jsonl")
    p.add_argument("--with-generate", action="store_true", help="是否加载完整模型并实际生成一次来统计输出图像token数量。")
    p.add_argument("--image-token-start", type=int, default=DEFAULT_IMAGE_TOKEN_START)
    p.add_argument("--image-token-count", type=int, default=DEFAULT_IMAGE_TOKEN_COUNT)
    p.add_argument("--image-token-length", type=int, default=DEFAULT_IMAGE_TOKEN_LENGTH)
    p.add_argument("--max-new-tokens", type=int, default=1200)
    p.add_argument("--min-new-tokens", type=int, default=900)
    p.add_argument("--save-json", default="data/anole_image_token_probe.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    # 轻量模式：默认只加载 config/tokenizer，不加载整模型
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)

    image_token_ids = image_token_ids_from_range(args.image_token_start, args.image_token_count)
    image_token_ids_sorted = sorted(set(int(x) for x in image_token_ids))
    min_id = image_token_ids_sorted[0]
    max_id = image_token_ids_sorted[-1]
    count = len(image_token_ids_sorted)
    expected_count = max_id - min_id + 1
    is_contiguous = expected_count == count

    # 从测试样本估计“每张图目标token数”
    expected_img_tokens_from_data = None
    try:
        samples = load_jsonl(args.test_jsonl)
        if samples and "labels" in samples[0]:
            image_set = set(image_token_ids_sorted)
            counts = []
            for s in samples[: min(32, len(samples))]:
                labels = s.get("labels", [])
                counts.append(sum(1 for x in labels if x in image_set))
            if counts:
                # labels里通常还包含eos，因此常见是 1025；图像token一般是 1024
                expected_img_tokens_from_data = int(round(sum(counts) / len(counts)))
    except Exception:
        pass

    generated_total = None
    generated_image_count = None
    first_40_ids: List[int] = []
    first_40_image_ids: List[int] = []

    if args.with_generate:
        model = ChameleonForConditionalGeneration.from_pretrained(
            args.model,
            local_files_only=args.local_files_only,
            device_map="auto",
            dtype=torch.bfloat16,
        )
        model.eval()
        eos_id = int(tokenizer.eos_token_id)
        logits_processor = LogitsProcessorList([ImageOnlyLogitsProcessor(image_token_ids_sorted, eos_id)])

        prompt = build_probe_prompt(tokenizer).to(model.device)
        with torch.no_grad():
            out = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                min_new_tokens=args.min_new_tokens,
                logits_processor=logits_processor,
                eos_token_id=eos_id,
                pad_token_id=eos_id,
            )

        new_ids = out[0][prompt.shape[1] :].tolist()
        image_set = set(image_token_ids_sorted)
        image_new_ids = [int(x) for x in new_ids if int(x) in image_set]
        generated_total = len(new_ids)
        generated_image_count = len(image_new_ids)
        first_40_ids = [int(x) for x in new_ids[:40]]
        first_40_image_ids = image_new_ids[:40]

    report: Dict[str, Any] = {
        "model": args.model,
        "num_image_tokens": count,
        "min_image_token_id": min_id,
        "max_image_token_id": max_id,
        "is_contiguous_range": is_contiguous,
        "range_size_if_contiguous": expected_count,
        "first_20_image_token_ids": image_token_ids_sorted[:20],
        "last_20_image_token_ids": image_token_ids_sorted[-20:],
        "expected_tokens_from_test_labels": expected_img_tokens_from_data,
        "expected_image_token_length": args.image_token_length,
        "with_generate": bool(args.with_generate),
        "generated_new_tokens_total": generated_total,
        "generated_image_tokens_count": generated_image_count,
        "generated_first_40_ids": first_40_ids,
        "generated_first_40_image_ids": first_40_image_ids,
    }

    out_path = Path(args.save_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"Saved probe report to: {out_path.resolve()}")


if __name__ == "__main__":
    main()



# 起始token：8712
# token总数：7672
# 需要1024个token组成image
