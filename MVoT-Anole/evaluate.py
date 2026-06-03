# import json
# import numpy as np
# from PIL import Image
# from pathlib import Path
# from typing import Any, Dict, List

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from peft import PeftModel
# from omegaconf import OmegaConf
# from transformers import ChameleonForConditionalGeneration, ChameleonProcessor, LogitsProcessor, LogitsProcessorList

# from anole_utils import (
#     DEFAULT_IMAGE_TOKEN_COUNT,
#     DEFAULT_IMAGE_TOKEN_LENGTH,
#     DEFAULT_IMAGE_TOKEN_START,
#     image_token_ids_from_range,
#     load_jsonl,
# )

# # =====================================================================
# # 🛠️ 本地资源配置
# # =====================================================================
# MODEL_NAME_OR_PATH = "/root/autodl-tmp/Anole-7b-v0.1-hf"                   
# ADAPTER_PATH = "./MVoT-Anole/outputs/anole_mvot_lora/checkpoint-50"             
# TEST_JSONL = "./MVoT-Anole/data/测试样本_anole.jsonl" 
# OUT_DIR = "./MVoT-Anole/data/anole_mvot_eval_测试结果"                         

# # 🌟 核心：你找回来的两个本地文件
# VQGAN_CONFIG = "./vqgan/config.yaml"
# VQGAN_CKPT = "./vqgan/model.ckpt"

# MAX_SAMPLES = 10                                                         
# IMAGE_TOKEN_START = DEFAULT_IMAGE_TOKEN_START # 8192
# # =====================================================================

# # ---------------------------------------------------------------------
# # 🎨 极简 VQGAN 解码逻辑重构 (零依赖版本)
# # ---------------------------------------------------------------------
# class SimpleVQDecoder(nn.Module):
#     def __init__(self, config_path, ckpt_path, device):
#         super().__init__()
#         self.device = device
#         conf = OmegaConf.load(config_path)
#         # 自动提取参数
#         params = conf.model.params if "model" in conf else conf.params
#         ddconfig = params.ddconfig
        
#         # 1. 这里的重点是加载码本 (Codebook)
#         embed_dim = params.embed_dim
#         n_embed = params.n_embed
#         self.codebook = nn.Embedding(n_embed, embed_dim)
        
#         # 2. 手动构造骨架：由于写全量 ResNet 卷积层太长，我们直接使用映射后的特征
#         # 注意：既然已经有了 ckpt，我们利用 torch.load 动态构建
#         self.sd = torch.load(ckpt_path, map_location="cpu")
#         if "state_dict" in self.sd: self.sd = self.sd["state_dict"]
        
#         # 3. 这里的骚操作：虽然我们要手动出图，但为了避免重写几百行卷积层代码，
#         # 我们利用一个最简单的逻辑：如果我们无法重建整个网络，我们起码要拿到查表后的特征。
#         # 幸好你手里有 ckpt，我们可以尝试实例化 taming 的类（如果 pip install 成功过）
#         # 如果还是不行，我们就必须借助外部最简实现的 VQGAN 结构。
        
#         from taming.models.vqgan import VQModel
#         self.model = VQModel(**params)
#         self.model.load_state_dict(self.sd, strict=False)
#         self.model.to(device).eval()

#     @torch.no_grad()
#     def decode_tokens(self, token_ids):
#         ids = list(token_ids[:1024])
#         if len(ids) < 1024: ids += [ids[-1] if ids else 0] * (1024 - len(ids))
#         clean_ids = torch.tensor([max(0, int(x) - IMAGE_TOKEN_START) for x in ids], device=self.device)
        
#         # 执行查表与解码
#         z_q = self.model.quantize.get_codebook_entry(clean_ids, shape=(1, 32, 32, -1))
#         x_recon = self.model.decode(z_q)
        
#         # 像素转换
#         x_recon = torch.clamp((x_recon + 1.0) / 2.0, min=0.0, max=1.0)
#         img_np = x_recon.cpu().squeeze(0).permute(1, 2, 0).float().numpy()
#         return Image.fromarray((img_np * 255).astype(np.uint8))

# # ---------------------------------------------------------------------

# def main() -> None:
#     print(f"🚀 启动评估 (本地外挂解码版)...")
#     out_dir = Path(OUT_DIR)
#     out_dir.mkdir(parents=True, exist_ok=True)
#     target_device = torch.device("cuda:0")

#     # 1. 加载语言模型部分 (只负责想数字)
#     base = ChameleonForConditionalGeneration.from_pretrained(
#         MODEL_NAME_OR_PATH, device_map={"": 0}, torch_dtype=torch.bfloat16
#     )
#     model = PeftModel.from_pretrained(base, ADAPTER_PATH)
#     model.eval()

#     # 2. 🌟 挂载独立的“纯净解码工厂”
#     try:
#         my_decoder = SimpleVQDecoder(VQGAN_CONFIG, VQGAN_CKPT, target_device)
#         print("✅ 独立解码工厂组装成功！")
#     except Exception as e:
#         print(f"🚨 解码工厂组装失败，请检查 taming-transformers 是否正确安装: {e}")
#         return

#     # 3. 准备数据
#     image_token_ids = image_token_ids_from_range(IMAGE_TOKEN_START, 16384)
#     samples = load_jsonl(TEST_JSONL)[:MAX_SAMPLES]

#     for idx, sample in enumerate(samples):
#         sample_id = sample.get("id", f"s_{idx}")
#         print(f"  -> 正在生成样本: {sample_id}")
        
#         prompt_ids = [v for v, y in zip(sample["input_ids"], sample["labels"]) if y == -100]
#         true_img_ids = [v for v, y in zip(sample["input_ids"], sample["labels"]) if y != -100 and v in set(image_token_ids)]

#         with torch.no_grad():
#             gen = model.generate(
#                 input_ids=torch.tensor([prompt_ids], device=target_device),
#                 max_new_tokens=len(true_img_ids) + 5,
#                 do_sample=False
#             )
        
#         pred_ids = gen[0][len(prompt_ids):].tolist()
#         pred_ids = [v for v in pred_ids if v in set(image_token_ids)][:1024]

#         # 4. 🌟 使用独立工厂绘图
#         for tag, ids in [("pred", pred_ids), ("gt", true_img_ids)]:
#             try:
#                 img = my_decoder.decode_tokens(ids)
#                 img.save(out_dir / f"{sample_id}_{tag}.png")
#                 print(f"      ✨ {tag} 图片已落地！")
#             except Exception as e:
#                 print(f"      ⚠️ {tag} 绘图失败: {e}")

#     print(f"\n✅ 全部完成！图片保存目录：{out_dir.resolve()}")

# if __name__ == "__main__":
#     main()



# # python MVoT-Anole/evaluate.py



import json
import math
from pathlib import Path
from typing import Any, Dict, List

import torch
from peft import PeftModel
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor, LogitsProcessor, LogitsProcessorList

from anole_utils import (
    DEFAULT_IMAGE_TOKEN_COUNT,
    DEFAULT_IMAGE_TOKEN_LENGTH,
    DEFAULT_IMAGE_TOKEN_START,
    decode_token_ids_to_image,
    image_token_ids_from_range,
    load_jsonl,
    load_vqgan,
)

MODEL_NAME_OR_PATH = "/root/autodl-tmp/Anole-7b-v0.1-hf"
ADAPTER_PATH = "./MVoT-Anole/outputs/anole_mvot_lora_final"
TEST_JSONL = "./MVoT-Anole/data/sft_测试样本_anole.jsonl"
DISTANCE_MATRIX = "./MVoT-Anole/data/MSE查询表_Anole.pt"
VQGAN_CONFIG = "./MVoT-Anole/vqgan/config.yaml"
VQGAN_CKPT = "./MVoT-Anole/vqgan_finetuned/maze_vqgan_epoch_25.ckpt"
OUT_DIR = "./MVoT-Anole/outputs/anole_mvot_eval"
MAX_SAMPLES = 10
LOCAL_FILES_ONLY = False
IMAGE_TOKEN_START = 8192  # 🚨 强行锁死正确的词表偏移起点
IMAGE_TOKEN_COUNT = 16384  # VQGAN 码本总数
IMAGE_TOKEN_LENGTH = DEFAULT_IMAGE_TOKEN_LENGTH


class ImageOnlyLogitsProcessor(LogitsProcessor):
    def __init__(self, image_token_ids: List[int], eos_id: int):
        self.allowed = sorted(set(int(x) for x in image_token_ids) | {int(eos_id)})

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        mask = torch.full_like(scores, -float("inf"))
        mask[:, self.allowed] = 0
        return scores + mask


def _align_to_nearest_square(token_list: List[int], target_std_len: int, name: str) -> List[int]:
    """
    底层自适应对齐防线：强制将任意 Token 序列安全裁剪为最近的完全平方数。
    """
    curr_len = len(token_list)
    if curr_len == 0:
        print(f"     ⚠️ 警告: {name} 序列为空！")
        return []

    # 优先尝试使用预设的标准长度截断
    std_side = math.isqrt(target_std_len)
    if std_side * std_side == target_std_len and curr_len >= target_std_len:
        return token_list[:target_std_len]

    # 否则，动态向下寻找当前真实存在的最近完全平方数进行抢救
    side = math.isqrt(curr_len)
    valid_len = side * side
    if valid_len != curr_len:
        print(f"     ⚠️ 自动校准: {name} 长度 ({curr_len}) 不是平方数，强制裁剪至安全网格 {valid_len} ({side}x{side})")
    
    return token_list[:valid_len]


def main() -> None:
    out_dir = Path(OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    processor = ChameleonProcessor.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=LOCAL_FILES_ONLY)
    tokenizer = processor.tokenizer
    base = ChameleonForConditionalGeneration.from_pretrained(
        MODEL_NAME_OR_PATH, local_files_only=LOCAL_FILES_ONLY, device_map="auto", dtype=torch.bfloat16
    )
    model = PeftModel.from_pretrained(base, ADAPTER_PATH)
    model.eval()

    vq_model, _ = load_vqgan(VQGAN_CONFIG, VQGAN_CKPT, device="cpu")
    image_token_ids = image_token_ids_from_range(IMAGE_TOKEN_START, IMAGE_TOKEN_COUNT)
    image_token_set = set(image_token_ids)
    eos = int(tokenizer.eos_token_id)
    logits_processor = LogitsProcessorList([ImageOnlyLogitsProcessor(image_token_ids, eos)])

    dist_obj = torch.load(DISTANCE_MATRIX, map_location=model.device)
    distance_matrix = dist_obj["distance_matrix"].to(model.device)
    id_to_code = {int(t): i for i, t in enumerate(image_token_ids)}

    # 安全校验全局设定的 IMAGE_TOKEN_LENGTH 本身是否合法
    std_len = IMAGE_TOKEN_LENGTH
    if math.isqrt(std_len) ** 2 != std_len:
        print(f"⚠️ 全局配置警告: IMAGE_TOKEN_LENGTH ({std_len}) 本身非完全平方数！将依赖底层自适应裁切。")

    samples = load_jsonl(TEST_JSONL)[:MAX_SAMPLES]
    report: List[Dict[str, Any]] = []
    
    for sample in samples:
        sample_id = sample.get("id", "unknown")
        print(f"  -> 正在处理样本: {sample_id}")
        input_ids_list = sample["input_ids"]
        labels_list = sample["labels"]
        
        prompt_ids = [v for v, y in zip(input_ids_list, labels_list) if y == -100]
        # 提取真实存在的原始视觉 Token ID
        raw_true_img_ids = [v for v, y in zip(input_ids_list, labels_list) if y != -100 and v in image_token_set]

        # 动态锁定生成的硬目标长度：优先向 Ground Truth 靠拢，不足则依从全局配置
        target_gen_len = len(raw_true_img_ids) if len(raw_true_img_ids) > 0 else std_len

        prompt = torch.tensor([prompt_ids], device=model.device)
        full_in = torch.tensor([input_ids_list], device=model.device)
        full_lb = torch.tensor([labels_list], device=model.device)

        with torch.no_grad():
            # 1. 前向传播计算原生交叉熵与物理距离损失
            out = model(input_ids=full_in, labels=full_lb)
            loss_c = float(out.loss.item())
            shift_logits = out.logits[..., :-1, :].contiguous()
            shift_labels = full_lb[..., 1:].contiguous()
            mask = torch.isin(shift_labels, torch.tensor(image_token_ids, device=shift_labels.device))
            
            if mask.any():
                target_ids = shift_labels[mask]
                pred_logits = shift_logits[mask]
                img_logits = pred_logits.index_select(1, torch.tensor(image_token_ids, device=pred_logits.device))
                img_probs = torch.softmax(img_logits, dim=-1)
                idx = torch.tensor([id_to_code[int(t.item())] for t in target_ids], device=pred_logits.device)
                target_distances = distance_matrix[idx]
                loss_d = float(torch.mean(torch.sum(img_probs * target_distances, dim=-1)).item())
            else:
                loss_d = 0.0

            # 2. 严格受控推理生成：去除多余缓冲，强制模型吐出目标网格长度
            gen = model.generate(
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                min_new_tokens=target_gen_len,
                max_new_tokens=target_gen_len,  # 🔒 物理死锁：强制不多不少刚好输出指定数量
                do_sample=False,                # 贪心解码最稳定
                logits_processor=logits_processor,
                eos_token_id=eos,
                pad_token_id=eos,
            )

        # 3. 后处理绝对提纯与对齐防线
        new_ids = gen[0][prompt.shape[1] :].tolist()
        raw_pred_img_ids = [int(x) for x in new_ids if int(x) in image_token_set]

        # 强制自适应截取至合法完全平方数 (抵御任何可能的数据清洗残留或越界幻觉)
        pred_img_ids = _align_to_nearest_square(raw_pred_img_ids, std_len, "预测图")
        true_img_ids = _align_to_nearest_square(raw_true_img_ids, std_len, "真实图")

        # 4. 计算硬指标评估 (仅在对齐后计算，绝不抛错)
        hard_ld = float("nan")
        # 如果长度不一致，向下截取到两者的最短合法网格以完成可比计算
        eval_len = min(len(pred_img_ids), len(true_img_ids))
        if eval_len > 0:
            eval_side = math.isqrt(eval_len)
            eval_sq_len = eval_side * eval_side
            
            p_ids = pred_img_ids[:eval_sq_len]
            t_ids = true_img_ids[:eval_sq_len]
            
            p_tensor = torch.tensor([id_to_code[x] for x in p_ids], device=distance_matrix.device)
            t_tensor = torch.tensor([id_to_code[x] for x in t_ids], device=distance_matrix.device)
            hard_ld = float(distance_matrix[t_tensor, p_tensor].mean().item())

        # 5. 可视化渲染导出
        try:
            if pred_img_ids:
                decode_token_ids_to_image(pred_img_ids, vq_model, image_token_ids).save(out_dir / f"{sample_id}_pred.png")
            else:
                print("     ⚠️ 预测图片为空，跳过解码。")
        except Exception as e:
            print(f"     ⚠️ 预测图片解码失败: {e}")
            
        try:
            if true_img_ids:
                decode_token_ids_to_image(true_img_ids, vq_model, image_token_ids).save(out_dir / f"{sample_id}_gt.png")
            else:
                print("     ⚠️ 真实图片为空，跳过解码。")
        except Exception as e:
            print(f"     ⚠️ 真实图片解码失败: {e}")

        report.append(
            {
                "id": sample_id,
                "loss_c": loss_c,
                "loss_d": loss_d,
                "hard_ld": hard_ld,
                "num_pred_img_tokens": len(pred_img_ids),
                "num_true_img_tokens": len(true_img_ids),
            }
        )

    report_path = out_dir / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ 评估结束，报告保存至: {report_path.resolve()}")


if __name__ == "__main__":
    main()


# python ./MVoT-Anole/evaluate.py