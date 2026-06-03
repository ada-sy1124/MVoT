# import json
# from pathlib import Path
# from typing import Any, Dict, List

# import torch
# from tqdm import tqdm

# from anole_utils import (
#     encode_image_to_token_ids,
#     find_vq_model,
#     image_token_ids_from_range,
#     load_anole_model,
#     load_jsonl,
#     save_jsonl,
# )

# # =====================================================================
# # 🛠️ 全局参数配置区 (修改这里，然后直接运行 python 你的脚本名.py)
# # =====================================================================
# MODEL_NAME_OR_PATH = "/root/autodl-tmp/Anole-7b-v0.1-hf" 
# # MODEL_NAME_OR_PATH = "../MVoT/hf_cache/models--leloy--Anole-7b-v0.1-hf/snapshots/96df52301e844d8a624a13953051ead4c008343b"                 # 模型路径
# INPUT_JSONL = "data/训练样本/maze_dataset_merged_600.jsonl"   # 原始动作轨迹数据
# IMAGE_ROOT = "data/600个样本单步image"                        # 迷宫图像存放的根目录
# OUTPUT_JSONL = "MVoT-Anole/data/测试样本_anole.jsonl"        # 生成的 SFT 训练集保存路径
# # META_JSON = "data/训练样本/sft_dataset_anole.meta.json"       # 生成的元数据保存路径

# START_INDEX = 200                                               # 处理样本的起始索引
# END_INDEX = 201                                              # 处理样本的结束索引

# # 🚨 核心图像词表参数 (已根据最新探针结果自动更新！)
# IMAGE_TOKEN_START = 8712                                      # 图像 Token 的真实起始 ID
# IMAGE_TOKEN_COUNT = 8192                                     # 图像 Token 的总数量
# IMAGE_TOKEN_LENGTH = 1024                                     # 一张图像由多少个 Token 组成 (固定1024)

# LOCAL_FILES_ONLY = False                                      # 是否只使用本地缓存
# # =====================================================================


# def find_frame_path(sample_dir: Path, frame_index: int) -> Path:
#     matches = sorted(sample_dir.glob(f"*_move_{frame_index:04d}.png"))
#     if not matches:
#         return sample_dir / f"missing_move_{frame_index:04d}.png"
#     return matches[0]


# def collect_image_paths(sample_index: int, frame_count: int, image_root: str) -> List[str]:
#     sample_dir = Path(image_root) / f"sample_{sample_index:06d}"
#     return [str(find_frame_path(sample_dir, i)) for i in range(frame_count)]


# def main() -> None:
#     print(f"🚀 开始构建 Anole 专属格式的 SFT 训练集...")
#     print(f"📦 加载模型: {MODEL_NAME_OR_PATH}")
#     print(f"📊 处理范围: 第 {START_INDEX} 个 到 第 {END_INDEX} 个样本")

#     samples = load_jsonl(INPUT_JSONL)

#     # 1. 加载模型和处理器
#     processor, model = load_anole_model(
#         model_name_or_path=MODEL_NAME_OR_PATH,
#         local_files_only=LOCAL_FILES_ONLY,
#         # device_map="auto",
#         device_map="cpu",
#         # dtype=torch.bfloat16,
#         dtype=torch.float32,
#     )
#     model.eval()
#     tokenizer = processor.tokenizer
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # 2. 提取 VQ 模块和 Token 范围
#     vq_model = find_vq_model(model)
#     image_token_ids = image_token_ids_from_range(IMAGE_TOKEN_START, IMAGE_TOKEN_COUNT)

#     # 3. 准备 System Prompt
#     system_prompt = (
#         "system: You are a deterministic visual world model. "
#         "Map the given observation and action to the next visual state.\nuser: "
#     )
#     sys_ids = tokenizer.encode(system_prompt, add_special_tokens=False)

#     records: List[Dict[str, Any]] = []
#     selected = samples[START_INDEX:END_INDEX]
    
#     # 4. 开始遍历样本
#     for local_idx, sample in enumerate(tqdm(selected, desc="Building Anole SFT")):
#         sample_index = START_INDEX + local_idx
#         moves = sample["moves"]
#         frame_paths = collect_image_paths(sample_index, len(moves) + 1, IMAGE_ROOT)
#         encoded_frames = []
        
#         # 4.1 预先将所有的图像帧编码为 Token IDs
#         for p in frame_paths:
#             ids = encode_image_to_token_ids(p, processor, vq_model, model.device, image_token_ids)
#             if len(ids) != IMAGE_TOKEN_LENGTH:
#                 raise ValueError(
#                     f"Image token length mismatch for {p}: got {len(ids)}, "
#                     f"expected {IMAGE_TOKEN_LENGTH}."
#                 )
#             encoded_frames.append(ids)

#         # 4.2 组装 SFT 数据的 Input 和 Label
#         for step_idx, action in enumerate(moves):
#             img1_ids = encoded_frames[step_idx]       # 当前帧 (t)
#             img2_ids = encoded_frames[step_idx + 1]   # 下一帧 (t+1)
            
#             user_text_after_img = f"\nAction: {action}\nassistant: "
#             user_ids = tokenizer.encode(user_text_after_img, add_special_tokens=False)

#             prompt_ids = sys_ids + img1_ids + user_ids
#             target_ids = img2_ids + [tokenizer.eos_token_id]
            
#             # input_ids 是完整序列，labels 将 prompt 部分遮蔽 (设为 -100)，只计算目标图像的 Loss
#             input_ids = prompt_ids + target_ids
#             labels = [-100] * len(prompt_ids) + target_ids

#             records.append(
#                 {
#                     "id": f"maze_env_transition_{sample_index:06d}_{step_idx:04d}",
#                     "input_ids": input_ids,
#                     "labels": labels,
#                     "metadata": {
#                         "sample_index": sample_index,
#                         "step_index": step_idx,
#                         "action": action,
#                         "maze_id": sample.get("id"),
#                     },
#                 }
#             )

#     # 5. 保存结果
#     save_jsonl(records, OUTPUT_JSONL)
#     meta = {
#         "model": MODEL_NAME_OR_PATH,
#         "num_records": len(records),
#         "image_token_length": IMAGE_TOKEN_LENGTH,
#         "num_image_tokens": len(image_token_ids),
#         "min_image_token_id": min(image_token_ids),
#         "max_image_token_id": max(image_token_ids),
#         "start_index": START_INDEX,
#         "end_index": END_INDEX,
#     }
#     Path(META_JSON).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    
#     print("\n✅ 数据集构建完成！")
#     print(f"💾 成功保存 {len(records)} 条样本 -> {Path(OUTPUT_JSONL).resolve()}")
#     print(f"📄 元数据已保存 -> {Path(META_JSON).resolve()}")


# if __name__ == "__main__":
#     main()

# # python ./MVoT-Anole/prepare_sft_dataset.py



import json
import importlib
from pathlib import Path
from typing import Any, Dict, List

import torch
import yaml
from PIL import Image
from transformers import ChameleonProcessor
from tqdm import tqdm

# =====================================================================
# 🛠️ 运行配置区 (一键配置，直接运行)
# =====================================================================
MODEL_NAME_OR_PATH = "/root/autodl-tmp/Anole-7b-v0.1-hf"
VQGAN_CONFIG = "MVoT-Anole/vqgan/config.yaml"
VQGAN_CKPT = "./MVoT-Anole/vqgan_finetuned/maze_vqgan_epoch_25.ckpt"

INPUT_JSONL = "data/训练样本/maze_dataset_merged_600.jsonl"
IMAGE_ROOT = "data/600个单步样本image"

# OUTPUT_JSONL = "MVoT-Anole/data/sft_训练样本_anole.jsonl"
# META_JSON = "MVoT-Anole/data/sft_训练样本_anole.meta.json"
OUTPUT_JSONL = "MVoT-Anole/data/sft_测试样本_anole.jsonl"
META_JSON = "MVoT-Anole/data/sft_测试样本_anole.meta.json"

START_INDEX = 50
# END_INDEX = 250
END_INDEX = 51

# 🚨 词表对齐参数
IMAGE_TOKEN_START = 8192   # 必须与你模型 Embedding 层预留的图像起点一致
IMAGE_TOKEN_LENGTH = 1024  # 16x16的特征图展平通常是 1024

LOCAL_FILES_ONLY = False
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# =====================================================================

# ---------------------------------------------------------------------
# 🔧 内部工具函数 (脱离对 anole_utils 的依赖)
# ---------------------------------------------------------------------
def load_jsonl(path: str) -> List[Dict]:
    with open(path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(data: List[Dict], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

def instantiate_from_config(config: Dict[str, Any]) -> Any:
    target = config.get("target")
    if target == "ldm.models.autoencoder.VQModel":
        target = "taming.models.vqgan.VQModel"
    module_name, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    params = dict(config.get("params", {}))
    if cls_name == "VQModel" and "lossconfig" in params:
        params["lossconfig"] = {"target": "torch.nn.Identity"}
    return cls(**params)

def load_vqgan(config_path: str, ckpt_path: str, device: str):
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model = instantiate_from_config(cfg["model"])
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    return model.to(device).eval(), cfg

# def encode_image_to_token_ids(image_path: str, vq_model: Any, resolution: int, device: str, offset: int) -> List[int]:
#     # 1. 预处理图像
#     img = Image.open(image_path).convert("RGB").resize((resolution, resolution), Image.BICUBIC)
#     x = torch.tensor(list(img.getdata()), dtype=torch.float32).view(resolution, resolution, 3)
#     x = (x.permute(2, 0, 1).unsqueeze(0) / 255.0 * 2.0 - 1.0).to(device)
    
#     # 2. 抽取原始 Token (0 ~ 16383)
#     with torch.no_grad():
#         encoded = vq_model.encode(x)
#         z = encoded[0] if isinstance(encoded, tuple) else encoded
#         q_out = vq_model.quantize(z)
        
#         # 🌟 修复点：精准解析 taming 库的嵌套 tuple
#         # q_out[2] 是 info tuple: (perplexity, min_encodings, min_encoding_indices)
#         info = q_out[2]
#         if isinstance(info, tuple) and len(info) >= 3:
#             raw_indices_tensor = info[2]
#         elif torch.is_tensor(info):
#             raw_indices_tensor = info
#         else:
#             raise RuntimeError("无法从 quantize 输出中解析出 Token 索引。")
            
#         raw_indices = raw_indices_tensor.view(-1).to(torch.long).cpu().tolist()
    
#     # 3. 加上偏移量，映射到 LLM 的大词表
#     mapped_indices = [idx + offset for idx in raw_indices]
#     return mapped_indices

def encode_image_to_token_ids(image_path: str, vq_model: Any, resolution: int, device: str, offset: int) -> List[int]:
    # 1. 预处理图像
    img = Image.open(image_path).convert("RGB").resize((resolution, resolution), Image.BICUBIC)
    x = torch.tensor(list(img.getdata()), dtype=torch.float32).view(resolution, resolution, 3)
    x = (x.permute(2, 0, 1).unsqueeze(0) / 255.0 * 2.0 - 1.0).to(device)
    
    # 2. 抽取原始 Token (0 ~ 16383)
    with torch.no_grad():
        encoded = vq_model.encode(x)
        z = encoded[0] if isinstance(encoded, tuple) else encoded
        q_out = vq_model.quantize(z)
        
        info = q_out[2]
        if isinstance(info, tuple) and len(info) >= 3:
            raw_indices_tensor = info[2]
        elif torch.is_tensor(info):
            raw_indices_tensor = info
        else:
            raise RuntimeError("无法从 quantize 输出中解析出 Token 索引。")
            
        # 🌟 核心修复点：显式重塑为正确的空间维度 (1, 32, 32)，并强制物理连续化
        # VQGAN 在 resolution=256 时，标准下采样特征图必须是 32x32
        expected_side = resolution // 8
        total_tokens = expected_side * expected_side
        
        # 确保无论量化器底层输出的是一维还是三维，都强制统一排布
        grid_indices = raw_indices_tensor.reshape(1, expected_side, expected_side)
        
        # 强制按标准行扫描顺序 (从左到右，从上到下) 连续展平
        raw_indices = grid_indices.contiguous().view(-1).to(torch.long).cpu().tolist()
        
        # 终极物理保险：确保数量绝对准确
        if len(raw_indices) != total_tokens:
            raise ValueError(f"严重空间解析错误: 期望 {total_tokens} 个Token，实际提取了 {len(raw_indices)} 个。")
    
    # 3. 加上偏移量，映射到 LLM 的大词表
    mapped_indices = [idx + offset for idx in raw_indices]
    return mapped_indices


# ---------------------------------------------------------------------

def find_frame_path(sample_dir: Path, frame_index: int) -> Path:
    matches = sorted(sample_dir.glob(f"*_move_{frame_index:04d}.png"))
    if not matches:
        return sample_dir / f"missing_move_{frame_index:04d}.png"
    return matches[0]

def collect_image_paths(sample_index: int, frame_count: int, image_root: str) -> List[str]:
    sample_dir = Path(image_root) / f"sample_{sample_index:06d}"
    return [str(find_frame_path(sample_dir, i)) for i in range(frame_count)]

def main() -> None:
    print(f"🚀 开始构建多模态 SFT 训练集 (文本 + 图像)")
    print(f"💻 运行设备: {DEVICE}")
    
    samples = load_jsonl(INPUT_JSONL)
    selected = samples[START_INDEX:END_INDEX]

    # 📚 加载处理“文本”的家伙
    print("📦 加载文本分词器 (Tokenizer)...")
    processor = ChameleonProcessor.from_pretrained(MODEL_NAME_OR_PATH, local_files_only=LOCAL_FILES_ONLY)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 👁️ 加载处理“图像”的家伙
    print("🎨 加载图像编码器 (VQGAN)...")
    vq_model, vq_cfg = load_vqgan(VQGAN_CONFIG, VQGAN_CKPT, device=DEVICE)
    resolution = int(vq_cfg["model"]["params"]["ddconfig"]["resolution"])

    system_prompt = (
        "system: You are a deterministic visual world model. "
        "Map the given observation and action to the next visual state.\nuser: "
    )
    # 把文本变成 ID
    sys_ids = tokenizer.encode(system_prompt, add_special_tokens=False)

    records: List[Dict[str, Any]] = []
    
    for local_idx, sample in enumerate(tqdm(selected, desc="Building SFT")):
        sample_index = START_INDEX + local_idx
        moves = sample["moves"]
        frame_paths = collect_image_paths(sample_index, len(moves) + 1, IMAGE_ROOT)
        
        # 批量把这一局的图片全变成加上了偏移量的 ID
        encoded_frames = []
        for p in frame_paths:
            ids = encode_image_to_token_ids(
                image_path=p,
                vq_model=vq_model,
                resolution=resolution,
                device=DEVICE,
                offset=IMAGE_TOKEN_START # 加上 8192
            )
            if len(ids) != IMAGE_TOKEN_LENGTH:
                raise ValueError(f"尺寸不匹配: {p} 产生了 {len(ids)} 个 token, 预期 {IMAGE_TOKEN_LENGTH}")
            encoded_frames.append(ids)

        # 拼接数据结构
        for step_idx, action in enumerate(moves):
            img1_ids = encoded_frames[step_idx]       # 状态图 N
            img2_ids = encoded_frames[step_idx + 1]   # 状态图 N+1
            
            # 把动作文本变成 ID
            user_ids = tokenizer.encode(f"\nAction: {action}\nassistant: ", add_special_tokens=False)
            
            # 🌟 文本 ID 和 图像 ID 完美融合
            prompt_ids = sys_ids + img1_ids + user_ids
            target_ids = img2_ids + [tokenizer.eos_token_id]
            
            input_ids = prompt_ids + target_ids
            
            # -100 表示这部分不计算 Loss (只让模型学 assistant 后面的内容)
            labels = [-100] * len(prompt_ids) + target_ids
            
            records.append(
                {
                    "id": f"maze_env_transition_{sample_index:06d}_{step_idx:04d}",
                    "input_ids": input_ids,
                    "labels": labels,
                    "metadata": {
                        "sample_index": sample_index,
                        "step_index": step_idx,
                        "action": action,
                        "maze_id": sample.get("id"),
                    },
                }
            )

    save_jsonl(records, OUTPUT_JSONL)
    
    # 写入 Meta 数据
    meta = {
        "model": MODEL_NAME_OR_PATH,
        "vqgan_config": VQGAN_CONFIG,
        "vqgan_ckpt": VQGAN_CKPT,
        "num_records": len(records),
        "image_token_length": IMAGE_TOKEN_LENGTH,
        "image_token_offset": IMAGE_TOKEN_START,
        "start_index": START_INDEX,
        "end_index": END_INDEX,
    }
    Path(META_JSON).write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    
    print(f"\n✅ 成功将 {len(selected)} 局游戏的帧转化为了 {len(records)} 条 SFT 动作转移样本！")
    print(f"📂 数据已保存至 -> {Path(OUTPUT_JSONL).resolve()}")

if __name__ == "__main__":
    main()


# python ./MVoT-Anole/prepare_sft_dataset.py