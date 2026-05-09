# import json
# from pathlib import Path
# import torch

# # 如果需要导入你自己的工具包，请保留这行
# from anole_utils import image_token_ids_from_range, load_anole_model

# # =====================================================================
# # 🛠️ 全局参数配置区 (在这里直接改参数，然后无脑运行 python 脚本.py 即可)
# # =====================================================================

# MODEL_NAME_OR_PATH = "../MVoT/hf_cache/models--leloy--Anole-7b-v0.1-hf/snapshots/96df52301e844d8a624a13953051ead4c008343b"  # 底座模型路径
# OUTPUT_FILE_PATH = "data/MSE查询表_anole.pt"   # 矩阵保存路径
# IS_NORMALIZE = False                            # 是否进行归一化 (压缩到 0-1 之间，强烈建议 True)
# LOCAL_FILES_ONLY = False                       # 是否只使用本地文件不联网

# # 🚨 核心图像词表参数 (根据最新探针结果已更新)
# IMAGE_TOKEN_START = 8712                       # 图像 Token 的真实起始 ID
# IMAGE_TOKEN_COUNT = 8192                       # 图像 Token 的总数量

# # =====================================================================

# def main() -> None:
#     print(f"🚀 开始生成物理距离查询表...")
#     print(f"📦 模型: {MODEL_NAME_OR_PATH}")
#     print(f"🎯 图像词表范围: 起点 {IMAGE_TOKEN_START}, 总数 {IMAGE_TOKEN_COUNT}")

#     # 1. 加载模型
#     processor, model = load_anole_model(
#         model_name_or_path=MODEL_NAME_OR_PATH,
#         local_files_only=LOCAL_FILES_ONLY,
#         device_map="cpu", # 算矩阵用 CPU 就够了
#         dtype=torch.float32,
#     )
#     _ = processor.tokenizer
    
#     # 2. 获取目标 Token IDs
#     image_token_ids = image_token_ids_from_range(IMAGE_TOKEN_START, IMAGE_TOKEN_COUNT)

#     # 3. 截取底层的特征向量 (Embeddings)
#     emb = model.get_input_embeddings().weight.detach().to(torch.float32)
#     idx = torch.tensor(image_token_ids, dtype=torch.long)
#     image_embeddings = emb.index_select(0, idx)

#     print("🧮 正在计算 7672x7672 维度的 L2 物理距离矩阵，请稍候...")
    
#     # 4. 计算距离矩阵 (MSE)
#     dist = torch.cdist(image_embeddings, image_embeddings, p=2) ** 2
    
#     # 5. 归一化处理
#     if IS_NORMALIZE:
#         dist = dist / (dist.max() + 1e-8)
#         print("⚖️ 矩阵已完成归一化 (数值压缩至 0~1)。")

#     # 6. 保存 .pt 矩阵文件
#     out = Path(OUTPUT_FILE_PATH)
#     out.parent.mkdir(parents=True, exist_ok=True)
#     torch.save(
#         {
#             "distance_matrix": dist,
#             "image_token_ids": image_token_ids,
#             "num_image_tokens": len(image_token_ids),
#             "model": MODEL_NAME_OR_PATH,
#             "normalized": IS_NORMALIZE,
#         },
#         out,
#     )

#     # 7. 保存配套的 .json 说明文件
#     meta_path = out.with_suffix(".json")
#     meta_path.write_text(
#         json.dumps(
#             {
#                 "model": MODEL_NAME_OR_PATH,
#                 "normalized": IS_NORMALIZE,
#                 "num_image_tokens": len(image_token_ids),
#                 "min_image_token_id": min(image_token_ids),
#                 "max_image_token_id": max(image_token_ids),
#             },
#             ensure_ascii=False,
#             indent=2,
#         ),
#         encoding="utf-8",
#     )
    
#     print("\n✅ 任务圆满完成！")
#     print(f"💾 矩阵文件已保存至 -> {out.resolve()}")
#     print(f"📄 元数据已保存至   -> {meta_path.resolve()}")

# if __name__ == "__main__":
#     main()


import json
from pathlib import Path
import torch

# =====================================================================
# 🛠️ VQGAN 物理距离矩阵生成配置区 (直接一键运行)
# =====================================================================
VQGAN_CKPT_PATH = "MVoT-Anole/vqgan/model.ckpt"   # 你的开源 VQGAN 权重路径
OUTPUT_FILE_PATH = "MVoT-Anole/data/MSE查询表_Anole.pt" # 输出矩阵路径
IS_NORMALIZE = False                              # 是否归一化
# =====================================================================

def main() -> None:
    print(f"🚀 开始从 VQGAN 提取视觉特征生成物理距离查询表...")
    print(f"📦 权重文件: {VQGAN_CKPT_PATH}")

    # 1. 直接强行加载本地 CKPT (无需实例化整个模型结构)
    print("⏳ 正在读取模型字典...")
    ckpt = torch.load(VQGAN_CKPT_PATH, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    # 2. 精准狙击“视觉字典”(Codebook) 的权重张量
    # VQGAN 的核心特征表通常叫这个名字
    cb_keys = [k for k in sd.keys() if "quantize.embedding.weight" in k or "quantize.codebook.weight" in k]
    if not cb_keys:
        raise RuntimeError("🚨 在权重中找不到 Codebook (quantize) 张量，请检查 ckpt 文件！")
    
    cb_key = cb_keys[0]
    codebook_embeddings = sd[cb_key].to(torch.float32).detach()
    
    num_tokens, dim = codebook_embeddings.shape
    print(f"🎯 成功提取视觉特征表! 共 {num_tokens} 个图像 Token，每个维度为 {dim}。")

    # 3. 计算物理距离矩阵 (L2 距离平方, 即 MSE 的核心部分)
    print(f"🧮 正在计算 {num_tokens}x{num_tokens} 维度的物理距离矩阵，请稍候...")
    dist = torch.cdist(codebook_embeddings, codebook_embeddings, p=2) ** 2
    
    # 4. 归一化处理
    if IS_NORMALIZE:
        dist = dist / (dist.max() + 1e-8)
        print("⚖️ 矩阵已完成归一化 (数值压缩至 0~1)。")

    # 5. 保存 .pt 矩阵文件
    out = Path(OUTPUT_FILE_PATH)
    out.parent.mkdir(parents=True, exist_ok=True)
    
    # 注意：纯 VQGAN 的 ID 默认就是从 0 开始的
    image_token_ids = list(range(num_tokens))
    
    torch.save(
        {
            "distance_matrix": dist,
            "image_token_ids": image_token_ids,
            "num_image_tokens": num_tokens,
            "model": VQGAN_CKPT_PATH,
            "normalized": IS_NORMALIZE,
        },
        out,
    )

    # 6. 保存配套的 .json 说明文件
    meta_path = out.with_suffix(".json")
    meta_path.write_text(
        json.dumps(
            {
                "model": VQGAN_CKPT_PATH,
                "normalized": IS_NORMALIZE,
                "num_image_tokens": num_tokens,
                "min_image_token_id": min(image_token_ids),
                "max_image_token_id": max(image_token_ids),
                "embedding_dim": dim,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    
    print("\n✅ 任务圆满完成！")
    print(f"💾 矩阵文件已保存至 -> {out.resolve()}")
    print(f"📄 元数据已保存至   -> {meta_path.resolve()}")

if __name__ == "__main__":
    main()



# python ./MVoT-Anole/compute_distance_table.py