# import json
# from pathlib import Path
# import os

# import torch
# from datasets import load_dataset
# from peft import LoraConfig, TaskType, get_peft_model
# from transformers import DataCollatorForSeq2Seq, TrainingArguments

# # 如果你有自定义的 utils 和 trainer，请保留这些导入
# from anole_utils import load_anole_model, image_token_ids_from_range
# from custom_trainer import AnoleMVoTTrainer

# # =====================================================================
# # 🛠️ 全局参数配置区 (修改这里，然后直接运行 python 你的脚本名.py)
# # =====================================================================

# # 模型与数据路径
# # MODEL_NAME_OR_PATH = "./MVoT/hf_cache/models--leloy--Anole-7b-v0.1-hf"                 # 底座模型路径
# MODEL_NAME_OR_PATH = "/root/autodl-tmp/Anole-7b-v0.1-hf"
# DATASET_JSONL = "./MVoT-Anole/data/sft_训练样本_anole.jsonl"       # 你刚刚生成的格式化 SFT 数据集
# DISTANCE_MATRIX = "./MVoT-Anole/data/MSE查询表_anole.pt"                   # 你刚才计算出来的物理距离表
# OUTPUT_DIR = "./MVoT-Anole/outputs/anole_mvot_lora1"                      # 训练过程中的检查点保存路径
# FINAL_DIR = "./MVoT-Anole/outputs/anole_mvot_lora_final"                 # 最终融合模型权重的保存路径

# # 训练超参数
# EPOCHS = 3                                                    # 训练轮数
# BATCH_SIZE = 2                                                # 每个 GPU 的 Batch Size (如果显存不够可以改成 1)
# GRAD_ACCUM = 4                                                # 梯度累积步数 (等效总 Batch Size = BATCH_SIZE * GRAD_ACCUM)
# LEARNING_RATE = 2e-4                                          # LoRA 学习率
# ALPHA = 0.1                                                   # 物理损失 (Ld) 的权重比例
# REPORT_TO = "wandb"                                           # 训练日志汇报目标 (如果不使用 wandb，可以改成 "none")

# # 🚨 核心图像词表参数 (已自动对齐最新数据！)
# IMAGE_TOKEN_START = 8712                                      # 图像 Token 的起始 ID
# IMAGE_TOKEN_COUNT = 8192                                      # 图像 Token 的总数量

# LOCAL_FILES_ONLY = False                                      # 是否只使用本地文件
# # =====================================================================


# def main() -> None:
#     print(f"🚀 开始启动 MVoT + LoRA 训练引擎...")
#     print(f"📦 挂载基座模型: {MODEL_NAME_OR_PATH}")
#     print(f"📊 读取数据集: {DATASET_JSONL}")
    
#     # 1. 加载模型与处理器
#     local_rank = int(os.environ.get("LOCAL_RANK", "0"))

#     processor, model = load_anole_model(
#         model_name_or_path=MODEL_NAME_OR_PATH,
#         local_files_only=LOCAL_FILES_ONLY,
#         # device_map="auto",
#         device_map={"": local_rank},
#         dtype=torch.bfloat16,
#     )
#     tokenizer = processor.tokenizer
#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token

#     # 2. 注入 LoRA 物理外挂适配器
#     print("⚙️ 正在注入 LoRA 适配器...")
#     lora_config = LoraConfig(
#         r=16,
#         lora_alpha=32,
#         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
#         lora_dropout=0.05,
#         bias="none",
#         task_type=TaskType.CAUSAL_LM,
#     )
#     model = get_peft_model(model, lora_config)
#     model.print_trainable_parameters()

#     # 3. 加载数据集
#     dataset = load_dataset("json", data_files=DATASET_JSONL, split="train")

#     # 4. 加载物理距离矩阵 (核心灵魂)
#     print(f"⏳ 加载物理距离矩阵: {DISTANCE_MATRIX}")
#     dist_obj = torch.load(DISTANCE_MATRIX, map_location=model.device)
#     distance_matrix = dist_obj["distance_matrix"].to(model.device)

#     if "image_token_ids" in dist_obj:
#         image_token_ids = [int(x) for x in dist_obj["image_token_ids"]]
#     else:
#         image_token_ids = image_token_ids_from_range(IMAGE_TOKEN_START, IMAGE_TOKEN_COUNT)

#     # 5. 配置训练参数
#     training_args = TrainingArguments(
#         output_dir=OUTPUT_DIR,
#         num_train_epochs=EPOCHS,
#         per_device_train_batch_size=BATCH_SIZE,
#         gradient_accumulation_steps=GRAD_ACCUM,
#         learning_rate=LEARNING_RATE,
#         logging_steps=5,
#         # save_strategy="epoch",
#         save_strategy="steps",
#         save_steps=50,
#         bf16=True,
#         report_to=REPORT_TO,
#         run_name="MVoT_Anole_LoRA",
#         gradient_checkpointing=True,
#         ddp_find_unused_parameters=False,
#     )

#     collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
#     # 6. 初始化自定义物理引力 Trainer
#     trainer = AnoleMVoTTrainer(
#         distance_matrix=distance_matrix,
#         image_token_ids=image_token_ids,
#         alpha=ALPHA,
#         model=model,
#         args=training_args,
#         train_dataset=dataset,
#         data_collator=collator,
#     )

#     # 7. 开始训练
#     print("🔥 引擎点火，开始训练！")
#     trainer.train()
    
#     # 8. 保存模型
#     trainer.save_model(FINAL_DIR)
#     tokenizer.save_pretrained(FINAL_DIR)
    
#     print("\n✅ 训练圆满结束！")
#     print(f"💾 最终的 LoRA 权重已安全保存至 -> {Path(FINAL_DIR).resolve()}")


# if __name__ == "__main__":
#     main()



# # python ./MVoT-Anole/train.py

# # PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=4 ./MVoT-Anole/train.py



import json
from pathlib import Path
import os

import torch
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import DataCollatorForSeq2Seq, TrainingArguments

# 如果你有自定义的 utils 和 trainer，请保留这些导入
from anole_utils import load_anole_model, image_token_ids_from_range
from custom_trainer import AnoleMVoTTrainer

# =====================================================================
# 🛠️ 全局参数配置区 (修改这里，然后直接运行 python 你的脚本名.py)
# =====================================================================

# 模型与数据路径
# MODEL_NAME_OR_PATH = "./MVoT/hf_cache/models--leloy--Anole-7b-v0.1-hf"                 # 底座模型路径
MODEL_NAME_OR_PATH = "/root/autodl-tmp/Anole-7b-v0.1-hf"
DATASET_JSONL = "./MVoT-Anole/data/sft_训练样本_anole.jsonl"       # 你刚刚生成的格式化 SFT 数据集
DISTANCE_MATRIX = "./MVoT-Anole/data/MSE查询表_Anole.pt"                   # 你刚才计算出来的物理距离表
OUTPUT_DIR = "./MVoT-Anole/outputs/anole_mvot_lora"                      # 训练过程中的检查点保存路径
FINAL_DIR = "./MVoT-Anole/outputs/anole_mvot_lora_final"                 # 最终融合模型权重的保存路径

# 训练超参数
EPOCHS = 3                                                    # 训练轮数
BATCH_SIZE = 2                                                # 每个 GPU 的 Batch Size (如果显存不够可以改成 1)
GRAD_ACCUM = 4                                                # 梯度累积步数 (等效总 Batch Size = BATCH_SIZE * GRAD_ACCUM)
LEARNING_RATE = 2e-4                                          # LoRA 学习率
ALPHA = 1                                                   # 物理损失 (Ld) 的权重比例
REPORT_TO = "wandb"                                           # 训练日志汇报目标 (如果不使用 wandb，可以改成 "none")

# 🚨 核心图像词表参数 (已自动对齐最新数据！)
IMAGE_TOKEN_START = 8712                                      # 图像 Token 的起始 ID
IMAGE_TOKEN_COUNT = 16384                                     # 图像 Token 的总数量（来自外部VQGAN codebook）

LOCAL_FILES_ONLY = False                                      # 是否只使用本地文件
# =====================================================================


def main() -> None:
    print(f"🚀 开始启动 MVoT + LoRA 训练引擎...")
    print(f"📦 挂载基座模型: {MODEL_NAME_OR_PATH}")
    print(f"📊 读取数据集: {DATASET_JSONL}")
    
    # 1. 加载模型与处理器
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    processor, model = load_anole_model(
        model_name_or_path=MODEL_NAME_OR_PATH,
        local_files_only=LOCAL_FILES_ONLY,
        # device_map="auto",
        device_map={"": local_rank},
        dtype=torch.bfloat16,
    )
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 2. 注入 LoRA 物理外挂适配器
    print("⚙️ 正在注入 LoRA 适配器...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # 3. 加载数据集
    dataset = load_dataset("json", data_files=DATASET_JSONL, split="train")

    # 4. 加载物理距离矩阵 (核心灵魂)
    print(f"⏳ 加载物理距离矩阵: {DISTANCE_MATRIX}")
    dist_obj = torch.load(DISTANCE_MATRIX, map_location=model.device)
    distance_matrix = dist_obj["distance_matrix"].to(model.device)

    if "image_token_ids" in dist_obj:
        image_token_ids = [int(x) for x in dist_obj["image_token_ids"]]
    else:
        image_token_ids = image_token_ids_from_range(IMAGE_TOKEN_START, IMAGE_TOKEN_COUNT)

    # 5. 配置训练参数
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        logging_steps=5,
        # save_strategy="epoch",
        save_strategy="steps",
        save_steps=100,
        bf16=True,
        report_to=REPORT_TO,
        run_name="MVoT_Anole_LoRA",
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,
    )

    collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)
    
    # 6. 初始化自定义物理引力 Trainer
    trainer = AnoleMVoTTrainer(
        distance_matrix=distance_matrix,
        image_token_ids=image_token_ids,
        alpha=ALPHA,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )

    # 7. 开始训练
    print("🔥 引擎点火，开始训练！")
    trainer.train()
    
    # 8. 保存模型
    trainer.save_model(FINAL_DIR)
    tokenizer.save_pretrained(FINAL_DIR)
    
    print("\n✅ 训练圆满结束！")
    print(f"💾 最终的 LoRA 权重已安全保存至 -> {Path(FINAL_DIR).resolve()}")


if __name__ == "__main__":
    main()



# python ./MVoT-Anole/train.py

# PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True torchrun --nproc_per_node=4 ./MVoT-Anole/train.py
