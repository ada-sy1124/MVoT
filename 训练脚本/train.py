import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType

from custom_trainer import ChameleonMVoTTrainer 

# ==========================================
# ⚙️ 全局物理常量配置 
# (必须与 compute_distance_table.py 保持绝对一致！)
# ==========================================
IMG_START_ID = 10000 
NUM_IMAGE_TOKENS = 8192
IMG_END_ID = IMG_START_ID + NUM_IMAGE_TOKENS - 1

def main():
    print("🚀 启动变色龙物理引擎微调流程...")

    # ==========================================
    # 步骤一：底座装载与分词器防错
    # ==========================================
    model_name_or_path = "facebook/chameleon-7b" 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    # 【补丁操作】许多原生模型的 tokenizer 没有定义“填充符(pad_token)”。
    # 如果不管它，多个不同长度的样本凑成一个 batch 时会报错。这里强制拿 eos_token 顶替。
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载大模型本体，bf16 精度是现代炼丹的标配，既不掉精度又能节省一半显存
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        torch_dtype=torch.bfloat16, 
        device_map="auto"           
    )

    # ==========================================
    # 步骤二：LoRA 低秩外挂注入
    # ==========================================
    # 解释：不更新大模型原有的几百亿参数，而是像插U盘一样，
    # 在所有核心投影层 (q_proj, k_proj等) 旁边并联小矩阵。
    # 这让单张 4090/3090 也能练得动 7B 模型。
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], 
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters() 

    # ==========================================
    # 步骤三：数据流水线与监督遮罩 (Masking)
    # ==========================================
    raw_dataset = load_dataset("json", data_files="data/maze_dataset.jsonl", split="train")

    def preprocess_function(examples):
        """
        【极其重要】大模型微调的核心奥义：只学答案，不学问题！
        """
        # 提取你的输入(环境状态+动作) 和 输出(下一帧图片)
        prompts = examples["prompt"]
        responses = examples["response"]
        
        model_inputs = {"input_ids": [], "attention_mask": [], "labels": []}
        
        for prompt, response in zip(prompts, responses):
            # 1. 把人类能看懂的文本/图片，翻译成机器认识的 ID 数字序列
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            response_ids = tokenizer.encode(response + tokenizer.eos_token, add_special_tokens=False)
            
            # 2. 拼接总长度
            input_ids = prompt_ids + response_ids
            
            # 3. 制作标签 (Labels) 遮罩：
            # 在 PyTorch 中，任何标签为 -100 的地方，都不会计算 Loss！
            # 我们把 Prompt 部分强制替换成 -100，迫使模型只能通过预测 Response (下一帧图像) 来降低 Loss。
            labels = [-100] * len(prompt_ids) + response_ids
            
            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append([1] * len(input_ids)) # 1 代表这是真实内容，不是后面的补齐0
            model_inputs["labels"].append(labels)
            
        return model_inputs

    print("⚙️ 正在对数据进行 Tokenization 和 Labels 遮罩处理...")
    # map 函数会高并发地处理所有数据，并丢弃掉不能转化为 Tensor 的原始文本列
    tokenized_dataset = raw_dataset.map(
        preprocess_function,
        batched=True,
        remove_columns=raw_dataset.column_names 
    )

    # ==========================================
    # 步骤四：装载真实的物理规则矩阵
    # ==========================================
    matrix_path = "data/chameleon_distance_table.pt"
    print(f"⏳ 正在加载预计算的物理 Codebook 矩阵: {matrix_path}")
    try:
        # 直接把之前算好的 .pt 文件读进显卡显存
        loaded_obj = torch.load(matrix_path)
        precomputed_dist_matrix = loaded_obj["distance_matrix"].to(model.device)
        img_start_id = int(loaded_obj["img_start_id"])
        img_end_id = int(loaded_obj["img_end_id"])
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到物理矩阵！请先运行 compute_distance_table.py 生成该文件。")

    # ==========================================
    # 步骤五：控制台面板参数设定
    # ==========================================
    training_args = TrainingArguments(
        output_dir="./outputs/chameleon_mvot_lora", 
        num_train_epochs=3,                  
        per_device_train_batch_size=2,       
        gradient_accumulation_steps=4,       
        learning_rate=2e-4,                  
        logging_steps=5,                     
        save_strategy="epoch",               
        bf16=True,                           
        report_to="wandb", # 推荐用 wandb 绝美 UI 看你的 Lc 和 Ld 是怎么下降的                 
        run_name="Maze_Physics_Engine_v1"    
    )

    # ==========================================
    # 步骤六：组装点火
    # ==========================================
    # DataCollator 的职责：如果你的批次里有一条长样本一条短样本，
    # 它会自动在短样本的末尾补上 pad_token，并且把对应的 labels 也标为 -100 防误伤。
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    # 注入我们魔改过的包含物理规则的 Trainer
    trainer = ChameleonMVoTTrainer(
        distance_matrix=precomputed_dist_matrix,
        img_start_id=img_start_id,  # 动态传入起止 ID，从此告别硬编码！
        img_end_id=img_end_id,      
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator 
    )

    print("🔥 矩阵就绪，数据并网，点火！...")
    trainer.train()

    # ==========================================
    # 步骤七：回收战利品
    # ==========================================
    final_save_path = "./outputs/chameleon_mvot_lora_final"
    # 只会保存几十兆的 LoRA 权重，方便你极速分享或部署
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"🎉 训练大圆满！你的专属物理环境引擎已诞生，保存在：{final_save_path}")

if __name__ == "__main__":
    main()
