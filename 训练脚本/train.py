import torch
from transformers import ChameleonForConditionalGeneration, AutoTokenizer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, TaskType
from custom_trainer import ChameleonMVoTTrainer 

def main():
    print("🚀 启动变色龙纯净物理引擎微调流程...")

    # ==========================================
    # 步骤一：底座装载与分词器防错
    # ==========================================
    model_name_or_path = "facebook/chameleon-7b" 
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ChameleonForConditionalGeneration.from_pretrained(
        model_name_or_path, 
        dtype=torch.bfloat16, 
        device_map="auto",
        attn_implementation="flash_attention_2"           
    )

    # ==========================================
    # 步骤二：LoRA 低秩外挂注入
    # ==========================================
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
    # 步骤三：数据直接装载 (抛弃所有正则与解析！)
    # ==========================================
    # 🚨 确保这里读取的是你上一步刚刚生成的、只包含纯数组的 JSONL 文件
    dataset_path = "data/训练样本/sft_dataset.jsonl" 
    print(f"⚙️ 正在直接挂载纯净版物理数据集: {dataset_path} ...")
    
    # 这一步读出来的数据，已经自带了 input_ids 和 labels，不需要任何 map 函数处理！
    tokenized_dataset = load_dataset("json", data_files=dataset_path, split="train")

    # ==========================================
    # 步骤四：装载真实的物理规则矩阵
    # ==========================================
    matrix_path = "data/MSE查询表_unorm1.pt"
    print(f"⏳ 正在加载预计算的物理 Codebook 矩阵: {matrix_path}")
    try:
        loaded_obj = torch.load(matrix_path)
        precomputed_dist_matrix = loaded_obj["distance_matrix"].to(model.device)
        img_start_id = int(loaded_obj["img_start_id"])
        img_end_id = int(loaded_obj["img_end_id"])
    except FileNotFoundError:
        raise FileNotFoundError(f"找不到物理矩阵！")

    # ==========================================
    # 步骤五：控制台面板参数设定
    # ==========================================
    training_args = TrainingArguments(
        output_dir="./outputs/chameleon_mvot_lora1", 
        num_train_epochs=3,                  
        per_device_train_batch_size=2,       
        gradient_accumulation_steps=4,       
        learning_rate=2e-4,                  
        logging_steps=5,                     
        save_strategy="epoch",               
        bf16=True,                           
        report_to="wandb",                  
        run_name="Maze_Physics_Engine_Pure"    
    )

    # ==========================================
    # 步骤六：组装点火
    # ==========================================
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

    trainer = ChameleonMVoTTrainer(
        distance_matrix=precomputed_dist_matrix,
        img_start_id=img_start_id,  
        img_end_id=img_end_id,      
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset, # 🚨 直接把读取的纯净数据丢进来！
        data_collator=data_collator 
    )

    print("🔥 矩阵就绪，数据并网，点火！...")
    trainer.train()

    # ==========================================
    # 步骤七：回收战利品
    # ==========================================
    final_save_path = "./outputs/chameleon_mvot_lora_final1"
    trainer.save_model(final_save_path)
    tokenizer.save_pretrained(final_save_path)
    print(f"🎉 训练大圆满！保存在：{final_save_path}")

if __name__ == "__main__":
    main()


# python ./纯净版本/训练脚本/train.py