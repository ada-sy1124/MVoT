import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import time

def generate_distance_table(
    model_name_or_path="facebook/chameleon-7b", 
    save_path="data/chameleon_distance_table.pt",
    img_start_id=10000, # 【外部传参】变色龙图像字典的第一页页码
    num_image_tokens=8192 # 【固定参数】变色龙图像字典的总页数
):
    print(f"📦 正在加载模型权重: {model_name_or_path}")
    
    # ==========================================
    # 第一步：轻量级加载（省显存技巧）
    # ==========================================
    # 解释：我们只需要提取词表，不需要做前向传播，所以用 device_map="cpu" 
    # 这样就算你没有显卡，在普通笔记本上也能瞬间跑完这个脚本。
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path, 
        device_map="cpu",       
        torch_dtype=torch.float32 # 必须用最高精度(float32)算距离，防止低精度带来误差
    )

    # ==========================================
    # 第二步：抽取大模型的“底层记忆库”
    # ==========================================
    print("🔍 正在抽取底层连续向量空间...")
    # get_input_embeddings() 拿到了所有文本和图像的底层高维向量。
    # .detach() 是告诉 PyTorch：我们只是拿数据看看，不需要算梯度（断开计算图，省内存）。
    embeddings = model.get_input_embeddings().weight.detach()

    # ==========================================
    # 第三步：精准切割“图像专属词典”
    # ==========================================
    vocab_size = tokenizer.vocab_size
    print(f"📊 模型总词表大小: {vocab_size}")
    
    img_end_id = img_start_id + num_image_tokens
    print(f"🎯 截取图像 Token 向量，范围: ID {img_start_id} 到 {img_end_id - 1}")
    
    # 【安全锁】防止你填错 ID 导致内存越界崩溃
    if img_end_id > embeddings.shape[0]:
        raise ValueError(f"配置的结束ID ({img_end_id}) 超出了模型的实际词表大小 ({embeddings.shape[0]})！")

    # 像切蛋糕一样，只把属于图像的那 8192 个向量切下来
    image_embeddings = embeddings[img_start_id:img_end_id]
    assert image_embeddings.shape[0] == num_image_tokens, "提取的 Token 数量错误！"

    # ==========================================
    # 第四步：暴力计算 8192 x 8192 的物理距离表
    # ==========================================
    print("🧮 正在计算 8192 x 8192 欧氏距离矩阵，请稍候...")
    start_time = time.time()
    
    # 魔法算子：torch.cdist 是底层 C++ 极度优化的函数，计算两两之间的欧氏距离
    euclidean_distances = torch.cdist(image_embeddings, image_embeddings, p=2)
    
    # 按照 MVoT 论文，我们需要的是均方误差 (MSE)，也就是欧氏距离的平方
    mse_matrix = euclidean_distances ** 2
    
    # ==========================================
    # 第五步：归一化（防止梯度爆炸的保命操作！）
    # ==========================================
    # 解释：高维空间的距离动辄几百上千，如果直接拿去当 Loss，反向传播的梯度会瞬间爆炸（变成 NaN）。
    # 所以我们把矩阵里最大的数字找出来，让所有人除以它，强行把距离压缩到 0.0 ~ 1.0 之间。
    max_distance = mse_matrix.max()
    normalized_matrix = mse_matrix / max_distance
    
    end_time = time.time()
    print(f"⚡ 计算完成！耗时: {end_time - start_time:.2f} 秒")

    # ==========================================
    # 第六步：固化到硬盘
    # ==========================================
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 保存为 .pt 文件，训练时就可以直接用 GPU 瞬间读取，省去了每次查表都要算的时间
    torch.save(
        {
            "distance_matrix": normalized_matrix,
            "img_start_id": int(img_start_id),
            "img_end_id": int(img_end_id - 1),
            "num_image_tokens": int(num_image_tokens),
        },
        save_path,
    )
    print(f"🎉 物理表已成功铸造！保存在: {os.path.abspath(save_path)}")

if __name__ == "__main__":
    # 【入口点】如果变色龙图像Token的真实起点是其他数字，请在这里修改！
    generate_distance_table(img_start_id=10000, num_image_tokens=8192)
