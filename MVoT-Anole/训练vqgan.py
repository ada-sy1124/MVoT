import os
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pathlib import Path
from PIL import Image
import yaml
import importlib
from tqdm import tqdm
import random

# =====================================================================
# 🛠️ VQGAN 微调配置区 (4卡并行加速版)
# =====================================================================
VQGAN_CONFIG = "MVoT-Anole/vqgan/config.yaml"
VQGAN_CKPT = "MVoT-Anole/vqgan_finetuned/maze_vqgan_epoch_19.ckpt"  
IMAGE_ROOT = "data/600个单步样本image"
OUTPUT_DIR = "MVoT-Anole/vqgan_finetuned"

# 🌟 关键修改 1：既然有 4 张卡，全局 Batch Size 可以直接拉满！
# 如果 16 还是爆显存，就改成 8。如果显存空闲，可以拉到 32。
BATCH_SIZE = 8       
LEARNING_RATE = 1e-4 
START_EPOCH = 19   
TOTAL_EPOCHS = 25       

# 基础设备仍然是 cuda，但待会我们会分配给所有卡
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# =====================================================================

def instantiate_from_config(config):
    target = config.get("target")
    if target == "ldm.models.autoencoder.VQModel":
        target = "taming.models.vqgan.VQModel"
    module_name, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    params = dict(config.get("params", {}))
    if cls_name == "VQModel":
        params["lossconfig"] = {"target": "torch.nn.Identity"}
    return cls(**params)

class MazeDataset(Dataset):
    def __init__(self, image_dir, resolution):
        self.image_paths = list(Path(image_dir).rglob("*.png")) + list(Path(image_dir).rglob("*.jpg"))
        self.transform = transforms.Compose([
            transforms.Resize((resolution, resolution), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        try:
            # 尝试正常打开图片
            img = Image.open(self.image_paths[idx]).convert("RGB")
            return self.transform(img)
        except Exception as e:
            # ⚠️ 捕获到坏图时的容错机制
            # print(f"\n⚠️ 跳过损坏的图片: {self.image_paths[idx]}")
            # 随机从数据集里再抽一张完好的图顶替当前位置
            new_idx = random.randint(0, len(self.image_paths) - 1)
            return self.__getitem__(new_idx)

def main():
    print("🚀 启动 VQGAN 专属画笔微调任务 (多卡加速版)...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(VQGAN_CONFIG, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    resolution = int(cfg["model"]["params"]["ddconfig"]["resolution"])

    print("📦 正在加载初始模型 (ImageNet 预训练权重)...")
    model = instantiate_from_config(cfg["model"])
    ckpt = torch.load(VQGAN_CKPT, map_location="cpu", weights_only=False)
    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)
    
    # 🌟 关键修改 2：检测并开启多卡数据并行 (DataParallel)
    model.to(DEVICE)
    gpu_count = torch.cuda.device_count()
    if gpu_count > 1:
        print(f"🔥 检测到 {gpu_count} 张物理显卡！正在启动 DataParallel 并行加速...")
        model = torch.nn.DataParallel(model)
    else:
        print("⚠️ 只检测到 1 张显卡或 CPU，将使用单卡训练。")
        
    model.train() 

    print(f"📂 正在扫描图片目录: {IMAGE_ROOT}")
    dataset = MazeDataset(IMAGE_ROOT, resolution)
    # num_workers 可以稍微调大一点保证多卡供数充足
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, drop_last=True)
    print(f"✅ 找到 {len(dataset)} 张图片，全局 Batch Size 为 {BATCH_SIZE}，共 {len(dataloader)} 个 Batch。")

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print("\n🔥 开始洗脑训练！(将粉色自然风转为黑白像素风)")

    for epoch in range(START_EPOCH, TOTAL_EPOCHS):
        total_loss = 0
        total_recon_loss = 0
        
        # 顺便把进度条的显示也改对，这样你在控制台看着才舒服
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{TOTAL_EPOCHS}")
        
        for batch in pbar:    
            imgs = batch.to(DEVICE)
            
            optimizer.zero_grad()
            
            x_recon, q_loss = model(imgs)
            # 原本只有这一行 MSE
            recon_loss = F.mse_loss(x_recon, imgs)
            
            # 多卡返回的 loss 是每张卡的 list，需要 mean() 合并一下
            loss = recon_loss.mean() + q_loss.mean()
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.mean().item()
            
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}", 
                "Recon (MSE)": f"{recon_loss.mean().item():.4f}"
            })

        avg_loss = total_loss / len(dataloader)
        print(f"🎯 Epoch {epoch+1} 完成! 平均 Loss: {avg_loss:.4f}")

        # 🌟 关键修改 3：保存时脱掉 DataParallel 的外衣，防止后续加载失败
        if isinstance(model, torch.nn.DataParallel):
            save_state_dict = model.module.state_dict()
        else:
            save_state_dict = model.state_dict()
            
        save_path = os.path.join(OUTPUT_DIR, f"maze_vqgan_epoch_{epoch+1}.ckpt")
        torch.save({"state_dict": save_state_dict}, save_path)
        print(f"💾 模型已保存至: {save_path}")

    print("\n🎉 训练彻底完成！你的专属黑白迷宫画笔已诞生。")

if __name__ == "__main__":
    main()


# python ./MVoT-Anole/训练vqgan.py