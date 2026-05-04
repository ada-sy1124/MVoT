import torch
from transformers import Trainer

class ChameleonMVoTTrainer(Trainer):
    def __init__(self, distance_matrix=None, img_start_id=None, img_end_id=None, *args, **kwargs):
        """
        构造函数（初始化）：
        我们要把在外面算好的“物理防抖字典”和“起止页码”强行塞进 Trainer 的肚子里。
        """
        super().__init__(*args, **kwargs)
        self.distance_matrix = distance_matrix
        self.img_start_id = img_start_id
        self.img_end_id = img_end_id
        
        # 【防御性编程】强制要求必须传入这三个参数，否则直接阻断训练，防止浪费显卡电费
        assert self.distance_matrix is not None, "必须传入距离矩阵！"
        assert self.img_start_id is not None and self.img_end_id is not None, "必须传入图像 Token 的起止 ID！"

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        """
        黑客拦截点：每次模型前向传播输出预测后，都会经过这个函数被打分（Loss）。
        我们要在这里把“语言规则分(Lc)”和“物理距离分(Ld)”揉在一起。
        """
        # 1. 放行：让模型正常跑完，拿到框架默认算好的交叉熵 Loss (Lc)
        outputs = model(**inputs)
        loss_c = outputs.loss 
        
        # 2. 截获：拿到模型对所有词汇的预测打分 (logits) 和 标准答案 (labels)
        logits = outputs.logits
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("inputs 中缺少 labels，无法计算训练损失。")
        
        # ==========================================
        # 🚨 核心修复 1：序列错位对齐 (Shift Alignment)
        # ==========================================
        # 变色龙是自回归模型，第 t 个位置的预测是为了生成第 t+1 个位置的 Token
        # 所以必须把 logits 掐尾，把 labels 去头，让它们在时间维度上严格对齐！
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # 3. 计算物理状态偏移的惩罚 Loss (Ld)，注意传入的是对齐后的 shift 变量！
        loss_d = self._compute_token_discrepancy(shift_logits, shift_labels)
        
       
        alpha = 1.0     
        total_loss = loss_c + alpha * loss_d
        
        # 5. UI 推流：推送到 Wandb 面板
        if self.model.training: 
            self.log({
                "loss_details/loss_c_nlp": loss_c.item(),
                "loss_details/loss_d_physics": loss_d.item(),
                "loss_details/alpha": alpha, # 把当前的 alpha 也监控起来
                "loss_details/total_loss": total_loss.item()
            })
        
        return (total_loss, outputs) if return_outputs else total_loss


    def _compute_token_discrepancy(self, logits, labels):
        img_mask = (labels >= self.img_start_id) & (labels <= self.img_end_id)
        
        if not img_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        target_img_ids = labels[img_mask] 
        pred_logits = logits[img_mask]    

        # ==========================================
        # 🚨 核心修复：先切片，再 Softmax！
        # ==========================================
        # 只把模型对 8192 个图像 Token 的原始打分（Logits）切下来
        img_logits = pred_logits[:, self.img_start_id : self.img_end_id + 1] 
        
        # 对这 8192 个打分做局部 Softmax。
        # 强制使得它们加起来等于 100%，无论模型原本多想输出文本，
        # 在这里我们只衡量它对“各个不同图像碎片”的偏好程度！
        img_probs = torch.softmax(img_logits, dim=-1) 
        
        # 索引降维（矩阵是从 0 开始的）
        matrix_indices = target_img_ids - self.img_start_id 
        
        # 极速查表
        target_distances = self.distance_matrix[matrix_indices]
        
        # 期望计算：现在的 img_probs 正常了，乘出来的距离期望才会是真实的物理偏差
        weighted_distances = img_probs * target_distances 
        expected_distances = torch.sum(weighted_distances, dim=-1) 
        
        custom_loss = torch.mean(expected_distances)
        
        return custom_loss
