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

    def compute_loss(self, model, inputs, return_outputs=False):
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
        
        # 3. 计算物理状态偏移的惩罚 Loss (Ld)
        loss_d = self._compute_token_discrepancy(logits, labels)
        
        # 4. 损失融合：alpha 是物理规则的权重。
        # 如果你发现模型画图极其扭曲，可以把 alpha 调大；如果发现模型变得不会说话了，就调小。
        alpha = 1.0
        total_loss = loss_c + alpha * loss_d
        
        # 5. UI 推流：把三种 Loss 强行拆开推送到 Wandb 面板，方便你在网页上观察是哪个 Loss 没降下去
        if self.model.training: 
            self.log({
                "loss_details/loss_c_nlp": loss_c.item(),
                "loss_details/loss_d_physics": loss_d.item(),
                "loss_details/total_loss": total_loss.item()
            })
        
        return (total_loss, outputs) if return_outputs else total_loss

    def _compute_token_discrepancy(self, logits, labels):
        """
        论文核心公式落地：在这里计算模型画错的像素，在物理空间里到底偏了多远。
        """
        # 1. 制造掩码 (Mask)：
        # labels 里有很多 -100（代表不计算的地方）和纯文本的 ID。
        # 我们只关心答案是“图像”的那些位置。
        img_mask = (labels >= self.img_start_id) & (labels <= self.img_end_id)
        
        # 如果这一句话里根本没有图（比如在寒暄），物理 Loss 直接给 0
        if not img_mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        # 2. 数据剥离：把纯图像位置的“模型预测分”和“标准答案”抽出来
        target_img_ids = labels[img_mask] 
        pred_logits = logits[img_mask]    

        # 3. 概率转换：用 softmax 把冷冰冰的打分变成 0~1 之间、加起来等于 1 的概率分布
        pred_probs = torch.softmax(pred_logits, dim=-1) 
        
        # 4. 字典切片：模型可能给了“狗”或者“你好”一些概率，我们不要。
        # 我们只把模型分配给那 8192 个图像单词的概率切下来。
        img_probs = pred_probs[:, self.img_start_id : self.img_end_id + 1] 
        
        # 5. 索引降维（极易错点！）：
        # 你的答案 ID 是 10000 起步的，但矩阵是从 0 开始的。
        # 所以必须减去起始 ID，比如把 10012 变成 12，这样才能去矩阵里查第 12 行！
        matrix_indices = target_img_ids - self.img_start_id 
        
        # 6. 极速查表：GPU 直接去矩阵里抽出这些答案对应的距离向量
        target_distances = self.distance_matrix[matrix_indices]
        
        # 7. 期望计算（核心魔法）：
        # 将模型给出的概率 * 真实的物理距离。
        # 模型如果把高概率给到了距离很远的错误 Token 上，乘出来的偏差值就会极大！
        weighted_distances = img_probs * target_distances 
        expected_distances = torch.sum(weighted_distances, dim=-1) 
        
        # 8. 汇总均值：求所有图像 Token 的平均偏差，返回给优化器去挨打
        custom_loss = torch.mean(expected_distances)
        
        return custom_loss