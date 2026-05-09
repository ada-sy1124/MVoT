# from typing import Sequence

# import torch
# from transformers import Trainer


# class AnoleMVoTTrainer(Trainer):
#     def __init__(
#         self,
#         distance_matrix: torch.Tensor,
#         image_token_ids: Sequence[int],
#         alpha: float = 1.0,
#         *args,
#         **kwargs,
#     ):
#         super().__init__(*args, **kwargs)
#         self.distance_matrix = distance_matrix
#         self.alpha = float(alpha)
#         self.image_token_ids = torch.tensor(list(image_token_ids), dtype=torch.long)
#         self.image_token_ids_sorted, _ = torch.sort(self.image_token_ids)

#         vocab_size = self.model.get_input_embeddings().weight.shape[0]
#         lookup = torch.full((vocab_size,), -1, dtype=torch.long)
#         for code_idx, token_id in enumerate(self.image_token_ids.tolist()):
#             if 0 <= token_id < vocab_size:
#                 lookup[token_id] = code_idx
#         self.lookup = lookup

#     def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
#         outputs = model(**inputs)
#         loss_c = outputs.loss

#         logits = outputs.logits[..., :-1, :].contiguous()
#         labels = inputs["labels"][..., 1:].contiguous()
#         loss_d = self._compute_token_discrepancy(logits, labels)
#         total = loss_c + self.alpha * loss_d

#         if self.model.training:
#             self.log(
#                 {
#                     "loss_details/loss_c_nlp": float(loss_c.item()),
#                     "loss_details/loss_d_physics": float(loss_d.item()),
#                     "loss_details/alpha": float(self.alpha),
#                     "loss_details/total_loss": float(total.item()),
#                 }
#             )
#         return (total, outputs) if return_outputs else total

#     def _compute_token_discrepancy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
#         # labels: [B, T], logits: [B, T, V]
#         image_ids = self.image_token_ids_sorted.to(labels.device)
#         mask = torch.isin(labels, image_ids)
#         if not mask.any():
#             return torch.tensor(0.0, device=logits.device, requires_grad=True)

#         target_ids = labels[mask]
#         pred_logits = logits[mask]  # [N, V]

#         image_token_ids = self.image_token_ids.to(pred_logits.device)
#         image_logits = pred_logits.index_select(1, image_token_ids)
#         image_probs = torch.softmax(image_logits, dim=-1)

#         lookup = self.lookup.to(target_ids.device)
#         matrix_indices = lookup[target_ids]
#         valid = matrix_indices >= 0
#         if not valid.any():
#             return torch.tensor(0.0, device=logits.device, requires_grad=True)

#         matrix_indices = matrix_indices[valid]
#         image_probs = image_probs[valid]
#         target_distances = self.distance_matrix.to(image_probs.device)[matrix_indices]
#         expected = torch.sum(image_probs * target_distances, dim=-1)
#         return torch.mean(expected)



from typing import Sequence

import torch
from transformers import Trainer


class AnoleMVoTTrainer(Trainer):
    def __init__(
        self,
        distance_matrix: torch.Tensor,
        image_token_ids: Sequence[int],
        alpha: float = 1.0,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.distance_matrix = distance_matrix
        self.alpha = float(alpha)
        self.image_token_ids = torch.tensor(list(image_token_ids), dtype=torch.long)
        self.image_token_ids_sorted, _ = torch.sort(self.image_token_ids)

        vocab_size = self.model.get_input_embeddings().weight.shape[0]
        lookup = torch.full((vocab_size,), -1, dtype=torch.long)
        for code_idx, token_id in enumerate(self.image_token_ids.tolist()):
            if 0 <= token_id < vocab_size:
                lookup[token_id] = code_idx
        self.lookup = lookup

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, **kwargs):
        outputs = model(**inputs)
        loss_c = outputs.loss

        logits = outputs.logits[..., :-1, :].contiguous()
        labels = inputs["labels"][..., 1:].contiguous()
        loss_d = self._compute_token_discrepancy(logits, labels)
        total = loss_c + self.alpha * loss_d

        if self.model.training:
            self.log(
                {
                    "loss_details/loss_c_nlp": float(loss_c.item()),
                    "loss_details/loss_d_physics": float(loss_d.item()),
                    "loss_details/alpha": float(self.alpha),
                    "loss_details/total_loss": float(total.item()),
                }
            )
        return (total, outputs) if return_outputs else total

    def _compute_token_discrepancy(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # labels: [B, T], logits: [B, T, V]
        image_ids = self.image_token_ids_sorted.to(labels.device)
        mask = torch.isin(labels, image_ids)
        if not mask.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        target_ids = labels[mask]
        pred_logits = logits[mask]  # [N, V]

        image_token_ids = self.image_token_ids.to(pred_logits.device)
        image_logits = pred_logits.index_select(1, image_token_ids)
        image_probs = torch.softmax(image_logits, dim=-1)

        lookup = self.lookup.to(target_ids.device)
        matrix_indices = lookup[target_ids]
        valid = matrix_indices >= 0
        if not valid.any():
            return torch.tensor(0.0, device=logits.device, requires_grad=True)

        matrix_indices = matrix_indices[valid]
        image_probs = image_probs[valid]
        target_distances = self.distance_matrix.to(image_probs.device)[matrix_indices]
        expected = torch.sum(image_probs * target_distances, dim=-1)
        return torch.mean(expected)
