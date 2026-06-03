# import json
# import math
# import re
# from pathlib import Path
# from typing import Any, Dict, List, Sequence, Tuple

# import torch
# from PIL import Image
# from transformers import ChameleonForConditionalGeneration, ChameleonProcessor

# DEFAULT_IMAGE_TOKEN_START = 8712
# DEFAULT_IMAGE_TOKEN_COUNT = 8192
# DEFAULT_IMAGE_TOKEN_LENGTH = 1024


# def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
#     records: List[Dict[str, Any]] = []
#     with Path(path).open("r", encoding="utf-8") as f:
#         for line_no, line in enumerate(f, start=1):
#             raw = line.strip()
#             if not raw:
#                 continue
#             try:
#                 records.append(json.loads(raw))
#             except json.JSONDecodeError as e:
#                 raise ValueError(f"Invalid JSON at {path}:{line_no}: {e}") from e
#     return records


# def save_jsonl(records: Sequence[Dict[str, Any]], path: str | Path) -> None:
#     out = Path(path)
#     out.parent.mkdir(parents=True, exist_ok=True)
#     with out.open("w", encoding="utf-8") as f:
#         for r in records:
#             f.write(json.dumps(r, ensure_ascii=False) + "\n")


# def load_anole_model(
#     model_name_or_path: str,
#     local_files_only: bool = False,
#     device_map: str = "auto",
#     dtype: torch.dtype = torch.bfloat16,
# ) -> Tuple[ChameleonProcessor, ChameleonForConditionalGeneration]:
#     processor = ChameleonProcessor.from_pretrained(
#         model_name_or_path,
#         local_files_only=local_files_only,
#     )
#     model = ChameleonForConditionalGeneration.from_pretrained(
#         model_name_or_path,
#         local_files_only=local_files_only,
#         device_map=device_map,
#         dtype=dtype,
#     )
#     return processor, model


# def find_vq_model(model: Any) -> Any:
#     candidates = [
#         ["model", "vqmodel"],
#         ["model", "vq_model"],
#         ["vqmodel"],
#         ["vq_model"],
#     ]
#     for chain in candidates:
#         cur = model
#         ok = True
#         for name in chain:
#             if not hasattr(cur, name):
#                 ok = False
#                 break
#             cur = getattr(cur, name)
#         if ok:
#             return cur
#     raise AttributeError("Cannot find VQ model (vqmodel/vq_model).")


# def get_image_token_ids(model: Any, tokenizer: Any) -> List[int]:
#     # 1) Preferred source: model.vocabulary_mapping.image_tokens
#     mapping = getattr(model, "vocabulary_mapping", None)
#     if mapping is not None and hasattr(mapping, "image_tokens"):
#         vals = getattr(mapping, "image_tokens")
#         if isinstance(vals, torch.Tensor):
#             ids = [int(x) for x in vals.view(-1).tolist()]
#         else:
#             ids = [int(x) for x in vals]
#         ids = sorted(set(ids))
#         if ids:
#             return ids

#     # 2) Fallback A: tokenizer token names like <image_123>
#     vocab = tokenizer.get_vocab()
#     pattern = re.compile(r"<image_(\d+)>$")
#     pairs: List[Tuple[int, int]] = []
#     for token, token_id in vocab.items():
#         m = pattern.match(token)
#         if m:
#             idx = int(m.group(1))
#             pairs.append((idx, int(token_id)))
#     if pairs:
#         pairs.sort(key=lambda x: x[0])
#         return [token_id for _, token_id in pairs]

#     # 3) Fallback B: reserved tokens, choose the most plausible contiguous block.
#     # Common in Chameleon/Anole-family tokenizers: <reserved08711> etc.
#     reserved_patterns = [
#         re.compile(r"<reserved(\d+)>$"),
#         re.compile(r"<\|reserved_(\d+)\|>$"),
#     ]
#     reserved_ids: List[int] = []
#     for token, token_id in vocab.items():
#         for rg in reserved_patterns:
#             if rg.match(token):
#                 reserved_ids.append(int(token_id))
#                 break

#     if reserved_ids:
#         reserved_ids = sorted(set(reserved_ids))
#         # Build contiguous segments
#         segments: List[List[int]] = []
#         cur: List[int] = [reserved_ids[0]]
#         for x in reserved_ids[1:]:
#             if x == cur[-1] + 1:
#                 cur.append(x)
#             else:
#                 segments.append(cur)
#                 cur = [x]
#         segments.append(cur)

#         # Target count from config if available
#         target = None
#         cfg = getattr(model, "config", None)
#         if cfg is not None:
#             vq_cfg = getattr(cfg, "vq_config", None)
#             if isinstance(vq_cfg, dict):
#                 for k in ["num_embeddings", "n_embed", "codebook_size"]:
#                     if k in vq_cfg:
#                         target = int(vq_cfg[k])
#                         break
#             elif vq_cfg is not None:
#                 for k in ["num_embeddings", "n_embed", "codebook_size"]:
#                     if hasattr(vq_cfg, k):
#                         target = int(getattr(vq_cfg, k))
#                         break

#         # Pick best segment: prefer one whose length is closest to target, else longest.
#         def score(seg: List[int]) -> Tuple[int, int]:
#             if target is not None:
#                 return (abs(len(seg) - target), -len(seg))
#             return (0, -len(seg))

#         best = sorted(segments, key=score)[0]
#         if len(best) >= 1024:
#             return best

#     raise RuntimeError("Cannot locate image token ids from model/tokenizer.")


# def image_token_maps(image_token_ids: Sequence[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
#     id_to_code = {int(token_id): int(code_idx) for code_idx, token_id in enumerate(image_token_ids)}
#     code_to_id = {int(code_idx): int(token_id) for code_idx, token_id in enumerate(image_token_ids)}
#     return id_to_code, code_to_id


# def image_token_ids_from_range(
#     start_id: int = DEFAULT_IMAGE_TOKEN_START,
#     count: int = DEFAULT_IMAGE_TOKEN_COUNT,
# ) -> List[int]:
#     return list(range(int(start_id), int(start_id) + int(count)))


# def encode_image_to_token_ids(
#     image_path: str | Path,
#     processor: ChameleonProcessor,
#     vq_model: Any,
#     device: torch.device,
#     image_token_ids: Sequence[int],
# ) -> List[int]:
#     id_count = len(image_token_ids)
#     code_to_id = {i: int(image_token_ids[i]) for i in range(id_count)}

#     img = Image.open(image_path).convert("RGB")
#     pixel_values = processor.image_processor(img, return_tensors="pt")["pixel_values"].to(
#         device=device, dtype=torch.float32
#     )
#     with torch.no_grad():
#         z = vq_model.encode(pixel_values)
#         z_latents = z.latents if hasattr(z, "latents") else z[0]
#         _, _, vq_indices = vq_model.quantize(z_latents)

#     code_indices = [int(x) for x in vq_indices.view(-1).tolist()]
#     max_idx = len(image_token_ids) - 1
#     if any(idx < 0 or idx > max_idx for idx in code_indices):
#         bad = [idx for idx in code_indices if idx < 0 or idx > max_idx][:5]
#         raise ValueError(
#             f"VQ index out of range for configured image token count={len(image_token_ids)}. "
#             f"Examples: {bad}"
#         )
#     return [code_to_id[idx] for idx in code_indices]


# def decode_token_ids_to_image(
#     token_ids: Sequence[int],
#     vq_model: Any,
#     image_token_ids: Sequence[int],
# ) -> Image.Image:
#     id_to_code, _ = image_token_maps(image_token_ids)
#     codes = [id_to_code[int(t)] for t in token_ids]
#     n = len(codes)
#     side = int(math.sqrt(n))
#     if side * side != n:
#         raise ValueError(f"Token length {n} is not a square number.")

#     vq_indices = torch.tensor(codes, dtype=torch.long).view(1, side, side)

#     with torch.no_grad():
#         if hasattr(vq_model, "decode_code"):
#             try:
#                 pixels = vq_model.decode_code(vq_indices)
#             except Exception:
#                 pixels = vq_model.decode_code(vq_indices.view(1, -1))
#         else:
#             if hasattr(vq_model.quantize, "get_codebook_entry"):
#                 embed_dim = getattr(getattr(vq_model, "config", None), "embed_dim", 256)
#                 z = vq_model.quantize.get_codebook_entry(
#                     vq_indices,
#                     shape=(1, side, side, embed_dim),
#                 )
#             else:
#                 z = vq_model.quantize.embedding(vq_indices)
#             z = z.permute(0, 3, 1, 2).contiguous()
#             if hasattr(vq_model, "decode"):
#                 pixels = vq_model.decode(z)
#             elif hasattr(vq_model, "decoder"):
#                 pixels = vq_model.decoder(z)
#             else:
#                 raise AttributeError("No decode/decode_code/decoder found in VQ model.")

#     pixels = torch.clamp((pixels + 1.0) / 2.0, 0.0, 1.0)
#     arr = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
#     return Image.fromarray(arr)




import importlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import torch
import yaml
from PIL import Image
from transformers import ChameleonForConditionalGeneration, ChameleonProcessor


DEFAULT_IMAGE_TOKEN_START = 0
DEFAULT_IMAGE_TOKEN_COUNT = 16384
DEFAULT_IMAGE_TOKEN_LENGTH = 1024


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            raw = line.strip()
            if not raw:
                continue
            records.append(json.loads(raw))
    return records


def save_jsonl(records: Sequence[Dict[str, Any]], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def image_token_ids_from_range(start_id: int = DEFAULT_IMAGE_TOKEN_START, count: int = DEFAULT_IMAGE_TOKEN_COUNT) -> List[int]:
    return list(range(int(start_id), int(start_id) + int(count)))


def image_token_maps(image_token_ids: Sequence[int]) -> Tuple[Dict[int, int], Dict[int, int]]:
    id_to_code = {int(token_id): int(code_idx) for code_idx, token_id in enumerate(image_token_ids)}
    code_to_id = {int(code_idx): int(token_id) for code_idx, token_id in enumerate(image_token_ids)}
    return id_to_code, code_to_id


def load_anole_model(
    model_name_or_path: str,
    local_files_only: bool = False,
    device_map: str | Dict[str, int] = "auto",
    dtype: torch.dtype = torch.bfloat16,
) -> Tuple[ChameleonProcessor, ChameleonForConditionalGeneration]:
    processor = ChameleonProcessor.from_pretrained(model_name_or_path, local_files_only=local_files_only)
    model = ChameleonForConditionalGeneration.from_pretrained(
        model_name_or_path,
        local_files_only=local_files_only,
        device_map=device_map,
        dtype=dtype,
    )
    return processor, model


def _instantiate_from_config(config: Dict[str, Any]) -> Any:
    target = config.get("target")
    if not target:
        raise ValueError("Config missing target.")
    if target == "ldm.models.autoencoder.VQModel":
        target = "taming.models.vqgan.VQModel"
    module_name, cls_name = target.rsplit(".", 1)
    module = importlib.import_module(module_name)
    cls = getattr(module, cls_name)
    params = dict(config.get("params", {}))
    if cls_name == "VQModel" and "lossconfig" in params:
        params["lossconfig"] = {"target": "torch.nn.Identity"}
    return cls(**params)


def load_vqgan(config_path: str | Path, ckpt_path: str | Path, device: str = "cpu") -> Tuple[Any, Dict[str, Any]]:
    cfg_path = Path(config_path)
    ckpt = Path(ckpt_path)
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    model_cfg = cfg["model"]
    vq = _instantiate_from_config(model_cfg)
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    state_dict = state["state_dict"] if isinstance(state, dict) and "state_dict" in state else state
    vq.load_state_dict(state_dict, strict=False)
    vq = vq.to(device).eval()
    return vq, cfg


def vqgan_codebook_size(vq_model: Any) -> int:
    quant = getattr(vq_model, "quantize", None)
    if quant is not None and hasattr(quant, "embedding") and hasattr(quant.embedding, "num_embeddings"):
        return int(quant.embedding.num_embeddings)
    raise RuntimeError("Cannot infer codebook size from VQGAN.")


def encode_image_to_token_ids(
    image_path: str | Path,
    vq_model: Any,
    image_token_ids: Sequence[int],
    resolution: int = 256,
    device: str = "cpu",
) -> List[int]:
    _, code_to_id = image_token_maps(image_token_ids)
    img = Image.open(image_path).convert("RGB").resize((resolution, resolution), Image.BICUBIC)
    x = torch.tensor(list(img.getdata()), dtype=torch.float32).view(resolution, resolution, 3)
    x = x.permute(2, 0, 1).unsqueeze(0) / 255.0
    x = (x * 2.0 - 1.0).to(device)

    with torch.no_grad():
        encoded = vq_model.encode(x)
    if isinstance(encoded, tuple) and len(encoded) >= 3 and isinstance(encoded[2], tuple) and len(encoded[2]) >= 3:
        indices = encoded[2][2].view(-1).to(torch.long)
    elif isinstance(encoded, tuple):
        z = encoded[0]
        q_out = vq_model.quantize(z)
        indices = q_out[2].view(-1).to(torch.long)
    else:
        q_out = vq_model.quantize(encoded)
        indices = q_out[2].view(-1).to(torch.long)

    code_indices = [int(v) for v in indices.tolist()]
    max_idx = len(image_token_ids) - 1
    bad = [idx for idx in code_indices if idx < 0 or idx > max_idx]
    if bad:
        raise ValueError(f"VQ index out of range for configured image token count={len(image_token_ids)}. Examples: {bad[:5]}")
    return [code_to_id[idx] for idx in code_indices]


# def decode_token_ids_to_image(token_ids: Sequence[int], vq_model: Any, image_token_ids: Sequence[int]) -> Image.Image:
#     id_to_code, _ = image_token_maps(image_token_ids)
#     codes = [id_to_code[int(t)] for t in token_ids]
#     n = len(codes)
#     side = int(math.sqrt(n))
#     if side * side != n:
#         raise ValueError(f"Token length {n} is not a square number.")

#     vq_indices = torch.tensor(codes, dtype=torch.long).view(1, side, side)
#     with torch.no_grad():
#         if hasattr(vq_model, "decode_code"):
#             try:
#                 pixels = vq_model.decode_code(vq_indices)
#             except Exception:
#                 pixels = vq_model.decode_code(vq_indices.view(1, -1))
#         else:
#             z = vq_model.quantize.get_codebook_entry(vq_indices, shape=(1, side, side, getattr(vq_model, "embed_dim", 4)))
#             z = z.permute(0, 3, 1, 2).contiguous()
#             pixels = vq_model.decode(z)
#     pixels = torch.clamp((pixels + 1.0) / 2.0, 0.0, 1.0)
#     arr = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
#     return Image.fromarray(arr)


def decode_token_ids_to_image(token_ids: Sequence[int], vq_model: Any, image_token_ids: Sequence[int]) -> Image.Image:
    id_to_code, _ = image_token_maps(image_token_ids)
    
    # 1. 严格过滤纯净视觉 ID
    valid_token_ids = [t for t in token_ids if int(t) in id_to_code]
    codes = [id_to_code[int(t)] for t in valid_token_ids]
    
    n = len(codes)
    if n == 0:
        raise ValueError("解码失败：输入的有效视觉 Token 数量为 0。")

    # 2. 动态自适应完美平方数校准
    side = int(math.isqrt(n))
    if side * side != n:
        target_side = 32 if n >= 1024 else (16 if n >= 256 else side)
        target_len = target_side * target_side
        print(f"⚠️ [底层防护] 输入长度 ({n}) 非正方形，强行截取前 {target_len} 个 Token 映射为 {target_side}x{target_side} 矩阵...")
        codes = codes[:target_len]
        side = target_side

    # 将一维 code 序列变成二维空间网格 indices: shape (1, side, side)
    device = next(vq_model.parameters()).device
    vq_indices = torch.tensor(codes, dtype=torch.long, device=device).view(1, side, side)
    
    # 3. 🚨 暴利直连解码核心 (彻底规避 embed_code 报错)
    with torch.no_grad():
        quantizer = getattr(vq_model, "quantize", None)
        
        # 路线 A：优先尝试直接走标准的 PyTorch nn.Embedding 查表 (最稳健，100% 不报错)
        if quantizer is not None and hasattr(quantizer, "embedding"):
            # 直接输入 (1, side, side) 查表得到 (1, side, side, embed_dim)
            z_q = quantizer.embedding(vq_indices)
            # 重排维度适配解码器卷积层要求：(Batch, Channel, Height, Width) -> (1, embed_dim, side, side)
            z_q = z_q.permute(0, 3, 1, 2).contiguous()
            pixels = vq_model.decode(z_q)
            
        # 路线 B：备用兼容原生 VQGAN 方法
        elif hasattr(vq_model, "decode_code"):
            try:
                pixels = vq_model.decode_code(vq_indices)
            except Exception:
                pixels = vq_model.decode_code(vq_indices.view(1, -1))
        else:
            # 极限兜底
            raise RuntimeError("VQGAN 解码器结构异常：既没有标准的 .embedding 属性，也没有 .decode_code 方法。")
            
    # 4. 图像张量后处理渲染
    pixels = torch.clamp((pixels + 1.0) / 2.0, 0.0, 1.0)
    arr = (pixels[0].permute(1, 2, 0).cpu().numpy() * 255).astype("uint8")
    
    # 确保输出单通道灰度图或 RGB 均兼容
    if arr.shape[2] == 1:
        arr = arr.squeeze(-1)
        return Image.fromarray(arr, mode="L")
    return Image.fromarray(arr, mode="RGB")