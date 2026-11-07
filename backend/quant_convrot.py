# int8 + convrot（Comfy-Kitchen "ConvRot"）权重的反旋转。
#
# 量化端（comfy_kitchen/tensor/int8_utils.py）：
#   H    = normalized regular Hadamard(convrot_groupsize)   # h4 反复 kron，再 /sqrt(size)
#   W_rot = W @ H^T         按 in_features 每 group 一组
#   x_rot = x @ H           在线激活旋转 → (xH)(WH^T)^T == xW^T
#
# 本文件把它还原成“未旋转”的 int8：W = W_rot @ H（H 对称正交），
# 再按原来的 per-output-channel weight_scale 重新量化（行范数不变 → scale 仍有效，
# 存储仍是 int8，不会因为转 bf16 而爆显存）。
#
# 单独放一个轻量模块，避免为了复用而 import backend.loader（那会拉起整套引擎/循环导入）。

import json
import logging

import torch

logger = logging.getLogger("loader")

_HADAMARD_CACHE = {}


def regular_hadamard(size: int, dtype=torch.float32) -> torch.Tensor:
    """ConvRot 用的 normalized regular Hadamard（size 必须是 4 的幂，如 256）。"""
    if size < 4 or (size & (size - 1)) != 0:
        raise ValueError("regular Hadamard size must be a power of 4, got {}".format(size))
    if size in _HADAMARD_CACHE:
        return _HADAMARD_CACHE[size].to(dtype)

    h4 = torch.tensor(
        [[1, 1, 1, -1], [1, 1, -1, 1], [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float32)
    h, cur = h4, 4
    while cur < size:
        h = torch.kron(h, h4)
        cur *= 4
    h = h / (size ** 0.5)
    _HADAMARD_CACHE[size] = h
    return h.to(dtype)


def _json_u8(obj) -> torch.Tensor:
    return torch.tensor(list(json.dumps(obj).encode("utf-8")), dtype=torch.uint8)


def detect_convrot(sd: dict):
    """扫描 `comfy_quant`，返回 (是否带 convrot, convrot_groupsize)。"""
    for k, v in sd.items():
        if not str(k).endswith(".comfy_quant") or not torch.is_tensor(v):
            continue
        try:
            obj = json.loads(v.detach().cpu().numpy().tobytes().decode("utf-8", "replace"))
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("convrot"):
            return True, int(obj.get("convrot_groupsize") or 256)
    return False, 256


def unrotate_convrot_weights(sd: dict) -> int:
    """把带 `comfy_quant: {"convrot": true}` 的 int8 权重原地还原成未旋转布局。

    返回处理的层数（0 表示这批权重里没有 convrot）。
    """
    targets = {}
    for k, v in list(sd.items()):
        if not str(k).endswith(".comfy_quant") or not torch.is_tensor(v):
            continue
        try:
            obj = json.loads(v.detach().cpu().numpy().tobytes().decode("utf-8", "replace"))
        except Exception:
            continue
        if isinstance(obj, dict) and obj.get("convrot"):
            targets[str(k)[: -len(".comfy_quant")]] = (obj, int(obj.get("convrot_groupsize") or 256))

    if not targets:
        return 0

    done = 0
    for layer, (obj, gs) in targets.items():
        w = sd.get(layer + ".weight")
        s = sd.get(layer + ".weight_scale")
        if not torch.is_tensor(w) or not torch.is_tensor(s) or w.dtype != torch.int8:
            continue
        out_f, in_f = int(w.shape[0]), int(w.shape[1])
        if gs <= 0 or in_f % gs != 0:
            logger.warning("[quant] convrot 跳过 %s（in_features=%d %% groupsize=%d != 0）",
                           layer, in_f, gs)
            continue
        try:
            h = regular_hadamard(gs)
            wf = (w.to(torch.float32) * s.to(torch.float32).reshape(out_f, 1))
            wf = (wf.reshape(out_f, in_f // gs, gs) @ h).reshape(out_f, in_f)
            # Hadamard 旋转会把原来的“尖峰”分散到整行，反旋转后单元素最大值可能超过
            # 原 scale 对应的 127 → 直接复用旧 scale 会被 clamp 削峰（实测 2%+ 误差）。
            # 因此重算 per-output-channel scale（行范数不变，量化精度与原来同一量级）。
            new_scale = (wf.abs().amax(dim=1, keepdim=True) / 127.0).clamp_min(1e-8)
            sd[layer + ".weight"] = (wf / new_scale).round().clamp_(-127, 127).to(torch.int8)
            sd[layer + ".weight_scale"] = new_scale.to(s.dtype).reshape(s.shape)
            sd[layer + ".comfy_quant"] = _json_u8(
                {k: v for k, v in obj.items() if k not in ("convrot", "convrot_groupsize")})
            done += 1
        except Exception as e:
            logger.warning("[quant] convrot 还原失败 %s: %s", layer, e)

    if done:
        logger.info("[quant] convrot 反旋转完成：%d 个 int8 权重已还原为未旋转布局", done)
    return done
