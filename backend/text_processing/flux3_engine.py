# FLUX 3 文本编码（Qwen3-VL 4B，取 8 个隐藏层拼接成 20480 维）。
# 对齐 ComfyUI comfy/text_encoders/flux.py Flux3*：
#   * instruct 模板只包一次，包住整段 prompt
#   * 右侧用 151643 补齐到 80 的倍数；pad 位置在 encoder 里被 mask 掉
#   * pad 的 rope 位置固定为 1（reference 锁定的 transformers 4.57 行为）
#   * 取第 4/8/12/16/20/24/28/32 层，不做 final norm，按特征维拼接
# DiT 会 attend 所有 token（含 pad），所以输出保持完整补齐长度。

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.text_processing import emphasis, parsing

TEMPLATE_HEAD = "<|im_start|>user\n"
TEMPLATE_TAIL = "<|im_end|>\n<|im_start|>assistant\n"
PAD_ID = 151643
PAD_MULTIPLE = 80
LAYERS = [4, 8, 12, 16, 20, 24, 28, 32]


def _emphasis_option_name() -> str:
    """延迟取 opts.emphasis：本模块在 backend.loader 导入期就会被加载，
    此时 modules.shared 可能还没初始化完，直接 import 会触发循环导入。"""
    try:
        from modules.shared import opts

        return getattr(opts, "emphasis", None) or "Original"
    except Exception:
        return "Original"


def escape_brackets(text: str) -> str:
    # ComfyUI 只对 ( ) 做权重；[ ] 保持普通文本（布局 JSON 里大量出现）
    return text.replace("[", "\\[").replace("]", "\\]")


class Flux3TextProcessingEngine:
    def __init__(self, text_encoder, tokenizer):
        self.emphasis = emphasis.get_current_option(_emphasis_option_name())()
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer

    def _ids(self, text: str) -> list[int]:
        return self.tokenizer([text])["input_ids"][0] if text else []

    def tokenize(self, texts):
        return [self._ids(TEMPLATE_HEAD + t + TEMPLATE_TAIL) for t in texts]

    def tokenize_line(self, line: str):
        parsed = parsing.parse_prompt_attention(
            escape_brackets(line), getattr(self.emphasis, "name", "Original"))
        if len(parsed) == 1 or all(w == 1.0 for _, w in parsed):
            text = "".join(t for t, _ in parsed)
            tokens = self._ids(TEMPLATE_HEAD + text + TEMPLATE_TAIL)
            multipliers = [1.0] * len(tokens)
        else:
            tokens, multipliers = [], []
            head = self._ids(TEMPLATE_HEAD)
            tokens += head
            multipliers += [1.0] * len(head)
            for text, weight in parsed:
                ids = self._ids(text)
                tokens += ids
                multipliers += [weight] * len(ids)
            tail = self._ids(TEMPLATE_TAIL)
            tokens += tail
            multipliers += [1.0] * len(tail)

        n_real = len(tokens)
        n_pad = -n_real % PAD_MULTIPLE
        tokens += [PAD_ID] * n_pad
        multipliers += [1.0] * n_pad
        return tokens, multipliers, n_real

    def __call__(self, texts: "SdConditioning"):
        self.emphasis = emphasis.get_current_option(_emphasis_option_name())()
        if any(emphasis.uses_emphasis(x) for x in texts):
            dynamic_args.last_extra_generation_params["Emphasis"] = getattr(
                self.emphasis, "name", "Original")

        zs = []
        cache = {}
        for line in texts:
            if line not in cache:
                tokens, multipliers, n_real = self.tokenize_line(line)
                cache[line] = self.process_tokens(tokens, multipliers, n_real)
            zs.append(cache[line])
        return zs

    def process_tokens(self, tokens, multipliers, n_real):
        device = memory_management.text_encoder_device()
        ids = torch.tensor([tokens], device=device, dtype=torch.long)
        embeds = self.text_encoder.get_input_embeddings()(ids)

        mult = torch.asarray([multipliers], device=device, dtype=embeds.dtype)
        z = self.emphasis(embeds, mult)
        if z is not None:
            embeds = z

        mask = torch.zeros((1, len(tokens)), device=device, dtype=torch.long)
        mask[:, :n_real] = 1
        position_ids = (mask.cumsum(-1) - 1).masked_fill(mask == 0, 1)

        _, z = self.text_encoder(
            None,
            attention_mask=mask,
            embeds=embeds,
            num_tokens=[n_real],
            intermediate_output=list(LAYERS),
            final_layer_norm_intermediate=False,
            position_ids=position_ids,
        )
        # z: (1, taps, seq, d) -> (seq, taps * d)，与 ComfyUI 的 movedim(1, 2).flatten(2) 同序
        b, n, seq, d = z.shape
        return z.permute(0, 2, 1, 3).reshape(b, seq, n * d)[0]
