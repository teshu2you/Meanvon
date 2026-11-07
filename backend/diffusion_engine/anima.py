from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch
from backend.huggingface_guess import model_list

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.anima_engine import Qwen06Engine
from modules.shared import opts


# ===== 参考图（ref_latents）总开关 =====
# Anima 的底座是 Cosmos-Predict2（MiniTrainDIT），上游没有“参考图”机制：
# 这里只能把参考 latent 当成额外一帧拼进时间轴。这个补丁能很好地把结构带过来，
# 但也会让模型倾向于“向参考帧插值” → 出图结构对、细节发糊。
#   False → 不使用参考帧，结构图回退到通用 img2img 初始 latent（默认，清晰）
#   True  → 使用参考帧（结构最准，细节偏软）
# 可用 config.default_anima_use_reference 覆盖。
ANIMA_USE_NATIVE_REFERENCE = False


def anima_reference_enabled() -> bool:
    try:
        import modules.config as _cfg
        v = getattr(_cfg, "default_anima_use_reference", None)
        if isinstance(v, bool):
            return v
    except Exception:
        pass
    return ANIMA_USE_NATIVE_REFERENCE


class Anima(ForgeDiffusionEngine):
    matched_guesses = [model_list.Anima]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3_06b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3_06b": huggingface_components["tokenizer"], "t5xxl": huggingface_components["tokenizer_2"]})

        vae = VAE(model=huggingface_components["vae"], is_wan=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen06Engine(
            text_encoder=clip.cond_stage_model.qwen3_06b,
            qwen_tokenizer=clip.tokenizer.qwen3_06b,
            t5_tokenizer=clip.tokenizer.t5xxl,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True
        self.use_shift = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        if not prompt.is_negative_prompt:
            if not opts.anima_do_reference or not anima_reference_enabled():
                dynamic_args.ref_latents.clear()
            else:
                _references = [*self.ref_latents]
                if self.ini_latent is not None:
                    _references.insert(0, self.ini_latent)
                    self.ini_latent = None
                dynamic_args.ref_latents = _references.copy()

        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt)[0])
        return token_count, max(512, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        samples: torch.Tensor = super().encode_first_stage(x)

        if opts.anima_do_reference and anima_reference_enabled():
            sample = samples[0].detach().clone().unsqueeze(0).cpu()
            if dynamic_args.is_referencing:
                self.ref_latents.append(sample)
            else:
                self.ini_latent = sample

        return samples
