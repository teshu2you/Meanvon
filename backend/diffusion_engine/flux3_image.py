# FLUX 3 Image engine（结构对齐 backend/diffusion_engine/flux2.py）。

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from modules.prompt_parser import SdConditioning

import torch

from backend import memory_management
from backend.args import dynamic_args
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.huggingface_guess import model_list
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.flux3_engine import Flux3TextProcessingEngine


class Flux3Image(ForgeDiffusionEngine):
    matched_guesses = [model_list.Flux3Image, model_list.Flux3Base]

    # loader 的 guess() 来自 modules_forge/packages 那份 model_list，
    # 与 backend 这份不是同一个类对象；两份都登记，避免每次加载都走
    # "matched by class-name fallback" 的兜底分支。
    # Flux3Base = Klein Base（非蒸馏，flow_map=False，shift 1.15）
    try:
        from modules_forge.packages.huggingface_guess import model_list as _mf_model_list

        matched_guesses.append(_mf_model_list.Flux3Image)
        matched_guesses.append(_mf_model_list.Flux3Base)
    except Exception:
        pass

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(
            model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]},
            tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]},
        )

        # FLUX.3 与 FLUX.2 共用 128 通道 / 16x 的 packed VAE
        vae = VAE(model=huggingface_components["vae"], is_flux2=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        self.text_processing_engine_flux3 = Flux3TextProcessingEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_4b,
            tokenizer=clip.tokenizer.qwen3vl_4b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_flux3 = True
        # cond / uncond 长度不同时，采样器不能做 batch 拼接
        dynamic_args.flux3 = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_flux3(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_flux3.tokenize([prompt])[0])
        return token_count, max(999, token_count)
