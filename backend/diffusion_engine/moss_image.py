import torch
from backend.huggingface_guess import model_list

from backend import memory_management
from backend.diffusion_engine.base import ForgeDiffusionEngine, ForgeObjects
from backend.patcher.clip import CLIP
from backend.patcher.unet import UnetPatcher
from backend.patcher.vae import VAE
from backend.text_processing.moss_image_engine import Qwen359BEngine


class MOSSImage(ForgeDiffusionEngine):
    matched_guesses = [model_list.MOSSImage]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen35_9b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen35_9b": huggingface_components["tokenizer"]})

        # MOSS-Image 用的是 AutoencoderKLFlux2 的「图像」用法：
        # 空间 16× 下采样 + 32ch × patch(2,2) = 128ch latent。
        # 不能复用 is_flux2=True（那条分支是给视频/打包路径的），
        # 否则 latent 通道与采样网格会与 DiT 的 RoPE 尺度错位 → 过曝。
        vae = VAE(model=huggingface_components["vae"], is_flux2_image=True)

        k_predictor = self._get_predictor()

        unet = UnetPatcher.from_model(model=huggingface_components["transformer"], diffusers_scheduler=None, k_predictor=k_predictor, config=estimated_config)

        self.text_processing_engine_qwen = Qwen359BEngine(
            text_encoder=clip.cond_stage_model.qwen35_9b,
            tokenizer=clip.tokenizer.qwen35_9b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.use_shift = True

    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: list[str]):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)
        return self.text_processing_engine_qwen(prompt)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)
