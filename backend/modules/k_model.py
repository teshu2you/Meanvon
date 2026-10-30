import math

import torch

from backend import args, attention, memory_management
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler



class _IdentityLatentFormat:
    """
    Flux / Flux2 / ZImage / QwenImage 的 latent 缩放已经在 predictor 内部
    （calculate_input / calculate_denoised）完成，这里只需要恒等映射，
    仅用于满足 Comfy 采样器 process_in / process_out 的调用约定。
    """
    latent_channels = 16

    def process_in(self, latent):
        return latent

    def process_out(self, latent):
        return latent


class _KModelConfig:
    def __init__(self, latent_format):
        self.latent_format = latent_format
        
class KModel(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, diffusers_scheduler, config, k_predictor):
        super().__init__()

        self.config = config

        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        _store = f"storage: {self.storage_dtype}"
        _compute = f"computation: {self.computation_dtype}"

        if args.dynamic_args.ops.endswith("FP8"):
            _compute += f" + {torch.float8_e4m3fn}"
        if args.dynamic_args.ops.startswith("Mixed"):
            _compute = "computation: Mixed"

        memory_management.logger.info(f"Diffusion Model: {{{_store}, {_compute}}}")

        self.diffusion_model = model
        self.diffusion_model.eval()
        self.diffusion_model.requires_grad_(False)

        if diffusers_scheduler is not None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

        # Comfy 的 CFGGuider.inner_sample 收尾会访问
        #   self.inner_model.model_config.latent_format.process_out(samples)
        # img2img 时还会访问 process_latent_in。KModel 之前没有这些属性，
        # 采样一结束就抛：AttributeError: 'KModel' object has no attribute 'model_config'
        # 注意：必须放在 self.diffusion_model 赋值之后。
        self.model_config = _KModelConfig(self._resolve_latent_format())

    def _resolve_latent_format(self):
        # 底层 config / diffusion_model 若自带 latent_format，优先尊重它
        for src in (self.config, self.diffusion_model):
            lf = getattr(src, "latent_format", None)
            if lf is not None and hasattr(lf, "process_out"):
                return lf
        return _IdentityLatentFormat()

    def process_latent_in(self, latent):
        return self.model_config.latent_format.process_in(latent)

    def process_latent_out(self, latent):
        return self.model_config.latent_format.process_out(latent)

    def apply_model(self, x, t, c_concat=None, c_crossattn=None, control=None, transformer_options={}, **kwargs):
        sigma = t
        xc = self.predictor.calculate_input(sigma, x)
        if c_concat is not None:
            xc = torch.cat([xc] + [c_concat], dim=1)

        context = c_crossattn
        dtype = self.computation_dtype

        xc = xc.to(dtype)
        t = self.predictor.timestep(t).float()
        context = context.to(dtype)
        extra_conds = {}
        for o in kwargs:
            extra = kwargs[o]
            if hasattr(extra, "dtype"):
                if extra.dtype != torch.int and extra.dtype != torch.long:
                    extra = extra.to(dtype)
            extra_conds[o] = extra

        model_output = self.diffusion_model(xc, t, context=context, control=control, transformer_options=transformer_options, **extra_conds).float()
        # 形状守卫：视频 DiT 有时会在出口对图像路径也补上 T 轴，
        # 造成 apply_model 返回 5D，而 xc 是 4D，calculate_denoised
        # 与 calc_cond_batch 的广播就会造出幽灵维。这里把多余的 1 维挤掉：
        if (
            isinstance(model_output, torch.Tensor)
            and xc.dim() == 4
            and model_output.dim() == 5
            and model_output.shape[:2] == xc.shape[:2]
            and model_output.shape[2] == 1
            and model_output.shape[3:] == xc.shape[2:]
        ):
            model_output = model_output.squeeze(2)

        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape: list[int]) -> float:
        """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/model_base.py#L354"""
        input_shapes = [input_shape]
        area = sum(map(lambda input_shape: input_shape[0] * math.prod(input_shape[2:]), input_shapes))

        if attention.attention_function is attention.attention_basic:
            _size = 8
        elif args.dynamic_args.ops.startswith("Mixed"):
            _size = 1
        else:
            _size = memory_management.dtype_size(self.computation_dtype)

        return (area * _size * 0.02 * self.config.memory_usage_factor) * (1024 * 1024)
