import torch
import backend
from backend import memory_management, attention
from backend.modules.k_prediction import k_prediction_from_diffusers_scheduler
from util.printf import printF, MasterName
import ldm_patched
from enum import Enum
from ldm_patched.ldm.modules.diffusionmodules.openaimodel import UNetModel, Timestep
from ldm_patched.modules.model_sampling import EPS, V_PREDICTION, EDM, ModelSamplingDiscrete, ModelSamplingContinuousEDM, StableCascadeSampling, ModelSamplingContinuousV
from ldm_patched.modules.model_base import BaseModel, model_sampling, ModelType, SDXL, SDXLRefiner, HunyuanDiT, Flux, SD15_instructpix2pix, SD21UNCLIP, SD3

class KModel(torch.nn.Module):
    def __init__(self, model, diffusers_scheduler, k_predictor=None, config=None):
        super().__init__()

        self.config = config
        self.model_config = self.config
        self.storage_dtype = model.storage_dtype
        self.computation_dtype = model.computation_dtype

        printF(name=MasterName.get_master_name(),
               info="K-Model Created: {}".format(dict(storage_dtype=self.storage_dtype, computation_dtype=self.computation_dtype))).printf()

        self.diffusion_model = model

        if k_predictor is None:
            self.predictor = k_prediction_from_diffusers_scheduler(diffusers_scheduler)
        else:
            self.predictor = k_predictor

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

        model_output = self.diffusion_model(xc, t, context=context, control=control,
                                            transformer_options=transformer_options, **extra_conds).float()
        return self.predictor.calculate_denoised(sigma, model_output, x)

    def memory_required(self, input_shape):
        area = input_shape[0] * input_shape[2] * input_shape[3]
        dtype_size = memory_management.dtype_size(self.computation_dtype)

        if attention.attention_function in [attention.attention_pytorch, attention.attention_xformers]:
            scaler = 1.28
        else:
            scaler = 1.65
            if attention.get_attn_precision() == torch.float32:
                dtype_size = 4

        return scaler * area * dtype_size * 16384