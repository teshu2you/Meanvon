import math
from typing import TYPE_CHECKING

from util.printf import MasterName, printF

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
from backend.text_processing.krea2_engine import Qwen3VL4BEngine
from modules.shared import opts


class Krea2(ForgeDiffusionEngine):
    matched_guesses = [model_list.Krea2]

    def __init__(self, estimated_config, huggingface_components):
        super().__init__(estimated_config, huggingface_components)

        clip = CLIP(model_dict={"qwen3vl_4b": huggingface_components["text_encoder"]}, tokenizer_dict={"qwen3vl_4b": huggingface_components["tokenizer"]})

        vae = VAE(model=huggingface_components["vae"])
        # vae = VAE(model=huggingface_components["vae"], latent_channels=16)
        
        # ================================================================
        # ★ 空 VAE 兜底
        # 日志证据（Krea2_Realism_V1.safetensors，split checkpoint）：
        #   [VAE-FILE] vae=VAE filename=None model_file=None
        #   [VAE-DEC/probe] decoder weight[0] std=0.000000e+00 zero_ratio=1.000
        # 说明 Forge 用 WanVAE() 默认构造了一个全 0 的空壳，没加载任何权重。
        # 结果：decoder 输出恒 0 → (0+1)/2 = 0.5 → 灰图。
        #
        # 处理：检测到空 VAE，从 models/vae/ 里找已知的 Wan VAE 文件手动加载。
        # 若所有候选都失败，给出鲜明警告。
        # ================================================================
        try:
            _fsm = vae.first_stage_model
            _w = getattr(getattr(_fsm, "decoder", None), "conv1", None)
            _w = getattr(_w, "weight", None) if _w is not None else None
            _is_empty = (_w is None) or (float(_w.float().std().cpu()) < 1e-6)

            if _is_empty:
                printF(name=MasterName.get_master_name(),
                       info="[KREA-VAE] 检测到空 VAE（checkpoint 未含 VAE 权重），"
                            "尝试从 models/vae/ 手动加载").printf()
                import os
                from modules.config import path_vae
                from backend.utils import load_torch_file

                _candidates = [
                    "wan_2.1_vae.safetensors",
                    "wan2.1_vae.safetensors",
                    "wan_2.1_vae.sft",
                    "wan_2.1_vae.pth",
                ]
                _ok = False
                for _name in _candidates:
                    _path = os.path.join(path_vae, _name)
                    if not os.path.exists(_path):
                        continue
                    try:
                        _sd = load_torch_file(_path, device="cpu")
                        # 兼容无前缀 key
                        if not any(k.startswith("decoder.") for k in _sd):
                            _sd = {k: v for k, v in _sd.items()
                                   if k.startswith(("decoder.", "encoder."))}
                        _miss, _unexp = _fsm.load_state_dict(_sd, strict=False)
                        _new_std = float(_fsm.decoder.conv1.weight.float().std().cpu())
                        printF(name=MasterName.get_master_name(),
                               info="[KREA-VAE] loaded {}: "
                                    "decoder.conv1.std={:.4e} "
                                    "missing={} unexpected={}".format(
                                        _name, _new_std,
                                        len(_miss), len(_unexp))).printf()
                        if _new_std > 1e-6:
                            _ok = True
                            vae.first_stage_model = _fsm
                            break
                    except Exception as _le:
                        printF(name=MasterName.get_master_name(),
                               info="[KREA-VAE] {} load failed: {}".format(
                                   _name, _le)).printf()

                if not _ok:
                    printF(name=MasterName.get_master_name(),
                           info="[KREA-VAE] ⚠️ 未能修复。请在 models/vae/ 放置真实的 "
                                "wan_2.1_vae.safetensors（~254 MB），"
                                "否则 decode 永远是 0.5 灰图。").printf()
            else:
                _std0 = float(_w.float().std().cpu())
                printF(name=MasterName.get_master_name(),
                       info="[KREA-VAE] OK decoder.conv1.std={:.4e}".format(_std0)).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[KREA-VAE] check failed: {}".format(_e)).printf()
            
        k_predictor = self._get_predictor()

        # ================================================================
        # ★ 兜底设置 shift
        # ----------------------------------------------------------------
        # Krea2 借用 Wan latent space（lf=Wan21, scale_factor=1.0）。
        # Wan 家族的 flow matching 必须有一个正 shift，否则:
        #     timestep(sigma) = sigma / (sigma + shift)   -> 全塌缩
        # 日志证据：24 步全程 t≈0.000997，std=1.13（应为 2~3），
        # 输出灰+局部彩。这是典型的 "shift=None" 症状。
        #
        # 尝试顺序：
        #   1) 若 k_predictor 已有非 None 的 shift，信任它
        #   2) 否则用 Wan21 官方 shift = 5.0
        #   3) 若仍无效，改 3.0（低分辨率常用）
        # ================================================================
        if k_predictor is not None:
            _cur_shift = getattr(k_predictor, "shift", None)
            if _cur_shift is None:
                for _attr in ("shift", "time_shift", "flux_shift"):
                    try:
                        setattr(k_predictor, _attr, 5.0)
                        break
                    except Exception:
                        continue
                printF(name=MasterName.get_master_name(),
                       info="[KREA-KP] shift 为空，兜底设置为 5.0（Wan21 默认）").printf()

        # 👇 原有探针（保留）
        printF(name=MasterName.get_master_name(),
               info="[KREA-KP] type={} module={} timescale={} shift={}".format(
                   type(k_predictor).__name__ if k_predictor is not None else "None",
                   type(k_predictor).__module__ if k_predictor is not None else "N/A",
                   getattr(k_predictor, "timescale", "N/A"),
                   getattr(k_predictor, "shift", "N/A"))).printf()

        unet = UnetPatcher.from_model(
            model=huggingface_components["transformer"],
            diffusers_scheduler=None,
            k_predictor=k_predictor,
            config=estimated_config,
        )

        self.text_processing_engine_qwen = Qwen3VL4BEngine(
            text_encoder=clip.cond_stage_model.qwen3vl_4b,
            tokenizer=clip.tokenizer.qwen3vl_4b,
        )

        self.forge_objects = ForgeObjects(unet=unet, clip=clip, vae=vae, clipvision=None)
        self.forge_objects_original = self.forge_objects.shallow_copy()
        self.forge_objects_after_applying_lora = self.forge_objects.shallow_copy()

        self.is_wan = True

    def _probe_out(self, out, prompt):
        try:
            if isinstance(out, (list, tuple)) and len(out) > 0:
                _t = out[0][0] if isinstance(out[0], (list, tuple)) else out[0]
                if isinstance(_t, torch.Tensor):
                    printF(name=MasterName.get_master_name(),
                           info="[KREA-COND/out] is_neg={} shape={} "
                                "std={:.15f} sum={:.6f} min={:.4f} max={:.4f} id={}".format(
                                    getattr(prompt, "is_negative_prompt", "?"),
                                    tuple(_t.shape),
                                    float(_t.std()), float(_t.sum()),
                                    float(_t.min()), float(_t.max()),
                                    id(_t))).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[KREA-COND/out] probe failed: {}".format(_e)).printf()
        return out
    
    @torch.inference_mode()
    def get_learned_conditioning(self, prompt: "SdConditioning"):
        memory_management.load_model_gpu(self.forge_objects.clip.patcher)

        try:
            _neg = getattr(prompt, "is_negative_prompt", "?")
            _txt = str(prompt)
            printF(name=MasterName.get_master_name(),
                   info="[KREA-COND/in] is_neg={} len={} head={!r}".format(
                       _neg, len(_txt), _txt[:120])).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[KREA-COND/in] probe failed: {}".format(_e)).printf()

        # 👇 加 3 行，就在这里
        print("[KREA-IN] is_neg={} id_prompt={} repr={!r}".format(
            getattr(prompt, "is_negative_prompt", "?"),
            id(prompt),
            str(prompt)[:100]))
        # 👆
        
        if not prompt.is_negative_prompt:
            _references = [*self.ref_latents]
            if self.ini_latent is not None:
                _references.insert(0, self.ini_latent)
                self.ini_latent = None

            if opts.krea2_do_reference and bool(_references):
                return self._probe_out(
                    self.get_learned_conditioning_with_image(prompt, _references), prompt)
            else:
                dynamic_args.ref_latents.clear()

        return self._probe_out(self.text_processing_engine_qwen(prompt), prompt)

    @torch.inference_mode()
    def get_learned_conditioning_with_image(self, prompt: list[str], images: list[torch.Tensor]):
        images_vl, ref_latents = [], []
        for image in images:
            v, r = self.encode_vision(image)
            images_vl.append(v)
            ref_latents.append(r.squeeze(2))

        dynamic_args.ref_latents = ref_latents.copy()
        return self.text_processing_engine_qwen(prompt, images=images_vl)

    @torch.inference_mode()
    def encode_vision(self, image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        samples = image.movedim(-1, 1)  # b, c, h, w

        total = int(768 * 768)
        scale_by = math.sqrt(total / (samples.shape[3] * samples.shape[2]))
        width = round(samples.shape[3] * scale_by)
        height = round(samples.shape[2] * scale_by)

        s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        _vision = s.movedim(1, -1)[:, :, :, :3]

        width = samples.shape[3]
        height = samples.shape[2]

        width = round(width / 16.0) * 16
        height = round(height / 16.0) * 16
        s = torch.nn.functional.interpolate(samples, size=(height, width), mode="area")
        sample = self.forge_objects.vae.encode(s.movedim(1, -1)[:, :, :, :3])
        _latent = self.forge_objects.vae.first_stage_model.process_in(sample)

        return (_vision, _latent)

    @torch.inference_mode()
    def get_prompt_lengths_on_ui(self, prompt: str) -> tuple[int, int]:
        token_count = len(self.text_processing_engine_qwen.tokenize(prompt))
        return token_count, max(999, token_count)

    @torch.inference_mode()
    def encode_first_stage(self, x: torch.Tensor):
        if opts.krea2_do_reference:
            start_image = x[0].movedim(0, -1).mul(0.5).add(0.5).unsqueeze(0)
            if dynamic_args.is_referencing:
                self.ref_latents.append(start_image.cpu())
            else:
                self.ini_latent = start_image.cpu()

        return super().encode_first_stage(x)
