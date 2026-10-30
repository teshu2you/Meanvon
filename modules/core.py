import sys
from backend import utils
import backend
from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.kolors import Kolors

from backend.diffusion_engine.krea import Krea2
from backend.modules.k_model import KModel
from modules import devices
from modules.rng import ImageRNG
from util.printf import printF, MasterName
import modules.constants as constants
import os
import einops
import torch
import numpy as np
import ldm_patched.modules.model_management
import ldm_patched.modules.model_detection
import ldm_patched.modules.model_patcher
import ldm_patched.modules.utils
import ldm_patched.modules.controlnet
import modules.sample_hijack
import ldm_patched.modules.samplers
import ldm_patched.modules.latent_formats
import modules.advanced_parameters
from ldm_patched.contrib.external_canny import Canny
from ldm_patched.contrib.external_freelunch import FreeU, FreeU_V2
from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.contrib.external import VAEDecode, EmptyLatentImage, VAEEncode, VAEEncodeTiled, VAEDecodeTiled, \
    VAEEncodeForInpaint, \
    ControlNetApplyAdvanced, ConditioningZeroOut, ConditioningAverage, CLIPVisionEncode, unCLIPConditioning, \
    ControlNetApplyAdvanced
from ldm_patched.modules.model_base import SDXL, SDXLRefiner
from modules.lora import match_lora
from ldm_patched.modules.lora import model_lora_keys_unet, model_lora_keys_clip, load_lora
from modules.config import path_embeddings, path_vae
from modules import sd_vae_approx, sd_vae_taesd
from modules.util import get_file_from_folder_list
from ldm_patched.contrib.external_post_processing import ImageScaleToTotalPixels
from ldm_patched.contrib.external_model_advanced import ModelSamplingDiscrete, ModelSamplingContinuousEDM
from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.flux2 import Flux2
from backend.diffusion_engine.zimage import ZImage
from backend.diffusion_engine.qwen import QwenImage
from backend.diffusion_engine.kolors import Kolors


opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opVAEEncode = VAEEncode()
opVAEDecodeTiled = VAEDecodeTiled()
opVAEEncodeTiled = VAEEncodeTiled()
opControlNetApplyAdvanced = ControlNetApplyAdvanced()
opModelSamplingDiscrete = ModelSamplingDiscrete()
opConditioningZeroOut = ConditioningZeroOut()
opConditioningAverage = ConditioningAverage()
opCLIPVisionEncode = CLIPVisionEncode()
opImageScaleToTotalPixels = ImageScaleToTotalPixels()
opCanny = Canny()
opFreeU = FreeU_V2()
opModelSamplingContinuousEDM = ModelSamplingContinuousEDM()

# 需要按潜在通道数 + 8 倍缩放自行构造噪声的 flow-matching 引擎
FORGE_FLOW_ENGINES = (Flux, Flux2, ZImage, QwenImage, Kolors, Krea2)

# ===== HunyuanDiT 1.2 灰度修复开关 =====
HUNYUAN_DEC_DIAG       = True    # 只读诊断
HUNYUAN_DEC_FP32       = False   # 关闭！手动强转 fp32 会撞上 vae.decode 内部的 bf16 转换
HUNYUAN_DEC_NORM       = False   # ✅ 用这个：纯数值缩放 + DC 校正，不碰 dtype
HUNYUAN_DEC_TARGET_STD = 1.40    # VAE 空间 std 目标（1.1~1.5 之间调）

# 若 vae.decode 内部不应用 process_out，把这里改成 True。
# 现在默认 False：假定 vae.decode 会应用 process_out，
# 所以 ksampler 需要把 sample() 的出口先 process_in 回模型空间。
KSAMPLER_SKIP_UNDO_PROCESS_OUT = False

def is_flow_engine(model) -> bool:
    """直接引擎 / StableDiffusionModel(model_ori=引擎) 都认。

    UnetPatcher 从 flow engine 里拆出来的场景无法可靠判定
    （内层 KModel 没有引擎类名标记），因此不在本函数覆盖范围内——
    这类路径应该在 process_diffusion 里就被分到 flow 分支。
    """
    if isinstance(model, FORGE_FLOW_ENGINES):
        return True
    ori = getattr(model, "model_ori", None)
    if isinstance(ori, FORGE_FLOW_ENGINES):
        return True
    return False


def get_latent_format(model, vae=None):
    """多路径 latent_format 解析。

    ZImage / Flux 系列：
        engine.forge_objects.vae.latent_format   ← 最可靠（VAE 与 process_out 同源）
        engine.forge_objects.unet.model.xxx      ← KModel 侧通常没有
    老 SD / SDXL：
        model.model.latent_format

    绝不回落到引擎对象本身（引擎没有 latent_format）。
    """
    def _from(obj):
        if obj is None:
            return None
        # 0 层
        lf = getattr(obj, "latent_format", None)
        if lf is not None:
            return lf
        # 1 层 (.model)
        inner = getattr(obj, "model", None)
        if inner is not None:
            lf = getattr(inner, "latent_format", None)
            if lf is not None:
                return lf
            # 2 层 (.model.model)
            inner2 = getattr(inner, "model", None)
            if inner2 is not None:
                lf = getattr(inner2, "latent_format", None)
                if lf is not None:
                    return lf
        # first_stage_model
        fsm = getattr(obj, "first_stage_model", None)
        if fsm is not None:
            lf = getattr(fsm, "latent_format", None)
            if lf is not None:
                return lf
        return None

    if is_flow_engine(model):
        fo = getattr(model, "forge_objects", None)
        for obj in (getattr(fo, "vae", None),
                    getattr(fo, "unet", None),
                    model):
            lf = _from(obj)
            if lf is not None:
                return lf

    lf = _from(model)
    if lf is not None:
        return lf

    if vae is not None:
        return _from(vae)
    return None

def should_use_prompt_expansion(model) -> bool:
    """Fooocus V2 扩写只对 SD1.5 / SDXL 的 CLIP 有意义。

    Flux / ZImage / QwenImage 用 T5 / Qwen 系 text encoder，吃自然语言，
    Fooocus 的标签串不生效，还会挤占 prompt 预算、破坏 token 布局。
    而且扩写会产出多个变体，每个变体单独跑一遍完整采样
    —— 这就是日志里 ZImage 跑三次、pos.shape 分别 75/48/84 的原因。

    注意：调用方拿到的可能是 StableDiffusionModel 包装，也可能是引擎本身，
    所以这里用鸭子类型判断，不只看 isinstance。
    """
    if model is None:
        # 模型还没解析出来时不要贸然关，交给上层原有逻辑
        return True

    # 1) 直接是 flow 引擎实例
    if isinstance(model, FORGE_FLOW_ENGINES):
        return False

    # 2) StableDiffusionModel 包装 → 看 model_ori
    ori = getattr(model, "model_ori", None)
    if isinstance(ori, FORGE_FLOW_ENGINES):
        return False
    if ori is not None and getattr(ori, "forge_objects", None) is not None:
        return False

    # 3) 鸭子类型：ForgeDiffusionEngine 都有 forge_objects
    if getattr(model, "forge_objects", None) is not None:
        return False

    return True
    
class StableDiffusionModel:
    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None, filename=None, vae_filename=None, model_ori=None):
        if isinstance(filename, str):
            # split checkpoint（FLUX / 部分 ZImage 等）可能没有 VAE、CLIP，
            # 或者拿到的是"空壳"：
            #   vae 非 None 但 vae.first_stage_model is None
            #   clip 非 None 但 clip.cond_stage_model is None
            # 原来的链式属性访问会直接炸：
            #   AttributeError: 'NoneType' object has no attribute 'model_file'
            # 逐个 getattr 解包 + 判空，能标记就标记，不能就跳过。
            _unet_inner = getattr(unet, "model", None) if unet is not None else None
            is_refiner = isinstance(_unet_inner, SDXLRefiner)
            if _unet_inner is not None:
                _unet_inner.model_file = dict(filename=filename, prefix='model')

            _clip_inner = getattr(clip, "cond_stage_model", None) if clip is not None else None
            if _clip_inner is not None:
                _clip_inner.model_file = dict(filename=filename,
                                              prefix='refiner_clip' if is_refiner else 'base_clip')

            _vae_inner = getattr(vae, "first_stage_model", None) if vae is not None else None
            if _vae_inner is not None:
                _vae_inner.model_file = dict(filename=filename, prefix='first_stage_model')
            elif vae is not None:
                printF(name=MasterName.get_master_name(),
                       info="[SDM] vae is a shell (first_stage_model=None), "
                            "model_file not stamped — 这通常是 split FLUX "
                            "checkpoint，需要另配默认 VAE。").printf()

        self.unet = unet
        
        if self.unet is not None:
            self.lora_key_map_unet = {}

            # 只有当 unet.model 是 ldm_patched 老模型（有 model_config.unet_config）
            # 才生成 lora key map；Forge 新引擎（KModel：Flux / ZImage / QwenImage …）
            # 不走老逻辑，Forge 在 backend/patcher 里自己处理 LoRA 映射。
            # 否则：AttributeError: 'KModel' object has no attribute 'model_config'
            _model = getattr(self.unet, "model", None)
            if _model is not None and hasattr(_model, "model_config") and \
                getattr(_model.model_config, "unet_config", None) is not None:
                self.lora_key_map_unet = model_lora_keys_unet(self.unet.model, self.lora_key_map_unet)
                
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.filename = filename
        self.vae_filename = vae_filename

        self.model_ori = model_ori

        self.unet_with_lora = self.unet
        self.clip_with_lora = self.clip
        self.visited_loras = ''

        self.lora_key_map_unet = {}
        self.lora_key_map_clip = {}

        if self.unet is not None:
            self.lora_key_map_unet = model_lora_keys_unet(self.unet.model, self.lora_key_map_unet)
            self.lora_key_map_unet.update({x: x for x in self.unet.model.state_dict().keys()})

        if self.clip is not None:
            self.lora_key_map_clip = model_lora_keys_clip(self.clip.cond_stage_model, self.lora_key_map_clip)
            self.lora_key_map_clip.update({x: x for x in self.clip.cond_stage_model.state_dict().keys()})

    def set_clip_skip(self, clip_skip):
        # 分体 FLUX / unet-only checkpoint 里 clip 可能是 None：
        # 权重中没有 text encoder，Forge 也没从 default 挂上来。
        # 原来的裸调用会：AttributeError: 'NoneType' object has no attribute 'clip_layer'
        _engine = self.model_ori
        _is_engine = isinstance(_engine, backend.diffusion_engine.base.ForgeDiffusionEngine)

        if _is_engine:
            _engine.set_clip_skip(clip_skip)
            return

        _clip = getattr(self, "clip", None)
        _cs = getattr(_clip, "cond_stage_model", None) if _clip is not None else None

        if _cs is not None and hasattr(_clip, "clip_layer"):
            _clip.clip_layer(clip_skip)
            return

        printF(name=MasterName.get_master_name(),
               info="[SDM] set_clip_skip skipped (no text encoder): "
                    "model_ori={} is_engine={} clip={} cond_stage_model={}. "
                    "后续 prompt 编码会失败，请补齐 FLUX 的默认 "
                    "text encoder + VAE，或换回真 SDXL all-in-one。".format(
                        type(_engine).__name__ if _engine is not None else "None",
                        _is_engine,
                        type(_clip).__name__ if _clip is not None else "None",
                        type(_cs).__name__ if _cs is not None else "None")).printf()

    @torch.no_grad()
    @torch.inference_mode()
    def refresh_loras(self, loras):
        assert isinstance(loras, list)

        if self.visited_loras == str(loras):
            printF(name=MasterName.get_master_name(),
                   info="[Warning] visited_loras: {} is same as filename:{}, just return".format(self.visited_loras,
                                                                                                 str(loras))).printf()
            return

        self.visited_loras = str(loras)

        if self.unet is None:
            return

        printF(name=MasterName.get_master_name(),
               info="Request to load LoRAs {} for model [{}].".format(loras, self.filename)).printf()

        loras_to_load = []

        for enable, name, weight in loras:
            if name == 'None':
                continue

            if os.path.exists(name):
                lora_filename = name
            else:
                lora_filename = os.path.join(modules.config.paths_loras[0], name)

            if not os.path.exists(lora_filename):
                printF(name=MasterName.get_master_name(),
                       info="Lora file not found: {}".format(lora_filename)).printf()
                continue

            loras_to_load.append((enable, lora_filename, weight))

        self.unet_with_lora = self.unet.clone() if self.unet is not None else None
        self.clip_with_lora = self.clip.clone() if self.clip is not None else None

        # format like: true,"sd_xl_offset_example-lora_1.0.safetensors",0.1
        for lora_enable, lora_filename, weight in loras_to_load:
            lora_unmatch = ldm_patched.modules.utils.load_torch_file(lora_filename, safe_load=False)
            lora_unet, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_unet)
            lora_clip, lora_unmatch = match_lora(lora_unmatch, self.lora_key_map_clip)

            if len(lora_unmatch) > 12:
                # model mismatch
                continue

            if len(lora_unmatch) > 0:
                printF(name=MasterName.get_master_name(),
                       info="Loaded LoRA [{}] for model [{}] with unmatched keys {}".format(lora_filename,
                                                                                            self.filename, list(
                               lora_unmatch.keys()))).printf()

            if "flux" not in lora_filename.lower():
                if self.unet_with_lora is not None and len(lora_unet) > 0:
                    loaded_keys = self.unet_with_lora.add_patches(lora_unet, weight)
                    printF(name=MasterName.get_master_name(),
                           info="Loaded LoRA [{}] for UNet [{}] with {} keys at weight {}.".format(
                               lora_filename, self.filename, len(loaded_keys), weight)).printf()
                    for item in lora_unet:
                        if item not in loaded_keys:
                            printF(name=MasterName.get_master_name(),
                                   info="UNet LoRA key skipped: {}".format(item)).printf()

                if self.clip_with_lora is not None and len(lora_clip) > 0:
                    loaded_keys = self.clip_with_lora.add_patches(lora_clip, weight)
                    printF(name=MasterName.get_master_name(),
                           info="Loaded LoRA [{}] for CLIP [{}] with {} keys at weight {}.".format(
                               lora_filename, self.filename, len(loaded_keys), weight)).printf()
                    for item in lora_clip:
                        if item not in loaded_keys:
                            printF(name=MasterName.get_master_name(),
                                   info="CLIP LoRA key skipped: {}".format(item)).printf()
            else:
                model_flag = type(self.unet.model).__name__ if self.unet is not None else 'default'

                if self.unet_with_lora is not None and len(lora_unet) > 0:
                    loaded_keys = self.unet_with_lora.add_patches(filename=lora_filename, patches=lora_unet, strength_patch=1.0, online_mode=False)
                    skipped_keys = [item for item in lora_unet if item not in loaded_keys]
                    if len(skipped_keys) > 12:
                        printF(name=MasterName.get_master_name(),
                               info="[LORA] Mismatch {} for {}-UNet with {} keys mismatched in {} keys".format(lora_filename, model_flag, len(skipped_keys), len(loaded_keys))).printf()
                    else:
                        printF(name=MasterName.get_master_name(),
                               info="[LORA] Loaded {} for {}-UNet with {}".format(lora_filename,model_flag, len(loaded_keys))).printf()

                if self.clip_with_lora is not None and len(lora_clip) > 0:
                    loaded_keys = self.clip_with_lora.add_patches(filename=lora_filename, patches=lora_clip, strength_patch=1.0, online_mode=False)
                    skipped_keys = [item for item in lora_clip if item not in loaded_keys]
                    if len(skipped_keys) > 12:
                        printF(name=MasterName.get_master_name(),
                               info="[LORA] Mismatch {} for {}-CLIP with {} keys mismatched in {} keys".format(lora_filename, model_flag,len(skipped_keys),  len(loaded_keys))).printf()
                    else:
                        printF(name=MasterName.get_master_name(),
                               info="[LORA] Loaded {} for {}-CLIP with {}".format(lora_filename, model_flag, len(loaded_keys))).printf()


@torch.no_grad()
@torch.inference_mode()
def apply_freeu(model, b1, b2, s1, s2):
    return opFreeU.patch(model=model, b1=b1, b2=b2, s1=s1, s2=s2)[0]


@torch.no_grad()
@torch.inference_mode()
def load_clip_vision(ckpt_filename):
    return ldm_patched.modules.clip_vision.load(ckpt_filename)


@torch.no_grad()
@torch.inference_mode()
def load_controlnet(ckpt_filename):
    return ldm_patched.modules.controlnet.load_controlnet(ckpt_filename)

@torch.no_grad()
@torch.inference_mode()
def detect_edge(image, low_threshold, high_threshold):
    return opCanny.detect_edge(image=image, low_threshold=low_threshold, high_threshold=high_threshold)[0]

# @torch.no_grad()
# @torch.inference_mode()
# def encode_prompt_condition(clip, prompt):
#     return opCLIPTextEncode.encode(clip=clip, text=prompt)[0]

@torch.no_grad()
@torch.inference_mode()
def encode_clip_vision(clip_vision, image):
    return opCLIPVisionEncode.encode(clip_vision=clip_vision, image=image)[0]


@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
    return opControlNetApplyAdvanced.apply_controlnet(positive=positive, negative=negative, control_net=control_net,
                                                      image=image, strength=strength, start_percent=start_percent,
                                                      end_percent=end_percent)

# Attention!!!!!!!
@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename, model_file_type=constants.TYPE_NORMAL, vae_filename=None):
    unet, clip, vae, vae_filename, clip_vision, model_ori = load_checkpoint_guess_config(
        ckpt_filename, embedding_directory=path_embeddings, model_options={}, te_model_options={},
        model_file_type=model_file_type, vae_filename_param=vae_filename)
    # HunyuanDiT（以及所有复用 SDXL VAE 的模型）：
    # 只要 model 有 latent_format 而 VAE 没有，就必须同步过去。
    # 否则 VAE.decode 里的 process_out（×(1/0.13025)）被跳过，
    # VAE 收到未缩放 latent → 整图 8×8 马赛克块。
    # ... existing code ...
    try:
        if vae is not None and getattr(vae, "latent_format", None) is None:
            _lf = getattr(getattr(unet, "model", None), "latent_format", None)
            if _lf is not None:
                vae.latent_format = _lf
                printF(name=MasterName.get_master_name(),
                       info="[VAE-latent_format] was None, synced from unet: {} (scale={})".format(
                           type(_lf).__name__, getattr(_lf, "scale_factor", "?"))).printf()
    except Exception as _e:
        printF(name=MasterName.get_master_name(),
               info="[VAE-latent_format] sync failed: {}".format(_e)).printf()

    printF(name=MasterName.get_master_name(),
           info="[LOAD] unet={} clip={} vae={} fsm={} model_ori={} engine={}".format(
               type(unet).__name__ if unet is not None else "None",
               type(clip).__name__ if clip is not None else "None",
               type(vae).__name__ if vae is not None else "None",
               type(getattr(vae, "first_stage_model", None)).__name__
                   if vae is not None else "None",
               type(model_ori).__name__ if model_ori is not None else "None",
               isinstance(model_ori, backend.diffusion_engine.base.ForgeDiffusionEngine)
           )).printf()

    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision,
                                filename=ckpt_filename, vae_filename=vae_filename, model_ori=model_ori)
@torch.no_grad()
@torch.inference_mode()
def load_sd_lora(model, lora_filename, strength_model=1.0, strength_clip=1.0):
    if strength_model == 0 and strength_clip == 0:
        return model

    lora = ldm_patched.modules.utils.load_torch_file(lora_filename, safe_load=False)

    if lora_filename.lower().endswith('.fooocus.patch'):
        loaded = lora
    else:
        key_map = model_lora_keys_unet(model.unet.model)
        key_map = model_lora_keys_clip(model.clip.cond_stage_model, key_map)
        loaded = load_lora(lora, key_map)

    new_modelpatcher = model.unet.clone()
    k = new_modelpatcher.add_patches(loaded, strength_model)

    new_clip = model.clip.clone()
    k1 = new_clip.add_patches(loaded, strength_clip)

    k = set(k)
    k1 = set(k1)
    for x in loaded:
        if (x not in k) and (x not in k1):
            print("Lora missed: ", x)

    unet, clip = new_modelpatcher, new_clip
    return StableDiffusionModel(unet=unet, clip=clip, vae=model.vae, clip_vision=model.clip_vision)


@torch.no_grad()
@torch.inference_mode()
def upscale(image, megapixels=1.0):
    return opImageScaleToTotalPixels.upscale(image=image, upscale_method='bicubic', megapixels=megapixels)[0]


@torch.no_grad()
@torch.inference_mode()
def generate_empty_latent(width=1024, height=1024, batch_size=1):
    return opEmptyLatentImage.generate(width=width, height=height, batch_size=batch_size)[0]


import inspect as _inspect

_VAE_APPLIES_PROCESS_OUT_CACHE = {}

def _vae_applies_process_out(vae):
    cls = type(vae)
    if cls in _VAE_APPLIES_PROCESS_OUT_CACHE:
        return _VAE_APPLIES_PROCESS_OUT_CACHE[cls]
    val = False
    try:
        src = _inspect.getsource(cls.decode)
        val = ("process_out" in src) or ("latent_format" in src)
        printF(name=MasterName.get_master_name(),
               info="[VAE-PROBE] {} .decode applies process_out: {} "
                    "(src_lines={})".format(cls.__name__, val,
                        len(src.splitlines()))).printf()
    except Exception as _e:
        printF(name=MasterName.get_master_name(),
               info="[VAE-PROBE] inspect failed: {}".format(_e)).printf()
    _VAE_APPLIES_PROCESS_OUT_CACHE[cls] = val
    return val


# ===== Wan2.1 latent 后处理参数 =====
WAN21_DC_CORRECT_BEFORE_PROCESS_OUT = True
# 🔧 关键旋钮：0.0=不做 DC（会过曝，但结构最完整）
#             0.25 / 0.5 / 0.75 = 折中
#             1.0 = 全 DC（图会变灰）
# 建议从 0.35 或 0.5 起，往一个方向微调：
#   - 图偏过曝/发绿 → 加大 strength
#   - 图变灰/暗淡 → 减小 strength
WAN21_DC_STRENGTH = 0.35
WAN21_DC_TRIGGER_THRESHOLD = 0.25

# ===== Wan2.1 latent 后处理参数 =====
# "standardize"  → per-channel 强制 mean=0, std=1（推荐，直接对齐 Wan 训练分布）
# "per_ch_dc"    → 只做 per-channel mean DC，保留原始 per-channel std（保守）
# "global_dc"    → 旧的全局 DC（已证明错，仅保留供对照）
# "none"         → 不处理（会绿色过曝）
WAN21_LATENT_NORM_MODE = "none"

# standardize 时，某通道 std < 此值的视为"噪声通道"，不做 std 归一化（防止放大噪声）
WAN21_MIN_PER_CH_STD = 0.1

# VAE 输出后是否做 percentile stretch（standardize 后最多轻微过曝，一般不需要）
WAN21_POST_DECODE_STRETCH = False

# ===== Wan2.1 官方 latent statics =====
_WAN21_LATENTS_MEAN = [-0.7571, -0.7089, -0.9113,  0.1075, -0.1745,  0.9653,
                       -0.1517,  1.5508,  0.4134, -0.0715,  0.5517, -0.3632,
                       -0.1922, -0.9497,  0.2503, -0.2921]
_WAN21_LATENTS_STD  = [ 2.8184,  1.4541,  2.3275,  2.6558,  1.2196,  1.7708,
                        2.6052,  2.0743,  3.2687,  2.1526,  2.8652,  1.5579,
                        1.6382,  1.1253,  2.8251,  1.9160]


def _wan21_process_out(x):
    """Wan2.1 官方 statics: y = x * std_c + mean_c，channel 在 dim=1。
    兼容 4D (B,C,H,W) 和 5D (B,C,T,H,W)。"""
    mean = torch.tensor(_WAN21_LATENTS_MEAN, dtype=x.dtype, device=x.device)
    std  = torch.tensor(_WAN21_LATENTS_STD,  dtype=x.dtype, device=x.device)
    view = (1, x.shape[1]) + (1,) * (x.dim() - 2)
    return x * std.view(view) + mean.view(view)


def _wan21_normalize_latent(x_4d):
    """把 4D latent (B,C,H,W) 对齐到 Wan decoder 训练分布（每通道 N(0,1)）。
    返回处理后的 4D tensor。"""
    mode = WAN21_LATENT_NORM_MODE
    if mode == "none":
        return x_4d

    _flt = x_4d.float()
    _pc_mean = _flt.mean(dim=(0, 2, 3), keepdim=True)   # (1,C,1,1)
    _g_mean  = float(_flt.mean())
    _g_std   = float(_flt.std())

    if mode == "global_dc":
        _out = _flt - _g_mean
        printF(name=MasterName.get_master_name(),
               info="[VAE-DEC/norm] global_dc: mean {:+.4f} -> {:+.4f} "
                    "std {:.4f} -> {:.4f}".format(
                        _g_mean, float(_out.mean()), _g_std, float(_out.std()))).printf()
        return _out.to(x_4d.dtype)

    if mode == "per_ch_dc":
        _out = _flt - _pc_mean
        printF(name=MasterName.get_master_name(),
               info="[VAE-DEC/norm] per_ch_dc: per_ch |mean|max={:.4f} -> 0 "
                    "std {:.4f} -> {:.4f}".format(
                        float(_pc_mean.abs().max()), _g_std, float(_out.std()))).printf()
        return _out.to(x_4d.dtype)

    if mode == "standardize":
        _pc_std = _flt.std(dim=(0, 2, 3), keepdim=True).clamp(min=1e-6)
        # 对 std 太小的通道，不做 std 归一化（保留原样），只做 mean DC
        _denom = _pc_std.clone()
        _noise_mask = (_pc_std < WAN21_MIN_PER_CH_STD)
        _denom[_noise_mask] = 1.0
        _out = (_flt - _pc_mean) / _denom
        # noise 通道只做 mean DC，std 保持
        if _noise_mask.any():
            _out = torch.where(
                _noise_mask.expand_as(_out),
                _flt - _pc_mean,
                _out,
            )
        printF(name=MasterName.get_master_name(),
               info="[VAE-DEC/norm] standardize: "
                    "in(mean={:+.4f},std={:.4f}) -> out(mean={:+.4f},std={:.4f}) | "
                    "pc_std[min={:.3f},max={:.3f}], noise_ch={}/{}".format(
                        _g_mean, _g_std,
                        float(_out.mean()), float(_out.std()),
                        float(_pc_std.min()), float(_pc_std.max()),
                        int(_noise_mask.sum()), _pc_std.numel())).printf()
        return _out.to(x_4d.dtype)

    # 未知 mode → 安全 fallback
    return x_4d


@torch.no_grad()
@torch.inference_mode()
def decode_vae(vae, latent_image, tiled=False):
    # split FLUX 的 vae 是非 None 的"空壳"（first_stage_model=None），
    # 现在能走到这里只会因为 prompt 已经编码成功。若 VAE 仍是空壳
    # 后续 vae.decode 会抛没有上下文的 AttributeError。

    _fsm = getattr(vae, "first_stage_model", None) if vae is not None else None

    print(f"[VAE-FILE] vae={type(vae).__name__} "
      f"filename={getattr(vae, 'filename', None)} "
      f"model_file={getattr(vae, 'model_file', None)} "
      f"fsm_id={id(_fsm)} vae_id={id(vae)}")


    if vae is None or _fsm is None:
        raise RuntimeError(
            "[decode_vae] VAE 未初始化（vae={} / first_stage_model={}）。\n"
            "  unet-only FLUX 不带 VAE，需要单独配置：\n"
            "    models/vae/ae.safetensors\n"
            "    default_flux_vae_name = 'ae.safetensors'（或 default_flux1_vae_name）\n"
            "  或换回带 VAE 的 all-in-one SDXL。".format(
                type(vae).__name__ if vae is not None else "None",
                type(_fsm).__name__ if _fsm is not None else "None"))

    if isinstance(latent_image, dict):
        latent_image = dict(latent_image)
        _s = latent_image["samples"]
    elif isinstance(latent_image, torch.Tensor):
        _s = latent_image
        latent_image = {"samples": latent_image}
    else:
        raise TypeError("decode_vae expected dict or Tensor, got {}".format(
            type(latent_image).__name__))

    _fsm = getattr(vae, "first_stage_model", None)
    _lf = getattr(vae, "latent_format", None)
    if _lf is None and _fsm is not None:
        _lf = getattr(_fsm, "latent_format", None)

    # flow 模型 model 空间 std 正常在 0.5 ~ 2.0。
    # > 2.0 基本可以判定是"上游没回滚 process_out"或"采样发散"，
    # 这种情况 VAE 解码结果一定是灰块/过曝，直接吼出来而不是静默出图。
    _in_std = float(_s.float().std())
    # 这条阈值是按「模型空间」写的，但 decode 收到的是 VAE 空间
    # （SDXL ≈ 4~6，Flux ≈ 更大），保留日志即可，别按 2.0 告警。
    if HUNYUAN_DEC_DIAG:
        printF(name=MasterName.get_master_name(),
               info="[VAE-DEC/in] std={:.4f} mean={:+.4f} dtype={} lf={}".format(
                   _in_std, float(_s.mean()), _s.dtype,
                   type(_lf).__name__ if _lf is not None else "None")).printf()

    # ===== 只读诊断 =====
    if HUNYUAN_DEC_DIAG:
        try:
            _vae_std = -1.0
            _vae_std_src = "none"
            _lf_name = type(_lf).__name__ if _lf is not None else "None"
            if _lf_name in ("Wan21", "Wan21V2") and _s.dim() >= 4:
                # 用硬编码 statics 正确 broadcast —— 避免 lf.process_out 在 4D
                # 输入上错 broadcast 成 5D×5D 得到虚假 2.75
                _vae_std = float(_wan21_process_out(_s.float()).std())
                _vae_std_src = "manual_wan21"
            elif _lf is not None and hasattr(_lf, "process_out"):
                _vae_std = float(_lf.process_out(_s.float()).std())
                _vae_std_src = "lf"
            printF(name=MasterName.get_master_name(),
                   info="[HY-DEC/in] model_std={:.4f} mean={:+.4f} "
                        "min={:+.3f} max={:+.3f} | vae_std~{:.4f}({}) | "
                        "samples.dtype={} | fsm.dtype={} | lf={}".format(
                            float(_s.std()), float(_s.mean()),
                            float(_s.min()), float(_s.max()),
                            _vae_std, _vae_std_src, _s.dtype,
                            next(_fsm.parameters()).dtype if _fsm is not None else None,
                            _lf_name,
                        )).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[HY-DEC/in] diag failed: {}".format(_e)).printf()
                            
    # ===== 5D 兜底：任何原因的 5D 都压回 4D =====
    if _s.dim() == 5:
        _s5 = tuple(_s.shape)
        _s = _s.squeeze(2) if _s.shape[2] == 1 else _s[:, :, 0]
        latent_image = {"samples": _s}
        printF(name=MasterName.get_master_name(),
               info="[VAE-DEC/squeeze] 5D {} -> 4D {}".format(_s5, tuple(_s.shape))).printf()
    # =====

    # ===== FP32：必须通过 Forge 设置（不走这里）
    # 手动 _fsm.to(float32) 会让 vae.decode 内部 bf16 样本撞 fp32 权重 → crash
    # 需要 fp32 VAE 请到 Forge UI → Settings → VAE precision = fp32
    # =====

    # ===== 归一化 + DC 校正（纯数值，保持原 dtype） =====
    if HUNYUAN_DEC_NORM and _lf is not None and hasattr(_lf, "process_out"):
        try:
            _vae_std = float(_lf.process_out(_s.float()).std())
            if _vae_std > 1e-6:
                _ratio = HUNYUAN_DEC_TARGET_STD / _vae_std
                if abs(_ratio - 1.0) > 0.05:
                    _s = _s - _s.mean()                 # DC 校正
                    _s = _s * _ratio                    # std 归一化
                    latent_image["samples"] = _s
                    printF(name=MasterName.get_master_name(),
                           info="[HY-DEC/norm] vae_std {:.4f} -> {:.4f} "
                                "(x{:.3f}), DC removed, dtype={}".format(
                                    _vae_std, HUNYUAN_DEC_TARGET_STD,
                                    _ratio, _s.dtype)).printf()
                else:
                    printF(name=MasterName.get_master_name(),
                           info="[HY-DEC/norm] vae_std {:.4f} ~= {:.4f}, skip".format(
                               _vae_std, HUNYUAN_DEC_TARGET_STD)).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[HY-DEC/norm] failed: {}".format(_e)).printf()

    # ===== 实际解码 =====
    # 崩溃点：
    #   backend/patcher/vae.py:166
    #     self.memory_used_decode = lambda shape, dtype:
    #       (2200 if shape[2] <= 4 else 7000) * shape[3] * shape[4] * (8*8) * ...
    #   该 lambda 只认 5D (B,C,T,H,W)，4D 时 shape[4] 越界 → IndexError。
    #
    # 还有第二个必须绕过 opVAEDecode 的原因：
    #   opVAEDecode.decode 内部有：
    #     if len(images.shape) == 5:
    #         images = images.reshape(-1, images.shape[-3], images.shape[-2], images.shape[-1])
    #   该 reshape 是按 (B,T,H,W,C) 写的；WanVAE 实际输出 (B,C,T,H,W)，
    #   会被错切为 (C, 1, H, W) → 3 张 1px 高灰度图 → "乱的图块"。
    #
    # 修法：
    #   1) 4D → 5D(T=1) 绕过 memory_used_decode 越界
    #   2) 直接调 vae.decode，拿原始 5D (B,C,T,H,W)，避开 opVAEDecode 的错位 reshape
    #   3) 取 T=0 压回 4D (B,C,H,W) 返回
    # 注意：仅对视频 VAE（Wan21 系）启用；图像 VAE 保持原 opVAEDecode 路径。
    _VIDEO_VAE_NAMES = ("Wan21", "Wan21V2", "Wan22", "HunyuanVideo")
    _is_video_vae = _lf is not None and type(_lf).__name__ in _VIDEO_VAE_NAMES

    _s_dec = latent_image["samples"]

    if _is_video_vae:
        _lf_name = type(_lf).__name__

        # ===== 统一压到 4D 处理 =====
        _s_dec_4d = None
        if _s_dec.dim() == 4:
            _s_dec_4d = _s_dec
        elif _s_dec.dim() == 5 and _s_dec.shape[2] == 1:
            _s_dec_4d = _s_dec.squeeze(2)

        if _s_dec_4d is not None:
            # ===== 诊断：把 4D latent 直接转成灰度图 dump 出来 =====
            if HUNYUAN_DEC_DIAG:
                try:
                    import os as _os
                    from PIL import Image as _PILImage
                    _lat_vis = _s_dec_4d[0].float().mean(0)  # (H,W)
                    _lo, _hi = float(_lat_vis.min()), float(_lat_vis.max())
                    if _hi > _lo + 1e-6:
                        _lat_vis = (_lat_vis - _lo) / (_hi - _lo)
                    _lat_u8 = (_lat_vis * 255).clamp(0, 255).byte().cpu().numpy()
                    _dump_path = _os.path.join(
                        _os.getcwd(), "latent_debug_pre_vae.png")
                    _PILImage.fromarray(_lat_u8).save(_dump_path)
                    printF(name=MasterName.get_master_name(),
                            info="[VAE-DEC/dump] latent 已存 {}".format(_dump_path)).printf()
                except Exception as _e:
                    printF(name=MasterName.get_master_name(),
                            info="[VAE-DEC/dump] failed: {}".format(_e)).printf()
            # =====
            
            if _lf_name in ("Wan21", "Wan21V2"):
                # ===== 对齐到 Wan decoder 训练分布（per-channel standardize）=====
                _old_std  = float(_s_dec_4d.float().std())
                _old_mean = float(_s_dec_4d.float().mean())
                _s_dec_4d = _wan21_normalize_latent(_s_dec_4d)

                # ===== Wan2.1 官方 statics =====
                _s_dec = _wan21_process_out(_s_dec_4d)
                printF(name=MasterName.get_master_name(),
                       info="[VAE-DEC/out] wan21 process_out: "
                            "std {:.4f}->{:.4f} mean {:+.4f}->{:+.4f} shape={} "
                            "(mode={})".format(
                                _old_std, float(_s_dec.float().std()),
                                _old_mean, float(_s_dec.float().mean()),
                                tuple(_s_dec.shape),
                                WAN21_LATENT_NORM_MODE)).printf()
            else:
                # 其他视频 VAE：5D 上调用 lf.process_out（5D 广播是正确方向）
                _s_dec = _s_dec_4d.unsqueeze(2)
                if _lf is not None and hasattr(_lf, "process_out"):
                    _old_std = float(_s_dec.float().std())
                    _old_mean = float(_s_dec.float().mean())
                    try:
                        _s_dec = _lf.process_out(_s_dec)
                        printF(name=MasterName.get_master_name(),
                               info="[VAE-DEC/out] {}.process_out(5D): "
                                    "std {:.4f}->{:.4f} mean {:+.4f}->{:+.4f}".format(
                                        _lf_name,
                                        _old_std, float(_s_dec.float().std()),
                                        _old_mean, float(_s_dec.float().mean()))).printf()
                    except Exception as _e:
                        printF(name=MasterName.get_master_name(),
                               info="[VAE-DEC/out] {}.process_out failed: {}".format(
                                   _lf_name, _e)).printf()

            # ===== 统一 wrap 成 5D，绕过 backend/patcher/vae.py 的 5D-only lambda =====
            if _s_dec.dim() == 4:
                _s_dec = _s_dec.unsqueeze(2)
                printF(name=MasterName.get_master_name(),
                       info="[VAE-DEC/wrap] 4D->5D shape={}".format(
                           tuple(_s_dec.shape))).printf()
        else:
            # 5D with T>1 或异常布局，pass through（图像生成不该到这里）
            printF(name=MasterName.get_master_name(),
                   info="[VAE-DEC/skip] 5D with T>1 或异常布局 {}，"
                        "跳过 DC/process_out，直接 decode".format(
                            tuple(_s_dec.shape))).printf()

        # T=4 replicate 只在需要修网格时开；standardize 后大概率网格不明显
        # 若网格仍明显，把 4 改成 4，同时把分辨率降到 768 避免 CPU offload
        WAN21_T_REPLICATE = 1   # 1 = 不 replicate（T=1 直接 decode，快）
        
        _t_orig = _s_dec.shape[2]
        _t_took_idx = 0
        if WAN21_T_REPLICATE > 1 and _t_orig == 1:
            _s_dec = _s_dec.repeat(1, 1, WAN21_T_REPLICATE, 1, 1)
            printF(name=MasterName.get_master_name(),
                   info="[VAE-DEC/T] T=1 -> T={} replicate".format(
                       WAN21_T_REPLICATE)).printf()
        # =====

        # ===== 探针 1：输入 latent 是否有 NaN/Inf =====
        _in_finite = bool(torch.isfinite(_s_dec).all().item())
        _in_inf_cnt = int((~torch.isfinite(_s_dec)).sum().item())
        printF(name=MasterName.get_master_name(),
            info="[VAE-DEC/probe] input finite={} nan_inf_count={} "
                    "absmax={:.4f} dtype={}".format(
                        _in_finite, _in_inf_cnt,
                        float(_s_dec.float().abs().max()), _s_dec.dtype)).printf()

        # ===== 探针 2：VAE 一个 conv 权重的 std（判断是不是全 0）=====
        try:
            _fsm_dev = next(_fsm.parameters()).device
            _w = _fsm.decoder.conv1.weight if hasattr(_fsm, "decoder") \
                    else next(_fsm.decoder.parameters())
            _w_std = float(_w.float().std().cpu())
            _w_absmax = float(_w.float().abs().max().cpu())
            printF(name=MasterName.get_master_name(),
                info="[VAE-DEC/probe] decoder weight[0] device={} dtype={} "
                        "std={:.6e} absmax={:.6e} zero_ratio={:.3f}".format(
                            _w.device, _w.dtype, _w_std, _w_absmax,
                            float((_w == 0).float().mean().cpu()))).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                info="[VAE-DEC/probe] weight 读取失败: {}".format(_e)).printf()

        # ===== 探针 3：显存状况 =====
        try:
            _free, _total = torch.cuda.mem_get_info()
            _alloc = torch.cuda.memory_allocated() / 1024**2
            _reserved = torch.cuda.memory_reserved() / 1024**2
            printF(name=MasterName.get_master_name(),
                info="[VAE-DEC/probe] VRAM free={:.0f}MB total={:.0f}MB "
                        "allocated={:.0f}MB reserved={:.0f}MB".format(
                            _free/1024**2, _total/1024**2, _alloc, _reserved)).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                info="[VAE-DEC/probe] VRAM probe failed: {}".format(_e)).printf()

        _out = vae.decode(_s_dec)
        
        # ===== T>1 时取中间帧 =====
        if _t_orig == 1 and isinstance(_out, torch.Tensor) and _out.dim() == 5 and _out.shape[1] > 1:
            _n = _out.shape[1]
            _t_took_idx = _n // 2
            _out = _out[:, _t_took_idx:_t_took_idx+1]
            printF(name=MasterName.get_master_name(),
                   info="[VAE-DEC/T] 取中间帧 idx={} (共 {} 帧)".format(
                       _t_took_idx, _n)).printf()

        # ===== 探针 4：decode 之后重新读权重 =====
        try:
            _w2 = _fsm.decoder.conv1.weight if hasattr(_fsm, "decoder") \
                    else next(_fsm.decoder.parameters())
            _w2_std = float(_w2.float().std().cpu())
            _w2_zero = float((_w2 == 0).float().mean().cpu())
            printF(name=MasterName.get_master_name(),
                info="[VAE-DEC/post-probe] decoder weight[0] device={} "
                        "std={:.6e} zero_ratio={:.3f}".format(
                            _w2.device, _w2_std, _w2_zero)).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                info="[VAE-DEC/post-probe] failed: {}".format(_e)).printf()

        # ===== 解码输出诊断 =====
        if isinstance(_out, torch.Tensor) and _out.numel() > 0:
            _o_finite = bool(torch.isfinite(_out).all().item())
            _o_std = float(_out.float().std())
            if (not _o_finite) or _o_std < 1e-5:
                printF(name=MasterName.get_master_name(),
                       info="[VAE-DEC/BAD] decoder 输出异常: "
                            "finite={} std={:.6f} shape={} → 会渲染成灰色常量图。"
                            "检查 process_out 方向 / VAE 权重 / latent_format".format(
                                _o_finite, _o_std, tuple(_out.shape))).printf()
            else:
                printF(name=MasterName.get_master_name(),
                       info="[VAE-DEC/OK] decoder 输出 std={:.4f} "
                            "min={:+.4f} max={:+.4f} shape={}".format(
                                _o_std, float(_out.min()), float(_out.max()),
                                tuple(_out.shape))).printf()

        # ===== 可选：percentile 拉伸（把过曝/压暗的分布拉回 0~1）=====
        if WAN21_POST_DECODE_STRETCH and isinstance(_out, torch.Tensor) and _out.numel() > 0:
            try:
                _o_f = _out.float()
                _o_min = float(_o_f.min()); _o_max = float(_o_f.max())
                if _o_max > 0.98 or _o_min < 0.02:
                    _flat = _o_f.flatten()
                    _lo = float(torch.quantile(_flat, 0.005))
                    _hi = float(torch.quantile(_flat, 0.995))
                    if _hi - _lo > 1e-6:
                        _out = ((_o_f - _lo) / (_hi - _lo)).clamp(0, 1).to(_out.dtype)
                        printF(name=MasterName.get_master_name(),
                               info="[VAE-DEC/stretch] [p0.5={:.4f}, p99.5={:.4f}] "
                                    "-> [0, 1]".format(_lo, _hi)).printf()
            except Exception as _e:
                printF(name=MasterName.get_master_name(),
                       info="[VAE-DEC/stretch] failed: {}".format(_e)).printf()
        # =====

        if isinstance(_out, torch.Tensor) and _out.dim() == 5:
            _osh = tuple(_out.shape)
            if _out.shape[-1] in (3, 4) and _out.shape[-3] > 4:
                _out = _out[:, 0]                       # BTHWC → BHWC
                _layout = "BTHWC"
            elif _out.shape[1] in (3, 4):
                _out = _out[:, :, 0].permute(0, 2, 3, 1)
                _layout = "BCTHW"
            else:
                _out = _out[:, 0]
                _layout = "fallback"
            printF(name=MasterName.get_master_name(),
                   info="[VAE-DEC/unwrap] video VAE {} -> {} (layout={})".format(
                       _osh, tuple(_out.shape), _layout)).printf()

        return _out

    # 图像 VAE 走原路径
    if tiled:
        return opVAEDecodeTiled.decode(samples=latent_image, vae=vae, tile_size=512)[0]
    else:
        return opVAEDecode.decode(samples=latent_image, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae(vae, pixels, tiled=False):
    if tiled:
        return opVAEEncodeTiled.encode(pixels=pixels, vae=vae, tile_size=512)[0]
    else:
        return opVAEEncode.encode(pixels=pixels, vae=vae)[0]


@torch.no_grad()
@torch.inference_mode()
def encode_vae_inpaint(vae, pixels, mask):
    assert mask.ndim == 3 and pixels.ndim == 4
    assert mask.shape[-1] == pixels.shape[-2]
    assert mask.shape[-2] == pixels.shape[-3]

    w = mask.round()[..., None]
    pixels = pixels * (1 - w) + 0.5 * w

    latent = vae.encode(pixels)
    B, C, H, W = latent.shape

    latent_mask = mask[:, None, :, :]
    latent_mask = torch.nn.functional.interpolate(latent_mask, size=(H * 8, W * 8), mode="bilinear").round()
    latent_mask = torch.nn.functional.max_pool2d(latent_mask, (8, 8)).round().to(latent)

    return latent, latent_mask

VAE_approx_models = {}

class VAEApprox(torch.nn.Module):
    def __init__(self, latent_channels=4):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(latent_channels, 8, (7, 7))
        self.conv2 = torch.nn.Conv2d(8, 16, (5, 5))
        self.conv3 = torch.nn.Conv2d(16, 32, (3, 3))
        self.conv4 = torch.nn.Conv2d(32, 64, (3, 3))
        self.conv5 = torch.nn.Conv2d(64, 32, (3, 3))
        self.conv6 = torch.nn.Conv2d(32, 16, (3, 3))
        self.conv7 = torch.nn.Conv2d(16, 8, (3, 3))
        self.conv8 = torch.nn.Conv2d(8, 3, (3, 3))
        self.current_type = None

    def forward(self, x):
        extra = 11
        x = torch.nn.functional.interpolate(x, (x.shape[2] * 2, x.shape[3] * 2))
        x = torch.nn.functional.pad(x, (extra, extra, extra, extra))
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.conv5, self.conv6, self.conv7, self.conv8]:
            x = layer(x)
            x = torch.nn.functional.leaky_relu(x, 0.1)
        return x


approximation_indexes = {"Full": 0, "Approx NN": 1, "Approx cheap": 2, "TAESD": 3}
def samples_to_images_tensor(sample, approximation=None, model=None):
    if approximation == 2:
        x_sample = sd_vae_approx.cheap_approximation(model, sample)
    elif approximation == 1:
        m = sd_vae_approx.model(model)
        if m is None:
            x_sample = sd_vae_approx.cheap_approximation(model, sample)
        else:
            x_sample = m(sample.to(devices.device, devices.dtype)).detach()
    elif approximation == 3:
        m = sd_vae_taesd.decoder_model(model)
        if m is None:
            x_sample = sd_vae_approx.cheap_approximation(model, sample)
        else:
            x_sample = m(sample.to(devices.device, devices.dtype)).detach()
            x_sample = x_sample * 2 - 1
    else:
        x_sample = model.decode_first_stage(sample)

    return x_sample

def decode_first_stage(model, x):
    approx_index = approximation_indexes.get("Full", 0)
    return samples_to_images_tensor(x, approx_index, model)


class DecodedSamples(list):
    already_decoded = True
def decode_latent_batch(model, batch, target_device=None, check_for_nans=False):
    samples = DecodedSamples()
    samples_pytorch = decode_first_stage(model, batch).to(target_device)

    for x in samples_pytorch:
        samples.append(x)

    return samples


@torch.no_grad()
@torch.inference_mode()
def get_previewer(model):
    global VAE_approx_models
    from modules.config import path_vae_approx

    if is_flow_engine(model):
        # 新引擎的 VAE 潜通道数不同（16 / 128），复用 FLUX 的 approx 网络
        vae_approx_filename = os.path.join(
            path_vae,
            getattr(modules.config, "default_flux1_vae_name",
                    getattr(modules.config, "default_flux_vae_name", "ae.safetensors")))
    elif isinstance(model, Kolors):
        vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth')
    else:
        _inner = getattr(model, "model", None)
        _lf = getattr(_inner, "latent_format", None) if _inner is not None else None
        if isinstance(_lf, ldm_patched.modules.latent_formats.SDXL):
            vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth')
        else:
            vae_approx_filename = os.path.join(path_vae_approx, 'vaeapp_sd15.pth')

    if vae_approx_filename in VAE_approx_models:
        VAE_approx_model = VAE_approx_models[vae_approx_filename]
    else:
        if is_flow_engine(model) or isinstance(model, Kolors):
            sd = utils.load_torch_file(vae_approx_filename, safe_load=False, device='cpu' if devices.device.type != 'cuda' else None)
            VAE_approx_model = VAEApprox(latent_channels=model.forge_objects.vae.latent_channels)
            VAE_approx_model.load_state_dict(sd, strict=False)
            del sd
            VAE_approx_model.eval()
            VAE_approx_model.to(devices.device, devices.dtype)
            # flow 分支漏设了 current_type，preview_function 里
            # x0.to(current_type) 会因 None 抛 TypeError。
            VAE_approx_model.current_type = devices.dtype
        else:
            sd = torch.load(vae_approx_filename, map_location='cpu')
            VAE_approx_model = VAEApprox()
            VAE_approx_model.load_state_dict(sd)
            del sd
            VAE_approx_model.eval()

            if ldm_patched.modules.model_management.should_use_fp16():
                VAE_approx_model.half()
                VAE_approx_model.current_type = torch.float16
            else:
                VAE_approx_model.float()
                VAE_approx_model.current_type = torch.float32

            VAE_approx_model.to(ldm_patched.modules.model_management.get_torch_device())
        VAE_approx_models[vae_approx_filename] = VAE_approx_model

    @torch.no_grad()
    @torch.inference_mode()
    def preview_function(x0, step, total_steps):
        with torch.no_grad():
            # current_type 可能仍是 None（旧缓存或异常分支），此时跳过 dtype 转换
            x_sample = x0 if VAE_approx_model.current_type is None else x0.to(VAE_approx_model.current_type)
            x_sample = VAE_approx_model(x_sample) * 127.5 + 127.5
            x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')[0]
            x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)

            return x_sample

    return preview_function


@torch.no_grad()
@torch.inference_mode()
def ksampler(model, positive, negative, latent, seed=None, steps=30, cfg=7.0, sampler_name='dpmpp_2m_sde_gpu',
             scheduler='karras', denoise=1.0, disable_noise=False, start_step=None, last_step=None,
             force_full_denoise=False, callback_function=None, refiner=None, refiner_switch=-1,
             previewer_start=None, previewer_end=None, sigmas=None, noise_mean=None, disable_preview=False,
             width=None, height=None):

    # fail-fast：走到这里 positive/negative 还是 None，意味着 prompt 根本没编码成功。
    # 继续往下会在 sampler_helpers.convert_cond() 里抛
    #   TypeError: 'NoneType' object is not iterable
    # ——既没有栈上下文（看不到是 cond），也指不出该补什么。
    if positive is None or negative is None:
        raise RuntimeError(
            "[KSAMPLER] positive/negative conditioning is None —— prompt 未被编码。\n"
            "  最常见原因：当前 checkpoint 是 unet-only 分体 FLUX（权重里没有 T5/CLIP-L），\n"
            "  且 Forge 默认 text encoder 没配置。加载日志里会有：\n"
            "    'no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded.'\n"
            "  处理方式（二选一）：\n"
            "    A) 补齐 FLUX 组件：\n"
            "         models/clip/t5xxl_fp16.safetensors\n"
            "         models/clip/clip_l.safetensors\n"
            "         models/vae/ae.safetensors\n"
            "       并在 modules/config.py / Settings 里设置：\n"
            "         default_flux_t5_name / default_flux_clip_l_name / default_flux_vae_name\n"
            "    B) 换回 all-in-one SDXL checkpoint（18:20 那次能出图的那份）。\n"
            "  当前 positive={} negative={} clip={} vae={}".format(
                type(positive).__name__ if positive is not None else "None",
                type(negative).__name__ if negative is not None else "None",
                type(getattr(model, "clip", None)).__name__,
                type(getattr(model, "vae", None)).__name__))

    # ===== 已撤销：Krea2 cond 4D unsqueeze =====
    # 之前为了修 backend/nn/krea.py::txtfusion 的 4D 硬解包，
    # 在 ksampler 里对 cond (L,N,D) 做了 unsqueeze(0) → (1,L,N,D)。
    # 副作用：4D cond 进入 ComfyUI 的 calc_cond_batch 后，batch 推断
    # 会污染 latent 维度，最终 x_in 变 5D：
    #   RuntimeError: output with shape [1,16,128,128]
    #   doesn't match the broadcast shape [1,16,16,128,128]
    #
    # 正确方案：cond 保持 3D (L,N,D)，在 backend/nn/krea.py::txtfusion.forward
    # 开头加：
    #     if x.dim() == 3:
    #         x = x.unsqueeze(0)      # (L, N, D) -> (1, L, N, D)
    #     b, l, n, d = x.shape
    # 那里是真正的边界，4D 只在 txtfusion 内部存在，出来后仍是 3D (B,L,D)。
    #
    # 下面是旧逻辑，加开关保留以便对照调试，默认关闭：
    _KREA2_UNSQ_COND = False
    if _KREA2_UNSQ_COND:
        _krea2_engine = model if isinstance(model, Krea2) else getattr(model, "model_ori", None)
        if isinstance(_krea2_engine, Krea2):
            def _krea2_add_batch(conds):
                if not isinstance(conds, list):
                    return conds
                fixed = []
                for c in conds:
                    if isinstance(c, (list, tuple)) and len(c) == 2:
                        t, extra = c
                        if isinstance(t, torch.Tensor) and t.dim() == 3:
                            t = t.unsqueeze(0)
                        fixed.append([t, extra])
                    else:
                        fixed.append(c)
                return fixed
            positive = _krea2_add_batch(positive)
            negative = _krea2_add_batch(negative)
    # ===== 撤销结束 =====
    
    # ===== 新增：flow 引擎 guidance/cfg 双重放大防护 =====
    # FLUX-dev 蒸馏模型会把 guidance_scale 写入 cond extra['guidance']。
    # 若 sample() 再用同一数值跑 CFG，等效于放大两遍：
    #   x0 = x0_neg + cfg·(x0_pos − x0_neg)  +  模型内部 guidance embedding
    # 实测证据（本轮日志 cfg=3.91，guidance=3.91）：
    #   [DIAG-STEP] step= 0  x0_std=3.55   x_std=0.99  ← 噪声 std~1 ✓
    #   [DIAG-STEP] step=25  x0_std=5.37   x_std=5.37  ← x0 收敛但值放大 ~5× ✗
    #   [KSAMPLER]  std=14.88 mean=-2.58              ← process_out 后 14.88 = 5.37/0.3611+0.1159
    #   → VAE 收到 std≈14.9 / mean≈-2.6 的 latent → 黑图
    #
    # 处理：cond 已带 guidance(≠1.0) 时，强制 cfg=1.0
    #      （对齐 ComfyUI Flux-dev 标准工作流：CLIPTextEncode → FluxGuidance → KSampler(cfg=1)）
    if is_flow_engine(model) and abs(cfg - 1.0) > 1e-4:
        _gval = None
        try:
            if isinstance(positive, list) and positive \
                    and isinstance(positive[0], (list, tuple)):
                _extra = positive[0][1]
                if isinstance(_extra, dict) and "guidance" in _extra:
                    _g = _extra["guidance"]
                    if isinstance(_g, torch.Tensor) and _g.numel() > 0:
                        _gval = float(_g.flatten()[0])
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[KSAMPLER][WARN] 读取 guidance 失败: {}".format(_e)).printf()

        if _gval is not None and abs(_gval - 1.0) > 1e-4:
            printF(name=MasterName.get_master_name(),
                   info="[KSAMPLER] flow engine guidance={:.4f} 已在 cond 中，"
                        "cfg 从 {:.4f} 强制为 1.0（避免双重放大 → 黑图）".format(
                            _gval, cfg)).printf()
            cfg = 1.0
            # guidance 本身过大也会发散：FLUX-dev 典型 3.5，>5 基本会毁掉画面
            if _gval > 5.0:
                printF(name=MasterName.get_master_name(),
                       info="[KSAMPLER][WARN] guidance={:.2f} 偏高（FLUX-dev 典型 3.5），"
                            "仍可能发散，建议 UI 里调到 2.5~3.5".format(_gval)).printf()
    # ===== 新增结束 =====
    
    # ===== 新增：flow 引擎 SDE / ancestral / LCM 采样器防护 =====
    # 用户日志（Flux1_dev, 28 steps）：
    #   [Sampler] sigma_min=3.157e-4, sigma_max=1.0
    #   step=0   x0_std=1.2235  x_std=1.0006  ← 起点完美
    #   step=5   x0_std=3.9682  x_std=2.9979  ← 5 步内涨 3.3×
    #   step=25  x0_std=4.8933  x_std=4.8913  ← 稳定在 4.89（应为 0.9~1.2）
    #   → process_out: 4.8933/0.3611+0.1159 = 13.55 → VAE 黑图
    #
    # 原因：dpmpp_2m_sde 是随机采样器，每步按 √(2·σᵢ·|Δσ|) 注入噪声。
    #      karras 对 flow（线性 sigma）生成非均匀序列，前 5 步 σ 跳变达 0.16，
    #      注噪量 ≈ √(2×1.0×0.16) ≈ 0.57，是 simple 调度（√(2×1.0×0.037) ≈ 0.27）
    #      的 2.1 倍，且集中落在 σ 最高的不稳定区间 → x 从 1.0 涨到 3.0。
    #      Flux 训练时用线性 sigma，SDE + karras 双重违反训练假设。
    #
    # 处理：SDE / ancestral / LCM → euler；karras / exponential → simple。
    # 对齐 ComfyUI Flux1-dev 标准工作流。
    # 不碰 SD/SDXL：is_flow_engine(model) 为 False 时整块跳过。
    if is_flow_engine(model):
        _BAD_SAMPLERS = {
            "dpmpp_2m_sde", "dpmpp_2m_sde_gpu",
            "dpmpp_3m_sde", "dpmpp_3m_sde_gpu",
            "dpmpp_sde", "dpmpp_sde_gpu",
            "euler_ancestral", "euler_ancestral_cfg_pp",
            "dpmpp_2s_ancestral", "dpmpp_2s_ancestral_cfg_pp",
            "dpmpp_2s_ancestral_gpu",
            "ddpm", "lcm", "lcm_gpu",
        }
        if sampler_name in _BAD_SAMPLERS:
            printF(name=MasterName.get_master_name(),
                   info="[KSAMPLER] flow engine 不兼容 SDE/ancestral 采样器 '{}'，"
                        "自动切换到 euler（flow matching 会因持续加噪发散 → 黑图）。".format(
                            sampler_name)).printf()
            sampler_name = "euler"

        if scheduler in ("karras", "exponential"):
            printF(name=MasterName.get_master_name(),
                   info="[KSAMPLER] flow engine + scheduler '{}' → 'simple'"
                        "（euler + simple 是 Flux1-dev 标准组合，karras 的 "
                        "非均匀 sigma 违反 flow 的线性训练假设）".format(scheduler)).printf()
            scheduler = "simple"
    # ===== 新增结束 =====
            
    latent_image = latent["samples"]

    if sigmas is not None:
        sigmas = sigmas.clone().to(ldm_patched.modules.model_management.get_torch_device())

    if is_flow_engine(model):
        # 支持 4D（Flux 风格图像 latent [B,C,H,W]）与 5D（Wan2.1 风格
        # 视频 VAE latent [B,C,T,H,W]）两种布局。图像生成时 T=1，
        # 但模型内部的 Conv3d 仍按 5D 处理，noise 必须同维。
        if latent_image.dim() == 5:
            _b, _ch, _tt, _hh, _ww = latent_image.shape
        else:
            _b = latent_image.shape[0] if latent_image.dim() == 4 else 1
            _ch = latent_image.shape[-3]
            _hh = latent_image.shape[-2]
            _ww = latent_image.shape[-1]
            _tt = 1

        rng = ImageRNG((_ch, _hh, _ww), [seed], subseeds=None, subseed_strength=0,
                       seed_resize_from_h=0, seed_resize_from_w=0)
        noise = rng.next()
        if not isinstance(noise, torch.Tensor):
            noise = torch.as_tensor(np.asarray(noise))

        if latent_image.dim() == 5:
            # 5D：把单帧 noise 扩展到 T 维
            noise = noise.reshape(1, _ch, 1, _hh, _ww)
            if _tt > 1:
                noise = noise.repeat(1, 1, _tt, 1, 1)
        else:
            noise = noise.reshape(1, _ch, _hh, _ww)
        noise = noise.to(dtype=latent_image.dtype)

        if disable_noise:
            noise = torch.zeros_like(noise)
        elif "noise_mask" not in latent and denoise >= 0.999 and not force_full_denoise:
            latent_image = torch.zeros_like(noise)
        # 其余情况（img2img / inpaint / denoise<1）必须保留入参 latent，
        # 原来无条件置零会把输入图彻底丢掉。
    else:
        if disable_noise:
            noise = torch.zeros(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, device="cpu")
        else:
            batch_inds = latent["batch_index"] if "batch_index" in latent else None
            noise = ldm_patched.modules.sample.prepare_noise(latent_image, seed, batch_inds)

        if isinstance(noise_mean, torch.Tensor):
            noise = noise + noise_mean - torch.mean(noise, dim=1, keepdim=True)

    noise_mask = None
    if "noise_mask" in latent:
        noise_mask = latent["noise_mask"]

    previewer = get_previewer(model)

    if previewer_start is None:
        previewer_start = 0

    if previewer_end is None:
        previewer_end = steps

    def callback(step, x0, x, total_steps):
        ldm_patched.modules.model_management.throw_exception_if_processing_interrupted()
        # ===== 临时：每 5 步打印一次 x0/x 的 std，看模型是否在降噪 =====
        if step % 5 == 0:
            try:
                printF(name=MasterName.get_master_name(),
                       info="[DIAG-STEP] step={:>3}/{:<3} | x0_std={:.4f} x_std={:.4f}".format(
                           step, total_steps,
                           float(x0.std()), float(x.std()))).printf()
            except Exception:
                pass
        # =====
        y = None
        if previewer is not None and not disable_preview:
            y = previewer(x0, previewer_start + step, previewer_end)
        if callback_function is not None:
            callback_function(previewer_start + step, x0, x, previewer_end, y)

    # ===== 临时诊断：model_sampling + cond 结构 =====
    try:
        _mm = getattr(model, "model", model)
        _ms = getattr(_mm, "model_sampling", None)
        if _ms is not None:
            _parts = ["type={}".format(type(_ms).__name__)]
            _sig = getattr(_ms, "sigmas", None)
            if isinstance(_sig, torch.Tensor) and _sig.numel() > 0:
                _parts.append("sigmas[0]={:.4f}".format(float(_sig[0])))
                _parts.append("sigmas[-1]={:.4f}".format(float(_sig[-1])))
            for _a in ("zsnr", "sigma_min", "sigma_max", "sampling"):
                _v = getattr(_ms, _a, None)
                if _v is not None:
                    _parts.append("{}={}".format(_a, _v))
            printF(name=MasterName.get_master_name(),
                   info="[DIAG-SAMPLING] " + " | ".join(_parts)).printf()
    except Exception as _e:
        printF(name=MasterName.get_master_name(),
               info="[DIAG-SAMPLING] failed: {}".format(_e)).printf()
           

    def _dump_cond(tag, conds):
        try:
            if conds is None:
                printF(name=MasterName.get_master_name(),
                       info="[DIAG-COND] {} is None —— prompt 编码失败，"
                            "clip 大概率为 None".format(tag)).printf()
                return
            if not conds:
                printF(name=MasterName.get_master_name(),
                       info="[DIAG-COND] {} EMPTY list".format(tag)).printf()
                return
            for i, c in enumerate(conds):
                if not isinstance(c, (list, tuple)) or len(c) != 2:
                    printF(name=MasterName.get_master_name(),
                           info="[DIAG-COND] {}[{}] not [[t,extra]]: {}".format(
                               tag, i, type(c).__name__)).printf()
                    continue
                t, extra = c
                tshape = tuple(t.shape) if isinstance(t, torch.Tensor) else type(t).__name__
                tstd = float(t.std()) if isinstance(t, torch.Tensor) else "?"
                ekeys = list(extra.keys()) if isinstance(extra, dict) else "not_dict"
                printF(name=MasterName.get_master_name(),
                       info="[DIAG-COND] {}[{}] ct={} ct_std={} extra_keys={}".format(
                           tag, i, tshape, tstd, ekeys)).printf()
                if isinstance(extra, dict):
                    for k, v in extra.items():
                        if isinstance(v, torch.Tensor):
                            _is_float = v.dtype in (torch.float16, torch.float32,
                                                    torch.float64, torch.bfloat16)
                            _stat = ("{:.4f}".format(float(v.float().std()))
                                     if _is_float and v.numel() > 0 else "int")
                            printF(name=MasterName.get_master_name(),
                                   info="  [DIAG-COND]   {}: shape={} std={} dtype={}".format(
                                       k, tuple(v.shape), _stat, v.dtype)).printf()
                        else:
                            printF(name=MasterName.get_master_name(),
                                   info="  [DIAG-COND]   {}: {}({})".format(
                                       k, type(v).__name__, v)).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[DIAG-COND] {} failed: {}".format(tag, _e)).printf()

    # ===== pos vs neg 是否完全一致 =====
    try:
        if positive and negative and len(positive) > 0 and len(negative) > 0:
            _p = positive[0][0] if isinstance(positive[0], (list, tuple)) else None
            _n = negative[0][0] if isinstance(negative[0], (list, tuple)) else None
            if isinstance(_p, torch.Tensor) and isinstance(_n, torch.Tensor):
                _same_obj  = (_p is _n)              
                _same_data = torch.equal(_p, _n)     
                printF(name=MasterName.get_master_name(),
                       info="[DIAG-COND-HASH] pos_is_neg={} equal={} | "
                            "pos_id={} neg_id={} | "
                            "pos_sum={:.6f} neg_sum={:.6f}".format(
                                _same_obj, _same_data,
                                id(_p), id(_n),
                                float(_p.sum()), float(_n.sum()))).printf()
    except Exception as _e:
        printF(name=MasterName.get_master_name(),
               info="[DIAG-COND-HASH] failed: {}".format(_e)).printf()
    # =====
    
    _dump_cond("pos", positive)
    _dump_cond("neg", negative)

    # ===== 全零 cond 显式告警：adaLN 调制失效 → 输出严重偏离 =====
    def _check_cond_zero(tag, conds):
        try:
            if not conds:
                return
            for i, c in enumerate(conds):
                if not isinstance(c, (list, tuple)) or len(c) != 2:
                    continue
                extra = c[1]
                if not isinstance(extra, dict):
                    continue
                for k in ("pooled_output", "vector", "y"):
                    v = extra.get(k)
                    if isinstance(v, torch.Tensor) and v.numel() > 0:
                        if float(v.float().abs().max()) < 1e-6:
                            printF(name=MasterName.get_master_name(),
                                   info="[KSAMPLER/WARN] {}[{}]['{}'] 全零！"
                                        "adaLN 调制完全无效 → 生成内容会严重偏离。"
                                        "去查 Krea2 engine 的 cond 编码 / "
                                        "tap 索引（不是 decode 问题）".format(
                                            tag, i, k)).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[KSAMPLER/WARN] _check_cond_zero failed: {}".format(_e)).printf()

    _check_cond_zero("pos", positive)
    _check_cond_zero("neg", negative)
    
        # ============================================================
    # 【临时测试】y=zeros 是否是死灰主因
    # 只在 y / pooled_output / vector 全零时，注入小额噪声。
    # 如果图像明显改善 → y=zeros 确诊，去修 krea2_engine.py
    # 如果毫无变化 → y=zeros 是模型设计（罕见），转去查 context 4D→3D
    # 测完删除本段。
    # ============================================================
    _TEST_Y_ZEROS = False            # 👈 测试开关，修好后删这段
    if _TEST_Y_ZEROS:
        try:
            _inj_count = 0
            for _tag, _conds in (("pos", positive), ("neg", negative)):
                for _c in _conds:
                    if isinstance(_c, (list, tuple)) and len(_c) == 2 and isinstance(_c[1], dict):
                        for _k in ("y", "pooled_output", "vector"):
                            _v = _c[1].get(_k)
                            if isinstance(_v, torch.Tensor) and float(_v.abs().max()) < 1e-6:
                                # 用标准正态 ×0.1 注入，保持尺度合理
                                _c[1][_k] = _v + torch.randn_like(_v) * 0.1
                                _inj_count += 1
                                printF(name=MasterName.get_master_name(),
                                       info="[TEST-Y] {}.{} ZERO -> noise std={:.3f}".format(
                                           _tag, _k, float(_c[1][_k].std()))).printf()
            if _inj_count == 0:
                printF(name=MasterName.get_master_name(),
                       info="[TEST-Y] y/pooled/vector 均非零，跳过注入").printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[TEST-Y] failed: {}".format(_e)).printf()
    # ============================================================
    
    
    # Forge 引擎（Flux / Flux2 / ZImage / QwenImage / Kolors / Krea2）是 FlowEngine 实例，
    # 但 None / StableDiffusionModel / UnetPatcher 都没有 forge_objects；
    # 后者本身已经是 ModelPatcher，可直接给 sample()。
    if is_flow_engine(model) and hasattr(model, "forge_objects"):
        sampler_model = model.forge_objects.unet
    else:
        sampler_model = model


    # ===== 临时诊断：apply_model 进出口形状 =====
    # 只在 apply_model 外面包一层打印，不改任何 tensor。
    # 用实例属性做守卫，避免同一 sampler_model 被反复 wrap（多次出图会叠层）。
    if not getattr(sampler_model, "_meanvon_diag_wrapped", False) and \
            hasattr(sampler_model, "apply_model"):
        sampler_model._meanvon_diag_wrapped = True
        _diag_prev_apply = sampler_model.apply_model
        _diag_cnt = {"n": 0}

        def _diag_apply_model(x, t, *a, **kw):
            _diag_cnt["n"] += 1
            _n = _diag_cnt["n"]
            out = _diag_prev_apply(x, t, *a, **kw)
            try:
                _cc = kw.get("c_concat", a[0] if len(a) > 0 else None)
                _ctrl = kw.get("control", None)
                # 👇 改这一行：t 打数值不打 shape
                if isinstance(t, torch.Tensor) and t.numel() > 0:
                    _t_str = "val={:.4f}(min={:.4f},max={:.4f},n={})".format(
                        float(t.flatten()[0]), float(t.min()),
                        float(t.max()), t.numel())
                else:
                    _t_str = str(t)
                printF(name=MasterName.get_master_name(),
                       info="[DIAG-APPLY] #{} | in x={} dim={} | t={} | "
                            "c_concat={} | control={} | OUT={} dim={}".format(
                                _n,
                                tuple(x.shape), x.dim(),
                                _t_str,
                                tuple(_cc.shape) if isinstance(_cc, torch.Tensor) else None,
                                tuple(_ctrl.shape) if isinstance(_ctrl, torch.Tensor) else
                                    (type(_ctrl).__name__ if _ctrl is not None else None),
                                tuple(out.shape) if isinstance(out, torch.Tensor) else type(out).__name__,
                                out.dim() if isinstance(out, torch.Tensor) else -1)).printf()
            except Exception as _e:
                printF(name=MasterName.get_master_name(),
                       info="[DIAG-APPLY] print failed: {}".format(_e)).printf()
            return out

        sampler_model.apply_model = _diag_apply_model
        
    disable_pbar = True   
    modules.sample_hijack.current_refiner = refiner
    modules.sample_hijack.refiner_switch_step = refiner_switch
    ldm_patched.modules.samplers.sample = modules.sample_hijack.sample_hacked

    try:
        samples = ldm_patched.modules.sample.sample(sampler_model,
                                                    noise, steps, cfg, sampler_name, scheduler,
                                                    positive, negative, latent_image,
                                                    denoise=denoise, disable_noise=disable_noise,
                                                    start_step=start_step,
                                                    last_step=last_step,
                                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask,
                                                    callback=callback,
                                                    disable_pbar=disable_pbar, seed=seed, sigmas=sigmas)

        out = latent.copy()
        
        # ===== 临时止血：sample() 出口若有 5D，压回 4D =====
        if isinstance(samples, torch.Tensor) and samples.dim() == 5:
            _shape5d = tuple(samples.shape)
            # 先诊断各 T 切片差异
            if samples.shape[2] > 1:
                _diff_max = float(
                    (samples[:, :, 1:] - samples[:, :, :1]).abs().max())
                printF(name=MasterName.get_master_name(),
                       info="[KSAMPLER/diag] 5D 输出 T={} > 1，"
                            "各 T 切片最大差异 max={:.6f} （≈0 表示 broadcast 污染，"
                            "取 T=0 无损；非 0 表示真实多帧，期望形状需重新审视）".format(
                                samples.shape[2], _diff_max)).printf()

            if samples.shape[2] == 1:
                # T=1，直接 squeeze —— 无损
                samples = samples.squeeze(2)
            else:
                # T>1：图像模型（Krea2）不该有 T 维，几乎必为 cond 4D broadcast。
                # 取 T=0；若上一步 diff_max 明显非 0，需要查 backend/nn/krea.py
                # 与 context 4D→3D 处理。
                samples = samples[:, :, 0]
            printF(name=MasterName.get_master_name(),
                   info="[KSAMPLER/squeeze] {} -> {} (T dim collapse)".format(
                       _shape5d, tuple(samples.shape))).printf()

        _vae = (getattr(getattr(model, "forge_objects", None), "vae", None)
                if is_flow_engine(model) else getattr(model, "vae", None))
        _lf = get_latent_format(model, vae=_vae)

        _std_before = float(samples.std())
        printF(name=MasterName.get_master_name(),
               info="[KSAMPLER] std={:.4f} lf={} scale={} shift={} "
                    "samples.shape={} batch={} seed={} steps={} cfg={}".format(
                        _std_before,
                        type(_lf).__name__ if _lf is not None else "None",
                        getattr(_lf, "scale_factor", "?"),
                        getattr(_lf, "shift_factor", "?"),
                        tuple(samples.shape),
                        samples.shape[0] if samples.ndim > 0 else 0,
                        seed, steps, cfg)).printf()

        # ===== 这里绝不做 latent 空间换算 =====
        # 实测：sample_hijack.sample_hacked 的出口已经是 process_out 之后
        # 的 VAE 原生空间 latent（本次日志：0.6352 × (1/0.13025) = 4.8768
        # 与 std_before=4.8770 完全吻合），而 opVAEDecode → vae.decode
        # 内部不会再 process_out，samples 被原样交给 first_stage_model。
        #
        # 因此：
        #   process_in  → 0.635 → VAE 收到缩小 7.7 倍的 latent → 整图发灰
        #   process_out → 26.29 → 放大两次 → 过曝发白
        # 正确做法是原样透传。
        # SDXL VAE 原生空间 std 正常 3~6；flow 模型（Flux 等）另有一套量级。
        # 仅作提示，不做任何修改。
        if _std_before < 1.0 and isinstance(_lf, ldm_patched.modules.latent_formats.SDXL):
            printF(name=MasterName.get_master_name(),
                   info="[KSAMPLER][WARN] latent std={:.3f} 对 SDXL VAE 空间偏低"
                        "（正常 3~6），decode 可能发灰。".format(_std_before)).printf()

        out["samples"] = samples

    finally:
        modules.sample_hijack.current_refiner = None
        
    # ==== 加在 return 之前 ====
    printF(name=MasterName.get_master_name(),
           info="[FINAL-LATENT] mean={:+.4f} std={:.4f} min={:+.4f} max={:+.4f}".format(
               float(samples.mean()), float(samples.std()),
               float(samples.min()), float(samples.max()))).printf()
    return out


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    out = []
    for y in x:
        arr = y.float().cpu().numpy()
        # 视频单帧：VAEDecode 有时改不动 5D，兜底
        if arr.ndim == 5:                    # (C, T, H, W)
            arr = arr[:, 0]                  # → (C, H, W)
        # 剥前导 batch
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]                     # → (C, H, W) 或 (T, H, W, C)
        # CHW → HWC（只在首维是 C 且尾维不像 C 时转）
        if arr.ndim == 3 and arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = arr.transpose(1, 2, 0)     # → (H, W, C)
        out.append(np.clip(255. * arr, 0, 255).astype(np.uint8))
    return out


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y
