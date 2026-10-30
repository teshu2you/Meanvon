import copy
import sys
import re
import backend
from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.flux2 import Flux2
from backend.diffusion_engine.zimage import ZImage
from backend.diffusion_engine.qwen import QwenImage
from backend.diffusion_engine.krea import Krea2     # ← 新增
from backend.diffusion_engine.base import ForgeDiffusionEngine
import lark
from modules.core import should_use_prompt_expansion
from backend.diffusion_engine.kolors import Kolors
from backend.patcher.clip import JointTextEncoder
from backend.patcher_diffusion.kolors.models.tokenization_chatglm import ChatGLMTokenizer
from ldm_patched.text_encoders.hydit import *
import modules.core as core
import os
import torch
import modules.patch
import modules.config
import modules.constants as constants
import ldm_patched.modules.model_management
import ldm_patched.modules.latent_formats
import modules.inpaint_worker
import extras.vae_interpose as vae_interpose
from extras.expansion import FooocusExpansion
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from collections import namedtuple
# from modules.upscaler import model
from modules.util import get_current_log_path, get_previous_log_path, show_cuda_info, free_cuda_mem, free_cuda_cache, \
    get_file_from_folder_list, get_enabled_loras

from diffusers import (
    StableDiffusionXLPipeline,
    UNet2DConditionModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    KDPM2AncestralDiscreteScheduler,
)

from ldm_patched.modules.model_base import SDXL, SDXLRefiner, HunyuanDiT
from modules.sample_hijack import clip_separate
from ldm_patched.k_diffusion.sampling import BrownianTreeNoiseSampler
from modules.settings import default_settings
from util.printf import printF, MasterName

# xl_base: core.StableDiffusionModel = None
# xl_base_hash = ''
#
# xl_base_patched: core.StableDiffusionModel = None
# xl_base_patched_hash = ''

# xl_refiner: core.StableDiffusionModel = None
# xl_refiner_hash = ''

model_base = core.StableDiffusionModel()
model_refiner = core.StableDiffusionModel()

clip_vision: core.StableDiffusionModel = None
clip_vision_hash = ''

controlnet_canny: core.StableDiffusionModel = None
controlnet_canny_hash = ''

controlnet_depth: core.StableDiffusionModel = None
controlnet_depth_hash = ''

final_expansion = None
final_unet = None
final_clip = None
final_model_ori = None
final_vae = None
final_refiner_unet = None
final_refiner_vae = None
final_loras = None

loaded_ControlNets = {}


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnets(model_paths):
    global loaded_ControlNets
    cache = {}
    for p in model_paths:
        if p is not None:
            if p in loaded_ControlNets:
                cache[p] = loaded_ControlNets[p]
            else:
                cache[p] = core.load_controlnet(p)
    loaded_ControlNets = cache
    return


@torch.no_grad()
@torch.inference_mode()
def assert_model_integrity():
    global model_base
    error_message = None

    if model_base.unet_with_lora is None:
        return

    # if not isinstance(model_base.unet_with_lora.model, SDXL):
    #     error_message = 'You have selected base model other than SDXL. This is not supported yet.'

    if error_message is not None:
        raise NotImplementedError(error_message)

    return True

# ===== FLUX.2 辅助权重解析 =====
_FLUX2_TE_CONFIG_KEYS = {
    "klein4b": ("default_flux2_text_encoder_qwen3_4b", "default_flux2_text_encoder"),
    "klein9b": ("default_flux2_text_encoder_qwen3_8b", "default_flux2_text_encoder"),
    "dev":     ("default_flux2_text_encoder_mistral",  "default_flux2_text_encoder"),
}

_FLUX2_TE_DISPLAY = {"klein4b": "Qwen3-4B", "klein9b": "Qwen3-8B", "dev": "Mistral-Small"}

_FLUX2_TE_NAME_HINTS = {
    "klein4b": ("qwen3_4b", "qwen3-4b", "qwen34b"),
    "klein9b": ("qwen3_8b", "qwen3-8b", "qwen38b"),
    "dev":     ("mistral",),
}

_TEXT_ENCODER_EXTS = (".safetensors", ".sft", ".pt", ".pth", ".bin", ".gguf", ".ckpt")

# basename(checkpoint) -> 已通过实际加载验证的变体。
# klein 的 4B/9B 在文件名里经常没有标记，只能靠试一次来定，试出来就记住，
# 避免每次切模型都多付一次失败加载。
_FLUX2_RESOLVED = {}


# ===== Krea2 =====
# Krea2 用的是 FLUX 架构（double_blocks/single_blocks），但：
#   - 文本编码器 = Qwen3-VL-4B（视觉-语言模型，需要用完整的 TE 权重）
#   - VAE        = Wan2.1 专用 VAE（16ch latent，与 Flux 的 16ch 相近但并非同一份）
# 两者都不在 checkpoint 里，必须外部提供，否则 backend/loader.py 会在
#   load_huggingface_component(..., "text_encoder", ...)
# 处 assert isinstance(state_dict, dict) and len(state_dict) > 16 失败，
# 抛出 "You do not have Qwen3 state dict!" —— 这个报错完全不含缺件线索，
# 所以在 refresh_base_model 这一层就要把缺件拦住并说清楚。
_KREA2_TYPE = getattr(constants, "TYPE_Krea2", "krea2")

_KREA2_TE_HINTS = (
    "qwen3vl_4b", "qwen3vl-4b", "qwen3vl4b", "qwen3vl",
    "qwen3vl_4b_transformer",
)
_KREA2_VAE_HINTS = (
    "wan21_vae", "wan2.1_vae", "wan21-vae", "wan2.1-vae",
    "wan_vae", "wan2.1vae",
    "krea2_vae", "krea_vae", "krea2-vae", "krea-vae",
)


def _is_krea2_name(name_lower):
    """名字层面判定 Krea2。

    只按 'krea2' / 'krea-2' / 'krea_2' 判定；
    故意不把 'flux' 作为参考——Krea2 的权重前缀本来就是 double_blocks.，
    一旦和 Flux 混在一起，用户又会回到 'Could not detect' 里打转。
    merged 权重（比如 'flux-krea2-merged'）请自行改名或在 config 里
    显式指定，本文件不做名字以外的结构判定。
    """
    return ("krea2" in name_lower) or ("krea-2" in name_lower) or ("krea_2" in name_lower)

def _norm_aux_name(s):
    """抹掉大小写与所有分隔符：'qwen_3_4b.safetensors' -> 'qwen34bsafetensors'"""
    return re.sub(r"[^a-z0-9]", "", str(s or "").lower())


def _aux_path(name, folders):
    if not name:
        return None
    path = get_file_from_folder_list(name, folders)
    # get_file_from_folder_list 找不到时会返回含 "Not Exist" 的字符串
    if not path or "Not Exist" in str(path):
        printF(name=MasterName.get_master_name(),
               info="[Warning] auxiliary file NOT found: {}".format(name)).printf()
        return None
    return path


def _name_matches_variant(path, variant):
    n = _norm_aux_name(os.path.basename(str(path)))
    return any(_norm_aux_name(h) in n for h in _FLUX2_TE_NAME_HINTS[variant])


def _detect_flux2_variant(model_name):
    """从 checkpoint 名猜 FLUX.2 变体；名字里没有尺寸信息时返回 None。

    backend 的 guess（日志里的 Flux2K9B）是读张量形状得出的，比文件名可靠。
    这里只在名字里明确写了 4b/9b 时才下结论，否则交给 _flux2_aux_candidates
    按候选逐个试 —— klein 的 4B 与 9B hidden dim 不同，猜错必然加载失败。
    """
    l = _norm_aux_name(model_name)
    if "9b" in l:
        return "klein9b"
    if "4b" in l:
        return "klein4b"
    if "mistral" in l:
        return "dev"
    if "klein" in l or "flux2" in l:
        return None
    return "dev"


def _iter_text_encoder_folders(folders):
    """config.path_text_encoder 可能是 list，也可能被写成 str，统一成 list[str]。"""
    if folders is None:
        return []
    if isinstance(folders, (str, bytes, os.PathLike)):
        folders = [folders]
    out = []
    for f in folders:
        if not f:
            continue
        f = os.fspath(f)
        if os.path.isdir(f):
            out.append(f)
    return out


def _scan_text_encoder_folder(hints, folders, max_depth=3):
    """在 text_encoder 目录里按关键词找文本编码器。

    支持两种存放方式：
      1) 平铺单文件： text_encoders/Qwen3-8B.safetensors
      2) HF 目录：    text_encoders/Qwen3-8B/{config.json, *.safetensors}
    """
    hints = [_norm_aux_name(h) for h in hints]

    def _match(s):
        n = _norm_aux_name(s)
        return any(h and h in n for h in hints)

    for root in _iter_text_encoder_folders(folders):
        base_depth = root.rstrip("\\/").count(os.sep)
        for cur, dirnames, filenames in os.walk(root):
            if cur.rstrip("\\/").count(os.sep) - base_depth > max_depth:
                dirnames[:] = []
                continue

            lower_files = [fn.lower() for fn in filenames]

            if "config.json" in lower_files and any(
                fn.endswith(_TEXT_ENCODER_EXTS) for fn in lower_files
            ):
                if _match(os.path.basename(cur)) or _match(cur):
                    printF(name=MasterName.get_master_name(),
                           info="[Info] auxiliary auto-discovered (HF dir): {}".format(cur)).printf()
                    return cur

            for fn in sorted(filenames):
                if fn.lower().endswith(_TEXT_ENCODER_EXTS) and _match(fn):
                    p = os.path.join(cur, fn)
                    printF(name=MasterName.get_master_name(),
                           info="[Info] auxiliary auto-discovered: {}".format(p)).printf()
                    return p
    return None


def _describe_aux_dirs(folders):
    """打印 text_encoder 目录下实际内容，方便对照命名。"""
    lines = []
    for root in _iter_text_encoder_folders(folders):
        lines.append("  [{}]".format(root))
        try:
            entries = sorted(os.listdir(root))
        except OSError as e:
            lines.append("    <unreadable: {}>".format(e))
            continue
        if not entries:
            lines.append("    <empty>")
        for e in entries[:25]:
            p = os.path.join(root, e)
            lines.append("    {}{}".format(e, "/" if os.path.isdir(p) else ""))
        if len(entries) > 25:
            lines.append("    ... ({} more)".format(len(entries) - 25))
    if not lines:
        lines.append("  <config.path_text_encoder 里没有任何有效目录>")
    return "\n".join(lines)


def _flux2_te_path(variant):
    """取指定变体的文本编码器路径。

    注意：config 键指向的文件名若明显是**另一个**变体（9B 拿到 4B），
    不要盲目采用 —— 尺寸不符会在 forge_loader 里以空 state dict 的形式炸掉，
    报错信息还完全没有指向性。
    """
    cfg = modules.config
    for key in _FLUX2_TE_CONFIG_KEYS[variant]:
        p = _aux_path(getattr(cfg, key, None), cfg.path_text_encoder)
        if not p:
            continue
        if _name_matches_variant(p, variant):
            return p
        printF(name=MasterName.get_master_name(),
               info="[Warning] config 键 '{}' 指向 {}，不是 {} 变体，跳过".format(
                   key, os.path.basename(str(p)), _FLUX2_TE_DISPLAY[variant])).printf()

    return _scan_text_encoder_folder(_FLUX2_TE_NAME_HINTS[variant],
                                     getattr(cfg, "path_text_encoder", None))


def _flux2_aux_candidates(model_name, checkpoint_filename=None):
    """返回 [(variant, [vae_path, te_path])]，按尝试优先级排序。

    - 名字含 'klein' → 只试 Qwen3（klein9b → klein4b）。
      绝不退到 dev/Mistral：Mistral 与 Qwen3 维度不同，喂错只会在 backend 里
      触发无意义的 shape assert（replace_state_dict: assert weight.shape[0] == 3072），
      把「缺 qwen 文件」这个真问题盖住。
    - 名字含 'dev' / 'mistral' → 先 dev，再 qwen 兜底（防命名笔误）。
    - 无标记 → klein9b → klein4b → dev 全试。

    成功的变体会写入 _FLUX2_RESOLVED，下次切回同一 checkpoint 直接命中。
    """
    cfg = modules.config
    vae = _aux_path(getattr(cfg, "default_flux2_vae_name", None), cfg.path_vae)

    name_norm = _norm_aux_name(model_name)
    cache_key = os.path.basename(str(checkpoint_filename or model_name or ""))

    if "klein" in name_norm:
        base_order = ["klein9b", "klein4b"]
    elif "dev" in name_norm or "mistral" in name_norm:
        base_order = ["dev", "klein9b", "klein4b"]
    else:
        base_order = ["klein9b", "klein4b", "dev"]

    order = []
    for v in (_FLUX2_RESOLVED.get(cache_key),
              _detect_flux2_variant(model_name),
              *base_order):
        if v and v not in order:
            order.append(v)

    out = []
    skipped = []
    for v in order:
        te = _flux2_te_path(v)
        if te is None:
            skipped.append(v)
            continue
        out.append((v, [p for p in (vae, te) if p]))

    printF(name=MasterName.get_master_name(),
           info="[Info] FLUX.2 候选顺序: {} | 可用: {} | 缺文本编码器: {}".format(
               order,
               [v for v, _ in out] or "<none>",
               skipped or "<none>")).printf()
    return out

def _get_aux_state_dicts(model_file_type, model_name=None, flux2_variant=None):
    """
    返回需要从外部补齐的组件权重路径列表（VAE / 文本编码器 / LLM）。
    这些路径由 core.load_model -> load_checkpoint_guess_config 作为
    additional_state_dicts 传给 backend.loader.forge_loader。

    unet-only 的 checkpoint（Flux / Flux2 / ZImage / QwenImage / HunyuanDiT）
    必须提供，否则 backend/loader.py 会断言失败（如
    "You do not have Qwen3 state dict!"），异常 message 里不含任何
    缺失文件线索，极难排查 —— 所以在这一层就把缺件拦住并报清楚。

    model_name: 原始 checkpoint 名，仅 FLUX.2 需要（区分 klein 4B/9B 与 dev 32B）。
    """
    cfg = modules.config
    paths = []
    missing = []

    def _p(name, folders):
        if not name:
            return None
        path = get_file_from_folder_list(name, folders)
        # get_file_from_folder_list 找不到时会返回含 "Not Exist" 的字符串
        if not path or "Not Exist" in str(path):
            printF(name=MasterName.get_master_name(),
                   info="[Warning] auxiliary file NOT found: {}".format(name)).printf()
            return None
        return path

    if model_file_type == constants.TYPE_Flux1:
        paths = [
            _p(getattr(cfg, "default_flux1_vae_name", None), cfg.path_vae),
            _p(getattr(cfg, "default_flux1_text_encoder_clip", None), cfg.path_text_encoder),
            _p(getattr(cfg, "default_flux1_text_encoder_t5xxl", None), cfg.path_text_encoder),
        ]

    elif model_file_type == constants.TYPE_Flux2:
        variant = flux2_variant or _detect_flux2_variant(model_name) or "klein9b"
        keys = _FLUX2_TE_CONFIG_KEYS[variant]

        vae_path = _aux_path(getattr(cfg, "default_flux2_vae_name", None), cfg.path_vae)
        te_path = _flux2_te_path(variant)

        if vae_path is None:
            missing.append(("FLUX.2 VAE", "default_flux2_vae_name"))
        if te_path is None:
            missing.append((
                "FLUX.2 text encoder ({})".format(_FLUX2_TE_DISPLAY[variant]),
                " / ".join(keys) + "，或文件名含 " + str(_FLUX2_TE_NAME_HINTS[variant]),
            ))

        paths = [vae_path, te_path]

    elif model_file_type == constants.TYPE_ZImage:
        paths = [
            _p(getattr(cfg, "default_zimage_vae_name", None), cfg.path_vae),
            _p(getattr(cfg, "default_zimage_text_encoder_clip", None), cfg.path_text_encoder),
        ]

    elif model_file_type == constants.TYPE_QwenImage:
        # QwenImage：Qwen2.5-VL 文本编码器 + 独立 VAE
        paths = [
            _p(getattr(cfg, "default_qwenimage_vae_name", None), cfg.path_vae),
            _p(getattr(cfg, "default_qwenimage_text_encoder", None), cfg.path_text_encoder),
        ]

    elif model_file_type == _KREA2_TYPE:
        # Krea2 = FLUX.1 16ch 架构（krea.py forward: x=(1,16,H,W)）
        # ★ 正确 VAE = FLUX.1 的 ae.safetensors（AutoencoderKL，16ch）
        # ★ 错误 VAE：
        #   - wan2.1_vae  → 16ch 但 Wan21 latents_mean/std 偏移完全不同 → 灰+局部彩
        #   - flux2-vae   → 32ch latent，backend 用 Qwen2DVAE 加载 → 全部 conv 尺寸失配崩溃
        vae_path = _p(getattr(cfg, "default_krea2_vae_name", None), cfg.path_vae)
        te_path = _p(getattr(cfg, "default_krea2_text_encoder", None), cfg.path_text_encoder)

        paths = [vae_path, te_path]

    elif model_file_type == constants.TYPE_Kolors:
        paths = [
            _p(getattr(cfg, "default_kolors_chatglm_name", None), cfg.path_llm),
        ]
        
    elif model_file_type == constants.TYPE_HunyuanDiT:
        # HunyuanDiT 使用 CLIP-L + mT5-XXL 双文本编码器 + SDXL VAE。
        # checkpoint 若为 unet-only，这三份必须外部提供，
        # 否则 CLIP 会加载成 SDXL 形态 / mT5 缺失 → cond 无语义 → 花屏。
        paths = [
            _p(getattr(cfg, "default_hunyuandit_vae_name", None), cfg.path_vae),
            _p(getattr(cfg, "default_hunyuandit_text_encoder_clip", None), cfg.path_text_encoder),
            _p(getattr(cfg, "default_hunyuandit_text_encoder_mt5", None), cfg.path_text_encoder),
        ]

    if missing:
        detail = "; ".join("{} ← 缺少配置或文件（试过: {}）".format(n, k) for n, k in missing)
        raise RuntimeError(
            "[AuxWeights] {0} 需要外部权重，但没找齐：{1}\n"
            "text_encoder 目录实际内容:\n{2}\n"
            "处理方式：把缺失的权重放进上面的目录（单文件或 HF 目录均可，"
            "文件名需含 qwen3_4b / qwen3_8b / mistral 一类关键词），"
            "或在 modules/config.py 里把上面列出的 config 键指向实际文件名。".format(
                model_file_type, detail,
                _describe_aux_dirs(getattr(cfg, "path_text_encoder", None)),
            )
        )

    return [p for p in paths if p]


def get_model_file_type(name):
    """
    按「模型族」优先判定，避免 'flux-turbo' 被判成 turbo、
    'zimage-turbo' 被判成 turbo 之类的误判。
    返回的常量与 refresh_base_model 中分支比较的常量必须完全一致。
    """
    l_name = name.lower()

    # --- Krea2 最优先 ---
    # Krea2 与 Flux 共用 unet 权重键，但 TE / VAE 完全不同。
    # 如果先命中 "flux" 分支，_get_aux_state_dicts 就会按 Flux1 的
    # clip_l + t5xxl 去准备 aux，Qwen3-VL-4B 依然找不到，
    # 又会回到 "You do not have Qwen3 state dict!"。
    if _is_krea2_name(l_name):
        return _KREA2_TYPE
    
    # --- 模型族优先（顺序不可乱）---
    if "flux" in l_name:
        # FLUX.2 使用 Mistral-Small 文本编码器，与 FLUX.1 完全不同
        if "flux2" in l_name or "flux.2" in l_name or "flux-2" in l_name:
            return constants.TYPE_Flux2
        return constants.TYPE_Flux1

    if "zimage" in l_name or "z-image" in l_name or "z_image" in l_name:
        return constants.TYPE_ZImage

    if "qwen" in l_name:                     # QwenImage / Qwen-Image
        return constants.TYPE_QwenImage

    if "kolors" in l_name:
        return constants.TYPE_Kolors

    if "hunyuandit" in l_name:
        return constants.TYPE_HunyuanDiT

    # --- 加速/蒸馏类放最后 ---
    if "lcm" in l_name:
        return constants.TYPE_LCM
    if "lightning" in l_name:
        return constants.TYPE_LIGHTNING
    if "turbo" in l_name:
        return constants.TYPE_TURBO

    return constants.TYPE_NORMAL

_TE_MISMATCH_SIGNATURES = (
    "state dict!",            # You do not have Qwen3 state dict!
    "state_dict!",
    "assert weight.shape[0] == 3072",
)

def _looks_like_te_mismatch(err):
    """只有这类错误才值得换下一个变体重试（TE 尺寸/前缀不匹配）。
    量化元数据解析、文件损坏、OOM 等与变体无关，重试只会掩埋真因。"""
    msg = str(err)
    if any(s in msg for s in _TE_MISMATCH_SIGNATURES):
        return True
    return "text_encoder" in msg and ("size mismatch" in msg or "shape" in msg)

@torch.no_grad()
@torch.inference_mode()
def refresh_base_model(name, performance_selection=None):
    printF(name=MasterName.get_master_name(), info="[Function] Enter-> refresh_base_model").printf()

    global model_base
    printF(name=MasterName.get_master_name(), info="[Parameters] Base Model = {}".format(name)).printf()
    printF(name=MasterName.get_master_name(),
           info="[Parameters] performance_selection = {}".format(performance_selection)).printf()

    # filename = os.path.abspath(os.path.realpath(os.path.join(modules.config.paths_checkpoints[0], name)))
    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)
    if model_base.filename == filename:
        printF(name=MasterName.get_master_name(),
               info="model_base.filename: {} is same as filename:{}, just return".format(model_base.filename,
                                                                                         filename)).printf()
        return

    if "Not Exist" in name:
        name = "None"
        printF(name=MasterName.get_master_name(),
               info="{} is NOT Exist!, changed to None".format(name)).printf()

    if name in ('None', 'none', 'NONE', '') or name is None:
        printF(name=MasterName.get_master_name(), info="[Warning] Base Model unloaded.").printf()
        return

    if constants.TYPE_LIGHTNING in name.lower() and 1 == 2:
        printF(name=MasterName.get_master_name(), info="[Warning] Lightning model, no need to load here!").printf()
        return

    if "sd_xl_base_1.0" in name and performance_selection is not None and constants.TYPE_LIGHTNING in performance_selection.value.lower() and 1 == 2:
        printF(name=MasterName.get_master_name(),
               info="[Warning] Lightning mode, unload #sd_xl_base_1.0# base model!").printf()
        free_cuda_mem()
        return

    # vae_filename is in valid only when model is Flux or Kolors, other model type use old method
    model_file_type = get_model_file_type(name)

    if model_file_type == constants.TYPE_Flux2:
        attempts = _flux2_aux_candidates(name, filename)
        if not attempts:
            # 一个候选都凑不出 → 让 _get_aux_state_dicts 抛带目录清单的详细错误
            _get_aux_state_dicts(model_file_type, name, flux2_variant="klein9b")
            raise RuntimeError("[AuxWeights] FLUX.2 没有任何可用文本编码器")
    else:
        attempts = [(None, _get_aux_state_dicts(model_file_type, name))]

    model_base = core.StableDiffusionModel()
    errors = []
    for variant, aux in attempts:
        label = "{} ({})".format(model_file_type, variant) if variant else str(model_file_type)
        printF(name=MasterName.get_master_name(),
               info="[Parameters] try {} aux files = {}".format(label, aux)).printf()
        try:
            model_base = core.load_model(filename, model_file_type=model_file_type, vae_filename=aux)
        except Exception as e:
            model_base = core.StableDiffusionModel()
            errors.append((variant, e))
            printF(name=MasterName.get_master_name(),
                   info="[Warning] load with {} failed: {}".format(label, e)).printf()
            try:
                free_cuda_mem()
            except Exception:
                pass

            if model_file_type == constants.TYPE_Flux2 and not _looks_like_te_mismatch(e):
                printF(name=MasterName.get_master_name(),
                       info="[Error] {} 的失败与文本编码器无关（{}），"
                            "停止切换变体；真正原因见上一条 traceback".format(label, type(e).__name__)).printf()
                raise
            continue

        # 👇 加载后立刻核对：实际用了哪个 vae 文件
        try:
            _v = model_base.vae
            _fsm = getattr(_v, "first_stage_model", None)
            _sd = _fsm.state_dict() if _fsm is not None else {}
            _keys = list(_sd.keys())
            _path = None
            for _src in (_fsm, _v, getattr(_v, "patcher", None)):
                if _src is None: continue
                for _a in ("filename", "ckpt_name", "path", "model_path", "_filename"):
                    _val = getattr(_src, _a, None)
                    if _val:
                        _path = _val; break
                if _path: break
            printF(name=MasterName.get_master_name(),
                   info="[POST-LOAD] vae_file={} | fsm={} | keys={} | "
                        "has encoder.conv1.weight={} | has conv1.weight={} | "
                        "first_encoder_conv={}".format(
                            _path or "?", type(_fsm).__name__, len(_keys),
                            "encoder.conv1.weight" in _sd,
                            "conv1.weight" in _sd,
                            tuple(_sd["encoder.conv1.weight"].shape) if "encoder.conv1.weight" in _sd else None,
                        )).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[POST-LOAD] probe failed: {}".format(_e)).printf()
            
        if variant:
            _FLUX2_RESOLVED[os.path.basename(str(filename))] = variant
            printF(name=MasterName.get_master_name(),
                   info="[Info] FLUX.2 变体锁定为 {}（缓存，下次直接命中）".format(variant)).printf()
        break
    else:
        if model_file_type == constants.TYPE_Flux2:
            printF(name=MasterName.get_master_name(),
                   info="[Hint] FLUX.2 全部变体都因「文本编码器不匹配」失败：{}".format(
                       [(v, type(e).__name__, str(e)[:120]) for v, e in errors])).printf()
        # 抛第一个错误：通常是最有针对性的那个（klein9b），而不是最后的兜底失败
        raise errors[0][1] if errors else RuntimeError("[AuxWeights] no attempt made")


def is_base_sdxl():
    assert model_base is not None
    return isinstance(model_base.unet.model, SDXL)


# @torch.no_grad()
# @torch.inference_mode()
# def refresh_refiner_model(name, performance_selection=None):
#     printF(name=MasterName.get_master_name(), info="[Function] Enter-> refresh_refiner_model").printf()
#     global model_refiner

#     # filename = os.path.abspath(os.path.realpath(os.path.join(modules.config.paths_checkpoints[0], name)))

#     filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

#     printF(name=MasterName.get_master_name(), info="[Parameters] Refiner Model = {}".format(name)).printf()
#     printF(name=MasterName.get_master_name(),
#            info="[Parameters] performance_selection = {}".format(performance_selection)).printf()
#     if model_refiner.filename == filename:
#         printF(name=MasterName.get_master_name(),
#                info="model_refiner.filename: {} is same as filename:{}, just return".format(model_refiner.filename,
#                                                                                             filename)).printf()
#         return

#     if "Not Exist" in name:
#         name = "None"
#         printF(name=MasterName.get_master_name(),
#                info="{} is NOT Exist!, changed to None".format(name)).printf()

#     if name in ('None', 'none', 'NONE', '') or name is None:
#         printF(name=MasterName.get_master_name(), info="[Warning] Refiner unloaded.").printf()
#         return

#     if constants.TYPE_LIGHTNING in name.lower() and 1 == 2:
#         printF(name=MasterName.get_master_name(), info="[Warning] Lightning model, no need to load here!").printf()
#         return

#     if "sd_xl_base_1.0" in name and performance_selection is not None and constants.TYPE_LIGHTNING in performance_selection.value.lower() and 1 == 2:
#         printF(name=MasterName.get_master_name(),
#                info="[Warning] Lightning mode, unload #sd_xl_base_1.0# base model!").printf()
#         free_cuda_mem()
#         return

#     model_refiner = core.load_model(filename, model_file_type=get_model_file_type(name))
#     printF(name=MasterName.get_master_name(),
#            info="[Warning] Refiner model loaded: {}".format(model_refiner.filename)).printf()

#     if isinstance(model_refiner.unet.model, SDXL):
#         model_refiner.clip = None
#         model_refiner.vae = None
#     elif isinstance(model_refiner.unet.model, SDXLRefiner):
#         model_refiner.clip = None
#         model_refiner.vae = None
#     else:
#         model_refiner.clip = None

#     return


@torch.no_grad()
@torch.inference_mode()
def synthesize_refiner_model():
    global model_base, model_refiner
    printF(name=MasterName.get_master_name(),
           info="Synthetic Refiner Activated").printf()
    model_refiner = core.StableDiffusionModel(
        unet=model_base.unet,
        vae=model_base.vae,
        clip=model_base.clip,
        clip_vision=model_base.clip_vision,
        filename=model_base.filename
    )
    model_refiner.vae = None
    model_refiner.clip = None
    model_refiner.clip_vision = None

    return


@torch.no_grad()
@torch.inference_mode()
def refresh_loras(loras, base_model_additional_loras=None, performance_selection=None):
    printF(name=MasterName.get_master_name(), info="[Function] Enter-> refresh_loras").printf()
    global model_base, model_refiner, final_loras

    if not isinstance(base_model_additional_loras, list):
        base_model_additional_loras = []

    show_cuda_info()
    printF(name=MasterName.get_master_name(),
           info="[Parameters] performance_selection = {}".format(performance_selection)).printf()
    printF(name=MasterName.get_master_name(), info="[Parameters] loras = {}".format(loras)).printf()

    if loras == [[]]:
        return

    for k in loras:
        for enable, name, weight in loras:
            if "Not Exist" in name:
                name = "None"
                enable = False
                print(f'{name} is NOT Exist!, changed to None')

            if name in ('None', 'none', 'NONE', '') or name is None:
                continue

            if os.path.exists(name):
                filename = name
            else:
                filename = os.path.join(modules.config.lorafile_path[0], name)

            printF(name=MasterName.get_master_name(), info="[Parameters] filename = {}".format(filename)).printf()
            assert os.path.exists(filename), 'Lora file not found!'

    if performance_selection is not None and constants.TYPE_LIGHTNING in performance_selection.value.lower() and 1 == 2:
        printF(name=MasterName.get_master_name(), info="[Warning] Lightning model, no need to load here!").printf()
        return

    new_loras = []
    flag = False
    for ll in loras:
        if ll[1] == 'None':
            ll = base_model_additional_loras
            flag = True
            break
    if flag:
        new_loras = copy.deepcopy(loras) + copy.deepcopy(base_model_additional_loras)
    else:
        new_loras = copy.deepcopy(loras)
    printF(name=MasterName.get_master_name(), info="[Parameters] new_loras = {}".format(new_loras)).printf()

    if len(new_loras) > 5:
        model_base.refresh_loras(new_loras)
    else:
        model_base.refresh_loras(loras)

    model_refiner.refresh_loras(loras)

    final_loras = new_loras

    return


@torch.no_grad()
@torch.inference_mode()
def clip_encode_single(clip, text, verbose=False):
    # 一次性诊断（只打第一次）
    if not getattr(clip, "_diag_done", False):
        clip._diag_done = True
        tk = getattr(clip, "tokenizer", None)
        printF(name=MasterName.get_master_name(),
               info=f"[DIAG] final_clip={type(clip).__name__} "
                    f"has tokenize={hasattr(clip,'tokenize')} "
                    f"has encode_from_tokens={hasattr(clip,'encode_from_tokens')} "
                    f"tokenizer={type(tk).__name__ if tk is not None else None} "
                    f"tokenizer.callable={callable(tk)} "
                    f"cond_stage_model={type(getattr(clip,'cond_stage_model',None)).__name__}").printf()
        
    cached = clip.fcs_cond_cache.get(text, None)
    if cached is not None:
        if verbose:
            printF(name=MasterName.get_master_name(), info="[CLIP Cached] = {}".format(text)).printf()
        return cached

    def tokenize_and_encode(clip, text, return_pooled=True):
        if isinstance(clip.tokenizer.clip_l, ChatGLMTokenizer):
            tokens = clip.tokenizer.clip_l.tokenize(text)
        elif callable(getattr(clip, "tokenize", None)):
            tokens = clip.tokenize(text)
        else:
            # Forge 的 CLIP 壳（经 forge_loader 转出来）没有 tokenize 便捷方法，
            # 只有 tokenizer 容器，这里直接透传一层。
            tk = getattr(clip, "tokenizer", None)
            if tk is None:
                raise AttributeError(
                    f"{type(clip).__name__} 既没有 tokenize 也没有 tokenizer，无法分词。"
                )
            if callable(tk):
                tokens = tk(text)
            elif callable(getattr(tk, "tokenize", None)):
                tokens = tk.tokenize(text)
            else:
                raise AttributeError(
                    f"tokenizer 类型 {type(tk).__name__} 不是可调用对象，也没有 .tokenize()"
                )

        if isinstance(clip.cond_stage_model, JointTextEncoder):
            result = clip.text_processing_engine.encode_with_transformers(tokens)
        else:
            result = clip.encode_from_tokens(tokens, return_pooled=return_pooled, return_dict=True)

        return result

    # 优化后的代码逻辑
    if isinstance(clip.cond_stage_model, HyditModel):
        tokens = clip.tokenizer(text) if not callable(getattr(clip, "tokenize", None)) else clip.tokenize(text)

        # ===== 探针：token 与 encode 输出的实际 shape =====
        def _shp(v):
            if isinstance(v, torch.Tensor):
                return "Tensor{}".format(tuple(v.shape))
            if isinstance(v, (list, tuple)):
                return "list[{}]".format(len(v))
            return type(v).__name__
        _tok = {k: _shp(v) for k, v in tokens.items()} if isinstance(tokens, dict) else _shp(tokens)
        printF(name=MasterName.get_master_name(),
               info="[PROBE-TOK] text={!r} tokens={}".format(text[:30], _tok)).printf()

        result = clip.encode_from_tokens(tokens, return_pooled=False, return_dict=True)

        printF(name=MasterName.get_master_name(),
               info="[PROBE-ENC] text={!r} cond={} mask={} mt5={} mt5_mask={}".format(
                   text[:30],
                   _shp(result.get("cond")),
                   _shp(result.get("attention_mask")),
                   _shp(result.get("conditioning_mt5xl")),
                   _shp(result.get("attention_mask_mt5xl")))).printf()
        # =====
    else:
        result = tokenize_and_encode(clip, text, return_pooled=True)

    clip.fcs_cond_cache[text] = result
    if verbose:
        printF(name=MasterName.get_master_name(), info="[CLIP Encoded] = {}".format(text)).printf()
    return result


@torch.no_grad()
@torch.inference_mode()
def clone_cond(conds):
    """
    深拷贝条件，兼容不同模型家族：
      - SDXL / SDXLRefiner：extra = {"pooled_output": tensor}
      - HunyuanDiT：extra = {"attention_mask": ..., "conditioning_mt5xl": ...,
                              "attention_mask_mt5xl": ...}（无 pooled_output）
      - Flux / ZImage / QwenImage：extra 里可能有 vector / guidance 等
    因此不能写死 p["pooled_output"]，必须逐 key 克隆。
    """
    results = []

    for c, p in conds:
        if isinstance(c, torch.Tensor):
            c = c.clone()

        # p 可能是 dict（正常 extra 容器），也可能是旧的单 tensor
        if isinstance(p, dict):
            new_p = {
                k: (v.clone() if isinstance(v, torch.Tensor) else v)
                for k, v in p.items()
            }
        elif isinstance(p, torch.Tensor):
            # 向后兼容：老的单 tensor 形式 = pooled_output
            new_p = {"pooled_output": p.clone()}
        else:
            # 兜底（例如 pooled_output 是 int 哨兵 0 的历史路径）
            new_p = {"pooled_output": p}

        results.append([c, new_p])

    return results


schedule_parser = lark.Lark(r"""
!start: (prompt | /[][():]/+)*
prompt: (emphasized | scheduled | alternate | plain | WHITESPACE)*
!emphasized: "(" prompt ")"
        | "(" prompt ":" prompt ")"
        | "[" prompt "]"
scheduled: "[" [prompt ":"] prompt ":" [WHITESPACE] NUMBER [WHITESPACE] "]"
alternate: "[" prompt ("|" [prompt])+ "]"
WHITESPACE: /\s+/
plain: /([^\\\[\]():|]|\\.)+/
%import common.SIGNED_NUMBER -> NUMBER
""")


def get_learned_conditioning_prompt_schedules(prompts, base_steps):
    int_offset = 0
    flt_offset = 0
    steps = base_steps

    def collect_steps(steps, tree):
        res = [steps]

        class CollectSteps(lark.Visitor):
            def scheduled(self, tree):
                s = tree.children[-2]
                v = float(s)
                v = v * steps if v < 1 else v
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            def alternate(self, tree):
                res.extend(range(1, steps + 1))

        CollectSteps().visit(tree)
        return sorted(set(res))

    def at_step(step, tree):
        class AtStep(lark.Transformer):
            def scheduled(self, args):
                before, after, _, when, _ = args
                yield before or () if step <= when else after

            def alternate(self, args):
                args = ["" if not arg else arg for arg in args]
                yield args[(step - 1) % len(args)]

            def start(self, args):
                def flatten(x):
                    if isinstance(x, str):
                        yield x
                    else:
                        for gen in x:
                            yield from flatten(gen)

                return ''.join(flatten(args))

            def plain(self, args):
                yield args[0].value

            def __default__(self, data, children, meta):
                for child in children:
                    yield child

        return AtStep().transform(tree)

    def get_schedule(prompt):
        try:
            tree = schedule_parser.parse(prompt)
        except lark.exceptions.LarkError:
            if 0:
                import traceback
                traceback.print_exc()
            return [[steps, prompt]]
        return [[t, at_step(t, tree)] for t in collect_steps(steps, tree)]

    promptdict = {prompt: get_schedule(prompt) for prompt in set(prompts)}
    return [promptdict[prompt] for prompt in prompts]


class SdConditioning(list):
    """
    A list with prompts for stable diffusion's conditioner model.
    Forge 引擎（Flux / Flux2 / QwenImage / ZImage）在
    get_learned_conditioning 中会访问 is_negative_prompt 与
    distilled_cfg_scale，因此这里必须提供这两个属性。
    """

    def __init__(self, prompts, is_negative_prompt=False, width=None, height=None, copy_from=None):
        super().__init__()
        self.extend(prompts)

        self.is_negative_prompt = is_negative_prompt
        self.width = width
        self.height = height
        self.distilled_cfg_scale = None

        if copy_from is None:
            copy_from = prompts


def _encode_with_engine(engine, texts, is_negative_prompt=False):
    """调用 Forge 引擎的 get_learned_conditioning。

    注意：不同引擎签名不同（Flux 无 skip_flag；Kolors/SD15 有 skip_flag），
    这里统一只传 SdConditioning，避免 TypeError。
    """
    cond = SdConditioning(texts, is_negative_prompt=is_negative_prompt)
    return engine.get_learned_conditioning(cond)


def _looks_like_prompt_expansion(texts):
    """粗判：Fooocus V2 扩写通常是「原文 + 追加形容词/画风」，
    所以扩展文本通常以原文开头。多行 prompt 则互不为前缀。
    """
    if len(texts) != 2:
        return False
    if not all(isinstance(t, str) for t in texts):
        return False
    a, b = texts[0].strip().lower(), texts[1].strip().lower()
    if not a or not b:
        return False
    return b.startswith(a) and len(b) > len(a)


@torch.no_grad()
@torch.inference_mode()
def clip_encode(texts, pool_top_k=1, steps=30):
    global final_clip, final_model_ori

    # ===== 探针：确认每次调用 clip_encode 收到的 texts =====
    printF(name=MasterName.get_master_name(),
           info="[CLIP-INPUT] texts={}".format(
               [t[:40] if isinstance(t, str) else type(t).__name__ for t in texts]
           )).printf()
    # =====
    
    if final_clip is None:
        return None
    if not isinstance(texts, list):
        return None
    if len(texts) == 0:
        return None

    if isinstance(final_model_ori, ForgeDiffusionEngine):
        result_ces = _encode_with_engine(final_model_ori, texts)

        # 一次性诊断：引擎到底吐了哪些 key（value 是 list 还是 tensor）
        if isinstance(result_ces, dict) and not getattr(final_model_ori, "_cond_diag_done", False):
            final_model_ori._cond_diag_done = True
            try:
                desc = ", ".join(
                    "{}={}{}".format(
                        k, type(v).__name__,
                        "[{}]".format(len(v)) if isinstance(v, (list, tuple)) else "",
                    )
                    for k, v in result_ces.items()
                )
                printF(name=MasterName.get_master_name(),
                       info="[DIAG] engine cond keys: {}".format(desc)).printf()
            except Exception:
                pass

        def _ensure_batched(x):
            if isinstance(x, torch.Tensor) and x.dim() == 2:
                return x.unsqueeze(0)
            return x

        def _ensure_1d_batched(x):
            if isinstance(x, torch.Tensor) and x.dim() == 1:
                return x.unsqueeze(0)
            return x

        def _unwrap_list(v, n):
            if isinstance(v, (list, tuple)):
                return list(v)
            if v is None:
                return [None] * n
            # 引擎可能把 vector 作为「batch=n 的整块 tensor」返回，
            # 而 crossattn 是 list[n]。此时若直接 [v]*n，每条 cond 的 y
            # 都会是整块 [n, dim]，batch 与 img(batch=1) 不一致，崩：
            #   output with shape [1, 4032, 3072] doesn't match [3, ...]
            if isinstance(v, torch.Tensor) and v.dim() >= 2 and v.shape[0] == n and n > 1:
                return [v[i:i + 1] for i in range(n)]
            return [v] * n

        def _vec_dtype_device(ref_ct):
            # 关键：
            #   crossattn / context 会在上层 apply_model 里被搬到 cuda，
            #   但 pooled_output(y) / guidance 不会——它们一路原样传到
            #   flux.forward_orig，最后 vector_in.linear(y, weight, bias)。
            #   所以 y 的 device 必须显式对齐模型权重，而不是 ref_ct。
            dtype = None
            device = None
            try:
                w = final_model_ori.forge_objects.unet.model.diffusion_model.vector_in.weight
                dtype = w.dtype
                device = w.device
            except Exception:
                pass

            if dtype is None:
                dtype = ref_ct.dtype if isinstance(ref_ct, torch.Tensor) else torch.float32

            # 权重不在 cuda（比如 offload 到 CPU）时，也不要让 y 停留在 CPU——
            # 采样时模型会被 load_models_gpu 拉回 cuda，用 get_torch_device 兜底最稳。
            if device is None or device.type != "cuda":
                try:
                    device = ldm_patched.modules.model_management.get_torch_device()
                except Exception:
                    if isinstance(ref_ct, torch.Tensor):
                        device = ref_ct.device
                    else:
                        device = torch.device("cpu")
            return dtype, device

        def _vec_in_dim():
            try:
                d = final_model_ori.forge_objects.unet.model.diffusion_model.vec_in_dim
                if isinstance(d, int) and d > 0:
                    return d
            except Exception:
                pass
            return 768

        def _fallback_y(ref_ct):
            dtype, device = _vec_dtype_device(ref_ct)
            return torch.zeros((1, _vec_in_dim()), dtype=dtype, device=device)

        if isinstance(result_ces, dict):
            crossattn = result_ces.get("crossattn")
            pooled = result_ces.get("vector", None)
                            
            if pooled is None:
                if isinstance(crossattn, torch.Tensor) and crossattn.dim() == 4:
                    # 对 Krea2 的 4D 条件 (B, seq, layers, dim) 进行池化作为 y
                    # 使用 mean 池化生成 (B, dim) 的全局向量
                    pooled = crossattn.mean(dim=2).mean(dim=1, keepdim=True)  # (B, 1, dim)    
                else:
                    pooled = result_ces.get("pooled_output", None)
                
                
            guidance = result_ces.get("guidance", None)

            if isinstance(crossattn, (list, tuple)):
                crossattn_list = list(crossattn)
            elif isinstance(crossattn, torch.Tensor) and crossattn.dim() >= 3 \
                    and crossattn.shape[0] == len(texts) and len(texts) > 1:
                # 引擎把多 prompt 的 crossattn 直接 batch 返回（而不是 list）
                # 这里按 batch 拆成 list[1, seq, dim]，让 n 正确识别成多 prompt。
                crossattn_list = [crossattn[i:i + 1] for i in range(crossattn.shape[0])]
            elif crossattn is not None:
                crossattn_list = [crossattn]
            else:
                crossattn_list = []

            ct_list = []
            for ct in crossattn_list:
                ct = _ensure_batched(ct)
                if isinstance(ct, torch.Tensor):
                    ct_list.append(ct)
            if not ct_list:
                return None

            n = len(ct_list)
            pooled_list = _unwrap_list(pooled, n)
            if len(pooled_list) < n:
                pooled_list += [None] * (n - len(pooled_list))

            def _get_y(i, ref_ct):
                p = pooled_list[i] if i < len(pooled_list) else None
                p = _ensure_1d_batched(p)
                if isinstance(p, torch.Tensor):
                    dtype, device = _vec_dtype_device(ref_ct)
                    p = p.to(device=device, dtype=dtype)
                    # 逐条采样时 img 的 batch 恒为 1，强制 y 的 batch=1：
                    # 即便 _unwrap_list 拿到的仍是整块 tensor，也不会让
                    # modulation 广播到 [n, 4032, 3072]。
                    if p.dim() >= 2 and p.shape[0] != 1:
                        if p.shape[0] == n and i < p.shape[0]:
                            p = p[i:i + 1]
                        else:
                            p = p[:1]
                    return p
                return _fallback_y(ref_ct)

            def _make_extra(i, ref_ct):
                # y 永远是 Tensor，且 device/dtype 对齐 vector_in 权重。
                y = _get_y(i, ref_ct)
                extra = {"pooled_output": y, "vector": y, "y": y}

                if guidance is not None:
                    g = guidance[i] if isinstance(guidance, (list, tuple)) and i < len(guidance) else guidance
                    if isinstance(g, torch.Tensor):
                        dtype, device = _vec_dtype_device(ref_ct)
                        g = g.to(device=device, dtype=dtype)
                        # guidance 会被加到 vec 上，同样必须 batch=1
                        if g.dim() >= 1 and g.shape[0] != 1:
                            if g.shape[0] == n and i < g.shape[0]:
                                g = g[i:i + 1]
                            else:
                                g = g[:1]
                        extra["guidance"] = g
                        
                return extra

            if n == 1:
                return [[ct_list[0], _make_extra(0, ct_list[0])]]

            # Fooocus V2 扩写：pad 后沿 seq 维 cat，单张图
            if _looks_like_prompt_expansion(texts):
                max_len = max(int(t.shape[1]) for t in ct_list)
                aligned = [
                    t if int(t.shape[1]) == max_len
                    else torch.nn.functional.pad(t, (0, 0, 0, max_len - int(t.shape[1])))
                    for t in ct_list
                ]
                cat = torch.cat(aligned, dim=1)
                return [[cat, _make_extra(0, cat)]]

            # 多行 prompt 批量出图：每条独立 cond，每条都用真实 y。
            # 「是否逐条采样」由 process_diffusion 里 len > 1 识别，无需 sentinel。
            return [[ct_list[i], _make_extra(i, ct_list[i])] for i in range(n)]

        # 引擎直接返回 list/tuple（QwenImage / ZImage 之类逐 prompt 的张量）——保持原样
        if isinstance(result_ces, (list, tuple)):
            items = []
            for t in result_ces:
                if isinstance(t, torch.Tensor):
                    if t.dim() == 2:
                        t = t.unsqueeze(0)
                    items.append(t)

            if not items:
                return None

            def _zero_y(ref_ct):
                dtype, device = _vec_dtype_device(ref_ct)
                return torch.zeros((1, _vec_in_dim()), dtype=dtype, device=device)

            if len(items) == 1:
                y0 = _zero_y(items[0])
                return [[items[0], {"pooled_output": y0, "vector": y0, "y": y0}]]

            if _looks_like_prompt_expansion(texts):
                max_len = max(int(t.shape[1]) for t in items)
                aligned = [
                    t if int(t.shape[1]) == max_len
                    else torch.nn.functional.pad(t, (0, 0, 0, max_len - int(t.shape[1])))
                    for t in items
                ]
                cat = torch.cat(aligned, dim=1)
                yc = _zero_y(cat)
                return [[cat, {"pooled_output": yc, "vector": yc, "y": yc}]]

            # 保持与原语义一致：返回 list 会把 [单层 [tensor, extra]] 展开为多层，
            # 外层是 num 条 cond；每条都用 zero-y。
            out = []
            for t in items:
                yt = _zero_y(t)
                out.append([t, {"pooled_output": yt, "vector": yt, "y": yt}])
            return out

    # ===== legacy 路径 =====
    cond_list = []
    pooled_acc = None
    per_prompt_meta = []   # 逐条保留 Hydit 的配套流（mask / mt5 / mt5_mask）
    any_tuple = False

    for i, text in enumerate(texts):
        result_ces = clip_encode_single(final_clip, text, verbose=False)
        if isinstance(result_ces, dict):
            cond = result_ces.get("cond")
            pooled = result_ces.get("pooled_output", None)

            per_prompt_meta.append({
                "attention_mask": result_ces.get("attention_mask"),
                "conditioning_mt5xl": result_ces.get("conditioning_mt5xl"),
                "attention_mask_mt5xl": result_ces.get("attention_mask_mt5xl"),
            })

            if isinstance(final_clip.cond_stage_model if final_clip else None, HyditModel) \
                    and not getattr(clip_encode, "_hydit_diag_done", False):
                clip_encode._hydit_diag_done = True

                def _fmt(v):
                    if v is None:
                        return "None"
                    if isinstance(v, torch.Tensor):
                        return f"Tensor{tuple(v.shape)}@{v.dtype}"
                    return f"{type(v).__name__}({v})"

                printF(name=MasterName.get_master_name(),
                       info="[DIAG] HunyuanDiT conds: cond={}, pooled={}, att_mask={}, "
                            "conditioning_mt5xl={}, attention_mask_mt5xl={}".format(
                                _fmt(cond), _fmt(pooled),
                                _fmt(per_prompt_meta[-1].get("attention_mask")),
                                _fmt(per_prompt_meta[-1].get("conditioning_mt5xl")),
                                _fmt(per_prompt_meta[-1].get("attention_mask_mt5xl")))).printf()
        else:
            cond, pooled = result_ces
            any_tuple = True
            per_prompt_meta.append({})

        cond_list.append(cond)
        if i < pool_top_k and isinstance(pooled, torch.Tensor):
            pooled_acc = pooled if pooled_acc is None else (pooled_acc + pooled)

    # ===== HunyuanDiT：只使用主 prompt，不做 Fooocus 扩写拼接 =====
    # Fooocus V2 的扩写多 variant 是为 SDXL 设计的（SDXL 可以吃掉 N×77 token 的 cond，
    # 因为没有配对的 attention_mask / mT5）。
    # HunyuanDiT 架构是 CLIP-L(77) 与 mT5(256) 严格配对的双流；
    # 拼接 N 条 cond 到 N×77 但 mask 仍是 77、mT5 仍是 256，就会让 cross-attn mask
    # 广播错位，输出纯噪声。因此 Hydit 直接走主 prompt（index 0 = 用户真实输入）即可。
    if isinstance(final_clip.cond_stage_model if final_clip else None, HyditModel):
        meta0 = per_prompt_meta[0] if per_prompt_meta else {}
        extra = {}
        if meta0.get("attention_mask") is not None:
            extra["attention_mask"] = meta0["attention_mask"]
        if meta0.get("conditioning_mt5xl") is not None:
            extra["conditioning_mt5xl"] = meta0["conditioning_mt5xl"]
        if meta0.get("attention_mask_mt5xl") is not None:
            extra["attention_mask_mt5xl"] = meta0["attention_mask_mt5xl"]
        return [[cond_list[0], extra]]

    # ===== SDXL / SD15 老路径 =====
    extra = {}
    if pooled_acc is not None:
        extra["pooled_output"] = pooled_acc
    if per_prompt_meta and per_prompt_meta[0].get("attention_mask") is not None:
        extra["attention_mask"] = per_prompt_meta[0]["attention_mask"]

    return [[torch.cat(cond_list, dim=1), extra]]


# @torch.no_grad()
# @torch.inference_mode()
# def set_clip_skip(clip_skip: int):
#     global final_clip
#
#     if final_clip is None:
#         return
#
#     final_clip.clip_layer(-abs(clip_skip))
#     return

@torch.no_grad()
@torch.inference_mode()
def clear_all_caches():
    if final_clip is not None:
        final_clip.fcs_cond_cache = {}


@torch.no_grad()
@torch.inference_mode()
def prepare_text_encoder(async_call=True):
    from backend.patcher.base import ModelPatcher as BackendModelPatcher
    if async_call:
        pass
    assert_model_integrity()
    if final_clip is None:
        printF(name=MasterName.get_master_name(), info="[final_clip] = {}".format(final_clip)).printf()
        return

    clip_patcher = getattr(final_clip, "patcher", None)
    if clip_patcher is None:
        return

    # 复用模块级 helper（去掉原来的局部定义）
    forge_clip_patcher = clip_patcher if _is_backend_model_patcher(clip_patcher) else None
    if forge_clip_patcher is None:
        inner = getattr(clip_patcher, "model", None)
        if _is_backend_model_patcher(inner):
            forge_clip_patcher = inner

    is_forge_engine = (
        isinstance(final_model_ori, ForgeDiffusionEngine)
        or hasattr(final_model_ori, "forge_objects")
        or forge_clip_patcher is not None
    )

    if is_forge_engine and forge_clip_patcher is not None:
        try:
            # 只把 CLIP 交给 Forge 内存管理器；expansion 由
            # _preload_expansion_legacy 自己按 patcher 类型路由。
            backend.memory_management.load_models_gpu([forge_clip_patcher])
            _preload_expansion_legacy()
            return
        except (AttributeError, TypeError, RuntimeError) as e:
            printF(name=MasterName.get_master_name(),
                   info="[Warning] Forge text-encoder load failed ({}); falling back to legacy".format(e)).printf()

    try:
        ldm_patched.modules.model_management.load_models_gpu([clip_patcher])
    except (AttributeError, TypeError, RuntimeError) as e:
        printF(name=MasterName.get_master_name(),
               info="[Warning] text-encoder not preloaded: {}".format(e)).printf()

    _preload_expansion_legacy()
    return

# ===== 模块级 patcher 类型判定 =====
# prepare_text_encoder / _preload_expansion_legacy 共用。
def _is_backend_model_patcher(p):
    """判定 p 是新 backend ModelPatcher 还是 legacy ModelPatcher。

    backend（backend.patcher.base.ModelPatcher）：
        - 特征属性: model_patches_models
        - 该走: backend.memory_management.load_models_gpu
        - 传给 legacy memmgr 会抛:
              'ModelPatcher' object has no attribute 'current_loaded_device'

    legacy（ldm_patched.modules.model_patcher.ModelPatcher）：
        - 特征方法: current_loaded_device()
        - 该走: ldm_patched.modules.model_management.load_models_gpu
    """
    if p is None:
        return False
    try:
        from backend.patcher.base import ModelPatcher as _BMP
        if isinstance(p, _BMP):
            return True
    except Exception:
        # 同一模块被不同 sys.path 导入时 isinstance 会失效，
        # 用 model_patches_models 做鸭子类型兜底
        pass
    return hasattr(p, "model_patches_models")


def _preload_expansion_legacy():
    """Fooocus V2 扩写模型（transformers GPT2）预加载。

    修订：原实现无条件走 legacy 管理器：
        ldm_patched.modules.model_management.load_models_gpu([expansion_patcher])
    但 FooocusExpansion(flag=True) 在 flag_new_engine 分支下构造的是
    backend ModelPatcher（没有 current_loaded_device），于是抛：
        'ModelPatcher' object has no attribute 'current_loaded_device'
    异常被 except 吞掉 —— 预加载从未成功，只是打了一条 warning。
    （首次生成时靠懒加载兜底，多等 2~5s。）

    正确做法：按 patcher 类型选 memmgr。
    """
    if final_expansion is None:
        return
    expansion_patcher = getattr(final_expansion, "patcher", None)
    if expansion_patcher is None:
        return

    if _is_backend_model_patcher(expansion_patcher):
        try:
            backend.memory_management.load_models_gpu([expansion_patcher])
            printF(name=MasterName.get_master_name(),
                   info="[Expansion] preloaded via backend memmgr").printf()
            return
        except (AttributeError, TypeError, RuntimeError) as e:
            printF(name=MasterName.get_master_name(),
                   info="[Warning] expansion preload (backend) failed: {} "
                        "—— 走懒加载，不影响出图".format(e)).printf()
            return

    # legacy 路径
    try:
        ldm_patched.modules.model_management.load_models_gpu([expansion_patcher])
        printF(name=MasterName.get_master_name(),
               info="[Expansion] preloaded via legacy memmgr").printf()
    except (AttributeError, TypeError, RuntimeError) as e:
        printF(name=MasterName.get_master_name(),
               info="[Warning] expansion preload (legacy) failed: {} "
                    "—— 走懒加载，不影响出图".format(e)).printf()


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(base_model_name, loras,
                       base_model_additional_loras=None, use_synthetic_refiner=False, performance_selection=None):
    global final_unet, final_clip, final_vae, final_expansion, final_loras, final_model_ori

    final_unet = None
    final_clip = None
    final_vae = None
    final_model_ori = None

    final_loras = loras

    if use_synthetic_refiner == 'None':
        printF(name=MasterName.get_master_name(), info="[Warning] Synthetic Refiner Activated").printf()
        refresh_base_model(base_model_name, performance_selection)
        synthesize_refiner_model()
    else:
        refresh_base_model(base_model_name, performance_selection)

    refresh_loras(loras, base_model_additional_loras=base_model_additional_loras,
                  performance_selection=performance_selection)
    printF(name=MasterName.get_master_name(), info="[Function] Enter-> assert_model_integrity").printf()
    assert_model_integrity()

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_model_ori = model_base.model_ori
    final_vae = model_base.vae

    flag_new_engine = False
    if isinstance(final_unet, backend.patcher.unet.UnetPatcher):
        flag_new_engine = True

    if final_expansion is None:
        final_expansion = FooocusExpansion(flag=flag_new_engine)

    prepare_text_encoder(async_call=True)
    clear_all_caches()
    
    # 👇 一次性 VAE 往返测试（默认关闭，设环境变量为 1 时启用）
    import os as _os
    if _os.environ.get("MEANVON_VAE_RT", "0") == "1":
        try:
            import torch as _t
            _vae = model_base.vae
            _fsm = getattr(_vae, "first_stage_model", None)
            _dev = ldm_patched.modules.model_management.get_torch_device()
            # 三色条：R/G/B 各 1/3
            _img = _t.zeros(1, 3, 512, 512, dtype=_t.float32, device=_dev)
            _img[:, 0, :, :170] = 1.0
            _img[:, 1, :, 170:340] = 1.0
            _img[:, 2, :, 340:] = 1.0
            # VAE 可能要用 (B, C, T, H, W) 或 (B, C, H, W)，两种都试
            with _t.no_grad():
                try:
                    _lat = _vae.encode(_img)
                    _out = _vae.decode(_lat)
                except Exception:
                    _img5 = _img.unsqueeze(2)  # (1,3,1,512,512)
                    _lat = _vae.encode(_img5)
                    _out = _vae.decode(_lat)
            _o = _out if isinstance(_out, _t.Tensor) else _out.samples
            if _o.dim() == 5:  # (B,C,T,H,W) 或 (B,T,H,W,C)
                _o = _o.squeeze(2) if _o.shape[2] == 1 else _o
            if _o.dim() == 4 and _o.shape[1] == 3:
                for _c, _nm in enumerate(("R(应为1)", "G(应为1)", "B(应为1)")):
                    printF(name=MasterName.get_master_name(),
                           info="[VAE-RT] {} ch_mean={:+.4f} min={:+.4f} max={:+.4f}".format(
                               _nm, float(_o[:, _c].mean()),
                               float(_o[:, _c].min()), float(_o[:, _c].max()))).printf()
            printF(name=MasterName.get_master_name(),
                   info="[VAE-RT] output shape={} range=[{:.3f},{:.3f}]".format(
                       tuple(_o.shape), float(_o.min()), float(_o.max()))).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[VAE-RT] failed: {}".format(_e)).printf()

    return

@torch.no_grad()
@torch.inference_mode()
def vae_parse(latent):
    if final_refiner_vae is None:
        return latent

    result = vae_interpose.parse(latent["samples"])
    return {'samples': result}


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnet_canny(name=None):
    global controlnet_canny, controlnet_canny_hash
    if controlnet_canny_hash == str(controlnet_canny):
        return

    model_name = modules.config.default_controlnet_canny_name if name == None else name
    filename = os.path.join(modules.config.controlnet_path, model_name)
    controlnet_canny = core.load_controlnet(filename)

    controlnet_canny_hash = model_name
    printF(name=MasterName.get_master_name(),
           info="[Parameters] ControlNet model loaded: {}".format(controlnet_canny_hash)).printf()

    return


@torch.no_grad()
@torch.inference_mode()
def refresh_controlnet_depth(name=None):
    global controlnet_depth, controlnet_depth_hash
    if controlnet_depth_hash == str(controlnet_depth):
        return

    model_name = modules.config.default_controlnet_depth_name if name == None else name
    filename = os.path.join(modules.config.controlnet_path, model_name)
    controlnet_depth = core.load_controlnet(filename)

    controlnet_depth_hash = model_name
    printF(name=MasterName.get_master_name(),
           info="[Parameters] ControlNet model loaded: {}".format(controlnet_canny_hash)).printf()

    return


@torch.no_grad()
@torch.inference_mode()
def refresh_clip_vision():
    global clip_vision, clip_vision_hash
    if clip_vision_hash == str(clip_vision):
        return

    model_name = modules.config.default_clip_vision_name
    filename = os.path.join(modules.config.clip_vision_path, model_name)
    clip_vision = core.load_clip_vision(filename)

    clip_vision_hash = model_name
    printF(name=MasterName.get_master_name(),
           info="[Parameters] CLIP Vision model loaded: {}".format(clip_vision_hash)).printf()

    return


@torch.no_grad()
@torch.inference_mode()
def set_clip_skips(base_clip_skip):
    global model_base, model_refiner

    model_base.set_clip_skip(base_clip_skip)
    return


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas_all(sampler, model, scheduler, steps):
    from ldm_patched.modules.samplers import calculate_sigmas_scheduler

    discard_penultimate_sigma = False
    if sampler in ['dpm_2', 'dpm_2_ancestral']:
        steps += 1
        discard_penultimate_sigma = True

    sigmas = calculate_sigmas_scheduler(model, scheduler, steps)

    if discard_penultimate_sigma:
        sigmas = torch.cat([sigmas[:-2], sigmas[-1:]])
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas(sampler, model, scheduler, steps, denoise):
    if denoise is None or denoise > 0.9999:
        sigmas = calculate_sigmas_all(sampler, model, scheduler, steps)
    else:
        new_steps = int(steps / denoise)
        sigmas = calculate_sigmas_all(sampler, model, scheduler, new_steps)
        sigmas = sigmas[-(steps + 1):]
    return sigmas


@torch.no_grad()
@torch.inference_mode()
def get_candidate_vae(steps, switch, denoise=1.0, refiner_swap_method='joint'):
    assert refiner_swap_method in ['joint', 'separate', 'vae']

    if final_refiner_vae is not None and final_refiner_unet is not None:
        if denoise > 0.9:
            return final_vae, final_refiner_vae
        else:
            if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                return final_vae, None
            else:
                return final_refiner_vae, None

    return final_vae, final_refiner_vae


@torch.no_grad()
@torch.inference_mode()
def process_diffusion(positive_cond, negative_cond, steps, switch, width, height, image_seed, callback, sampler_name,
                      scheduler_name, img2img, input_image, start_step, control_lora_canny, canny_edge_low,
                      canny_edge_high, canny_start, canny_stop, canny_strength,
                      control_lora_depth, depth_start, depth_stop, depth_strength,
                      latent=None, denoise=1.0, tiled=False, cfg_scale=7.0, refiner_swap_method='joint',
                      disable_preview=False):
    # === 多 prompt 批量出图：逐条递归采样 ===
    if (
        isinstance(positive_cond, (list, tuple))
        and len(positive_cond) > 1
        and all(
            isinstance(c, (list, tuple)) and len(c) == 2 and isinstance(c[0], torch.Tensor)
            for c in positive_cond
        )
    ):
        n = len(positive_cond)
        if isinstance(negative_cond, (list, tuple)) and len(negative_cond) == n:
            neg_list = [[c] for c in negative_cond]
        else:
            neg_list = [negative_cond] * n

        all_images = []
        for i, pc in enumerate(positive_cond):
            images_i = process_diffusion(
                positive_cond=[pc],
                negative_cond=neg_list[i],
                steps=steps, switch=switch, width=width, height=height,
                image_seed=image_seed + i,
                callback=callback, sampler_name=sampler_name,
                scheduler_name=scheduler_name, img2img=img2img,
                input_image=input_image, start_step=start_step,
                control_lora_canny=control_lora_canny, canny_edge_low=canny_edge_low, canny_edge_high=canny_edge_high,
                canny_start=canny_start, canny_stop=canny_stop, canny_strength=canny_strength,
                control_lora_depth=control_lora_depth,
                depth_start=depth_start, depth_stop=depth_stop, depth_strength=depth_strength,
                latent=latent, denoise=denoise, tiled=tiled, cfg_scale=cfg_scale,
                refiner_swap_method=refiner_swap_method, disable_preview=disable_preview,
            )
            if images_i:
                all_images.extend(images_i)
        return all_images
    # === 多 prompt 分支结束 ===

    target_unet, target_vae, target_refiner_unet, target_refiner_vae, target_clip, target_model_ori \
        = final_unet, final_vae, final_refiner_unet, final_refiner_vae, final_clip, final_model_ori

    assert refiner_swap_method in ['joint', 'separate', 'vae']

    target_is_flow = isinstance(target_model_ori, (Flux, Flux2, Kolors, ZImage, QwenImage, Krea2))

    if latent is None:
        if target_is_flow:
            latent_channels = target_model_ori.forge_objects.vae.latent_channels
            initial_latent = {
                "samples": torch.zeros(
                    (1, latent_channels, height // 8, width // 8),
                    dtype=torch.float32,
                    device=ldm_patched.modules.model_management.get_torch_device(),
                )
            }
        else:
            initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    else:
        initial_latent = latent

    if target_is_flow:
        target_unet = target_model_ori

        alphas_cumprod_modifiers = target_unet.forge_objects.unet.model_options.get('alphas_cumprod_modifiers', [])
        alphas_cumprod_backup = None

        if len(alphas_cumprod_modifiers) > 0:
            alphas_cumprod_backup = target_unet.alphas_cumprod
            for modifier in alphas_cumprod_modifiers:
                target_unet.alphas_cumprod = modifier(target_unet.alphas_cumprod)
            target_unet.forge_objects.unet.model.model_sampling.set_sigmas(
                ((1 - target_unet.alphas_cumprod) / target_unet.alphas_cumprod) ** 0.5)
    else:
        if final_refiner_vae is not None and final_refiner_unet is not None:
            if denoise > 0.9:
                refiner_swap_method = 'vae'
            else:
                refiner_swap_method = 'joint'
                if denoise > (float(steps - switch) / float(steps)) ** 0.834:  # karras 0.834
                    target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                        = final_unet, final_vae, None, None
                    printF(name=MasterName.get_master_name(),
                           info="[Sampler] only use Base because of partial denoise.").printf()
                else:
                    positive_cond = clip_separate(positive_cond, target_model=final_refiner_unet.model,
                                                  target_clip=final_clip)
                    negative_cond = clip_separate(negative_cond, target_model=final_refiner_unet.model,
                                                  target_clip=final_clip)
                    target_unet, target_vae, target_refiner_unet, target_refiner_vae \
                        = final_refiner_unet, final_refiner_vae, None, None
                    printF(name=MasterName.get_master_name(),
                           info="[Sampler] only use Refiner because of partial denoise.").printf()

    printF(name=MasterName.get_master_name(),
           info="[Sampler] refiner_swap_method = {}".format(refiner_swap_method)).printf()

    minmax_sigmas = calculate_sigmas(sampler=sampler_name, scheduler=scheduler_name, model=final_unet.model,
                                     steps=steps, denoise=denoise)
    sigma_min, sigma_max = minmax_sigmas[minmax_sigmas > 0].min(), minmax_sigmas.max()
    sigma_min = float(sigma_min.cpu().numpy())
    sigma_max = float(sigma_max.cpu().numpy())
    printF(name=MasterName.get_master_name(),
           info="[Sampler] sigma_min = {}, sigma_max = {}".format(sigma_min, sigma_max)).printf()

    try:
        modules.patch.BrownianTreeNoiseSamplerPatched.global_sampler = None
    except Exception:
        pass

    modules.patch.BrownianTreeNoiseSamplerPatched.global_init(
        initial_latent['samples'].to(ldm_patched.modules.model_management.get_torch_device()),
        sigma_min, sigma_max, seed=image_seed, cpu=False)

    decoded_latent = None

    if control_lora_canny and input_image is None:
        edges_image = core.detect_edge(input_image, canny_edge_low, canny_edge_high)
        positive_cond, negative_cond = core.apply_controlnet(positive_cond, negative_cond,
                                                             controlnet_canny, edges_image, canny_strength, canny_start,
                                                             canny_stop)

    if control_lora_depth and input_image is None:
        positive_cond, negative_cond = core.apply_controlnet(positive_cond, negative_cond,
                                                             controlnet_depth, input_image, depth_strength, depth_start,
                                                             depth_stop)

    # =================================================================
    # 诊断工具（临时）
    # 关键修复：
    #   ① 所有引用改为 target_vae / target_unet（不再用未定义的 vae / self）
    #   ② 统一由 _diag_pre_decode / _diag_post_decode 两个 helper 承担
    #   ③ [FIX/PRE-DECODE] 真正跑到 decode 之前
    #   ④ 对 joint / separate / vae 三条路径都调 _diag_post_decode
    # =================================================================
    def _diag_pre_decode(sampled, unet, vae_used):
        try:
            _s = sampled["samples"] if isinstance(sampled, dict) else sampled
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[FIX/PRE-DECODE] fatal extract: {}".format(_e)).printf()
            return

        # ---- 1) latent 数值 + 5D 结构 ----
        try:
            _info = "shape={} dtype={} std={:.4f} min={:.4f} max={:.4f}".format(
                tuple(_s.shape), _s.dtype, float(_s.std()),
                float(_s.min()), float(_s.max()))
            if _s.ndim == 5:
                _t0, _t1 = _s[:, :, 0], _s[:, :, -1]
                _info += " | 5D B={} C={} T={} | T=0 vs T=-1 identical={}".format(
                    _s.shape[0], _s.shape[1], _s.shape[2],
                    bool(torch.allclose(_t0, _t1)))
            printF(name=MasterName.get_master_name(),
                   info="[FIX/PRE-DECODE] {}".format(_info)).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[FIX/PRE-DECODE] stats failed: {}".format(_e)).printf()

        try:
            printF(name=MasterName.get_master_name(),
                   info="[DIAG-PIPE] after ksampler: samples std={:.4f}".format(
                       float(_s.std()))).printf()
        except Exception:
            pass

        # ---- 2) unet vs vae latent_format ----
        def _resolve_lf(obj):
            if obj is None:
                return None
            for cand in (obj,
                         getattr(obj, "model", None),
                         getattr(obj, "first_stage_model", None),
                         getattr(obj, "patcher", None)):
                if cand is None:
                    continue
                lf = getattr(cand, "latent_format", None)
                if lf is not None:
                    return lf
            # ForgeDiffusionEngine: .forge_objects.{unet,vae}.model
            fo = getattr(obj, "forge_objects", None)
            if fo is not None:
                for inner in (getattr(fo, "unet", None), getattr(fo, "vae", None)):
                    if inner is None:
                        continue
                    for cand in (inner, getattr(inner, "model", None)):
                        if cand is None:
                            continue
                        lf = getattr(cand, "latent_format", None)
                        if lf is not None:
                            return lf
            return None

        try:
            _lf_m = _resolve_lf(unet)
            _lf_v = _resolve_lf(vae_used)
            printF(name=MasterName.get_master_name(),
                   info="[DIAG-DECODE] unet={} | model.lf={} scale={} || "
                        "vae={} | vae.lf={} scale={} ch={} | samples={} std={:.4f}".format(
                            type(unet).__name__,
                            type(_lf_m).__name__ if _lf_m else "None",
                            getattr(_lf_m, "scale_factor", None),
                            type(vae_used).__name__,
                            type(_lf_v).__name__ if _lf_v else "None",
                            getattr(_lf_v, "scale_factor", None),
                            getattr(vae_used, "latent_channels", None),
                            tuple(_s.shape),
                            float(_s.std()),
                        )).printf()
            if _lf_v is not None and hasattr(_lf_v, "process_out"):
                _after = _lf_v.process_out(_s)
                printF(name=MasterName.get_master_name(),
                    info="[DIAG-DECODE] after process_out: mean={:+.4f} std={:.4f} "
                            "min={:+.4f} max={:+.4f} | scale_factor={} | "
                            "has latents_mean={} latents_std={}".format(
                                float(_after.mean()), float(_after.std()),
                                float(_after.min()), float(_after.max()),
                                getattr(_lf_v, "scale_factor", None),
                                hasattr(_lf_v, "latents_mean"),
                                hasattr(_lf_v, "latents_std"))).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[DIAG-DECODE] failed: {}".format(_e)).printf()

        # ---- 3) VAE 内部指纹 ----
        try:
            _fsm = getattr(vae_used, "first_stage_model", None)
            _fk, _shape = None, "N/A"
            if _fsm is not None and hasattr(_fsm, "state_dict"):
                _sd = _fsm.state_dict()
                _fk = next(iter(_sd), None)
                if _fk is not None:
                    _shape = tuple(_sd[_fk].shape)
            printF(name=MasterName.get_master_name(),
                   info="[VAE-PROBE] fsm_class={} | first_key={} | shape={}".format(
                       type(_fsm).__name__ if _fsm is not None else "None",
                       _fk, _shape)).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[VAE-PROBE] failed: {}".format(_e)).printf()

        # ---- 4) VAE 加载路径 ----
        try:
            _path = "?"
            for _src in (vae_used,
                         getattr(vae_used, "first_stage_model", None),
                         getattr(vae_used, "patcher", None),
                         getattr(vae_used, "sd_checkpoint_info", None)):
                if _src is None:
                    continue
                for _attr in ("filename", "ckpt_name", "path", "model_path", "_filename"):
                    _val = getattr(_src, _attr, None)
                    if _val:
                        _path = _val
                        break
                if _path != "?":
                    break
            printF(name=MasterName.get_master_name(),
                   info="[VAE-LOADED] path={} | class={} | fsm_class={}".format(
                       _path,
                       type(vae_used).__name__,
                       type(getattr(vae_used, "first_stage_model", None)).__name__)).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[VAE-LOADED] failed: {}".format(_e)).printf()

    def _diag_post_decode(decoded):
        try:
            _od = decoded if isinstance(decoded, torch.Tensor) else \
                getattr(decoded, "samples", decoded)
            printF(name=MasterName.get_master_name(),
                   info="[FIX/POST-DECODE] type={} shape={} dtype={} range=[{:.3f},{:.3f}]".format(
                       type(decoded).__name__,
                       tuple(_od.shape) if hasattr(_od, "shape") else "<no shape>",
                       getattr(_od, "dtype", None),
                       float(_od.min()) if hasattr(_od, "min") else float("nan"),
                       float(_od.max()) if hasattr(_od, "max") else float("nan"))).printf()
        except Exception as _e:
            printF(name=MasterName.get_master_name(),
                   info="[FIX/POST-DECODE] failed: {}".format(_e)).printf()

    # =================================================================
    # 采样 + 解码
    # =================================================================
    if refiner_swap_method == 'joint':
        sampled_latent = core.ksampler(
            model=target_unet,
            refiner=target_refiner_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            steps=steps, start_step=0, last_step=steps, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            refiner_switch=switch,
            previewer_start=0,
            previewer_end=steps,
            disable_preview=disable_preview,
            width=width, height=height
        )
        _diag_pre_decode(sampled_latent, target_unet, target_vae)
        decoded_latent = core.decode_vae(
            vae=target_vae, latent_image=sampled_latent, tiled=tiled)
        _diag_post_decode(decoded_latent)

    if refiner_swap_method == 'separate':
        sampled_latent = core.ksampler(
            model=target_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=False,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=0,
            previewer_end=steps,
            disable_preview=disable_preview,
            width=width, height=height
        )
        printF(name=MasterName.get_master_name(),
               info="[Warning] Refiner swapped by changing ksampler. Noise preserved.").printf()

        target_model = target_refiner_unet
        if target_model is None:
            target_model = target_unet
            printF(name=MasterName.get_master_name(),
                   info="[Warning] Use base model to refine itself - this may because of developer mode.").printf()

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=target_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=target_clip),
            latent=sampled_latent,
            steps=steps, start_step=switch, last_step=steps, disable_noise=True, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=switch,
            previewer_end=steps,
            disable_preview=disable_preview,
            width=width, height=height
        )

        target_model = target_refiner_vae
        if target_model is None:
            target_model = target_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)
        _diag_post_decode(decoded_latent)

    if refiner_swap_method == 'vae':
        modules.patch.patch_settings[os.getpid()].eps_record = 'vae'

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.unswap()

        sampled_latent = core.ksampler(
            model=target_unet,
            positive=positive_cond,
            negative=negative_cond,
            latent=initial_latent,
            steps=steps, start_step=0, last_step=switch, disable_noise=False, force_full_denoise=True,
            seed=image_seed,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=0,
            previewer_end=steps,
            disable_preview=disable_preview,
            width=width, height=height
        )
        printF(name=MasterName.get_master_name(), info="[Warning] Fooocus VAE-based swap.").printf()

        target_model = target_refiner_unet
        if target_model is None:
            target_model = target_unet
            printF(name=MasterName.get_master_name(),
                   info="[Warning] Use base model to refine itself - this may because of developer mode.").printf()

        sampled_latent = vae_parse(sampled_latent)

        k_sigmas = 1.4
        sigmas = calculate_sigmas(sampler=sampler_name,
                                  scheduler=scheduler_name,
                                  model=target_model.model,
                                  steps=steps,
                                  denoise=denoise)[switch:] * k_sigmas
        len_sigmas = len(sigmas) - 1

        noise_mean = torch.mean(modules.patch.patch_settings[os.getpid()].eps_record, dim=1, keepdim=True)

        if modules.inpaint_worker.current_task is not None:
            modules.inpaint_worker.current_task.swap()

        sampled_latent = core.ksampler(
            model=target_model,
            positive=clip_separate(positive_cond, target_model=target_model.model, target_clip=target_clip),
            negative=clip_separate(negative_cond, target_model=target_model.model, target_clip=target_clip),
            latent=sampled_latent,
            steps=len_sigmas, start_step=0, last_step=len_sigmas, disable_noise=False, force_full_denoise=True,
            seed=image_seed + 1,
            denoise=denoise,
            callback_function=callback,
            cfg=cfg_scale,
            sampler_name=sampler_name,
            scheduler=scheduler_name,
            previewer_start=switch,
            previewer_end=steps,
            sigmas=sigmas,
            noise_mean=noise_mean,
            disable_preview=disable_preview,
            width=width, height=height
        )

        target_model = target_refiner_vae
        if target_model is None:
            target_model = target_vae
        decoded_latent = core.decode_vae(vae=target_model, latent_image=sampled_latent, tiled=tiled)
        _diag_post_decode(decoded_latent)

    images = core.pytorch_to_numpy(decoded_latent)
    modules.patch.patch_settings[os.getpid()].eps_record = None
    return images

@torch.no_grad()
@torch.inference_mode()
def lightning_process_diffusion(positive_cond, negative_cond, steps, width, height, image_seed,
                                scheduler_name, cfg_scale=7.0):
    global model_base, model_refiner, final_loras

    model_path = './models/Lightning/'
    base_model_name = model_base.filename
    distilled_model_refiner_name = model_refiner.filename
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"

    SCHEDULERS = {
        "ddim": DDIMScheduler,
        "DPMSolverMultistep": DPMSolverMultistepScheduler,
        "heun": HeunDiscreteScheduler,
        "euler_ancestral": EulerAncestralDiscreteScheduler,
        "euler": EulerDiscreteScheduler,
        "pndm": PNDMScheduler,
        "dpmpp_2m_sde": KDPM2AncestralDiscreteScheduler,
    }

    # pipe = StableDiffusionXLPipeline.from_pretrained(
    #     base,
    #     cache_dir=model_path,
    #     torch_dtype=torch.float16,
    #     variant="fp16",
    #     local_files_only=False).to("cuda")
    flag = False

    for kk in final_loras:
        if "lightning" in kk[0]:
            flag = True

    if flag:
        pipe = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path=base, cache_dir=model_path,
                                                         torch_dtype=torch.float16, variant="fp16").to("cuda")
        pipe.load_lora_weights(
            hf_hub_download(repo_id=repo, cache_dir=model_path, filename=distilled_model_refiner_name))
        pipe.fuse_lora()
    else:
        unet = UNet2DConditionModel.from_config(pretrained_model_name_or_path=base, subfolder="unet").to("cuda",
                                                                                                         torch.float16)
        unet.load_state_dict(
            load_file(hf_hub_download(repo_id=repo, cache_dir=model_path, filename=distilled_model_refiner_name),
                      device="cuda"))
        pipe = StableDiffusionXLPipeline.from_pretrained(pretrained_model_name_or_path=base, unet=unet,
                                                         torch_dtype=torch.float16, variant="fp16").to(
            "cuda")

    generator = torch.Generator("cuda").manual_seed(image_seed)

    pipe.scheduler = SCHEDULERS[scheduler_name].from_config(
        pipe.scheduler.config, timestep_spacing="trailing"
    )

    common_args = {
        "prompt": [positive_cond] * 1,
        "negative_prompt": [negative_cond] * 1,
        "guidance_scale": cfg_scale,
        "generator": generator,
        "num_inference_steps": steps,
        "width": width,
        "height": height,
    }

    # pipe.unet.load_state_dict(load_file(hf_hub_download(repo, distilled_model_refiner_name), device="cuda"))

    output = pipe(**common_args)

    return output.images
