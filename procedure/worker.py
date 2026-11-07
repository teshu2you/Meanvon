import math
import sys
import os
import threading
import traceback
from extras.censor import default_censor
import adapter
from adapter import args_manager
from util.printf import printF, MasterName
import json
import modules
import modules.meta_parser
from PIL import Image, ImageOps
from modules.resolutions import annotate_resolution_string, get_resolution_string, resolutions, string_to_dimensions
import modules.default_pipeline as pipeline
import modules.patch as patch
import extras
import shared
import modules.flags as flags
import modules.core as core
import ldm_patched.modules.model_management
import modules.inpaint_worker as inpaint_worker
import modules.config as config
import modules.advanced_parameters as advanced_parameters
import extras.preprocessors as preprocessors
import extras.ip_adapter as ip_adapter
import extras.face_crop as face_crop
import ldm_patched.modules.model_management as model_management
from modules.util import remove_empty_str, resize_image, HWC3, set_image_shape_ceil, get_image_shape_ceil, \
    get_shape_ceil, resample_image, erode_or_dilate, ordinal_suffix
from modules.private_logger import log
from modules.upscaler import perform_upscale
from extras.expansion import safe_str
from modules.sdxl_styles import apply_style, fooocus_expansion, apply_wildcards, apply_arrays
from version import main_version
import modules.constants as constants
from modules.flags import Performance
import copy
import random
import time
import numpy as np
import modules.util as util
import torch
import re
import logging
import cv2
from typing import List
from util.file import save_output_file
from adapter.parameters import GenerationFinishReason, ImageGenerationResult, ImageGenerationParams
from adapter.task_queue import QueueTask, TaskQueue, TaskOutputs

from enhanced.backend import comfyd, comfyclient_pipeline as comfypipeline
from enhanced.comfy_task import get_comfy_task, default_kolors_base_model_name

worker_queue: TaskQueue = None
queue_task: QueueTask = None
last_model_name = None


def _gallery_entry_path(entry):
    """Gallery 项兼容三种形态：
    Gradio 5 返回 (path_or_dict, caption) 元组；老版本返回 dict({'name': path})
    或纯路径 str。统一提取出本地路径。"""
    if isinstance(entry, (tuple, list)) and len(entry) > 0:
        entry = entry[0]
    if isinstance(entry, dict):
        return entry.get('name') or entry.get('image') or entry.get('path')
    if isinstance(entry, str):
        return entry
    if isinstance(entry, (tuple, list)) and len(entry) > 1:
        return entry[1]
    return None


def _blend_init_images(structure_image, semantic_image, semantic_weight=0.5):
    """把“结构参考图”和“语义参考图”融合成一张 img2img 初始图。

    两侧都是 (1, H, W, 3)、取值 [0,1] 的张量。语义图会先缩放到结构图尺寸，
    再按 semantic_weight 加权平均，从而在无原生语义参考通道的模型上，
    让 image_prompt_gallery 也真正影响图生图结果。
    """
    try:
        w = float(semantic_weight)
        w = max(0.0, min(1.0, w))
        sem = semantic_image
        if sem.shape[1:3] != structure_image.shape[1:3]:
            sem = sem.permute(0, 3, 1, 2)
            sem = torch.nn.functional.interpolate(
                sem, size=(structure_image.shape[1], structure_image.shape[2]),
                mode="bilinear", align_corners=False)
            sem = sem.permute(0, 2, 3, 1)
        blended = structure_image.to(torch.float32) * (1.0 - w) + sem.to(torch.float32) * w
        return blended.clamp(0.0, 1.0)
    except Exception as e:
        printF(name=MasterName.get_master_name(),
               info="[Reference] blend init failed: {}".format(e)).printf()
        return structure_image


def resolve_img2img_denoise(model_type_selector, model_name, user_denoise):
    """按模型族把 img2img denoise 校到该模型能用的区间。

    为什么必须动态调：不同模型族的噪声语义完全不同
      * flow matching（Flux / ZImage / Anima / Moss / Krea2 / QwenImage …）
        起始 sigma ≈ denoise，初始 latent（结构图）权重 = 1 - denoise
        → 1.00 权重 0%（等于纯文生图，“图生图没生效”就是这个）
        → 0.94 权重 6%（几乎看不出结构）
      * SD / SDXL（VP 噪声，A1111 语义）：0.30~0.70 常见，0.94 也几乎等于重画
      * 蒸馏模型（Flux-Schnell / ZImage-Turbo / Moss-SFT）：去噪预算少，
        denoise 太高会崩/糊
    具体数值见 modules/config.py::img2img_denoise_profiles。

    规则（可预测优先）：
      * 用户值 >= 0.999（滑块最大值 = “完全重画”，通常只是默认值）→ 用推荐值
      * 超出该模型族的 [minimum, maximum] → 夹到区间边界
      * 其余情况尊重用户值

    返回 (最终 denoise, 日志文本)。
    """
    prof = config.get_img2img_denoise_profile(model_type_selector)
    if prof.get("matched") == "default" and model_name:
        # UI 选择器和实际权重文件不一致时，按权重文件名兜底判定
        try:
            prof = config.get_img2img_denoise_profile(
                pipeline.get_model_file_type(model_name))
        except Exception:
            pass

    lo = float(prof["minimum"])
    hi = float(prof["maximum"])
    rec = float(prof["recommend"])
    try:
        val = float(user_denoise)
    except (TypeError, ValueError):
        val = 1.0

    if val >= 0.999:
        final = rec
        reason = "denoise=1.0 会完全丢弃初始 latent，改用推荐值"
    elif val > hi:
        final = hi
        reason = "超过该模型族上限，已夹到 {}".format(hi)
    elif val < lo:
        final = lo
        reason = "低于该模型族下限，已抬到 {}".format(lo)
    else:
        final = val
        reason = "在推荐区间内，保持不变"

    msg = ("[img2img] model_type={} (family={}) denoise {:.2f} -> {:.2f} | "
           "推荐 {:.2f}，可用 {:.2f}~{:.2f} | {} | {}").format(
        model_type_selector, prof["matched"], val, final, rec, lo, hi,
        prof["note"], reason)
    return final, msg


class taskManager:
    from modules.patch import PatchSettings, patch_settings, patch_all
    patch_all()

    async_tasks = []

    class AsyncTask:
        def __init__(self, args):
            self.args = args
            self.yields = []
            self.results = []
            self.last_stop = False
            self.processing = False
            # comfy 分支需要：任务交给 comfy 后端时的执行方法
            # （例如 't2i' / 'i2i' / None）。
            # 之前从没被设过，进 comfy 分支必炸：
            #   AttributeError: 'AsyncTask' object has no attribute 'task_method'
            # 给安全缺省，同时让 do_comfy 三重门闸能取到。
            self.task_method = None

    def __init__(self, request_source="webui"):
        self.direct_return = False
        self.api_first_run_flag = True
        self.read_wildcards_in_order = None
        self.save_extension = "png"
        self.img_paths = None
        self.base_model_name_prefix = ""
        self.fixed_steps = 4
        self.img2img_ctrls = None
        self.image_factory_checkbox = None
        self.inpaint_mask_image = None
        self.switch_sampler = None
        self.disable_seed_increment = None
        self.disable_intermediate_results = None
        self.disable_preview = None
        self.inpaint_respective_field = None
        self.controlnet_softness = None
        self.canny_low_threshold = None
        self.canny_high_threshold = None
        self.inpaint_strength = None
        self.inpaint_mask_upload_checkbox = None
        self.invert_mask_checkbox = None
        self.inpaint_erode_or_dilate = None
        self.metadata_scheme = flags.MetadataScheme.FOOOCUS
        self.save_metadata_to_images = None
        self.current_task = None
        self.seed = 0
        self.max_seed = int(1024 * 1024 * 1024)
        self.guidance_scale = 4.0
        self.pid = None
        self.outpaint_distance_bottom = None
        self.outpaint_distance_right = None
        self.outpaint_distance_top = None
        self.outpaint_distance_left = None
        self.upscale_value = None
        self.request_source = request_source
        self.seed_value = None

        self.use_synthetic_refiner = False
        self.ip_adapter_face_path = ""
        self.inpaint_head_model_path = ""
        self.inpaint_patch_model_path = ""
        self.base_model_additional_loras = []
        self.inpaint_disable_initial_latent = None
        self.skipping_cn_preprocessor = True
        self.inpaint_additional_prompt = ""
        self.inpaint_parameterized = "v1"
        self.debugging_inpaint_preprocessor = None
        self.inpaint_mask_image_upload = None

        self.execution_start_time = time.perf_counter()
        self.execution_end_time = self.execution_start_time

        self.buffer = []
        self.m_flag = False
        self.outputs = []
        self.global_results = []

        self.final_unet = None
        self.freeu_enabled = False
        self.loras_raw = [[]]
        self.freeu_b1 = 1.01
        self.freeu_b2 = 1.02
        self.freeu_s1 = 0.99
        self.freeu_s2 = 0.95

        self.raw_style_selections = ""
        self.use_expansion = ""

        self.initial_latent = None
        self.denoising_strength = 1.0
        self.tiled = False
        self.skip_prompt_processing = False
        self.refiner_swap_method = "joint"

        self.raw_prompt = ""
        self.raw_negative_prompt = ""

        self.inpaint_image = ""
        self.inpaint_mask = ""
        self.inpaint_head_model_path = ""
        self.controlnet_canny_path = modules.config.downloading_controlnet_canny()
        self.controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
        self.clip_vision_path = ""
        self.ip_negative_path = ""
        self.ip_adapter_path = ""

        self.goals = []
        self.tasks = []
        self.steps: int = 30
        self.switch = 20
        self.mixing_image_prompt_and_vary_upscale = False
        self.overwrite_step = -1
        self.overwrite_switch = -1
        self.overwrite_width = -1
        self.overwrite_height = -1
        self.width = 1024
        self.height = 1024

        self.results = []
        self.metadata_strings = []
        self.input_gallery_size = 0
        self.guidance_scale = 7.0
        self.imgs = ""
        self.use_style = ""

        self.adaptive_cfg = 7.0
        self.cfg_scale = 7.0
        self.adm_scaler_positive = 1.5
        self.adm_scaler_negative = 0.8
        self.adm_scaler_end = 0.3
        self.mixing_image_prompt_and_inpaint = False
        self.inpaint_engine = "v2.6"
        self.overwrite_vary_strength = -1
        self.start_step = 0
        self.denoise = 1.0
        self.input_image_filename = ""
        self.debugging_cn_preprocessor = False
        self.revision_images_filenames = ""
        self.overwrite_upscale_strength = -1
        self.revision_gallery_size = 0
        self.input_image_path = ""

        # 参考图（图生图）能力状态：
        #   _ref_kind               : 'clip_vision' / 'ref_latents' / None（由当前模型决定）
        #   _engine_revision_used   : 本次实际写入引擎 ref_latents 的语义参考图数量
        #   _semantic_fallback_path : 引擎无原生语义参考时，退化用该图做 img2img 初始图
        #   _semantic_fallback_active: 本轮是否处于语义回退 img2img
        self._ref_kind = None
        self._engine_revision_used = 0
        self._semantic_fallback_path = None
        self._semantic_fallback_active = False
        self._semantic_denoise = 0.6
        # 参考编辑型引擎（Kontext/Krea2/Klein）：结构图已作为 ini_latent 注入，
        # 采样从噪声全量生成（denoise=1），不再使用通用 img2img 初值。
        self._native_structure_ref = False

        self.prompt = "1girl"
        self.negative_prompt = "nsfw"
        self.style_selections = []
        self.performance_selection = Performance.SPEED
        self.aspect_ratios_selection = "1024×1024 (1:1)"
        self.image_number = 1
        self.image_seed = 0
        self.sharpness = 2.0
        self.sampler_name = "dpmpp_2m_sde_gpu"
        self.scheduler_name = "karras"
        self.generate_image_grid = True
        self.custom_steps = 4
        self.custom_switch = 0.4
        self.base_model_name = ""
        self.base_clip_skip = -2
        self.loras = [[]]
        self.save_metadata_json = True
        self.save_metadata_image = True
        self.img2img_mode = False
        self.img2img_start_step = 0.06
        self.img2img_denoise = 0.94
        self.img2img_scale = 1.0
        self._effective_img2img = False
        self.revision_mode = False
        self.positive_prompt_strength = 1
        self.negative_prompt_strength = 1
        self.revision_strength_1 = 1.0
        self.revision_strength_2 = 1.0
        self.revision_strength_3 = 1.0
        self.revision_strength_4 = 1.0
        self.same_seed_for_all = False
        self.output_format = "png"
        self.control_lora_canny = False
        self.canny_edge_low = 0.2
        self.canny_edge_high = 0.8
        self.canny_start = 0.0
        self.canny_stop = 0.4
        self.canny_strength = 0.8
        self.canny_model = "control-lora-canny-rank128.safetensors"
        self.control_lora_depth = False
        self.depth_start = 0.0
        self.depth_stop = 0.4
        self.depth_strength = 0.8
        self.depth_model = "control-lora-depth-rank128.safetensors"
        self.image_factory_checkbox = False
        self.current_tab = "uov"
        self.uov_method = "Disabled"
        self.uov_input_image = None
        self.outpaint_selections = []
        self.outpaint_expansion_ratio = 0.3
        self.inpaint_input_image = None
        self.cn_tasks = {'Image Prompt': ['none', 0.6, 0.5], 'PyraCanny': ['none', 0.6, 0.5],
                         'CPDS': ['none', 0.6, 0.5]}
        self.black_out_nsfw = 0
        self.input_gallery = []
        self.revision_gallery = []
        self.output_gallery = []                              # 新增：以前缺少此属性
        self.keep_input_names = False
        self.default_model_type = "SDXL"
        
        self.backfill_prompt = False
        self.translation_methods = ''
        self.comfyd_active_checkbox = False
        
                # ComfyUI 后端所需的字段。
        # 之前 generate_images 的 comfy 分支访问这三个：
        #   self.params_backend   ← 本轮炸点
        #   self.task_name        ← 修完上面后下一个
        #   async_task.task_method ← 再下一个
        # 都没在 __init__ 里兜底，进入分支必然连续 AttributeError。
        # 这里给安全缺省：
        self.task_class = "default"
        self.task_name = None
        self.params_backend = {}
  
          # AUDIT 只检查字段存在性，补 None 即可消警；若下游需要，按需赋 '' 或 'txt2img'
        if not hasattr(taskManager, "task_method"):
            taskManager.task_method = None
            
        # 自检：generate_images + process_diffusion 路径上所有
        # 可能被引用、但容易被漏初始化的字段。
        # 未来某个分支新增 self.xxx 而 __init__ 忘了补时，立即报出来。
        _audit = ['cfg_scale', 'params_backend', 'task_name', 'task_method',
                  'task_class', 'comfyd_active_checkbox',
                  'base_model_name_prefix', 'guidance_scale',
                  'adaptive_cfg', 'guidance_scale', 'denoising_strength']
        _missing = [a for a in _audit if not hasattr(self, a)]
        if _missing:
            printF(name=MasterName.get_master_name(),
                   info="[AUDIT] taskManager 缺字段: {}".format(_missing)).printf()

    def init_param(self, obj):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> init_param").printf()
        self.pid = os.getpid()
        printF(name=MasterName.get_master_name(), info="Started worker with PID {}".format(self.pid)).printf()
        if self.request_source == "api":
            params = obj.req_param
            self.prompt = params.prompt
            self.negative_prompt = params.negative_prompt
            self.style_selections = params.style_selections
            self.performance_selection = params.performance_selection
            self.aspect_ratios_selection = params.aspect_ratios_selection
            self.image_number = params.image_number
            self.image_seed = None if params.image_seed == -1 else params.image_seed
            self.sharpness = params.sharpness
            self.guidance_scale = params.guidance_scale
            self.refiner_swap_method = params.refiner_swap_method if hasattr(params, "refiner_swap_method") else "joint"
            self.base_model_name = params.base_model_name
            self.loras = params.loras
            self.image_factory_checkbox = params.uov_input_image is not None or params.inpaint_input_image is not None or len(
                params.image_prompts) > 0
            self.current_tab = 'uov' if params.uov_method != flags.disabled else 'ip' if len(
                params.image_prompts) > 0 else 'inpaint' if params.inpaint_input_image is not None else None
            self.uov_method = params.uov_method
            self.upscale_value = params.upscale_value
            self.uov_input_image = params.uov_input_image
            self.outpaint_selections = params.outpaint_selections
            self.outpaint_distance_left = params.outpaint_distance_left
            self.outpaint_distance_top = params.outpaint_distance_top
            self.outpaint_distance_right = params.outpaint_distance_right
            self.outpaint_distance_bottom = params.outpaint_distance_bottom
            self.inpaint_input_image = params.inpaint_input_image
            self.inpaint_additional_prompt = params.inpaint_additional_prompt
            self.inpaint_mask_image_upload = None
            self.save_extension = params.save_extension
            if self.inpaint_additional_prompt is None:
                self.inpaint_additional_prompt = ''

            self.image_seed = self.refresh_seed(self.image_seed is None, self.image_seed)
            self.default_model_type = params.default_model_type
            self.black_out_nsfw = params.black_out_nsfw

            for img_prompt in params.image_prompts:
                cn_img, cn_stop, cn_weight, cn_type = img_prompt
                self.cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])
                
            self.backfill_prompt = params.backfill_prompt
            self.translation_methods = params.translation_methods
            self.comfyd_active_checkbox = params.comfyd_active_checkbox

        elif self.request_source == "webui":
            obj.processing = True
            args = obj.args

            printF(name=MasterName.get_master_name(), info="WebUI reversed args: {}".format(len(args))).printf()

            groups = []
            names_dict = {}
            num = 4
            string = ""
            for inx, val in enumerate(args):
                if num <= 4:
                    string += "|{0:<2}| - {1:<20}".format(inx, str(val))
                    num -= 1
                if num == 0 and string != "":
                    groups.append(string)
                    string = ""
                    num = 4

            for kk in groups:
                printF(name=MasterName.get_master_name(), info="{}".format(kk)).printf()

            # ctrls sort
            # ctrls = [currentTask, generate_image_grid]
            # ctrls += [
            #     prompt, negative_prompt, style_selections,
            #     performance_selection, aspect_ratios_selection, image_number, image_seed,
            #     sharpness, switch_sampler, sampler_name, scheduler_name, fixed_steps, custom_steps, custom_switch,
            #     guidance_scale
            # ]
            #
            # ctrls += [base_model, base_clip_skip] + lora_ctrls
            # ctrls += [image_factory_checkbox, current_tab]
            # ctrls += [uov_method, uov_input_image]
            # ctrls += [outpaint_selections, inpaint_input_image, inpaint_additional_prompt, inpaint_mask_image]
            # ctrls += [disable_preview, disable_intermediate_results, disable_seed_increment]
            # ctrls += [adm_scaler_positive, adm_scaler_negative, adm_scaler_end, adaptive_cfg]
            # ctrls += [sampler_name, scheduler_name]
            # ctrls += [overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength]
            # ctrls += [overwrite_upscale_strength, mixing_image_prompt_and_vary_upscale, mixing_image_prompt_and_inpaint]
            # ctrls += [debugging_cn_preprocessor, skipping_cn_preprocessor, canny_low_threshold, canny_high_threshold]
            # ctrls += [refiner_swap_method, controlnet_softness]
            # ctrls += freeu_ctrls
            # ctrls += inpaint_ctrls
            # ctrls += [save_metadata_json, save_metadata_image] + img2img_ctrls + [same_seed_for_all, output_format]
            # ctrls += canny_ctrls + depth_ctrls
            # ctrls += ip_ctrls
            # ctrls += [model_type_selector]
            # ctrls += [nsfw_filter]
            # if not adapter.args_manager.args.disable_metadata:
            #     ctrls += [save_metadata_to_images, metadata_scheme]

            args.reverse()

            # self.current_task = args.pop()
            self.generate_image_grid = args.pop()

            self.prompt = args.pop()
            self.negative_prompt = args.pop()
            self.style_selections = args.pop()
            self.performance_selection = Performance(args.pop())
            self.aspect_ratios_selection = args.pop()
            self.image_number = args.pop()
            self.image_seed = args.pop()
            self.sharpness = args.pop()
            # index - 10
            self.switch_sampler = args.pop()
            self.sampler_name = args.pop()
            self.scheduler_name = args.pop()
            self.fixed_steps = args.pop()
            self.custom_steps = args.pop()
            self.custom_switch = args.pop()
            # cfg ======= guidance_scale
            self.guidance_scale = args.pop()
            # index - 17
            self.base_model_name = args.pop()
            self.base_clip_skip = args.pop()

            # index - 22
            self.loras = self.apply_enabled_loras([[bool(args.pop()), str(args.pop()), float(args.pop()), ] for _ in
                                                   range(modules.config.default_max_lora_number)])
            printF(name=MasterName.get_master_name(), info="[Parameters] self.loras = {}".format(self.loras)).printf()
            self.image_factory_checkbox = args.pop()
            self.current_tab = args.pop()

            self.uov_method = args.pop()
            self.uov_input_image = args.pop()

            self.outpaint_selections = args.pop()
            self.outpaint_expansion_ratio = args.pop()
            self.inpaint_input_image = args.pop()
            self.inpaint_additional_prompt = args.pop()
            self.inpaint_mask_image = args.pop()

            self.disable_preview = args.pop()
            self.disable_intermediate_results = args.pop()
            self.disable_seed_increment = args.pop()

            self.adm_scaler_positive = args.pop()
            self.adm_scaler_negative = args.pop()
            self.adm_scaler_end = args.pop()
            self.adaptive_cfg = args.pop()

            self.overwrite_step = args.pop()
            self.overwrite_switch = args.pop()
            self.overwrite_width = args.pop()
            self.overwrite_height = args.pop()
            self.overwrite_vary_strength = args.pop()

            self.overwrite_upscale_strength = args.pop()
            self.mixing_image_prompt_and_vary_upscale = args.pop()
            self.mixing_image_prompt_and_inpaint = args.pop()

            self.debugging_cn_preprocessor = args.pop()
            self.skipping_cn_preprocessor = args.pop()
            self.canny_low_threshold = args.pop()
            self.canny_high_threshold = args.pop()

            self.refiner_swap_method = args.pop()
            self.controlnet_softness = args.pop()

            self.freeu_enabled = args.pop()
            self.freeu_b1 = args.pop()
            self.freeu_b2 = args.pop()
            self.freeu_s1 = args.pop()
            self.freeu_s2 = args.pop()

            self.debugging_inpaint_preprocessor = args.pop()
            self.inpaint_disable_initial_latent = args.pop()
            # index 75 inpaint_engine
            self.inpaint_engine = args.pop()
            self.inpaint_strength = args.pop()
            self.inpaint_respective_field = args.pop()
            self.inpaint_mask_upload_checkbox = args.pop()
            self.invert_mask_checkbox = args.pop()
            self.inpaint_erode_or_dilate = args.pop()

            self.save_metadata_json = args.pop()
            # conflict with fun in metadata to image, remove --> save_metadata_image
            # self.save_metadata_image = args.pop()

            self.img2img_mode = args.pop()
            self.img2img_start_step = args.pop()
            self.img2img_denoise = args.pop()
            self.img2img_scale = args.pop()
            self.revision_mode = args.pop()
            self.positive_prompt_strength = args.pop()
            self.negative_prompt_strength = args.pop()
            self.revision_strength_1 = args.pop()
            self.revision_strength_2 = args.pop()
            self.revision_strength_3 = args.pop()
            self.revision_strength_4 = args.pop()

            self.same_seed_for_all = args.pop()
            self.output_format = args.pop()

            self.control_lora_canny = args.pop()
            self.canny_edge_low = args.pop()
            self.canny_edge_high = args.pop()
            self.canny_start = args.pop()
            self.canny_stop = args.pop()
            self.canny_strength = args.pop()
            self.canny_model = args.pop()

            self.control_lora_depth = args.pop()
            self.depth_start = args.pop()
            self.depth_stop = args.pop()
            self.depth_strength = args.pop()
            self.depth_model = args.pop()

            self.cn_tasks = {x: [] for x in flags.ip_list}
            for _ in range(flags.controlnet_image_count):
                cn_img = args.pop()
                cn_stop = args.pop()
                cn_weight = args.pop()
                cn_type = args.pop()
                printF(name=MasterName.get_master_name(),
                       info="cn_img：{}, cn_stop：{}, cn_weight：{}，cn_type：{}".format(cn_img, cn_stop, cn_weight,
                                                                                    cn_type)).printf()
                if cn_img is not None:
                    self.cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

            # index 80 save_metadata_to_images
            # self.save_metadata_to_images = args.pop() if not args_manager.args.disable_metadata else False
            # self.metadata_scheme = flags.MetadataScheme(
            #     args.pop()) if not args_manager.args.disable_metadata else flags.MetadataScheme.FOOOCUS

            self.default_model_type = args.pop()
            self.black_out_nsfw = args.pop()
            self.save_metadata_to_images = args.pop()
            self.metadata_scheme = flags.MetadataScheme(args.pop())

            # 归一化：不管 Gradio 返回 str / dict / (file, caption) 元组，
            # 统一提取成本地路径列表，后续 len() / 索引永远正确。
            self.input_gallery = [p for p in map(_gallery_entry_path, (args.pop() or [])) if p]
            self.revision_gallery = [p for p in map(_gallery_entry_path, (args.pop() or [])) if p]
            self.keep_input_names = args.pop()

            printF(name=MasterName.get_master_name(),
                   info="[Gallery] input_gallery: type=list len={} paths={}".format(
                       len(self.input_gallery), self.input_gallery[:2])).printf()
            printF(name=MasterName.get_master_name(),
                   info="[Gallery] revision_gallery: len={}".format(len(self.revision_gallery))).printf()
            
            
            self.input_gallery_size = len(self.input_gallery) # 立即计算
            self.revision_gallery_size = len(self.revision_gallery)

            if self.performance_selection in [Performance(Performance.Custom)]:
                self.steps = self.custom_steps
            else:
                self.steps = self.fixed_steps

            self.cfg_scale = self.adaptive_cfg if self.adaptive_cfg > 0 else self.guidance_scale

    @torch.no_grad()
    @torch.inference_mode()
    def process_generate(self, async_task, wq: TaskQueue, qt: QueueTask):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> process_generate").printf()
        global worker_queue, queue_task

        worker_queue = wq
        queue_task = qt

        if self.request_source == "api":
            printF(name=MasterName.get_master_name(),
                   info="[Parameters] api_first_run_flag = {}".format(self.api_first_run_flag)).printf()
            if self.api_first_run_flag:
                queue_task.is_finished = True
                worker_queue.finish_task(queue_task.job_id)
                self.api_first_run_flag = False
                return

            if queue_task is None or queue_task.req_param is None:
                return

            self.init_param(obj=queue_task)
            global last_model_name
            if last_model_name is None:
                last_model_name = queue_task.req_param.base_model_name
            if last_model_name != queue_task.req_param.base_model_name:
                model_management.cleanup_models()  # key1
                model_management.unload_all_models()
                model_management.soft_empty_cache()  # key2
                last_model_name = queue_task.req_param.base_model_name
        elif self.request_source == "webui":
            self.init_param(obj=async_task)

        async_task.last_stop = False
        async_task.processing = False
        async_task.yields = []
        async_task.results = []

        print(f"queue_task:{queue_task.__dict__}")
        print(f"worker_queue:{worker_queue.__dict__}")
        print(f"async_task:{async_task.__dict__}")
        worker_queue.start_task(queue_task.job_id)
        printF(name=MasterName.get_master_name(),
               info="[Task Queue] Task queue start task, job_id={}".format(queue_task.job_id)).printf()
        queue_task.is_finished = False
        execution_start_time = time.perf_counter()
        # self.outputs = TaskOutputs(async_task)
        self.outputs = []
        self.results = []
        self.tasks = []
        self.goals = []
        self.direct_return = False
        self.initial_latent = None

        procedure_list = [
            self.pre_process,
            self.download_image_func_models,
            self.encode_prompts,
            self.manage_cns,
            self.get_advanced_parameters,
            self.check_vary_in_goals,
            self.check_upscale_in_goals,
            self.check_inpaint_in_goals,
            self.check_cn_in_goals,
            self.generate_images,
            self.post_process
        ]
        try:
            for x in procedure_list:
                print('-' * 200)
                printF(name=MasterName.get_master_name(), info=x).printf()
                if self.direct_return:
                    printF(name=MasterName.get_master_name(),
                           info="[Return directly, ignore ...] self.direct_return: {}".format(
                               self.direct_return)).printf()
                    continue
                start_time = time.perf_counter()
                print(f"{x} -- async_task -- {async_task.__dict__}")
                x(async_task=async_task)
                cost_time = time.perf_counter() - start_time
                print(
                    f'\n                          Cost Time: <<<<<<<<<<<<<< {cost_time:.2f} seconds >>>>>>>>>>>>>>>>>')

            self.yield_result(async_task, self.results)
            queue_task.is_finished = True
            return
        except Exception as e:
            printF(name=MasterName.get_master_name(),
                   info="[Worker error] {}".format(e)).printf()
            logging.exception(e)
            if not queue_task.is_finished:
                queue_task.set_result(self.results, True, str(e))
                worker_queue.finish_task(queue_task.job_id)
                printF(name=MasterName.get_master_name(),
                       info="[Task Queue] Finish task with error, seq={}".format(queue_task.job_id)).printf()
            return []

    def refresh_seed(self, r, seed_string):
        if r:
            return random.randint(constants.MIN_SEED, constants.MAX_SEED)
        else:
            try:
                self.seed_value = int(seed_string)
                if constants.MIN_SEED <= self.seed_value <= constants.MAX_SEED:
                    return self.seed_value
            except ValueError:
                pass
            return random.randint(constants.MIN_SEED, constants.MAX_SEED)

    def progressbar(self, async_task, number, text):
        print(f'[MeanVon] {text}')
        if self.request_source == "api":
            self.outputs.append(['preview', (number, text, None)])
        elif self.request_source == "webui":
            async_task.yields.append(['preview', (number, text, None, 0, 0)])

    def yield_result(self, async_task, imgs, do_not_show_finished_images=False):
        if not isinstance(imgs, list):
            imgs = [imgs]
            
        if self.black_out_nsfw == "1":
            printF(name=MasterName.get_master_name(),
                               info="[Task Queue] Checking for NSFW content...").printf()
            imgs = default_censor(imgs)
            
        if self.request_source == "webui":
            async_task.results = async_task.results + imgs

            if do_not_show_finished_images:
                return

            for r in async_task.results[:]:
                if isinstance(r, ImageGenerationResult):
                    async_task.results.remove(r)

            async_task.yields.append(['results', async_task.results])
            return
        elif self.request_source == "api":
            results = []
            for i, im in enumerate(imgs):
                print(f"im:{im}")    # 原为 print("im:{im}")
                seed = -1 if len(self.tasks) == 0 else self.tasks[i]['task_seed']
                # img_filename = save_output_file(img=im, extension=self.save_extension)
                results.append(
                    ImageGenerationResult(im=im, seed=str(seed),
                                          finish_reason=GenerationFinishReason.success))
            async_task.set_result(results, False)
            print(f"async_task:{async_task.__dict__}")
            worker_queue.finish_task(async_task.job_id)
            printF(name=MasterName.get_master_name(),
                   info="[Task Queue] Finish task, job_id={}".format(async_task.job_id)).printf()

            self.outputs.append(['results', imgs])
            pipeline.prepare_text_encoder(async_call=True)

    def get_config_from_file(self, file):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> get_config_from_file").printf()
        with open(file, "r", encoding='utf-8') as f:
            try:
                cf = json.load(f)
                print(f"{cf}")
                return cf
            except Exception as e:
                printF(name=MasterName.get_master_name(), info="[Error] load_settings, e: {}".format(e)).printf()
            finally:
                f.close()

    def get_config_key(self, config):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> get_config_key").printf()
        self.pid = config.get("pid", os.getpid())
        self.prompt = config.get("prompt")
        self.negative_prompt = config.get("negative_prompt")
        self.style_selections = config.get("style_selections")
        self.performance_selection = config.get("performance_selection")
        self.aspect_ratios_selection = config.get("aspect_ratios_selection")
        self.image_number = config.get("image_number")
        self.image_seed = config.get("image_seed")

        self.sharpness = config.get("sharpness")
        self.sampler_name = config.get("sampler_name")
        self.scheduler_name = config.get("scheduler_name")
        self.custom_steps = config.get("custom_steps")
        self.custom_switch = config.get("custom_switch")
        self.guidance_scale = config.get("guidance_scale")

        self.base_model_name = config.get("base_model_name")
        self.base_clip_skip = config.get("base_clip_skip")

        self.loras = config.get("loras")
        self.save_metadata_json = config.get("save_metadata_json")
        # self.save_metadata_image = config.get("save_metadata_image")

        self.img2img_mode = config.get("img2img_mode")
        self.img2img_start_step = config.get("img2img_start_step")
        self.img2img_denoise = config.get("img2img_denoise")
        self.img2img_scale = config.get("img2img_scale")
        self.revision_mode = config.get("revision_mode")
        self.positive_prompt_strength = config.get("positive_prompt_strength")
        self.negative_prompt_strength = config.get("negative_prompt_strength")
        self.revision_strength_1 = config.get("revision_strength_1")
        self.revision_strength_2 = config.get("revision_strength_2")
        self.revision_strength_3 = config.get("revision_strength_3")
        self.revision_strength_4 = config.get("revision_strength_4")

        self.same_seed_for_all = config.get("same_seed_for_all")
        self.output_format = config.get("output_format")

        self.control_lora_canny = config.get("control_lora_canny")
        self.canny_edge_low = config.get("canny_edge_low")
        self.canny_edge_high = config.get("canny_edge_high")
        self.canny_start = config.get("canny_start")
        self.canny_stop = config.get("canny_stop")
        self.canny_strength = config.get("canny_strength")
        self.canny_model = config.get("canny_model")

        self.depth_start = config.get("depth_start")
        self.depth_stop = config.get("depth_stop")
        self.depth_strength = config.get("depth_strength")
        self.depth_model = config.get("depth_model")

        self.image_factory_checkbox = config.get("image_factory_checkbox")
        self.current_tab = config.get("current_tab")
        self.uov_method = config.get("uov_method")
        self.uov_input_image = config.get("uov_input_image")
        self.outpaint_selections = config.get("outpaint_selections")
        self.outpaint_expansion_ratio = config.get("outpaint_expansion_ratio")
        self.inpaint_input_image = config.get("inpaint_input_image")

        self.cn_tasks = config.get("cn_tasks")
        print(self.cn_tasks.__str__())
        if isinstance(self.cn_tasks, str):
            self.cn_tasks = dict(eval(self.cn_tasks))
            print(self.cn_tasks.__str__())
        self.input_gallery = config.get("input_gallery")
        self.revision_gallery = config.get("revision_gallery")
        self.keep_input_names = config.get("keep_input_names")

        self.freeu_enabled = config.get("freeu_enabled")
        self.loras_raw = config.get("loras_raw")
        self.freeu_b1 = config.get("freeu_b1")
        self.freeu_b2 = config.get("freeu_b2")
        self.freeu_s1 = config.get("freeu_s1")    # 原为 freeu_b3
        self.freeu_s2 = config.get("freeu_s2")    # 原为 freeu_b

        self.raw_style_selections = config.get("raw_style_selections")
        self.use_expansion = config.get("use_expansion")

        self.initial_latent = config.get("initial_latent")
        self.tiled = config.get("tiled")
        self.skip_prompt_processing = config.get("skip_prompt_processing")
        self.refiner_swap_method = config.get("refiner_swap_method")

        self.raw_prompt = config.get("raw_prompt")
        self.raw_negative_prompt = config.get("raw_negative_prompt")

        self.inpaint_image = config.get("inpaint_image")
        self.inpaint_mask = config.get("inpaint_mask")
        self.inpaint_head_model_path = config.get("inpaint_head_model_path")
        self.controlnet_canny_path = config.get("controlnet_canny_path")
        self.controlnet_cpds_path = config.get("controlnet_cpds_path")
        self.clip_vision_path = config.get("clip_vision_path")
        self.ip_negative_path = config.get("ip_negative_path")
        self.ip_adapter_path = config.get("ip_adapter_path")

        self.goals = config.get("goals")
        self.tasks = config.get("tasks")
        self.steps = config.get("steps")
        self.switch = config.get("switch")
        self.mixing_image_prompt_and_vary_upscale = config.get("mixing_image_prompt_and_vary_upscale")
        self.overwrite_step = config.get("overwrite_step")
        self.overwrite_switch = config.get("overwrite_switch")
        self.overwrite_width = config.get("overwrite_width")
        self.overwrite_height = config.get("overwrite_height")
        self.width = config.get("width")
        self.height = config.get("height")

        self.results = config.get("results")
        self.metadata_strings = config.get("metadata_strings")
        self.input_gallery_size = config.get("input_gallery_size")
        self.guidance_scale = config.get("guidance_scale")
        self.imgs = config.get("imgs")
        self.use_style = config.get("use_style")

        self.adaptive_cfg = config.get("adaptive_cfg")
        self.adm_scaler_positive = config.get("adm_scaler_positive")
        self.adm_scaler_negative = config.get("adm_scaler_negative")
        self.adm_scaler_end = config.get("adm_scaler_end")
        self.mixing_image_prompt_and_inpaint = config.get("mixing_image_prompt_and_inpaint")
        self.inpaint_engine = config.get("inpaint_engine")
        self.overwrite_vary_strength = config.get("overwrite_vary_strength")
        self.start_step = config.get("start_step")
        self.denoise = config.get("denoise")
        self.input_image_filename = config.get("input_image_filename")
        self.debugging_cn_preprocessor = config.get("debugging_cn_preprocessor")
        self.revision_images_filenames = config.get("revision_images_filenames")
        self.control_lora_depth = config.get("control_lora_depth")
        self.overwrite_upscale_strength = config.get("overwrite_upscale_strength")
        self.revision_gallery_size = config.get("revision_gallery_size")
        self.input_image_path = config.get("input_image_path")

        self.generate_image_grid = config.get("generate_image_grid")
        self.use_synthetic_refiner = config.get("use_synthetic_refiner")
        self.ip_adapter_face_path = config.get("ip_adapter_face_path")
        self.inpaint_head_model_path = config.get("inpaint_head_model_path")
        self.inpaint_patch_model_path = config.get("inpaint_patch_model_path")
        self.base_model_additional_loras = config.get("base_model_additional_loras")
        self.inpaint_disable_initial_latent = config.get("inpaint_disable_initial_latent")
        self.skipping_cn_preprocessor = config.get("skipping_cn_preprocessor")
        self.inpaint_additional_prompt = config.get("inpaint_additional_prompt")
        self.inpaint_parameterized = config.get("inpaint_engine") != 'None'
        self.debugging_inpaint_preprocessor = config.get("debugging_inpaint_preprocessor")
        self.inpaint_mask_image_upload = config.get("inpaint_mask_image_upload")

    def build_image_wall(self, async_task):
        if not self.generate_image_grid:
            return

        results = []

        if len(async_task.results) < 2:
            return

        for img in async_task.results:
            if isinstance(img, str) and os.path.exists(img):
                img = cv2.imread(img)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if not isinstance(img, np.ndarray):
                return
            if img.ndim != 3:
                return
            results.append(img)

        H, W, C = results[0].shape

        for img in results:
            Hn, Wn, Cn = img.shape
            if H != Hn:
                return
            if W != Wn:
                return
            if C != Cn:
                return

        cols = float(len(results)) ** 0.5
        cols = int(math.ceil(cols))
        rows = float(len(results)) / float(cols)
        rows = int(math.ceil(rows))

        wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(results):
                    img = results[y * cols + x]
                    wall[y * H:y * H + H, x * W:x * W + W, :] = img

        # must use deep copy otherwise gradio is super laggy. Do not use list.append() .
        async_task.results = async_task.results + [wall]
        return

    def apply_enabled_loras(self, loras):
        enabled_loras = []
        for lora_enabled, lora_model, lora_weight in loras:
            if lora_enabled:
                enabled_loras.append([lora_enabled, lora_model, lora_weight])
        printF(name=MasterName.get_master_name(),
               info="[Parameters] apply_enabled_loras --> enabled_loras: {}".format(enabled_loras)).printf()
        return enabled_loras

    def pre_process(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> pre_process").printf()
        self.raw_prompt = self.prompt
        self.raw_negative_prompt = self.negative_prompt
        self.outpaint_selections = [o.lower() for o in self.outpaint_selections]

        if not self.loras:
            self.loras = [[]]
        self.loras_raw = copy.deepcopy(self.loras)
        self.raw_style_selections = copy.deepcopy(self.style_selections)
        self.uov_method = self.uov_method.lower() if self.uov_method is not None else ""

        printF(name=MasterName.get_master_name(), info="[Parameters] raw_prompt: {}".format(self.raw_prompt)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] raw_negative_prompt: {}".format(self.raw_negative_prompt)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] outpaint_selections: {}".format(self.outpaint_selections)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] loras_raw: {}".format(self.loras_raw)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] raw_style_selections: {}".format(self.raw_style_selections)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] uov_method: {}".format(self.uov_method)).printf()

        if fooocus_expansion in self.style_selections:
            self.use_expansion = True
            self.style_selections.remove(fooocus_expansion)
        else:
            self.use_expansion = False

        self.use_style = len(self.style_selections) > 0

        printF(name=MasterName.get_master_name(), info="[Parameters] use_style: {}".format(self.use_style)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] use_expansion: {}".format(self.use_expansion)).printf()

        printF(name=MasterName.get_master_name(),
               info="[Parameters] performance_selection: {}".format(self.performance_selection)).printf()

        # assert self.performance_selection in [Performance.SPEED, Performance.QUALITY, Performance.LCM,Performance.TURBO, Performance.Lightning, Performance.HYPER_SD, Performance.Custom]

        # self.fixed_steps = Performance(self.performance_selection)

        if self.performance_selection == Performance.SPEED:
            self.fixed_steps = constants.STEPS_SPEED
            self.switch = constants.SWITCH_SPEED

        if self.performance_selection == Performance.QUALITY:
            self.fixed_steps = constants.STEPS_QUALITY
            self.switch = constants.SWITCH_QUALITY

        if self.performance_selection == Performance.LCM:
            self.fixed_steps = constants.STEPS_LCM
            printF(name=MasterName.get_master_name(), info="[Warning] Enter LCM mode.").printf()
            self.progressbar(async_task, 1, 'Downloading LCM components ...')
            if self.loras == [[]]:
                self.loras[0] = [True, modules.config.downloading_sdxl_lcm_lora(), 1.0]
            else:
                self.loras += [[True, modules.config.downloading_sdxl_lcm_lora(), 1.0]]


            if not self.switch_sampler:
                self.sampler_name = advanced_parameters.sampler_name = 'lcm'
                self.scheduler_name = advanced_parameters.scheduler_name = 'lcm'
            modules.patch.sharpness = self.sharpness = 0.0
            self.guidance_scale = 1.0
            modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
            modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
            modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
            modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0
            # no need to sync
            # self.steps = self.fixed_steps

        if self.performance_selection == Performance.TURBO:
            self.fixed_steps = constants.STEPS_TURBO
            printF(name=MasterName.get_master_name(), info="[Warning] Enter TURBO mode.").printf()
            self.progressbar(async_task, 1, 'Downloading TURBO components ...')
            if self.loras == [[]]:
                self.loras[0] = [True, modules.config.downloading_sdxl_turbo_lora(), 1.0]
            else:
                self.loras += [[True, modules.config.downloading_sdxl_turbo_lora(), 1.0]]


            if not self.switch_sampler:
                self.sampler_name = advanced_parameters.sampler_name = 'euler_ancestral'
                self.scheduler_name = advanced_parameters.scheduler_name = 'karras'
            modules.patch.sharpness = self.sharpness = 0.0
            self.guidance_scale = 1.0
            modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
            modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
            modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
            modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0
            # no need to sync
            # self.steps = self.fixed_steps

        if self.performance_selection == Performance.Lightning:
            self.fixed_steps = constants.STEPS_LIGHTNING
            printF(name=MasterName.get_master_name(), info="[Warning] Enter Lightning mode.").printf()
            self.progressbar(async_task, 1, 'Downloading Lightning components ...')
            printF(name=MasterName.get_master_name(), info="[Warning] force to replace the 1st lora.").printf()
            print(f"self.loras: {self.loras}")
            # lightning -- ignore adding default loras.
            # if self.loras == [[]]:
            #     self.loras[0] = [True, modules.config.downloading_sdxl_lightning_lora(), 1.0]
            # else:
            #     self.loras += [[True, modules.config.downloading_sdxl_lightning_lora(), 1.0]]

            if not self.switch_sampler:
                self.sampler_name = advanced_parameters.sampler_name = 'euler'
                self.scheduler_name = advanced_parameters.scheduler_name = 'sgm_uniform'

            modules.patch.sharpness = self.sharpness = 2.0
            self.guidance_scale = 1.0
            modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
            modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
            modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
            modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0
            # no need to sync
            # self.steps = self.fixed_steps

        if self.performance_selection == Performance.HYPER_SD:
            self.fixed_steps = constants.STEPS_HYPER_SD
            printF(name=MasterName.get_master_name(), info="[Warning] Enter Hyper-SD mode.").printf()
            self.progressbar(async_task, 1, 'Downloading Hyper-SD components ...')
            printF(name=MasterName.get_master_name(), info="[Warning] force to replace the 1st lora.").printf()
            print(f"self.loras: {self.loras}")

            if self.loras == [[]]:
                self.loras[0] = [True, modules.config.downloading_sdxl_hyper_sd_lora(), 0.8]
            else:
                self.loras += [[True, modules.config.downloading_sdxl_hyper_sd_lora(), 0.8]]


            if not self.switch_sampler:
                self.sampler_name = advanced_parameters.sampler_name = 'dpmpp_sde_gpu'
                self.scheduler_name = advanced_parameters.scheduler_name = 'karras'

            modules.patch.sharpness = self.sharpness = 0.0
            self.guidance_scale = 1.0
            modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
            modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
            modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
            modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0

        if self.performance_selection == Performance.Custom:
            self.steps = self.custom_steps

        printF(name=MasterName.get_master_name(),
               info="[Parameters] performance_selection: {}".format(self.performance_selection)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] fixed_steps: {}".format(self.fixed_steps)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] steps: {}".format(self.steps)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] base_model_name: {}".format(self.base_model_name)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] sampler_name: {}".format(self.sampler_name)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] black_out_nsfw: {}".format(self.black_out_nsfw)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] scheduler_name: {}".format(self.scheduler_name)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] Sharpness = {}".format(self.sharpness)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] Adaptive CFG = {}".format(self.adaptive_cfg)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] ControlNet Softness = {}".format(self.controlnet_softness)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] ADM Scale = {} : {} : {}".format(self.adm_scaler_positive, self.adm_scaler_negative,
                                                                   self.adm_scaler_end)).printf()

        modules.patch.patch_settings[self.pid] = patch.PatchSettings(
            self.sharpness,
            self.adm_scaler_end,
            self.adm_scaler_positive,
            self.adm_scaler_negative,
            self.controlnet_softness,
            self.adaptive_cfg
        )

        printF(name=MasterName.get_master_name(),
               info="[Parameters] pid = {} | patch_settings = {}".format(self.pid, modules.patch.patch_settings[self.pid].__dict__)).printf()

        self.guidance_scale = float(self.guidance_scale)
        printF(name=MasterName.get_master_name(),
               info="[Parameters] guidance_scale = {}".format(self.guidance_scale)).printf()

        self.width, self.height = self.aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
        self.width, self.height = int(self.width), int(self.height)

        self.skip_prompt_processing = False

        self.raw_prompt = self.prompt
        self.raw_negative_prompt = self.negative_prompt

        printF(name=MasterName.get_master_name(),
               info="[Parameters] refiner_swap_method = {}".format(self.refiner_swap_method)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] raw_prompt = {}".format(self.raw_prompt)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] raw_negative_prompt = {}".format(self.raw_negative_prompt)).printf()

        inpaint_worker.current_task = None
        self.inpaint_parameterized = self.inpaint_engine != 'None'

        self.inpaint_image = None
        self.inpaint_mask = None
        self.inpaint_head_model_path = None

        self.use_synthetic_refiner = False

        self.controlnet_canny_path = None
        self.controlnet_cpds_path = None
        self.clip_vision_path, self.ip_negative_path, self.ip_adapter_path, self.ip_adapter_face_path = None, None, None, None

        self.seed = self.image_seed
        if not isinstance(self.seed, int):
            self.seed = random.randint(1, self.max_seed)
        if self.seed < 0:
            self.seed = - self.seed
        self.seed = self.seed % self.max_seed
        printF(name=MasterName.get_master_name(), info="[Parameters] seed = {}".format(self.seed)).printf()

        resolution = self.aspect_ratios_selection
        if self.aspect_ratios_selection not in resolutions:
            try:
                resolution = annotate_resolution_string(self.aspect_ratios_selection)
            except Exception as e:
                print(f'Problem with resolution definition: "{resolution}", reverting to default: ' +
                      modules.settings.default_settings[
                          'resolution'])
                resolution = modules.settings.default_settings['resolution']
        self.width, self.height = string_to_dimensions(resolution)
        printF(name=MasterName.get_master_name(),
               info="[Parameters] width = {} , height = {} ".format(self.width, self.height)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] aspect_ratios_selection = {}".format(self.aspect_ratios_selection)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] switch = {}".format(self.switch)).printf()

    def download_image_func_models(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> download_image_func_models").printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] current_tab = {}".format(self.current_tab)).printf()
        if not self.image_factory_checkbox:
            printF(name=MasterName.get_master_name(), info="[Warning] image_factory_checkbox is not ENABLED!").printf()
        else:
            if (self.current_tab == 'uov' or (
                    self.current_tab == 'ip' and self.mixing_image_prompt_and_vary_upscale)) \
                    and self.uov_method != modules.flags.disabled and self.uov_input_image is not None:
                self.uov_input_image = HWC3(self.uov_input_image)
                if 'vary' in self.uov_method:
                    self.goals.append('vary')
                elif 'upscale' in self.uov_method:
                    self.goals.append('upscale')
                    if 'fast' in self.uov_method:
                        self.skip_prompt_processing = True
                    else:
                        self.steps = 18
                        if self.performance_selection == 'Speed':
                            self.steps = 18
                        if self.performance_selection == 'Quality':
                            self.steps = 36
                        if self.performance_selection == 'LCM':
                            self.steps = 8
                        if self.performance_selection == 'Turbo':
                            self.steps = 8
                        if self.performance_selection == 'Lightning':
                            self.steps = 4

                    self.progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()

            if (self.current_tab == 'inpaint' or (
                    self.current_tab == 'ip' and self.mixing_image_prompt_and_inpaint)) \
                    and isinstance(self.inpaint_input_image, dict):
                self.inpaint_image = self.inpaint_input_image['image']
                self.inpaint_mask = self.inpaint_input_image['mask'][:, :, 0]

                if self.inpaint_mask_upload_checkbox:
                    if isinstance(self.inpaint_mask_image_upload, np.ndarray):
                        if self.inpaint_mask_image_upload.ndim == 3:
                            H, W, C = self.inpaint_image.shape
                            inpaint_mask_image_upload = resample_image(self.inpaint_mask_image_upload, width=W,
                                                                       height=H)
                            inpaint_mask_image_upload = np.mean(inpaint_mask_image_upload, axis=2)
                            inpaint_mask_image_upload = (inpaint_mask_image_upload > 127).astype(np.uint8) * 255
                            self.inpaint_mask = np.maximum(self.inpaint_mask, inpaint_mask_image_upload)

                if int(self.inpaint_erode_or_dilate) != 0:
                    self.inpaint_mask = erode_or_dilate(self.inpaint_mask, self.inpaint_erode_or_dilate)

                if self.invert_mask_checkbox:
                    self.inpaint_mask = 255 - self.inpaint_mask

                self.inpaint_image = HWC3(self.inpaint_image)
                if isinstance(self.inpaint_image, np.ndarray) and isinstance(self.inpaint_mask, np.ndarray) \
                        and (np.any(self.inpaint_mask > 127) or len(self.outpaint_selections) > 0):
                    self.progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
                    if self.inpaint_parameterized:
                        self.progressbar(async_task, 1, 'Downloading inpainter ...')
                        self.inpaint_head_model_path, self.inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                            self.inpaint_engine)
                        self.base_model_additional_loras += [(True, self.inpaint_patch_model_path, 1.0)]
                        print(f'[Inpaint] Current inpaint model is {self.inpaint_patch_model_path}')

                    else:
                        self.inpaint_head_model_path, self.inpaint_patch_model_path = None, None
                        print(f'[Inpaint] Parameterized inpaint is disabled.')
                    if self.inpaint_additional_prompt != '':
                        if self.prompt == '':
                            self.prompt = self.inpaint_additional_prompt
                        else:
                            self.prompt = self.inpaint_additional_prompt + '\n' + self.prompt
                    self.goals.append('inpaint')
                    # sampler_name = 'dpmpp_2m_sde_gpu'  # only support the patched dpmpp_2m_sde_gpu
            if self.current_tab == 'ip' or \
                    self.mixing_image_prompt_and_inpaint or \
                    self.mixing_image_prompt_and_vary_upscale:
                self.goals.append('cn')
                self.progressbar(async_task, 1, 'Downloading control models ...')
                if len(self.cn_tasks[flags.cn_canny]) > 0:
                    self.controlnet_canny_path = modules.config.downloading_controlnet_canny()
                if len(self.cn_tasks[flags.cn_cpds]) > 0:
                    self.controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
                if len(self.cn_tasks[flags.cn_ip]) > 0:
                    self.clip_vision_path, self.ip_negative_path, self.ip_adapter_path = modules.config.downloading_ip_adapters(
                        'ip')
                if len(self.cn_tasks[flags.cn_ip_face]) > 0:
                    self.clip_vision_path, self.ip_negative_path, self.ip_adapter_face_path = modules.config.downloading_ip_adapters(
                        'face')

                printF(name=MasterName.get_master_name(),
                       info="[Parameters] use_synthetic_refiner = {}".format(self.use_synthetic_refiner)).printf()
                printF(name=MasterName.get_master_name(), info="[Parameters] steps = {}".format(self.steps)).printf()
                printF(name=MasterName.get_master_name(), info="[Parameters] prompt = {}".format(self.prompt)).printf()
                printF(name=MasterName.get_master_name(),
                       info="[Parameters] controlnet_canny_path = {}".format(self.controlnet_canny_path)).printf()
                printF(name=MasterName.get_master_name(),
                       info="[Parameters] controlnet_cpds_path = {}".format(self.controlnet_cpds_path)).printf()
                printF(name=MasterName.get_master_name(),
                       info="[Parameters] ip_adapter_face_path = {}".format(self.ip_adapter_face_path)).printf()
                printF(name=MasterName.get_master_name(),
                       info="[Parameters] ip_negative_path = {}".format(self.ip_negative_path)).printf()
                printF(name=MasterName.get_master_name(),
                       info="[Parameters] clip_vision_path = {}".format(self.clip_vision_path)).printf()

                self.progressbar(async_task, 1, 'Loading control models ...')
        printF(name=MasterName.get_master_name(),
                   info="[Parameters] self.goals = {}".format(self.goals)).printf()

    def manage_cns(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> manage_cns").printf()

        # ===== 关键修复：确保所有 gallery 都是列表 =====
        if not hasattr(self, 'output_gallery'):
            self.output_gallery = []
        self.input_gallery = self.input_gallery or []
        self.revision_gallery = self.revision_gallery or []
        self.output_gallery = self.output_gallery or []

        # 计算尺寸前先保证非 None
        self.input_gallery_size = len(self.input_gallery)
        self.revision_gallery_size = len(self.revision_gallery)

        if self.input_gallery_size == 0:
            self.img2img_mode = False
            self.input_image_path = None
            self.control_lora_canny = False
            self.control_lora_depth = False

        if self.revision_gallery_size == 0:
            self.revision_mode = False

        # Load or unload CNs
        printF(name=MasterName.get_master_name(),
               info="[Parameters] controlnet_canny_path = {}".format(self.controlnet_canny_path)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] controlnet_cpds_path = {}".format(self.controlnet_cpds_path)).printf()

        pipeline.refresh_controlnets([self.controlnet_canny_path, self.controlnet_cpds_path])
        ip_adapter.load_ip_adapter(self.clip_vision_path, self.ip_negative_path, self.ip_adapter_path)
        ip_adapter.load_ip_adapter(self.clip_vision_path, self.ip_negative_path, self.ip_adapter_face_path)
        
        
    def get_advanced_parameters(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> get_advanced_parameters").printf()

        if self.overwrite_step > 0:
            self.steps = self.overwrite_step

        if self.overwrite_switch > 0:
            self.switch = self.overwrite_switch

        if self.overwrite_width > 0:
            self.width = self.overwrite_width

        if self.overwrite_height > 0:
            self.height = self.overwrite_height

        printF(name=MasterName.get_master_name(),
               info="[Parameters] sampler_name = {}".format(self.sampler_name)).printf()
        printF(name=MasterName.get_master_name(),
               info="[Parameters] scheduler_name = {}".format(self.scheduler_name)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] steps = {}".format(self.steps)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] switch = {}".format(self.switch)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] width = {}".format(self.width)).printf()
        printF(name=MasterName.get_master_name(), info="[Parameters] height = {}".format(self.height)).printf()

    def get_image(self, path, megapixels=1.0):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> get_image").printf()
        image = None
        with open(path, 'rb') as image_file:
            pil_image = Image.open(image_file)
            image = ImageOps.exif_transpose(pil_image)
            image_file.close()
            image = image.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            image = modules.core.upscale(image, megapixels)
        return image

    def encode_prompts(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> encode_prompts").printf()
        self.progressbar(async_task, 1, 'Initializing ...')
        if not self.skip_prompt_processing:
            printF(name=MasterName.get_master_name(),
                   info="[Warning] Remove empty string of prompts and negative_prompts...").printf()
            prompts = remove_empty_str([safe_str(p) for p in self.prompt.split('\n')], default='')
            negative_prompts = remove_empty_str(
                [safe_str(p) for p in self.negative_prompt.splitlines()], default='')

            # ===== 换行合并：prompt 里的换行不再拆成多 task =====
            # 原行为：第 1 行是主 prompt，其余行进 extra_positive_prompts，
            # 会被 clip_encode 编译成多条 cond，process_diffusion 的多 prompt
            # 分支就会逐条递归采样 —— 一次点击出 4 张（每条换行一张）。
            # 新行为：把所有行合并成一条 prompt（逗号连接），只采一次。
            if len(prompts) > 1:
                printF(name=MasterName.get_master_name(),
                       info="[Prompt] {} 行 prompt 合并为 1 条 task（换行不再拆分）".format(
                           len(prompts))).printf()
                merged = ', '.join(p.strip() for p in prompts if p.strip())
                prompts = [merged]
            if len(negative_prompts) > 1:
                merged_neg = ', '.join(p.strip() for p in negative_prompts if p.strip())
                negative_prompts = [merged_neg]
            # ===== 换行合并结束 =====

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                self.use_expansion = False

            self.progressbar(async_task, 3, 'Loading models ...')
            pipeline.refresh_everything(base_model_name=self.base_model_name,
                                        loras=self.loras,
                                        base_model_additional_loras=self.base_model_additional_loras,
                                        use_synthetic_refiner=self.use_synthetic_refiner,
                                        performance_selection=Performance(self.performance_selection))

            self.progressbar(async_task, 3, 'Processing prompts ...')

            try:
                seed = int(self.image_seed)
            except Exception as e:
                seed = -1
            if not isinstance(seed, int) or seed < constants.MIN_SEED or seed > constants.MAX_SEED:
                seed = random.randint(constants.MIN_SEED, constants.MAX_SEED)

            printF(name=MasterName.get_master_name(),
                   info="[Parameters] base_model_name = {}".format(self.base_model_name)).printf()
            printF(name=MasterName.get_master_name(), info="[Parameters] loras = {}".format(self.loras)).printf()

            is_sdxl = pipeline.is_base_sdxl()

            # ===== 参考图能力判定（结构 / 语义） =====
            # 结构参考(img2img_gallery)：所有模型统一走 process_diffusion 的通用 VAE 初始
            #   latent，无需区分模型类型。
            # 语义参考(image_prompt_gallery)：
            #   - SDXL / SD1.5      -> CLIP-Vision（同时保留结构 img2img 回退）
            #   - Flux-Kontext / Krea2 / Flux2-Klein / Anima / Qwen-Edit
            #                       -> 引擎原生 ref_latents
            #   - 纯 Flux / ZImage / MOSS -> 无原生通道，退化为“用参考图做 img2img 初始图”
            try:
                pipeline.clear_engine_references()
            except Exception:
                pass
            self._ref_kind = 'clip_vision' if is_sdxl else pipeline.engine_reference_kind()
            self._engine_revision_used = 0
            self._semantic_fallback_path = None
            self._semantic_fallback_active = False
            self._semantic_denoise = 0.6

            if not is_sdxl:
                printF(name=MasterName.get_master_name(),
                       info="[Reference] non-SDXL model: ref_kind={}, control-lora(canny/depth) "
                            "仍仅限 SDXL，已自动关闭。".format(self._ref_kind)).printf()
                # Control-LoRA (canny/depth) 只对 SDXL 有意义，其它模型保持关闭，避免误报错。
                self.control_lora_canny = False
                self.control_lora_depth = False

            pipeline.set_clip_skips(self.base_clip_skip)
            if is_sdxl and self.revision_mode:
                try:
                    pipeline.refresh_clip_vision()
                except Exception as _e:
                    printF(name=MasterName.get_master_name(),
                           info="[Reference] refresh_clip_vision failed, 将只用 img2img 语义回退：{}".format(
                               _e)).printf()
            if self.control_lora_canny:
                pipeline.refresh_controlnet_canny(self.canny_model)
            if self.control_lora_depth:
                pipeline.refresh_controlnet_depth(self.depth_model)

            clip_vision_outputs = []
            revision_images_paths = []
            revision_strengths = []

            # ===== 结构参考（img2img_gallery）=====
            # 参考编辑型引擎（Kontext/Krea2/Klein）：把结构图写成引擎 ini_latent，
            # 采样时从噪声全量生成并由该图条件引导 —— 这是这类模型正确的“图生图”方式。
            # 其它模型仍走通用部分去噪初值（见 process_diffusion）。
            self._native_structure_ref = False
            _input_gallery = self.input_gallery or []
            if self.img2img_mode and len(_input_gallery) > 0 and self._ref_kind == 'ref_latents':
                _first_input_path = _gallery_entry_path(_input_gallery[0])
                if _first_input_path:
                    try:
                        _ini_used = pipeline.set_engine_references(
                            [self.get_image(_first_input_path)], as_input=True)
                    except Exception as _e:
                        _ini_used = 0
                        printF(name=MasterName.get_master_name(),
                               info="[Reference] 结构参考(ini_latent) 注入失败: {}".format(_e)).printf()
                    if _ini_used > 0:
                        self._native_structure_ref = True
                    printF(name=MasterName.get_master_name(),
                           info="[Reference] 结构图作为引擎 ini_latent 参考：{}".format(
                               self._native_structure_ref)).printf()

            # encode_prompts 在 manage_cns 之前执行，这里直接用真实 gallery 重新计算，
            # 避免 config 里的 revision_gallery_size 缺失/过期（可能为 None）。
            _revision_gallery = self.revision_gallery or []
            self.revision_gallery_size = len(_revision_gallery)
            if self.revision_gallery_size > 0:
                revision_images_paths = [p for p in map(_gallery_entry_path, _revision_gallery) if p]
                self.revision_images_filenames = list(
                    map(lambda path: os.path.basename(path) if path else path, revision_images_paths))
                revision_strengths = [self.revision_strength_1, self.revision_strength_2,
                                      self.revision_strength_3,
                                      self.revision_strength_4]
            else:
                self.revision_images_filenames = []

            if self.revision_mode and len(revision_images_paths) > 0:
                if is_sdxl:
                    # 旧路径：CLIP-Vision 语义嵌入（保留原有行为；本分支历史上未接入采样，
                    # 因此这里再补一条通用的 img2img 语义回退，确保 SDXL 的 image_prompt 真正生效）
                    for i in range(min(len(revision_images_paths), self.revision_gallery_size)):
                        print(f'Revision for image {i + 1} ...')
                        if revision_strengths[i % 4] != 0:
                            try:
                                revision_image = self.get_image(revision_images_paths[i])
                                clip_vision_output = modules.core.encode_clip_vision(pipeline.clip_vision,
                                                                                     revision_image)
                                clip_vision_outputs.append(clip_vision_output)
                            except Exception as _e:
                                printF(name=MasterName.get_master_name(),
                                       info="[Reference] CLIP-Vision encode failed for image {}: {}".format(
                                           i + 1, _e)).printf()
                        print(f'Revision for image {i + 1} finished')
                    self._semantic_fallback_path = revision_images_paths[0]
                    try:
                        _s = float(revision_strengths[0]) if revision_strengths else 1.0
                    except (TypeError, ValueError):
                        _s = 1.0
                    self._semantic_denoise = max(0.30, min(0.90, 0.45 + 0.20 * abs(_s)))
                elif self._ref_kind == 'ref_latents':
                    # 引擎原生语义参考：写入 ref_latents（Krea2 / Kontext / Klein / Anima / Qwen-Edit）
                    _ref_imgs = []
                    for i in range(min(len(revision_images_paths), self.revision_gallery_size)):
                        if revision_strengths[i % 4] != 0:
                            _ref_imgs.append(self.get_image(revision_images_paths[i]))
                    self._engine_revision_used = pipeline.set_engine_references(_ref_imgs, as_input=False)
                    printF(name=MasterName.get_master_name(),
                           info="[Reference] 语义参考已写入引擎 ref_latents：{} 张".format(
                               self._engine_revision_used)).printf()
                    if self._engine_revision_used == 0:
                        # 引擎路径失败 -> 退化为 img2img 语义回退，保证 image_prompt 不空转
                        self._semantic_fallback_path = revision_images_paths[0]
                        try:
                            _s = float(revision_strengths[0]) if revision_strengths else 1.0
                        except (TypeError, ValueError):
                            _s = 1.0
                        self._semantic_denoise = max(0.30, min(0.90, 0.45 + 0.20 * abs(_s)))
                        printF(name=MasterName.get_master_name(),
                               info="[Reference] 原生语义参考写入失败，退化为 img2img 语义回退"
                                    "（denoise={:.2f}）".format(self._semantic_denoise)).printf()
                else:
                    # 无原生语义参考通道：退化为“用参考图做 img2img 初始图”
                    self._semantic_fallback_path = revision_images_paths[0]
                    try:
                        _s = float(revision_strengths[0]) if revision_strengths else 1.0
                    except (TypeError, ValueError):
                        _s = 1.0
                    self._semantic_denoise = max(0.30, min(0.90, 0.45 + 0.20 * abs(_s)))
                    printF(name=MasterName.get_master_name(),
                           info="[Reference] 当前引擎无原生语义参考，语义图将作为 img2img 初始图"
                                "（denoise={:.2f}）。如需真正的语义参考，请使用 Krea2/Kontext/Edit 模型。".format(
                                    self._semantic_denoise)).printf()

            for i in range(self.image_number):
                if self.disable_seed_increment:
                    task_seed = seed
                else:
                    task_seed = (seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not

                task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future

                task_prompt = apply_wildcards(prompt, task_rng, i, self.read_wildcards_in_order)
                task_prompt = apply_arrays(task_prompt, i)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng, i, self.read_wildcards_in_order)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng, i, self.read_wildcards_in_order) for pmt
                                               in extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng, i, self.read_wildcards_in_order) for pmt
                                               in extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                if self.use_style:
                    for s in self.style_selections:
                        p, n = apply_style(s, positive=task_prompt)
                        positive_basic_workloads = positive_basic_workloads + p
                        negative_basic_workloads = negative_basic_workloads + n
                else:
                    positive_basic_workloads.append(task_prompt)

                negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

                positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
                negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

                positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
                negative_basic_workloads = remove_empty_str(negative_basic_workloads,
                                                            default=task_negative_prompt)

                self.tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='\n'.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='\n'.join([task_negative_prompt] + task_extra_negative_prompts),
                ))

            if self.use_expansion:
                for i, t in enumerate(self.tasks):
                    self.progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    printF(name=MasterName.get_master_name(),
                           info="[Prompt Expansion] New suffix: {}".format(expansion)).printf()
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            for i, t in enumerate(self.tasks):
                self.progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'], steps=self.steps)

            for i, t in enumerate(self.tasks):
                if abs(float(self.guidance_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    self.progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'],
                                                   steps=self.steps, is_negative_prompt=True)

            # 诊断：prompt 编码后参考通道最终状态。
            # anima.py 只在正 prompt 编码时把 ini_latent 转入 dynamic_args.ref_latents；
            # 若这里 ref_latents 为空，说明转移没发生（开关丢失/负 prompt 后被清空等）。
            try:
                from backend import args as _dargs
                _n_ref = len(getattr(_dargs.dynamic_args, 'ref_latents', []) or [])
                _ini = getattr(pipeline.final_model_ori, 'ini_latent', None) is not None if \
                    getattr(pipeline, 'final_model_ori', None) is not None else False
                printF(name=MasterName.get_master_name(),
                       info="[Reference][DIAG] after encode_prompts: dynamic_args.ref_latents={}, "
                            "engine.ini_latent={}, img2img_mode={}".format(
                                _n_ref, _ini, getattr(self, 'img2img_mode', None))).printf()
            except Exception as _e:
                printF(name=MasterName.get_master_name(),
                       info="[Reference][DIAG] failed: {}".format(_e)).printf()

    def check_vary_in_goals(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> check_vary_in_goals").printf()
        if len(self.goals) > 0:
            self.progressbar(async_task, 13, 'Image processing ...')

        if 'vary' in self.goals:
            if self.uov_input_image is None:
                self.goals.remove('vary')
                return
            if 'subtle' in self.uov_method:
                self.denoising_strength = 0.5
            if 'strong' in self.uov_method:
                self.denoising_strength = 0.85
            if self.overwrite_vary_strength > 0:
                self.denoising_strength = self.overwrite_vary_strength
            shape_ceil = get_image_shape_ceil(self.uov_input_image)
            if shape_ceil < 1024:
                print(f'[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                print(f'[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            uov_input_image = set_image_shape_ceil(self.uov_input_image, shape_ceil)
            initial_pixels = modules.core.numpy_to_pytorch(uov_input_image)

            printF(name=MasterName.get_master_name(),
                   info="[Parameters] denoising_strength = {}".format(self.denoising_strength)).printf()
            printF(name=MasterName.get_master_name(),
                   info="[Parameters] initial_pixels = {}".format(initial_pixels)).printf()

            self.progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=self.steps,
                switch=self.switch,
                denoise=self.denoising_strength,
                refiner_swap_method=self.refiner_swap_method
            )

            self.initial_latent = modules.core.encode_vae(vae=candidate_vae, pixels=initial_pixels)

            B, C, H, W = self.initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            printF(name=MasterName.get_master_name(), info="[Parameters] width = {}".format(width)).printf()
            printF(name=MasterName.get_master_name(), info="[Parameters] height = {}".format(height)).printf()

    def check_upscale_in_goals(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> check_upscale_in_goals").printf()
        if 'upscale' in self.goals:
            if self.uov_input_image is None:
                self.goals.remove('upscale')
                return
            H, W, C = self.uov_input_image.shape
            self.progressbar(async_task, 13, f'Upscaling image from {str((H, W))} ...')

            uov_input_image = perform_upscale(self.uov_input_image)
            printF(name=MasterName.get_master_name(), info="[Warning] Image upscaled.").printf()

            if '1.5x' in self.uov_method:
                f = 1.5
            elif '2x' in self.uov_method:
                f = 2.0
            else:
                f = 1.0

            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                printF(name=MasterName.get_master_name(), info="[Upscale] Image is resized because it is too small.").printf()
                uov_input_image = set_image_shape_ceil(uov_input_image, 1024)
                shape_ceil = 1024
            else:
                uov_input_image = resample_image(uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

            if 'fast' in self.uov_method:
                self.direct_return = True
            elif image_is_super_large:
                printF(name=MasterName.get_master_name(),
                       info="[Warning] Image is too large. Directly returned the SR image. Usually directly return SR image at 4K resolution.yields better results than SDXL diffusion.").printf()
                self.direct_return = True
            else:
                self.direct_return = False

            if self.direct_return:
                d = [('Upscale (Fast)', 'upscale_fast', '2x')]
                uov_input_image_path = log(uov_input_image, d, output_format=self.output_format)
                self.yield_result(async_task, uov_input_image_path, do_not_show_finished_images=True)
                return

            self.tiled = True
            self.denoising_strength = 0.382

            if self.overwrite_upscale_strength > 0:
                self.denoising_strength = self.overwrite_upscale_strength

            initial_pixels = modules.core.numpy_to_pytorch(uov_input_image)
            self.progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=self.steps,
                switch=self.switch,
                denoise=self.denoising_strength,
                refiner_swap_method=self.refiner_swap_method
            )

            self.initial_latent = modules.core.encode_vae(
                vae=candidate_vae,
                pixels=initial_pixels, tiled=True)

            B, C, H, W = self.initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            printF(name=MasterName.get_master_name(),
                   info="[Parameters] denoising_strength = {}".format(self.denoising_strength)).printf()
            printF(name=MasterName.get_master_name(), info="[Parameters] width = {}".format(width)).printf()
            printF(name=MasterName.get_master_name(), info="[Parameters] height = {}".format(height)).printf()

    def check_inpaint_in_goals(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> check_inpaint_in_goals").printf()
        if 'inpaint' in self.goals:
            if len(self.outpaint_selections) > 0:
                H, W, C = self.inpaint_image.shape
                if 'top' in self.outpaint_selections:
                    self.inpaint_image = np.pad(self.inpaint_image,
                                                [[int(H * self.outpaint_expansion_ratio), 0], [0, 0], [0, 0]],
                                                mode='edge')
                    self.inpaint_mask = np.pad(self.inpaint_mask,
                                               [[int(H * self.outpaint_expansion_ratio), 0], [0, 0]],
                                               mode='constant',
                                               constant_values=255)
                if 'bottom' in self.outpaint_selections:
                    self.inpaint_image = np.pad(self.inpaint_image,
                                                [[0, int(H * self.outpaint_expansion_ratio)], [0, 0], [0, 0]],
                                                mode='edge')
                    self.inpaint_mask = np.pad(self.inpaint_mask,
                                               [[0, int(H * self.outpaint_expansion_ratio)], [0, 0]],
                                               mode='constant',
                                               constant_values=255)

                H, W, C = self.inpaint_image.shape
                if 'left' in self.outpaint_selections:
                    self.inpaint_image = np.pad(self.inpaint_image,
                                                [[0, 0], [int(W * self.outpaint_expansion_ratio), 0], [0, 0]],
                                                mode='edge')
                    self.inpaint_mask = np.pad(self.inpaint_mask,
                                               [[0, 0], [int(W * self.outpaint_expansion_ratio), 0]],
                                               mode='constant',
                                               constant_values=255)
                if 'right' in self.outpaint_selections:
                    self.inpaint_image = np.pad(self.inpaint_image,
                                                [[0, 0], [0, int(W * self.outpaint_expansion_ratio)], [0, 0]],
                                                mode='edge')
                    self.inpaint_mask = np.pad(self.inpaint_mask,
                                               [[0, 0], [0, int(W * self.outpaint_expansion_ratio)]],
                                               mode='constant',
                                               constant_values=255)

                self.inpaint_image = np.ascontiguousarray(self.inpaint_image.copy())
                self.inpaint_mask = np.ascontiguousarray(self.inpaint_mask.copy())

                self.inpaint_strength = 1.0
                self.inpaint_respective_field = 1.0

            self.denoising_strength = self.inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=self.inpaint_image,
                mask=self.inpaint_mask,
                use_fill=self.denoising_strength > 0.99,
                k=self.inpaint_respective_field
            )

            if self.debugging_inpaint_preprocessor:
                self.yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(),
                                  do_not_show_finished_images=True)
                return

            self.progressbar(async_task, 13, 'VAE Inpaint encoding ...')

            inpaint_pixel_fill = modules.core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = modules.core.numpy_to_pytorch(
                inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = modules.core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=self.steps,
                switch=self.switch,
                denoise=self.denoising_strength,
                refiner_swap_method=self.refiner_swap_method
            )

            latent_inpaint, latent_mask = modules.core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                self.progressbar(async_task, 13, 'VAE SD15 encoding ...')
                latent_swap = modules.core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            self.progressbar(async_task, 13, 'VAE encoding ...')
            latent_fill = modules.core.encode_vae(
                vae=candidate_vae,
                pixels=inpaint_pixel_fill)['samples']

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if self.inpaint_parameterized:
                pipeline.final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=self.inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=pipeline.final_unet
                )

            if not self.inpaint_disable_initial_latent:
                self.initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            self.height, self.width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            printF(name=MasterName.get_master_name(),
                   info="[Parameters] denoising_strength = {}".format(self.denoising_strength)).printf()
            printF(name=MasterName.get_master_name(),
                   info="[Parameters] image: final_width = {} , final_height = {} ".format(final_width, final_height)).printf()
            printF(name=MasterName.get_master_name(),
                   info="[Parameters] latent: width = {},  height = {}".format(self.width, self.height)).printf()

    def check_cn_in_goals(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> check_cn_in_goals").printf()
        if 'cn' in self.goals:
            printF(name=MasterName.get_master_name(),
                   info="[Parameters] skipping_cn_preprocessor = {}".format(self.skipping_cn_preprocessor)).printf()
            for task in self.cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=self.width, height=self.height)

                if not self.skipping_cn_preprocessor:
                    cn_img = preprocessors.canny_pyramid(cn_img, self.canny_low_threshold, self.canny_high_threshold)

                cn_img = HWC3(cn_img)
                task[0] = modules.core.numpy_to_pytorch(cn_img)
                if self.debugging_cn_preprocessor:
                    self.yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in self.cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=self.width, height=self.height)

                if not self.skipping_cn_preprocessor:
                    cn_img = preprocessors.cpds(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = modules.core.numpy_to_pytorch(cn_img)
                if self.debugging_cn_preprocessor:
                    self.yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in self.cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=self.ip_adapter_path)
                if self.debugging_cn_preprocessor:
                    self.yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in self.cn_tasks[flags.cn_ip_face]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                if not self.skipping_cn_preprocessor:
                    cn_img = extras.face_crop.crop_image(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)
                print(f"self.ip_adapter_face_path:{self.ip_adapter_face_path}")
                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=self.ip_adapter_face_path)
                if self.debugging_cn_preprocessor:
                    self.yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return

            all_ip_tasks = self.cn_tasks[flags.cn_ip] + self.cn_tasks[flags.cn_ip_face]

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

    def generate_images(self, async_task):
        global worker_queue, queue_task
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> generate_images").printf()
        self.execution_start_time = time.perf_counter()
        if self.freeu_enabled:
            printF(name=MasterName.get_master_name(), info="[Warning] FreeU ENABLED!").printf()
            pipeline.final_unet = modules.core.apply_freeu(
                pipeline.final_unet,
                self.freeu_b1,
                self.freeu_b2,
                self.freeu_s1,
                self.freeu_s2
            )

        all_steps = self.steps * self.image_number

        printF(name=MasterName.get_master_name(),
               info="[Parameters] denoising_strength = {}".format(self.denoising_strength)).printf()

        if isinstance(self.initial_latent, dict) and 'samples' in self.initial_latent:
            log_shape = self.initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(self.height, self.width)}'

        printF(name=MasterName.get_master_name(),
               info="[Parameters] Initial Latent shape: {}".format(log_shape)).printf()

        preparation_time = time.perf_counter() - self.execution_start_time
        print(f'')
        printF(name=MasterName.get_master_name(),
               info="Preparation time: {:.2f} seconds".format(preparation_time)).printf()

        final_sampler_name = self.sampler_name
        final_scheduler_name = self.scheduler_name

        # ===== 按模型族强制套用预设 sampler/scheduler =====
        # UI 的“Model Type Selector”切换时靠 Gradio 事件改下拉框，不一定生效；
        # switch_sampler 打开（默认由预设链设为 True）时这里以后端预设为准。
        if self.switch_sampler:
            try:
                _pairs = config.get_model_type_sampler_pairs(getattr(self, "default_model_type", None))
                _samp_ok = {str(s) for s in (flags.sampler_list or [])}
                _sched_ok = {str(s) for s in (flags.scheduler_list or [])}
                for _s, _d in _pairs:
                    if _s in _samp_ok and _d in _sched_ok:
                        if (_s, _d) != (final_sampler_name, final_scheduler_name):
                            printF(name=MasterName.get_master_name(),
                                   info="[Sampler] model_type={} 套用预设：sampler {} -> {} | "
                                        "scheduler {} -> {}".format(
                                            getattr(self, "default_model_type", None),
                                            final_sampler_name, _s,
                                            final_scheduler_name, _d)).printf()
                        final_sampler_name, final_scheduler_name = _s, _d
                        break
            except Exception as _e:
                printF(name=MasterName.get_master_name(),
                       info="[Sampler] 预设 sampler/scheduler 套用失败: {}".format(_e)).printf()

        if self.scheduler_name in ['lcm', 'tcd', 'Lightning']:
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = modules.core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling=self.scheduler_name,
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = modules.core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling=self.scheduler_name,
                    zsnr=False)[0]
        elif self.scheduler_name == 'edm_playground_v2.5':
            final_scheduler_name = 'karras'

            def patch_edm(unet):
                return core.opModelSamplingContinuousEDM.patch(
                    unet,
                    sampling=self.scheduler_name,
                    sigma_max=120.0,
                    sigma_min=0.002)[0]

            if pipeline.final_unet is not None:
                pipeline.final_unet = patch_edm(pipeline.final_unet)
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = patch_edm(pipeline.final_refiner_unet)

        printF(name=MasterName.get_master_name(), info="[Warning] Using {} scheduler.".format(self.scheduler_name)).printf()

        printF(name=MasterName.get_master_name(),
               info="Moving model to GPU ...for image No. {} process!".format(self.image_number)).printf()
        async_task.yields.append(['preview', (13, 'Moving model to GPU ...', None, 0, self.image_number)])

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * self.steps + step
            img_p = current_task_id + 1
            img_r = self.image_number - img_p
            async_task.yields.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}{ordinal_suffix(current_task_id + 1)} Sampling',
                y, img_p, img_r)])

        # print(f"self.tasks: {self.tasks}")
        for current_task_id, task in enumerate(self.tasks):
            execution_start_time = time.perf_counter()

            _structure_selected = (self.img2img_mode or self.control_lora_canny or self.control_lora_depth) \
                and self.input_gallery_size > 0
            self._semantic_fallback_active = False

            if self._native_structure_ref:
                # 参考编辑型引擎：结构图已在 encode_prompts 写成 ini_latent，
                # 这里从噪声全量生成（denoise=1），不再需要 img2img 初值。
                self.input_image_path = None
                self.input_image_filename = None
                self.start_step = 0
                self.denoise = 1.0
                self._effective_img2img = False

                # ===== 分辨率对齐：输出比例跟随结构参考图 =====
                # 否则参考图（如横向）被 interpolate 拉到 UI 比例（如竖向 832x1216），
                # 人物会被压扁/拉长 → 畸形。这里保持总像素不变，只改宽高比。
                try:
                    _ref_path = _gallery_entry_path((self.input_gallery or [])[0]) if \
                        (self.input_gallery or []) else None
                    if _ref_path and os.path.isfile(_ref_path):
                        with Image.open(_ref_path) as _pil:
                            _iw, _ih = ImageOps.exif_transpose(_pil).size
                        _target_mp = float(self.width) * float(self.height)
                        _ratio = float(_iw) / max(1, float(_ih))
                        _new_h = int(round((_target_mp / _ratio) ** 0.5 / 16.0) * 16)
                        _new_w = int(round((_target_mp * _ratio) ** 0.5 / 16.0) * 16)
                        _new_h = max(256, _new_h)
                        _new_w = max(256, _new_w)
                        if (_new_w, _new_h) != (int(self.width), int(self.height)):
                            printF(name=MasterName.get_master_name(),
                                   info="[Reference] 分辨率对齐参考图比例：{}x{} -> {}x{} "
                                        "（避免参考图被拉伸 → 畸形）".format(
                                            self.width, self.height, _new_w, _new_h)).printf()
                            self.width, self.height = _new_w, _new_h
                except Exception as _e:
                    printF(name=MasterName.get_master_name(),
                           info="[Reference] 分辨率对齐失败，沿用 UI 比例：{}".format(_e)).printf()
            else:
                if _structure_selected:
                    input_gallery_entry = self.input_gallery[current_task_id % self.input_gallery_size]
                    self.input_image_path = _gallery_entry_path(input_gallery_entry)
                    self.input_image_filename = None if self.input_image_path is None else os.path.basename(
                        self.input_image_path)
                elif self.revision_mode and self._semantic_fallback_path is not None:
                    # 引擎无原生语义参考：把 image_prompt_gallery 的参考图当作 img2img 初始图，
                    # 用中等 denoise 实现“参考图语义”图生图（对纯 Flux / ZImage / MOSS / SDXL 通用）。
                    self.input_image_path = self._semantic_fallback_path
                    self.input_image_filename = os.path.basename(self.input_image_path)
                    self._semantic_fallback_active = True
                else:
                    self.input_image_path = None
                    self.input_image_filename = None
                    self.keep_input_names = None

                if self.img2img_mode:
                    self.start_step = round(self.steps * self.img2img_start_step)
                    self.denoise = self.img2img_denoise
                    # ===== 按模型族动态校正 img2img denoise =====
                    # flow（Flux/ZImage/Anima/Moss…）的起始 sigma ≈ denoise、
                    # 初始 latent 权重 = 1 - denoise；SD/SDXL 是 A1111 语义；
                    # 蒸馏模型预算更少。同一个数值在不同模型上效果完全不同，
                    # 所以这里统一按 model_type_selector 校正（表见 modules/config.py）。
                    if self.input_image_path is not None:
                        self.denoise, _i2i_msg = resolve_img2img_denoise(
                            getattr(self, 'default_model_type', None),
                            self.base_model_name,
                            self.denoise)
                        printF(name=MasterName.get_master_name(),
                               info=_i2i_msg).printf()
                elif self._semantic_fallback_active:
                    self.start_step = 0
                    self.denoise = self._semantic_denoise
                else:
                    self.start_step = 0
                    self.denoise = self.denoising_strength

                # 传给 process_diffusion 的 img2img 开关：结构 img2img 或语义回退时都启用
                self._effective_img2img = bool(self.img2img_mode or self._semantic_fallback_active)

            input_image = None
            is_sdxl = pipeline.is_base_sdxl()

            # ===== 蒸馏模型 CFG 钐制（识别模型真实结构） =====
            # Klein / Schnell 是 guidance-embed 蒸馏模型，引导已内化在权重里：
            # 任何 >1.0 的外部 CFG 都是双重引导 → 过曝、色彩失真。
            # Base 模型不在名单内，保留用户的 CFG 设置。
            _bm_name = str(getattr(self, 'base_model_name', '') or '').lower()
            # klein_9b_base / klein_4b_base 等非蒸馏 Klein 变体归为 Base
            _is_klein_base = ('klein' in _bm_name) and not any(
                k in _bm_name for k in ('distill', 'schnell', 'turbo', 'dmd', 'lightning', 'base'))
            # 蒸馏判定必须在 Klein Base 排除之后：'klein_base' 不能命中 distilled
            _is_distilled = ('schnell' in _bm_name) or (
                ('klein' in _bm_name) and not _is_klein_base)

            if _is_distilled and float(self.guidance_scale) > 1.0:
                printF(name=MasterName.get_master_name(),
                       info="[CFG-Guard] 蒸馏模型 {} 检测到 guidance_scale={:.2f} > 1.0，"
                            "强制钳制为 1.0（防止双重引导 → 过曝）".format(
                                self.base_model_name, self.guidance_scale)).printf()
                self.guidance_scale = 1.0

            # ===== img2img 步数保障（Base 模型） =====
            # Base 模型引导未内化，img2img 需要 ≥20 步的去噪预算来重建结构；
            # 蒸馏版 4-10 步就够，不在此名单。用户手动设了更高步数则尊重用户。
            if (getattr(self, 'img2img_mode', False) and not _is_distilled
                    and int(self.steps) < 20):
                printF(name=MasterName.get_master_name(),
                       info="[Steps-Guard] Base 模型 {} img2img 检测到 steps={} < 20，"
                            "自动提升为 24（去噪预算不足 → 结构重建失败/形体崩坏）".format(
                                self.base_model_name, self.steps)).printf()
                self.steps = 24

            if self.input_image_path is not None:
                img2img_megapixels = self.width * self.height * self.img2img_scale ** 2 / 2 ** 20
                min_mp = constants.MIN_MEGAPIXELS if is_sdxl else constants.MIN_MEGAPIXELS_SD
                max_mp = constants.MAX_MEGAPIXELS if is_sdxl else constants.MAX_MEGAPIXELS_SD
                if img2img_megapixels < min_mp:
                    img2img_megapixels = min_mp
                elif img2img_megapixels > max_mp:
                    img2img_megapixels = max_mp
                input_image = self.get_image(path=self.input_image_path, megapixels=img2img_megapixels)

                # 结构参考 + 语义参考同时提供、且引擎没有原生语义通道时：
                # 把语义参考图融合进初始图，让 image_prompt_gallery 也生效。
                if (self.img2img_mode and self.revision_mode
                        and self._engine_revision_used == 0
                        and self._semantic_fallback_path is not None
                        and self.input_image_path != self._semantic_fallback_path):
                    try:
                        _sem_img = self.get_image(path=self._semantic_fallback_path,
                                                  megapixels=img2img_megapixels)
                        _s = float(self.revision_strength_1)
                        _w = max(0.15, min(0.85, 0.5 + 0.15 * _s))
                        input_image = _blend_init_images(input_image, _sem_img, semantic_weight=_w)
                        printF(name=MasterName.get_master_name(),
                               info="[Reference] 结构图与语义图已融合作为初始图（semantic_weight={:.2f}）".format(
                                   _w)).printf()
                    except Exception as e:
                        printF(name=MasterName.get_master_name(),
                               info="[Reference] 融合语义图失败，忽略：{}".format(e)).printf()

            if "kolors" in self.base_model_name_prefix.lower():
                self.task_class = "Kolors+"
            elif "flux" in self.base_model_name_prefix.lower():
                self.task_class = "Flux"
            elif "hydit" in self.base_model_name_prefix.lower():
                self.task_class = "HyDiT+"
            elif "sd3x" in self.base_model_name_prefix.lower():
                self.task_class = "SD3x"
            else:
                self.task_class = "default"

            try:
                if async_task.last_stop is not False:
                    ldm_patched.modules.model_management.interrupt_current_processing()
                positive_cond, negative_cond = task['c'], task['uc']
                
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] image_number = {}".format(self.image_number)).printf()
                printF(name=MasterName.get_master_name(), info="[Parameters] steps = {}".format(self.steps)).printf()
                printF(name=MasterName.get_master_name(), info="[Parameters] switch = {}".format(self.switch)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] img2img_mode = {}".format(self.img2img_mode)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] input_image = {}".format(input_image)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] start_step = {}".format(self.start_step)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] control_lora_canny = {}".format(self.control_lora_canny)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] canny_edge_low = {}".format(self.canny_edge_low)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] canny_edge_high = {}".format(self.canny_edge_high)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] canny_start = {}".format(self.canny_start)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] canny_stop = {}".format(self.canny_stop)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] canny_strength = {}".format(self.canny_strength)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] control_lora_depth = {}".format(self.control_lora_depth)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] depth_start = {}".format(self.depth_start)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] depth_stop = {}".format(self.depth_stop)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] depth_strength = {}".format(self.depth_strength)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] task['task_seed'] = {}".format(task['task_seed'])).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] final_sampler_name = {}".format(final_sampler_name)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] final_scheduler_name = {}".format(final_scheduler_name)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] initial_latent = {}".format(self.initial_latent)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] denoise = {}".format(self.denoise)).printf()
                printF(name=MasterName.get_master_name(), info="[Parameters] tiled = {}".format(self.tiled)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] guidance_scale = {}".format(self.guidance_scale)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] refiner_swap_method = {}".format(self.refiner_swap_method)).printf()
                printF(name=MasterName.get_master_name(),
                    info="[Parameters] loras = {}".format(self.loras)).printf()
                               

                # ===== comfy 分支的门闸：三重条件缺一不可 =====
                #
                # 之前写法是 `if self.task_class in flags.comfy_classes:`，
                # 只要 flags.comfy_classes 含 "default"（实测就是），
                # 所有 SDXL 出图都会进 comfy 分支，然后连锁 AttributeError：
                #   1) self.params_backend      （已补）
                #   2) self.task_name = None    （已补）
                #   3) async_task.task_method   （本轮）
                # 继续补单点只会在 get_comfy_task 内部 / process_flow 内部继续炸。
                #
                # 正确做法：comfyd_active_checkbox 是用户的显式意愿；
                #           没开就绝不进 comfy。
                #           task_name / task_method 是 comfy 工作流必须的键，
                #           不存在就不进。
                _task_method = getattr(async_task, 'task_method', None)
                _use_comfy = (
                    bool(self.comfyd_active_checkbox)
                    and self.task_class in flags.comfy_classes
                    and self.task_name is not None
                    and _task_method is not None
                )

                if _use_comfy:
                    printF(name=MasterName.get_master_name(),
                           info="[Comfy] Enter comfy backend: task_class={}, "
                                "task_name={}, task_method={}".format(
                                    self.task_class, self.task_name, _task_method)).printf()

                    combined_positive = ", ".join([x for x in task["positive"] if isinstance(x, str) and x.strip()])
                    combined_negative = ", ".join([x for x in task["negative"] if isinstance(x, str) and x.strip()])

                    default_params = dict(
                        prompt=combined_positive,
                        negative_prompt=combined_negative,
                        width=self.width,
                        height=self.height,
                        base_model=self.base_model_name,
                        sampler=self.sampler_name,
                        scheduler=final_scheduler_name,
                        cfg_scale=self.cfg_scale,
                        steps=self.steps,
                        denoise=self.denoising_strength,
                        seed=task['task_seed'],
                        )
                    default_params.update(self.params_backend or {})

                    comfy_task = get_comfy_task(self.task_name, _task_method,
                                                default_params, input_image)
                    self.imgs = comfypipeline.process_flow(
                        comfy_task.name, comfy_task.params, comfy_task.images,
                        callback=callback)

                    del task['c'], task['uc'], combined_positive, combined_negative  # Save memory

                else:
                    # 本地 Forge 路径（SDXL / 没启用 comfy 的 FLUX 都走这里）
                    if self.comfyd_active_checkbox and self.task_class in flags.comfy_classes \
                            and not _use_comfy:
                        # 用户开了 comfy，但配置不完整——主动吼出来，别静默回落
                        printF(name=MasterName.get_master_name(),
                               info="[Comfy][WARN] comfyd_active_checkbox={} "
                                    "task_class={} in comfy_classes={} "
                                    "task_name={} task_method={} "
                                    "—— 配置不全，自动回落到本地 pipeline。".format(
                                        self.comfyd_active_checkbox,
                                        self.task_class,
                                        self.task_class in flags.comfy_classes,
                                        self.task_name, _task_method)).printf()
                
                    if pipeline.model_base is not None:
                        # Resolve the active inner model
                        active_model = getattr(pipeline.model_base,
                            'inner_model', pipeline.model_base)
                        use_tiling = getattr(async_task,
                            'seamless_tiling', False)
                        util.apply_native_tiling(active_model,
                            enable=use_tiling)

                    if 'cn' in self.goals:
                        for cn_flag, cn_path in [
                            (flags.cn_canny, self.controlnet_canny_path),
                            (flags.cn_cpds, self.controlnet_cpds_path)
                        ]:
                            for cn_img, cn_stop, cn_weight in self.cn_tasks[cn_flag]:
                                positive_cond, negative_cond = modules.core.apply_controlnet(
                                    positive_cond, negative_cond,
                                    pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)
                    self.execution_end_time = time.perf_counter()

                    # printF(name=MasterName.get_master_name(),
                    #     info="[Parameters] positive_cond = {}".format(positive_cond)).printf()
                    # printF(name=MasterName.get_master_name(),
                    #     info="[Parameters] negative_cond = {}".format(negative_cond)).printf()


                    self.imgs = pipeline.process_diffusion(
                        positive_cond=positive_cond,
                        negative_cond=negative_cond,
                        steps=self.steps,
                        switch=self.switch,
                        width=self.width,
                        img2img=self._effective_img2img,  # 结构 img2img 或语义回退
                        input_image=input_image,  # ? -> latent
                        start_step=self.start_step,
                        control_lora_canny=self.control_lora_canny,
                        canny_edge_low=self.canny_edge_low,
                        canny_edge_high=self.canny_edge_high,
                        canny_start=self.canny_start,
                        canny_stop=self.canny_stop,
                        canny_strength=self.canny_strength,
                        control_lora_depth=self.control_lora_depth,
                        depth_start=self.depth_start,
                        depth_stop=self.depth_stop,
                        depth_strength=self.depth_strength,
                        height=self.height,
                        image_seed=task['task_seed'],
                        callback=callback,
                        sampler_name=final_sampler_name,
                        scheduler_name=final_scheduler_name,
                        latent=self.initial_latent,
                        denoise=self.denoise,
                        tiled=self.tiled,
                        cfg_scale=self.guidance_scale,
                        refiner_swap_method=self.refiner_swap_method
                    )

                    del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if modules.inpaint_worker.current_task is not None:
                    self.imgs = [modules.inpaint_worker.current_task.post_process(x) for x in self.imgs]

                self.log_meta_messages(task=task)
                self.yield_result(async_task, self.img_paths,
                                do_not_show_finished_images=len(self.tasks) == 1 or self.disable_intermediate_results)

            except ldm_patched.modules.model_management.InterruptProcessingException as e:
                if async_task.last_stop == 'skip':
                    print('User skipped')
                    async_task.last_stop = False
                    continue
                else:
                    print('User stopped')
                    if self.request_source == "api":
                        self.results.append(ImageGenerationResult(
                            im=None, seed=task['task_seed'], finish_reason=GenerationFinishReason.user_cancel))
                        queue_task.set_result(task_result=self.results, finish_with_error=True, error_message=str(e))
                    break
            except Exception as e:
                printF(name=MasterName.get_master_name(),
                       info="[Error] Process error: {}".format(e)).printf()
                logging.exception(e)
                if self.request_source == "api":
                    self.results.append(ImageGenerationResult(
                        im=None, seed=task['task_seed'], finish_reason=GenerationFinishReason.error))
                    queue_task.set_result(task_result=self.results, finish_with_error=True, error_message=str(e))
                break

            execution_time = time.perf_counter() - execution_start_time
            printF(name=MasterName.get_master_name(),
                   info="[Info] Generating and Saving time: {:.2f} seconds".format(execution_time)).printf()

    def get_meta_messages(self, task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> get_meta_messages").printf()
        execution_time = time.perf_counter() - self.execution_start_time
        printF(name=MasterName.get_master_name(),
               info="[Info] Diffusion time: {:.2f} seconds".format(execution_time)).printf()

        metadata_string = [('Prompt', 'prompt', task['log_positive_prompt']),
                           ('Negative Prompt', 'negative_prompt', task['log_negative_prompt']),
                           ('Fooocus V2 Expansion', 'prompt_expansion', task['expansion']),
                           ('Styles', 'styles', str(self.raw_style_selections)),
                           ('Real Prompt', 'real_prompt', task['positive']),
                           ('Real Negative Prompt', 'real_negative_prompt', task['negative']),
                           ('Performance', 'performance', Performance(self.performance_selection).value)]

        if Performance(self.performance_selection).steps() != self.steps:
            metadata_string.append(('Steps', 'steps', self.steps))

        metadata_string += [('Resolution', 'resolution', config.add_ratio(f"{self.width}*{self.height}")),
                            ('Guidance Scale', 'guidance_scale', self.guidance_scale),
                            ('Sharpness', 'sharpness', self.sharpness),
                            ('ADM Guidance', 'adm_guidance', str((
                                modules.patch.patch_settings[self.pid].positive_adm_scale,
                                modules.patch.patch_settings[self.pid].negative_adm_scale,
                                modules.patch.patch_settings[self.pid].adm_scaler_end))),
                            ('Model Type', 'model_type_selector', self.default_model_type),
                            ('Base Model', 'base_model', self.base_model_name),
                            ('CFG & CLIP Skips', "cfg_clip_skips",
                             (self.guidance_scale, self.base_clip_skip)),
                            ('Image-2-Image', "image_2_image",
                             (self.img2img_mode, self.start_step, self.denoise, self.img2img_scale,
                              self.input_image_filename) if self.img2img_mode else (
                                 self.img2img_mode)),
                            ('Revision', "revision",
                             (self.revision_mode, self.revision_strength_1, self.revision_strength_2,
                              self.revision_strength_3,
                              self.revision_strength_4,
                              self.revision_images_filenames) if self.revision_mode else (
                                 self.revision_mode)),
                            ('Prompt Strengths', 'prompt_strengths',
                             (self.positive_prompt_strength, self.negative_prompt_strength)),
                            ('Canny', 'canny',
                             (self.control_lora_canny, self.canny_edge_low, self.canny_edge_high, self.canny_start,
                              self.canny_stop,
                              self.canny_strength, self.canny_model,
                              self.input_image_filename) if self.control_lora_canny else (
                                 self.control_lora_canny)),
                            ('Depth', 'depth',
                             (self.control_lora_depth, self.depth_start, self.depth_stop, self.depth_strength,
                              self.depth_model,
                              self.input_image_filename) if self.control_lora_depth else self.control_lora_depth)
                            ]

        if modules.patch.patch_settings[self.pid].adaptive_cfg != modules.config.default_cfg_tsnr:
            metadata_string.append(
                ('CFG Mimicking from TSNR', 'adaptive_cfg', modules.patch.patch_settings[self.pid].adaptive_cfg))

        metadata_string.append(('Sampler', 'sampler', self.sampler_name))
        metadata_string.append(('Scheduler', 'scheduler', self.scheduler_name))
        metadata_string.append(('Seed', 'seed', str(task['task_seed'])))

        if self.freeu_enabled:
            metadata_string.append(
                ('FreeU', 'freeu', str((self.freeu_b1, self.freeu_b2, self.freeu_s1, self.freeu_s2))))

        if self.loras == [[]]:
            metadata_string.append(
                ('LoRA', 'lora', 'none:none'))
        else:
            for idx, mmm in enumerate(self.loras):
                if mmm[1] not in ['None', 'NONE', "Not Exist!->"]:
                    metadata_string.append((f'LoRA {idx + 1}', f'lora_combined_{idx + 1}', f'{mmm[1]}:{mmm[2]}'))
        execution_time = time.perf_counter() - self.execution_start_time
        metadata_string.append(('Execution Time', 'time', f'{execution_time:.2f} seconds'))

        metadata_string.append(('Metadata Scheme', 'metadata_scheme',
                                self.metadata_scheme.value if self.save_metadata_to_images else self.save_metadata_to_images))
        metadata_string.append(('Version', 'version', 'MeanVon v' + main_version))

        printF(name=MasterName.get_master_name(),
               info="[Parameters] metadata_string = {}".format(metadata_string)).printf()

        metadata_dict = {}
        for i in metadata_string:
            metadata_dict.update({
                i[1]: str(i[2])
            })
        self.metadata_strings = json.dumps(metadata_dict, ensure_ascii=False)

        return metadata_string

    def log_meta_messages(self, task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> log_meta_messages").printf()
        import modules.patch
        self.img_paths = []

        for x in self.imgs:
            d = self.get_meta_messages(task)
            metadata_parser = None
            if self.save_metadata_to_images:
                metadata_parser = modules.meta_parser.get_metadata_parser(self.metadata_scheme)
                metadata_parser.set_data(task['log_positive_prompt'], task['positive'],
                                         task['log_negative_prompt'], task['negative'],
                                         self.steps, self.base_model_name, self.loras)
            self.base_model_name_prefix = self.base_model_name.split(".")[0]

            self.img_paths.append(
                log(x, d, metadata_parser, self.output_format, self.save_metadata_json, self.input_image_filename,
                    self.keep_input_names,
                    self.base_model_name_prefix))

    def post_process(self, async_task):
        printF(name=MasterName.get_master_name(), info="[Function] Enter-> post_process").printf()
        async_task.yields.append(['metadatas', self.metadata_strings])
        async_task.yields.append(['results', self.results])

        pipeline.clear_all_caches()  # cleanup after generation
        pipeline.prepare_text_encoder(async_call=True)


def process_top():
    import ldm_patched.modules.model_management
    ldm_patched.modules.model_management.interrupt_current_processing()


@torch.no_grad()
@torch.inference_mode()
def task_schedule_loop(request_source="api"):
    printF(name=MasterName.get_master_name(), info="[Function] Enter-> task_schedule_loop").printf()
    global worker_queue, queue_task
    task_manager = taskManager(request_source=request_source)
    async_tasks = task_manager.async_tasks

    igp = ImageGenerationParams()

    print(f"worker_queue:{worker_queue.__dict__}")
    worker_queue.add_task(type=adapter.task_queue.TaskType.text_2_img, req_param=igp)
    queue_task = QueueTask(job_id=worker_queue.last_job_id, type=adapter.task_queue.TaskType.text_2_img,
                           req_param=igp,
                           in_queue_millis=int(round(time.time() * 1000)))
    worker_queue.queue.append(queue_task)
    worker_queue.queue[0].start_millis = 0

    if request_source == "api":
        while True:
            time.sleep(5)
            try:
                print(f"{worker_queue.__dict__}")
                if worker_queue is None or len(worker_queue.queue) == 0:
                    time.sleep(0.1)
                    continue

                if not async_tasks:
                    current_task = task_manager.AsyncTask(args=[])  # 原为 AsyncTask 没有括号
                    current_task.last_stop = False
                    current_task.processing = False
                    current_task.yields = []
                    current_task.results = []
                else:
                    current_task = async_tasks.pop(0)
                if worker_queue.queue[0].start_millis == 0:
                    print(
                        f"current_task:{current_task.__dict__} \nworker_queue:{worker_queue.__dict__}")
                    task_manager.process_generate(current_task, wq=worker_queue, qt=queue_task)
            except:
                traceback.print_exc()
                break


@torch.no_grad()
@torch.inference_mode()
def blocking_get_task_result(job_id: str) -> List[ImageGenerationResult]:
    waiting_sleep_steps: int = 0
    waiting_start_time = time.perf_counter()
    while not worker_queue.is_task_finished(job_id):
        if waiting_sleep_steps == 0:
            printF(name=MasterName.get_master_name(),
                   info="[Task Queue] Waiting for task finished, job_id={}".format(job_id)).printf()
        delay = 0.05
        time.sleep(delay)
        waiting_sleep_steps += 1
        if waiting_sleep_steps % int(10 / delay) == 0:
            waiting_time = time.perf_counter() - waiting_start_time
            printF(name=MasterName.get_master_name(),
                   info="[Task Queue] Already waiting for {0} seconds, job_id={1}".format(round(waiting_time, 1),
                                                                                          job_id)).printf()

    task = worker_queue.get_task(job_id, True)
    return task.task_result


if __name__ == "__main__":
    tm = taskManager()
