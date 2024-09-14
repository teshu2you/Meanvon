import copy
import sys

import backend
from backend.diffusion_engine.flux import Flux
import lark
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
from modules.upscaler import model
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


def get_model_file_type(name):
    model_file_type = None
    l_name = name.lower()
    if "lightning" in l_name:
        model_file_type = constants.TYPE_LIGHTNING
    elif "lcm" in l_name:
        model_file_type = constants.TYPE_LCM
    elif "turbo" in l_name:
        model_file_type = constants.TYPE_TURBO
    elif "hunyuandit" in l_name:
        model_file_type = constants.TYPE_HunyuanDiT
    elif "flux" in l_name:
        model_file_type = constants.TYPE_Flux
    else:
        model_file_type = constants.TYPE_NORMAL
    return model_file_type


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

    # model_base = core.StableDiffusionModel()
    vae_filename = [get_file_from_folder_list(modules.config.default_flux_vae_name, modules.config.path_vae), get_file_from_folder_list(modules.config.default_flux_text_encoder_clip, modules.config.path_text_encoder), get_file_from_folder_list(modules.config.default_flux_text_encoder_t5xxl, modules.config.path_text_encoder)]
    model_base = core.load_model(filename, model_file_type=get_model_file_type(name), vae_filename=vae_filename)
    printF(name=MasterName.get_master_name(),
           info="[Warning] Base model loaded: {}".format(model_base.filename)).printf()
    return


def is_base_sdxl():
    assert model_base is not None
    return isinstance(model_base.unet.model, SDXL)


@torch.no_grad()
@torch.inference_mode()
def refresh_refiner_model(name, performance_selection=None):
    printF(name=MasterName.get_master_name(), info="[Function] Enter-> refresh_refiner_model").printf()
    global model_refiner

    # filename = os.path.abspath(os.path.realpath(os.path.join(modules.config.paths_checkpoints[0], name)))

    filename = get_file_from_folder_list(name, modules.config.paths_checkpoints)

    printF(name=MasterName.get_master_name(), info="[Parameters] Refiner Model = {}".format(name)).printf()
    printF(name=MasterName.get_master_name(),
           info="[Parameters] performance_selection = {}".format(performance_selection)).printf()
    if model_refiner.filename == filename:
        printF(name=MasterName.get_master_name(),
               info="model_refiner.filename: {} is same as filename:{}, just return".format(model_refiner.filename,
                                                                                            filename)).printf()
        return

    if "Not Exist" in name:
        name = "None"
        printF(name=MasterName.get_master_name(),
               info="{} is NOT Exist!, changed to None".format(name)).printf()

    if name in ('None', 'none', 'NONE', '') or name is None:
        printF(name=MasterName.get_master_name(), info="[Warning] Refiner unloaded.").printf()
        return

    if constants.TYPE_LIGHTNING in name.lower() and 1 == 2:
        printF(name=MasterName.get_master_name(), info="[Warning] Lightning model, no need to load here!").printf()
        return

    if "sd_xl_base_1.0" in name and performance_selection is not None and constants.TYPE_LIGHTNING in performance_selection.value.lower() and 1 == 2:
        printF(name=MasterName.get_master_name(),
               info="[Warning] Lightning mode, unload #sd_xl_base_1.0# base model!").printf()
        free_cuda_mem()
        return

    model_refiner = core.load_model(filename, model_file_type=get_model_file_type(name))
    printF(name=MasterName.get_master_name(),
           info="[Warning] Refiner model loaded: {}".format(model_refiner.filename)).printf()

    if isinstance(model_refiner.unet.model, SDXL):
        model_refiner.clip = None
        model_refiner.vae = None
    elif isinstance(model_refiner.unet.model, SDXLRefiner):
        model_refiner.clip = None
        model_refiner.vae = None
    else:
        model_refiner.clip = None

    return


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
    cached = clip.fcs_cond_cache.get(text, None)
    if cached is not None:
        if verbose:
            printF(name=MasterName.get_master_name(), info="[CLIP Cached] = {}".format(text)).printf()
        return cached

    tokens = clip.tokenize(text)
    if isinstance(clip.cond_stage_model, HyditModel):
        result = clip.encode_from_tokens(tokens, return_pooled=False, return_dict=True)
    else:
        result = clip.encode_from_tokens(tokens, return_pooled=True)

    clip.fcs_cond_cache[text] = result
    if verbose:
        printF(name=MasterName.get_master_name(), info="[CLIP Encoded] = {}".format(text)).printf()
    return result


@torch.no_grad()
@torch.inference_mode()
def clone_cond(conds):
    results = []

    for c, p in conds:
        p = p["pooled_output"]

        if isinstance(c, torch.Tensor):
            c = c.clone()

        if isinstance(p, torch.Tensor):
            p = p.clone()

        results.append([c, {"pooled_output": p}])

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
                v = v*steps if v<1 else v
                tree.children[-2] = min(steps, int(v))
                if tree.children[-2] >= 1:
                    res.append(tree.children[-2])

            def alternate(self, tree):
                res.extend(range(1, steps+1))

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
    Can also specify width and height of created image - SDXL needs it.
    """
    def __init__(self, prompts, copy_from=None):
        super().__init__()
        self.extend(prompts)

        if copy_from is None:
            copy_from = prompts


@torch.no_grad()
@torch.inference_mode()
def clip_encode(texts, pool_top_k=1, steps=30):
    global final_clip, final_model_ori

    if final_clip is None:
        return None

    if not isinstance(texts, list):
        return None
    if len(texts) == 0:
        return None

    cond_list = []
    pooled_acc = 0
    flag = True

    if isinstance(final_model_ori, Flux):
        result_ces = final_model_ori.get_learned_conditioning(texts, skip_flag=False)
        cond = result_ces.get("crossattn")
        pooled = result_ces.get("vector", 0)

        return [[cond, {"pooled_output": pooled}]]

    else:
        for i, text in enumerate(texts):
            result_ces = clip_encode_single(final_clip, text, verbose=False)
            if isinstance(result_ces, dict):
                cond = result_ces.get("cond")
                pooled = result_ces.get("pooled_output", 0)
                att_mask = result_ces.get("attention_mask")
                cdt_mt5xl = result_ces.get("conditioning_mt5xl")
                att_mask_mt5xl = result_ces.get("attention_mask_mt5xl")
                flag = False
            else:
                cond, pooled = result_ces
                flag = True

            cond_list.append(cond)
            if i < pool_top_k and pooled is not None:
                pooled_acc += pooled

        if flag:
            return [[torch.cat(cond_list, dim=1),{"pooled_output": pooled_acc}]]
        else:
            return [[torch.cat(cond_list, dim=1),
                     {"pooled_output": pooled_acc, "attention_mask": att_mask, "attention_mask_mt5xl": att_mask_mt5xl,
                      "conditioning_mt5xl": cdt_mt5xl}]]

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
    if async_call:
        # TODO: make sure that this is always called in an async way so that users cannot feel it.
        pass
    assert_model_integrity()
    if final_clip is not None:
        if isinstance(final_clip, backend.patcher.vae.ModelPatcher):
            backend.memory_management.load_models_gpu([final_clip.patcher, final_expansion.patcher])
        else:
            ldm_patched.modules.model_management.load_models_gpu([final_clip.patcher, final_expansion.patcher])
    else:
        printF(name=MasterName.get_master_name(), info="[final_clip] = {}".format(final_clip)).printf()
    return


@torch.no_grad()
@torch.inference_mode()
def refresh_everything(refiner_model_name, base_model_name, loras,
                       base_model_additional_loras=None, use_synthetic_refiner=False, performance_selection=None):
    global final_unet, final_clip, final_vae, final_refiner_unet, final_refiner_vae, final_expansion, final_loras, final_model_ori

    final_unet = None
    final_clip = None
    final_vae = None
    final_model_ori = None
    final_refiner_unet = None
    final_refiner_vae = None

    final_loras = loras

    if use_synthetic_refiner and refiner_model_name == 'None':
        printF(name=MasterName.get_master_name(), info="[Warning] Synthetic Refiner Activated").printf()
        refresh_base_model(base_model_name, performance_selection)
        synthesize_refiner_model()
    else:
        refresh_refiner_model(refiner_model_name)
        refresh_base_model(base_model_name, performance_selection)

    refresh_loras(loras, base_model_additional_loras=base_model_additional_loras,
                  performance_selection=performance_selection)
    printF(name=MasterName.get_master_name(), info="[Function] Enter-> assert_model_integrity").printf()
    assert_model_integrity()

    final_unet = model_base.unet_with_lora
    final_clip = model_base.clip_with_lora
    final_model_ori = model_base.model_ori
    final_vae = model_base.vae

    final_refiner_unet = model_refiner.unet_with_lora
    final_refiner_vae = model_refiner.vae

    flag_new_engine = False
    if isinstance(final_unet, backend.patcher.unet.UnetPatcher):
        flag_new_engine = True

    if final_expansion is None:
        final_expansion = FooocusExpansion(flag=flag_new_engine)

    prepare_text_encoder(async_call=True)
    clear_all_caches()
    return


# refresh_everything(
#     refiner_model_name=modules.config.default_refiner_model_name,
#     base_model_name=modules.config.default_base_model_name,
#     loras=get_enabled_loras(modules.config.default_loras)
# )

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
def set_clip_skips(base_clip_skip, refiner_clip_skip):
    global model_base, model_refiner

    model_base.set_clip_skip(base_clip_skip)
    if model_refiner is not None:
        if model_refiner.clip is not None:
            model_refiner.set_clip_skip(refiner_clip_skip)
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
    target_unet, target_vae, target_refiner_unet, target_refiner_vae, target_clip, target_model_ori \
        = final_unet, final_vae, final_refiner_unet, final_refiner_vae, final_clip, final_model_ori

    assert refiner_swap_method in ['joint', 'separate', 'vae']

    if latent is None:
        initial_latent = core.generate_empty_latent(width=width, height=height, batch_size=1)
    else:
        initial_latent = latent

    if isinstance(target_model_ori, Flux):
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
            # Refiner Use Different VAE (then it is SD15)
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
        decoded_latent = core.decode_vae(vae=target_vae, latent_image=sampled_latent, tiled=tiled)

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


