import torch
import ldm_patched.modules.samplers
import ldm_patched.modules.model_management
from backend.modules.k_model import KModel
from ldm_patched.contrib.external_align_your_steps import AlignYourStepsScheduler
from collections import namedtuple
from ldm_patched.contrib.external_custom_sampler import SDTurboScheduler
from ldm_patched.k_diffusion import sampling as k_diffusion_sampling
from ldm_patched.modules.samplers import normal_scheduler, simple_scheduler, ddim_scheduler
from ldm_patched.modules.model_base import SDXLRefiner, SDXL, ModelSamplingContinuousEDM, ModelSamplingContinuousV, ModelSamplingDiscrete, BaseModel
from ldm_patched.modules.conds import CONDRegular
from ldm_patched.modules.sample import cleanup_additional_models
from ldm_patched.modules.samplers import resolve_areas_and_cond_masks, calculate_start_end_timesteps, \
    create_cond_with_same_area_if_none, pre_run_control, apply_empty_x_to_equal_area, encode_model_conds
from ldm_patched.modules.model_patcher import ModelPatcher
from ldm_patched.modules.samplers import CFGGuider
from util.printf import printF, MasterName

current_refiner = None
refiner_switch_step = -1


@torch.no_grad()
@torch.inference_mode()
def clip_separate_inner(c, p, target_model=None, target_clip=None):
    if target_model is None or isinstance(target_model, SDXLRefiner):
        c = c[..., -1280:].clone()
    elif isinstance(target_model, SDXL):
        c = c.clone()
    else:
        p = None
        c = c[..., :768].clone()

        final_layer_norm = target_clip.cond_stage_model.clip_l.transformer.text_model.final_layer_norm

        final_layer_norm_origin_device = final_layer_norm.weight.device
        final_layer_norm_origin_dtype = final_layer_norm.weight.dtype

        c_origin_device = c.device
        c_origin_dtype = c.dtype

        final_layer_norm.to(device='cpu', dtype=torch.float32)
        c = c.to(device='cpu', dtype=torch.float32)

        c = torch.chunk(c, int(c.size(1)) // 77, 1)
        c = [final_layer_norm(ci) for ci in c]
        c = torch.cat(c, dim=1)

        final_layer_norm.to(device=final_layer_norm_origin_device, dtype=final_layer_norm_origin_dtype)
        c = c.to(device=c_origin_device, dtype=c_origin_dtype)
    return c, p


@torch.no_grad()
@torch.inference_mode()
def clip_separate(cond, target_model=None, target_clip=None):
    results = []

    for c, px in cond:
        p = px.get('pooled_output', None)
        c, p = clip_separate_inner(c, p, target_model=target_model, target_clip=target_clip)
        p = {} if p is None else {'pooled_output': p.clone()}
        results.append([c, p])

    return results


@torch.no_grad()
@torch.inference_mode()
def clip_separate_after_preparation(cond, target_model=None, target_clip=None):
    results = []
    # print(f"cond: {cond}")
    for x in cond:
        # print(f"x: {x}")
        p = x.get('pooled_output', None)
        c = x['model_conds']['c_crossattn'].cond

        c, p = clip_separate_inner(c, p, target_model=target_model, target_clip=target_clip)

        result = {'model_conds': {'c_crossattn': CONDRegular(c)}}

        if p is not None:
            result['pooled_output'] = p.clone()

        results.append(result)
    return results


@torch.no_grad()
@torch.inference_mode()
def sample_hacked(model, noise, positive, negative, cfg, device, sampler, sigmas, model_options={}, latent_image=None, denoise_mask=None, callback=None, disable_pbar=False, seed=None):
    global current_refiner

    cfg_guider = CFGGuider(model)
    cfg_guider.set_conds(positive, negative)
    cfg_guider.set_model_sampler_cfg_function(sampler_cfg_function={"cond": positive, "uncond": negative, "cond_scale": cfg})
    cfg_guider.set_cfg(cfg)

    if current_refiner is not None and hasattr(current_refiner.model, 'extra_conds'):
        # print(f"current_refiner: {current_refiner.model.__dict__}")
        positive_refiner = clip_separate_after_preparation(cfg_guider.original_conds.get("positive"), target_model=current_refiner.model)
        negative_refiner = clip_separate_after_preparation(cfg_guider.original_conds.get("negative"), target_model=current_refiner.model)

        # print(f"current_refiner.model.extra_conds: {current_refiner.model.extra_conds}")
        positive_refiner = encode_model_conds(current_refiner.model.extra_conds, positive_refiner, noise, device, "positive", latent_image=latent_image, denoise_mask=denoise_mask)
        negative_refiner = encode_model_conds(current_refiner.model.extra_conds, negative_refiner, noise, device, "negative", latent_image=latent_image, denoise_mask=denoise_mask)

    def refiner_switch():
        def _match_item(x):
            _new_out = []
            if isinstance(x, list):
                y = x[0]
                for it in y.items():
                    if it[0] == "pooled_output":
                        _new_out.append({it[0]: it[1]})
                    else:
                        _new_out.append({it[0]: it[1]})
            return _new_out

        ldm_patched.modules.sampler_helpers.cleanup_additional_models(set(ldm_patched.modules.sampler_helpers.get_models_from_cond(positive, "control") + ldm_patched.modules.sampler_helpers.get_models_from_cond(negative, "control")))

        new_positive_refiner = _match_item(positive_refiner)
        new_negative_refiner = _match_item(negative_refiner)

        cfg_guider = CFGGuider(current_refiner)
        cfg_guider.set_conds([new_positive_refiner], [new_negative_refiner])

        # clear ip-adapter for refiner
        printF(name=MasterName.get_master_name(), info="Refiner Swapped").printf()
        return

    def callback_wrap(step, x0, x, total_steps):
        if step == refiner_switch_step and current_refiner is not None:
            refiner_switch()
        if callback is not None:
            callback(step, x0, x, total_steps)

    return cfg_guider.sample(noise, latent_image, sampler, sigmas, denoise_mask, callback_wrap, disable_pbar, seed)


@torch.no_grad()
@torch.inference_mode()
def calculate_sigmas_scheduler_hacked(obj, scheduler_name, steps):
    # model_sampling = ModelPatcher(model=model, load_device, offload_device).get_model_object("model_sampling")
    # print(f"model_sampling: {model.model_sampling}")
    if isinstance(obj, SDXL) or isinstance(obj, BaseModel):
        model_sampling = obj.model_sampling
    elif isinstance(obj, KModel):
        model_sampling = obj.predictor
    else:
        model_sampling = obj
    if scheduler_name == "karras":
        sigmas = k_diffusion_sampling.get_sigmas_karras(n=steps, sigma_min=float(model_sampling.sigma_min), sigma_max=float(model_sampling.sigma_max))
    elif scheduler_name == "exponential":
        sigmas = k_diffusion_sampling.get_sigmas_exponential(n=steps, sigma_min=float(model_sampling.sigma_min), sigma_max=float(model_sampling.sigma_max))
    elif scheduler_name == "normal":
        sigmas = normal_scheduler(model_sampling, steps)
    elif scheduler_name == "simple":
        sigmas = simple_scheduler(model_sampling, steps)
    elif scheduler_name == "ddim_uniform":
        sigmas = ddim_scheduler(model_sampling, steps)
    elif scheduler_name == "sgm_uniform":
        sigmas = normal_scheduler(model_sampling, steps, sgm=True)
    elif scheduler_name == "turbo":
        sigmas = SDTurboScheduler().get_sigmas(model=obj, steps=steps, denoise=1.0)[0]
    elif scheduler_name == "align_your_steps":
        model_type = 'SDXL' if isinstance(obj.latent_format, ldm_patched.modules.latent_formats.SDXL) else 'SD1'
        sigmas = AlignYourStepsScheduler().get_sigmas(model_type=model_type, steps=steps, denoise=1.0)[0]
    else:
        raise TypeError("error invalid scheduler")
    return sigmas


ldm_patched.modules.samplers.calculate_sigmas_scheduler = calculate_sigmas_scheduler_hacked
ldm_patched.modules.samplers.sample = sample_hacked
