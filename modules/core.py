import sys
from backend import utils
import backend
from backend.diffusion_engine.flux import Flux
from backend.diffusion_engine.kolors import Kolors

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

class StableDiffusionModel:
    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None, filename=None, vae_filename=None, model_ori=None):
        if isinstance(filename, str):
            is_refiner = isinstance(unet.model, SDXLRefiner)
            if unet is not None:
                unet.model.model_file = dict(filename=filename, prefix='model')
            if clip is not None:
                clip.cond_stage_model.model_file = dict(filename=filename,
                                                        prefix='refiner_clip' if is_refiner else 'base_clip')
            if vae is not None:
                vae.first_stage_model.model_file = dict(filename=filename, prefix='first_stage_model')
        self.unet = unet
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
        if isinstance(self.model_ori, backend.diffusion_engine.base.ForgeDiffusionEngine):
            self.model_ori.set_clip_skip(clip_skip)
        else:
            self.clip.clip_layer(clip_skip)

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
    unet, clip, vae, vae_filename, clip_vision, model_ori = load_checkpoint_guess_config(ckpt_filename, embedding_directory=path_embeddings,model_options={}, te_model_options={},
                                                                model_file_type=model_file_type, vae_filename_param=vae_filename)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision, filename=ckpt_filename, vae_filename=vae_filename, model_ori=model_ori)

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


@torch.no_grad()
@torch.inference_mode()
def decode_vae(vae, latent_image, tiled=False):
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
    if isinstance(model, Flux):
        vae_approx_filename = os.path.join(path_vae, modules.config.default_flux_vae_name)
    elif isinstance(model, Kolors):
        vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth')
    else:
        if isinstance(model.model.latent_format, ldm_patched.modules.latent_formats.SDXL):
            vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth')
        else:
            vae_approx_filename = os.path.join(path_vae_approx, 'vaeapp_sd15.pth')

    if vae_approx_filename in VAE_approx_models:
        VAE_approx_model = VAE_approx_models[vae_approx_filename]
    else:
        if isinstance(model, Flux) or isinstance(model, Kolors):
            # sd = model.forge_objects.clip.patcher.model
            sd = utils.load_torch_file(vae_approx_filename, safe_load=False, device='cpu' if devices.device.type != 'cuda' else None)
            VAE_approx_model = VAEApprox(latent_channels=model.forge_objects.vae.latent_channels)
            VAE_approx_model.load_state_dict(sd, strict=False)
            del sd
            VAE_approx_model.eval()
            VAE_approx_model.to(devices.device, devices.dtype)

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
            # if isinstance(model, Flux):
            #     x_sample = decode_latent_batch(model, x0, target_device=devices.cpu, check_for_nans=True)
            #     x_sample = torch.stack(x_sample).float()
            #     x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            #     x_sample = 255. * np.moveaxis(x0.cpu().numpy(), 0, 2)
            #     x_sample = x_sample.astype(np.uint8)
            # else:
            #     x_sample = x0.to(VAE_approx_model.current_type)
            #     x_sample = VAE_approx_model(x_sample) * 127.5 + 127.5
            #     x_sample = einops.rearrange(x_sample, 'b c h w -> b h w c')[0]
            #     x_sample = x_sample.cpu().numpy().clip(0, 255).astype(np.uint8)

            x_sample = x0.to(VAE_approx_model.current_type)
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
             previewer_start=None, previewer_end=None, sigmas=None, noise_mean=None, disable_preview=False, width=None, height=None):

    latent_image = latent["samples"]

    if sigmas is not None:
        sigmas = sigmas.clone().to(ldm_patched.modules.model_management.get_torch_device())

    if isinstance(model, Flux):
        latent_channels = model.forge_objects.vae.latent_channels
        rng = ImageRNG((latent_channels, height // 8, width // 8), [seed], subseeds=None, subseed_strength=0, seed_resize_from_h=0, seed_resize_from_w=0)
        noise = rng.next()
        latent_image = torch.zeros_like(noise)

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
        y = None
        if previewer is not None and not disable_preview:
            y = previewer(x0, previewer_start + step, previewer_end)
        if callback_function is not None:
            callback_function(previewer_start + step, x0, x, previewer_end, y)

    disable_pbar = False
    modules.sample_hijack.current_refiner = refiner
    modules.sample_hijack.refiner_switch_step = refiner_switch
    ldm_patched.modules.samplers.sample = modules.sample_hijack.sample_hacked

    try:
        samples = ldm_patched.modules.sample.sample(model,
                                                    noise, steps, cfg, sampler_name, scheduler,
                                                    positive, negative, latent_image,
                                                    denoise=denoise, disable_noise=disable_noise,
                                                    start_step=start_step,
                                                    last_step=last_step,
                                                    force_full_denoise=force_full_denoise, noise_mask=noise_mask,
                                                    callback=callback,
                                                    disable_pbar=disable_pbar, seed=seed, sigmas=sigmas)

        out = latent.copy()
        out["samples"] = samples
    finally:
        modules.sample_hijack.current_refiner = None

    return out


@torch.no_grad()
@torch.inference_mode()
def pytorch_to_numpy(x):
    return [np.clip(255. * y.cpu().numpy(), 0, 255).astype(np.uint8) for y in x]


@torch.no_grad()
@torch.inference_mode()
def numpy_to_pytorch(x):
    y = x.astype(np.float32) / 255.0
    y = y[None]
    y = np.ascontiguousarray(y.copy())
    y = torch.from_numpy(y).float()
    return y
