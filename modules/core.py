import sys

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

from ldm_patched.modules.sd import load_checkpoint_guess_config
from ldm_patched.contrib.external import VAEDecode, EmptyLatentImage, VAEEncode, VAEEncodeTiled, VAEDecodeTiled, \
    VAEEncodeForInpaint, \
    ControlNetApplyAdvanced, ConditioningZeroOut, ConditioningAverage, CLIPVisionEncode, unCLIPConditioning, \
    ControlNetApplyAdvanced
from ldm_patched.contrib.external_freelunch import FreeU_V2
from ldm_patched.modules.sample import prepare_mask
from modules.lora import match_lora
from ldm_patched.modules.lora import model_lora_keys_unet, model_lora_keys_clip
from modules.config import path_embeddings
from modules.util import get_file_from_folder_list
from ldm_patched.contrib.external_model_advanced import ModelSamplingDiscrete
from ldm_patched.contrib.external_post_processing import ImageScaleToTotalPixels

opEmptyLatentImage = EmptyLatentImage()
opVAEDecode = VAEDecode()
opVAEEncode = VAEEncode()
opVAEDecodeTiled = VAEDecodeTiled()
opVAEEncodeTiled = VAEEncodeTiled()
opControlNetApplyAdvanced = ControlNetApplyAdvanced()
opFreeU = FreeU_V2()
opModelSamplingDiscrete = ModelSamplingDiscrete()

opConditioningZeroOut = ConditioningZeroOut()
opConditioningAverage = ConditioningAverage()
opCLIPVisionEncode = CLIPVisionEncode()
opImageScaleToTotalPixels = ImageScaleToTotalPixels()


class StableDiffusionModel:
    def __init__(self, unet=None, vae=None, clip=None, clip_vision=None, filename=None):
        self.unet = unet
        self.vae = vae
        self.clip = clip
        self.clip_vision = clip_vision
        self.filename = filename
        self.unet_with_lora = unet
        self.clip_with_lora = clip
        self.visited_loras = ''

        self.lora_key_map_unet = {}
        self.lora_key_map_clip = {}

        if self.unet is not None:
            self.lora_key_map_unet = model_lora_keys_unet(self.unet.model, self.lora_key_map_unet)
            self.lora_key_map_unet.update({x: x for x in self.unet.model.state_dict().keys()})

        if self.clip is not None:
            self.lora_key_map_clip = model_lora_keys_clip(self.clip.cond_stage_model, self.lora_key_map_clip)
            self.lora_key_map_clip.update({x: x for x in self.clip.cond_stage_model.state_dict().keys()})

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

        printF(name=MasterName.get_master_name(), info="Request to load LoRAs {} for model [{}].".format(loras,self.filename)).printf()

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

            if self.unet_with_lora is not None and len(lora_unet) > 0:
                loaded_keys = self.unet_with_lora.add_patches(lora_unet, weight)
                printF(name=MasterName.get_master_name(),
                       info="Loaded LoRA [{}] for UNet [{}] with {} keys at weight {}.".format(
                           lora_filename, self.filename, len(loaded_keys), weight)).printf()
                print(f'')
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
def encode_clip_vision(clip_vision, image):
    return opCLIPVisionEncode.encode(clip_vision=clip_vision, image=image)[0]


@torch.no_grad()
@torch.inference_mode()
def apply_controlnet(positive, negative, control_net, image, strength, start_percent, end_percent):
    return opControlNetApplyAdvanced.apply_controlnet(positive=positive, negative=negative, control_net=control_net,
                                                      image=image, strength=strength, start_percent=start_percent,
                                                      end_percent=end_percent)


@torch.no_grad()
@torch.inference_mode()
def load_model(ckpt_filename, model_file_type=constants.TYPE_NORMAL):
    unet, clip, vae, clip_vision = load_checkpoint_guess_config(ckpt_filename, embedding_directory=path_embeddings,
                                                                model_file_type=model_file_type)
    return StableDiffusionModel(unet=unet, clip=clip, vae=vae, clip_vision=clip_vision, filename=ckpt_filename)


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


class VAEApprox(torch.nn.Module):
    def __init__(self):
        super(VAEApprox, self).__init__()
        self.conv1 = torch.nn.Conv2d(4, 8, (7, 7))
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


VAE_approx_models = {}


@torch.no_grad()
@torch.inference_mode()
def get_previewer(model):
    global VAE_approx_models

    from modules.config import path_vae_approx
    is_sdxl = isinstance(model.model.latent_format, ldm_patched.modules.latent_formats.SDXL)
    vae_approx_filename = os.path.join(path_vae_approx, 'xlvaeapp.pth' if is_sdxl else 'vaeapp_sd15.pth')

    if vae_approx_filename in VAE_approx_models:
        VAE_approx_model = VAE_approx_models[vae_approx_filename]
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
             previewer_start=None, previewer_end=None, sigmas=None, noise_mean=None, disable_preview=False):
    if sigmas is not None:
        sigmas = sigmas.clone().to(ldm_patched.modules.model_management.get_torch_device())

    latent_image = latent["samples"]

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
