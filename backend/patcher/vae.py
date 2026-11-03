# reference: https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/sd.py#L273

import itertools
import math

import torch

from backend import memory_management
from backend.patcher.base import ModelPatcher


@torch.inference_mode()
def tiled_scale_multidim(samples, function, tile=(64, 64), overlap=8, upscale_amount=4, out_channels=3, output_device="cpu", downscale=False, index_formulas=None):
    """https://github.com/comfyanonymous/ComfyUI/blob/v0.3.64/comfy/utils.py#L901"""
    dims = len(tile)

    if not (isinstance(upscale_amount, (tuple, list))):
        upscale_amount = [upscale_amount] * dims

    if not (isinstance(overlap, (tuple, list))):
        overlap = [overlap] * dims

    if index_formulas is None:
        index_formulas = upscale_amount

    if not (isinstance(index_formulas, (tuple, list))):
        index_formulas = [index_formulas] * dims

    def get_upscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale(dim, val):
        up = upscale_amount[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    def get_upscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return up * val

    def get_downscale_pos(dim, val):
        up = index_formulas[dim]
        if callable(up):
            return up(val)
        else:
            return val / up

    if downscale:
        get_scale = get_downscale
        get_pos = get_downscale_pos
    else:
        get_scale = get_upscale
        get_pos = get_upscale_pos

    def mult_list_upscale(a):
        out = []
        for i in range(len(a)):
            out.append(round(get_scale(i, a[i])))
        return out

    output = torch.empty([samples.shape[0], out_channels] + mult_list_upscale(samples.shape[2:]), device=output_device)

    for b in range(samples.shape[0]):
        s = samples[b : b + 1]

        if all(s.shape[d + 2] <= tile[d] for d in range(dims)):
            output[b : b + 1] = function(s).to(output_device)
            continue

        out = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)
        out_div = torch.zeros([s.shape[0], out_channels] + mult_list_upscale(s.shape[2:]), device=output_device)

        positions = [range(0, s.shape[d + 2] - overlap[d], tile[d] - overlap[d]) if s.shape[d + 2] > tile[d] else [0] for d in range(dims)]

        for it in itertools.product(*positions):
            s_in = s
            upscaled = []

            for d in range(dims):
                pos = max(0, min(s.shape[d + 2] - overlap[d], it[d]))
                l = min(tile[d], s.shape[d + 2] - pos)
                s_in = s_in.narrow(d + 2, pos, l)
                upscaled.append(round(get_pos(d, pos)))

            ps = function(s_in).to(output_device)
            mask = torch.ones_like(ps)

            for d in range(2, dims + 2):
                feather = round(get_scale(d - 2, overlap[d - 2]))
                if feather >= mask.shape[d]:
                    continue
                for t in range(feather):
                    a = (t + 1) / feather
                    mask.narrow(d, t, 1).mul_(a)
                    mask.narrow(d, mask.shape[d] - 1 - t, 1).mul_(a)

            o = out
            o_d = out_div
            for d in range(dims):
                o = o.narrow(d + 2, upscaled[d], mask.shape[d + 2])
                o_d = o_d.narrow(d + 2, upscaled[d], mask.shape[d + 2])

            o.add_(ps * mask)
            o_d.add_(mask)

        output[b : b + 1] = out / out_div
    return output


def tiled_scale(samples, function, tile_x=64, tile_y=64, overlap=8, upscale_amount=4, out_channels=3, output_device="cpu"):
    return tiled_scale_multidim(samples, function, (tile_y, tile_x), overlap=overlap, upscale_amount=upscale_amount, out_channels=out_channels, output_device=output_device)


def _weight_shape(module, name):
    """按 'a.b.weight' 逐级取属性，拿不到就返回 None。"""
    try:
        obj = module
        for part in name.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                return None
        return tuple(obj.shape)
    except Exception:
        return None


def _infer_flux2_latent_channels(model, fallback=16):
    """从 AutoencoderKLFlux2 的真实权重推导采样器用的 latent 通道数。

    打包规则：planar(32ch) → packed(128ch)，即 channel = 32 × patch²。
    decode 入口在 preprocess_decode() 之后，网络看到的已是 planar 通道，
    所以只能从 decoder.conv_in 的输入通道 × patch² 反推打包后的通道数。
    post_quant_conv 是 32→32 恒等通道，不能用来推断。
    """
    decoder = getattr(model, "decoder", None)

    shp = _weight_shape(decoder, "conv_in.weight")
    if shp is not None and len(shp) >= 2:
        planar = int(shp[1])
        pack = 1
        ps = getattr(model, "ps", None) or (2, 2)
        try:
            pack = int(ps[0]) * int(ps[1])
        except Exception:
            for p in tuple(getattr(getattr(model, "config", None), "patch_size", (2, 2)) or (2, 2)):
                pack *= int(p)
        return planar * pack

    return int(fallback)


class VAE:
    def __init__(self, model=None, device=None, dtype=None, no_init=False, *, is_wan=False, is_qwen21=False, is_flux2=False, is_flux2_image=False, is_mugen=False):
        if no_init:
            return

        self.output_channels = 3
        self.upscale_index_formula = None
        self.downscale_index_formula = None

        if is_qwen21:
            self.upscale_ratio = 16
            self.downscale_ratio = 16
            self.latent_dim = 2
            self.latent_channels = 64
            self.output_channels = 4
            self.pad_channel_value = 1.0
            self.memory_used_encode = lambda shape, dtype: (600 * shape[2] * shape[3]) * memory_management.dtype_size(dtype)
            self.memory_used_decode = lambda shape, dtype: (900 * shape[2] * shape[3] * (16 * 16)) * memory_management.dtype_size(dtype)

            dtype = dtype or memory_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float16, torch.float32])

        elif not is_wan:
            self.upscale_ratio = 8
            self.downscale_ratio = 8
            self.latent_dim = 2
            _latent_channels = 32 if is_mugen else int(getattr(model.config, 'latent_channels', 16))
            if is_flux2_image:
                # AutoencoderKLFlux2 的「真实 latent 通道」以网络为准：
                #   decoder.conv_in.weight = [C, latent_channels, 3, 3]  (latent_channels = 32)
                #   decoder.post_quant_conv.weight = [..., 128, ...]      (打包后 = 128)
                # config.json 里的 latent_channels 不是 DiT/采样器的通道数，
                # 之前直接取 config 导致 VAE 报告 16ch、DiT 却是 128ch。
                _latent_channels = _infer_flux2_latent_channels(model, fallback=_latent_channels)
            self.latent_channels = _latent_channels
            self.memory_used_encode = lambda shape, dtype: (1767 * shape[2] * shape[3]) * memory_management.dtype_size(dtype)
            self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * memory_management.dtype_size(dtype)

            if is_flux2:
                self.upscale_ratio = 16
                self.downscale_ratio = 16
                self.latent_channels = 128
                self.memory_used_decode = lambda shape, dtype: (2178 * shape[2] * shape[3] * 64) * memory_management.dtype_size(dtype) * 4.0

            if is_flux2_image:
                # FLUX.2 / MOSS-Image 这类「图像」VAE：空间 16× 下采样，
                # 但 latent 通道是 AutoencoderKLFlux2 的 bn 输出（patch_size 2x2）。
                # 必须与 latent_channels 一起从 config 推导，否则采样网格（H/W）
                # 会与 DiT 的 RoPE 训练尺度错位 → 过曝。
                self.upscale_ratio = 16
                self.downscale_ratio = 16
                memory_management.logger.info(
                    "VAE(flux2-image): latent_channels={}, ratio=16".format(_latent_channels)
                )

        else:
            self.upscale_ratio = (lambda a: max(0, a * 4 - 3), 8, 8)
            self.upscale_index_formula = (4, 8, 8)
            self.downscale_ratio = (lambda a: max(0, math.floor((a + 3) / 4)), 8, 8)
            self.downscale_index_formula = (4, 8, 8)
            self.latent_dim = 3
            self.latent_channels = int(model.config.z_dim)  # 16
            self.memory_used_encode = lambda shape, dtype: (1500 if shape[2] <= 4 else 6000) * shape[3] * shape[4] * memory_management.dtype_size(dtype)
            self.memory_used_decode = lambda shape, dtype: (2200 if shape[2] <= 4 else 7000) * shape[3] * shape[4] * (8 * 8) * memory_management.dtype_size(dtype)

            dtype = dtype or memory_management.vae_dtype(allowed_dtypes=[torch.bfloat16, torch.float16, torch.float32])

        self.first_stage_model = model.eval()

        self.device = device or memory_management.vae_device()
        offload_device = memory_management.vae_offload_device()

        self.vae_dtype = dtype or memory_management.vae_dtype()
        self.first_stage_model.to(self.vae_dtype)
        self.output_device = memory_management.intermediate_device()

        self.patcher = ModelPatcher(self.first_stage_model, load_device=self.device, offload_device=offload_device)
        self.is_wan = is_wan

    def clone(self):
        n = VAE(no_init=True)
        n.patcher = self.patcher.clone()
        n.memory_used_encode = self.memory_used_encode
        n.memory_used_decode = self.memory_used_decode
        n.downscale_ratio = self.downscale_ratio
        n.latent_channels = self.latent_channels
        n.first_stage_model = self.first_stage_model
        n.device = self.device
        n.vae_dtype = self.vae_dtype
        n.output_device = self.output_device
        n.is_wan = self.is_wan
        return n

    def decode_tiled_(self, samples, tile_x=64, tile_y=64, overlap=16):
        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
        output = self.process_output((tiled_scale(samples, decode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, output_device=self.output_device) + tiled_scale(samples, decode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, output_device=self.output_device) + tiled_scale(samples, decode_fn, tile_x, tile_y, overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, output_device=self.output_device)) / 3.0)
        return output

    def decode_tiled_3d(self, samples, tile_t=999, tile_x=32, tile_y=32, overlap=(1, 8, 8)):
        decode_fn = lambda a: self.first_stage_model.decode(a.to(self.vae_dtype).to(self.device)).float()
        return self.process_output(tiled_scale_multidim(samples, decode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.upscale_ratio, out_channels=self.output_channels, index_formulas=self.upscale_index_formula, output_device=self.output_device))

    def encode_tiled_(self, pixel_samples, tile_x=512, tile_y=512, overlap=64):
        encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).float()
        samples = tiled_scale(pixel_samples, encode_fn, tile_x, tile_y, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, tile_x * 2, tile_y // 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples += tiled_scale(pixel_samples, encode_fn, tile_x // 2, tile_y * 2, overlap, upscale_amount=(1 / self.downscale_ratio), out_channels=self.latent_channels, output_device=self.output_device)
        samples /= 3.0
        return samples

    def encode_tiled_3d(self, samples, tile_t=9999, tile_x=512, tile_y=512, overlap=(1, 64, 64)):
        encode_fn = lambda a: self.first_stage_model.encode((self.process_input(a)).to(self.vae_dtype).to(self.device)).float()
        return tiled_scale_multidim(samples, encode_fn, tile=(tile_t, tile_x, tile_y), overlap=overlap, upscale_amount=self.downscale_ratio, out_channels=self.latent_channels, downscale=True, index_formulas=self.downscale_index_formula, output_device=self.output_device)

    def decode(self, samples_in: torch.Tensor):
        if memory_management.VAE_ALWAYS_TILED:
            return self.decode_tiled(samples_in).to(self.output_device)

        pixel_samples = None
        _tile = False

        try:
            memory_used = self.memory_used_decode(samples_in.shape, self.vae_dtype)
            memory_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = memory_management.get_free_memory(self.device)
            batch_number = int(free_memory / memory_used)
            batch_number = max(1, batch_number)

            for x in range(0, samples_in.shape[0], batch_number):
                samples = samples_in[x : x + batch_number].to(device=self.device, dtype=self.vae_dtype)
                out = self.process_output(self.first_stage_model.decode(samples).to(device=self.output_device, dtype=torch.float32, copy=True))
                if pixel_samples is None:
                    pixel_samples = torch.empty((samples_in.shape[0],) + tuple(out.shape[1:]), device=self.output_device)
                pixel_samples[x : x + batch_number] = out
        except Exception as e:
            if not memory_management.is_oom(e):
                raise e
            memory_management.logger.warning("Encountered Out of Memory during VAE decoding; Retrying with Tiled VAE Decoding...")
            _tile = True

        if _tile:
            memory_management.soft_empty_cache()
            return self.decode_tiled(samples_in).to(self.output_device)

        pixel_samples = pixel_samples.to(self.output_device).movedim(1, -1)
        return pixel_samples

    def decode_tiled(self, samples: torch.Tensor, tile_x: int = 64, tile_y: int = 64, overlap: int = 16):
        memory_used = self.memory_used_decode(samples.shape, self.vae_dtype)
        memory_management.load_models_gpu([self.patcher], memory_required=memory_used)

        args = {
            "tile_x": tile_x,
            "tile_y": tile_y,
            "overlap": overlap,
        }

        if not self.is_wan:
            output = self.decode_tiled_(samples, **args)
        else:
            args["overlap"] = (1, overlap, overlap)
            output = self.decode_tiled_3d(samples, **args)

        return output.movedim(1, -1)

    def encode(self, pixel_samples: torch.Tensor):
        if memory_management.VAE_ALWAYS_TILED:
            return self.encode_tiled(pixel_samples)

        _samples = pixel_samples.movedim(-1, 1)
        if self.is_wan and _samples.ndim < 5:
            _samples = _samples.movedim(1, 0).unsqueeze(0)

        try:
            memory_used = self.memory_used_encode(_samples.shape, self.vae_dtype)
            memory_management.load_models_gpu([self.patcher], memory_required=memory_used)
            free_memory = memory_management.get_free_memory(self.device)
            batch_number = int(free_memory / max(1, memory_used))
            batch_number = max(1, batch_number)
            samples = None
            for x in range(0, _samples.shape[0], batch_number):
                pixels_in = self.process_input(_samples[x : x + batch_number]).to(self.vae_dtype).to(self.device)
                out = self.first_stage_model.encode(pixels_in).to(self.output_device).float()
                if samples is None:
                    samples = torch.empty((_samples.shape[0],) + tuple(out.shape[1:]), device=self.output_device)
                samples[x : x + batch_number] = out
            _tile = False
        except Exception as e:
            if not memory_management.is_oom(e):
                raise e
            memory_management.logger.warning("Encountered Out of Memory during VAE Encoding; Retrying with Tiled VAE Encoding...")
            _tile = True

        if _tile:
            memory_management.soft_empty_cache()
            return self.encode_tiled(pixel_samples)

        return samples

    def encode_tiled(self, pixel_samples: torch.Tensor, tile_x: int = 512, tile_y: int = 512, overlap: int = 64):
        pixel_samples = pixel_samples.movedim(-1, 1)
        if self.is_wan:
            pixel_samples = pixel_samples.movedim(1, 0).unsqueeze(0)

        memory_used = self.memory_used_encode(pixel_samples.shape, self.vae_dtype)
        memory_management.load_models_gpu([self.patcher], memory_required=memory_used)

        args = {
            "tile_x": tile_x,
            "tile_y": tile_y,
            "overlap": overlap,
        }

        if not self.is_wan:
            return self.encode_tiled_(pixel_samples, **args)

        args["tile_t"] = self.upscale_ratio[0](9999)
        args["overlap"] = (1, overlap, overlap)

        maximum = self.upscale_ratio[0](self.downscale_ratio[0](pixel_samples.shape[2]))
        return self.encode_tiled_3d(pixel_samples[:, :, :maximum], **args)

    @staticmethod
    def process_input(image: torch.Tensor):
        return image * 2.0 - 1.0

    @staticmethod
    def process_output(image: torch.Tensor):
        return image.add_(1.0).div_(2.0).clamp_(0.0, 1.0)
